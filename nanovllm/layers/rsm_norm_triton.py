"""Hand-written Triton kernels for RMSNorm.

This is the first lesson in the "fused kernels" series. The goal is to replace
the `@torch.compile`-based RMSNorm in ``layernorm.py`` with an explicit, fully
controllable Triton kernel that:

  * reads the input row ONCE into on-chip registers,
  * does square / mean / rsqrt / scale / weight-multiply all on-chip,
  * writes the result back ONCE.

Everything below is intentionally verbose and commented for learning purposes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr,              # *Pointer* to input  [n_rows, n_cols]
    w_ptr,              # *Pointer* to weight [n_cols]
    y_ptr,              # *Pointer* to output [n_rows, n_cols]
    x_row_stride,       # how many elements to jump to go to the next row of x
    y_row_stride,       # same for y
    n_cols,             # = hidden_size (real number of valid columns)
    eps,                # RMSNorm epsilon
    BLOCK_SIZE: tl.constexpr,  # compile-time constant, >= n_cols, power of 2
):
    # --- 1) which row am I responsible for? -------------------------------
    # grid = (n_rows,)  ->  each program handles exactly one token's hidden vec
    row = tl.program_id(0)

    # move the base pointers to the start of *this* row
    x_ptr += row * x_row_stride
    y_ptr += row * y_row_stride

    # --- 2) build the column index vector + boundary mask -----------------
    # A single program covers the whole row in one shot because BLOCK_SIZE is
    # chosen >= n_cols. `cols` is a *vector* [0, 1, ..., BLOCK_SIZE-1].
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols  # lanes past the real width must not read/write

    # --- 3) load the row ONCE into registers (accumulate in fp32) ---------
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # --- 4) the whole RMSNorm math, entirely on-chip ----------------------
    # mean of squares -> rstd. `tl.sum` is an on-chip (SRAM) reduction.
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w  # normalize + scale, all in fp32 registers

    # --- 5) write back ONCE (Triton casts fp32 -> y's dtype on store) -----
    tl.store(y_ptr + cols, y, mask=mask)


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Drop-in RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Args:
        x:      [..., hidden_size], any leading shape, fp16/bf16/fp32.
        weight: [hidden_size].
        eps:    epsilon.

    Returns:
        Tensor with the same shape/dtype as ``x``.
    """
    orig_shape = x.shape
    hidden_size = orig_shape[-1]

    # Triton kernels want a contiguous 2D [n_rows, n_cols] view.
    x2d = x.contiguous().view(-1, hidden_size)
    n_rows, n_cols = x2d.shape
    y2d = torch.empty_like(x2d)

    # BLOCK_SIZE must be a power of 2 and cover the whole row so a single
    # program can do the reduction without looping.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # num_warps: more warps = more parallelism per row, but diminishing returns.
    # 4 warps (128 threads) handling up to a few thousand cols is a good default.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    # Launch grid: one program per row/token.
    _rms_norm_fwd_kernel[(n_rows,)](
        x2d, weight, y2d,
        x2d.stride(0), y2d.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y2d.view(orig_shape)
