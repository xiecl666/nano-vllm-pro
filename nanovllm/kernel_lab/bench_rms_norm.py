"""Lesson 1 lab: verify + benchmark the hand-written Triton RMSNorm kernel.

Run on the GPU box:

    cd nano-vllm-pro
    python kernel_lab/bench_rms_norm.py

It checks correctness (vs an fp32 reference) and times three implementations:
  * eager PyTorch
  * torch.compile (what the project currently uses)
  * our hand-written Triton kernel
across the token counts you actually hit during decode (bs=1) up to prefill.
"""

import time

import torch

# Make "import nanovllm..." work when run from the repo root or this folder.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.rms_norm_triton import rms_norm_triton  # noqa: E402


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 reference, matches nanovllm.layers.layernorm.RMSNorm.rms_forward."""
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return xf.to(orig_dtype) * weight


@torch.compile
def rms_norm_compiled(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return xf.to(orig_dtype) * weight


def bench(fn, *args, iters: int = 200, warmup: int = 50) -> float:
    """Return per-call latency in microseconds (median-ish via mean of steady state)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    assert torch.cuda.is_available(), "need a GPU"
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    hidden = 1024          # Qwen3-0.6B hidden_size
    eps = 1e-6
    dtype = torch.bfloat16
    weight = torch.randn(hidden, dtype=dtype)

    print(f"hidden={hidden} dtype={dtype}  (eps={eps})")
    print(f"{'n_tok':>6} {'ok':>4} {'max_err':>10} | "
          f"{'eager(us)':>10} {'compile(us)':>12} {'triton(us)':>11} {'speedup':>8}")
    print("-" * 78)

    for n_tokens in [1, 4, 16, 64, 256, 512, 2048, 8192]:
        x = torch.randn(n_tokens, hidden, dtype=dtype)

        y_ref = rms_norm_ref(x, weight, eps)
        y_tri = rms_norm_triton(x, weight, eps)
        max_err = (y_ref.float() - y_tri.float()).abs().max().item()
        # bf16 tolerance: fp32-accumulate vs cast-then-mul weight differ slightly.
        ok = torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2)

        t_eager = bench(rms_norm_ref, x, weight, eps)
        t_comp = bench(rms_norm_compiled, x, weight, eps)
        t_tri = bench(rms_norm_triton, x, weight, eps)
        speedup = t_comp / t_tri

        print(f"{n_tokens:>6} {str(ok):>4} {max_err:>10.3e} | "
              f"{t_eager:>10.2f} {t_comp:>12.2f} {t_tri:>11.2f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
