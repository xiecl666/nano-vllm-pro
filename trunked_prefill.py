#!/usr/bin/env python3
"""
Test script for Chunked Prefill in a nano-vLLM fork.

It does two things:
  1) Correctness test: run the same long prompts with a large prefill budget
     and with a small prefill budget, then compare greedy outputs token-by-token.
  2) Interleave test: let short requests enter decode, then inject one long
     prompt and check whether decode steps continue while the long prompt is
     being prefilling in chunks. This is the key behavior of decode-first
     chunked prefill.

Typical usage:
  python test_chunked_prefill_nanovllm.py \
    --model ~/huggingface/Qwen3-0.6B \
    --mode both \
    --prompt-len 2048 \
    --chunk 512 \
    --max-tokens 32 \
    --max-model-len 4096 \
    --trace-out chunk_trace.jsonl

Notes:
  * The script assumes tensor_parallel_size=1 for the deterministic greedy
    sampler patch.
  * If your fork added Config fields such as enable_chunked_prefill or
    prefill_chunk_size, this script will pass them automatically. If those
    fields are missing, it still forces chunking by setting a small
    max_num_batched_tokens.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, fields
from typing import Any

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


class GreedySampler(torch.nn.Module):
    """Drop-in sampler that ignores temperature and returns argmax tokens."""

    @torch.inference_mode()
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor | None = None):
        return torch.argmax(logits.float(), dim=-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_field_names() -> set[str]:
    return {f.name for f in fields(Config)}


def filter_config_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    names = config_field_names()
    used = {k: v for k, v in kwargs.items() if k in names and v is not None}
    dropped = {k: v for k, v in kwargs.items() if k not in names and v is not None}
    return used, dropped


def build_llm(
    *,
    model: str,
    args: argparse.Namespace,
    max_num_batched_tokens: int,
    chunked: bool,
    label: str,
) -> tuple[LLM, dict[str, Any]]:
    raw_kwargs: dict[str, Any] = {
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "kvcache_block_size": args.block_size,
    }

    # These are used by the chunked-prefill implementation we discussed.
    # They will be ignored on upstream nano-vLLM where Config does not have them.
    if chunked:
        raw_kwargs.update(
            {
                "enable_chunked_prefill": True,
                "enable_mixed_prefill_decode": False,
                "prefill_chunk_size": args.chunk,
                "max_num_partial_prefills": args.max_partial_prefills,
            }
        )
    else:
        raw_kwargs.update(
            {
                "enable_chunked_prefill": False,
                "enable_mixed_prefill_decode": False,
            }
        )

    kwargs, dropped = filter_config_kwargs(raw_kwargs)
    if dropped and args.print_config:
        print(f"[{label}] ignored Config kwargs not present in this nano-vLLM fork: {sorted(dropped)}")
    print(f"[{label}] LLM kwargs: {kwargs}")

    llm = LLM(model, **kwargs)

    if args.greedy_patch:
        if args.tensor_parallel_size != 1:
            raise RuntimeError("--greedy-patch currently expects --tensor-parallel-size 1")
        llm.model_runner.sampler = GreedySampler()
        print(f"[{label}] installed GreedySampler patch for deterministic correctness testing")

    actual_max_model_len = getattr(llm.model_runner.config, "max_model_len", None)
    if actual_max_model_len is not None and args.prompt_len + args.max_tokens > actual_max_model_len:
        raise ValueError(
            f"prompt_len + max_tokens = {args.prompt_len + args.max_tokens} exceeds "
            f"effective max_model_len = {actual_max_model_len}. Increase --max-model-len "
            f"or reduce --prompt-len/--max-tokens."
        )
    return llm, dropped


def shutdown_llm(llm: LLM) -> None:
    """Shut down the engine and unregister its atexit hook when possible."""
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception as exc:
        print(f"[WARN] llm.exit() failed or was already called: {exc}", file=sys.stderr)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_prompt_tokens(tokenizer, target_len: int, salt: int = 0) -> list[int]:
    # Avoid random token IDs so that detokenization/debugging remains sane.
    prefix = f"Request {salt}. "
    body = (
        "This is a chunked prefill scheduling test for nano vLLM. "
        "The prompt is intentionally long and repetitive. "
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    if not body_ids:
        raise RuntimeError("Tokenizer produced empty body_ids")
    if len(prefix_ids) >= target_len:
        return prefix_ids[:target_len]
    need = target_len - len(prefix_ids)
    repeated = (body_ids * ((need + len(body_ids) - 1) // len(body_ids)))[:need]
    return prefix_ids + repeated


def make_prompts(tokenizer, prompt_len: int, num_prompts: int) -> list[list[int]]:
    return [make_prompt_tokens(tokenizer, prompt_len, salt=i) for i in range(num_prompts)]


def make_sampling_params(num_prompts: int, max_tokens: int) -> list[SamplingParams]:
    # temperature value is ignored when --greedy-patch is enabled. Keep it valid
    # for upstream nano-vLLM, whose SamplingParams rejects temperature=0.
    return [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=max_tokens) for _ in range(num_prompts)]


def unpack_schedule_output(out: Any):
    """Support both upstream `(seqs, is_prefill)` and a dataclass-style output."""
    if isinstance(out, tuple) and len(out) == 2:
        seqs, is_prefill = out
        return list(seqs), bool(is_prefill), not bool(is_prefill), "tuple"

    seqs = getattr(out, "seqs", None)
    if seqs is None:
        seqs = getattr(out, "scheduled_seqs", None)
    if seqs is None:
        raise TypeError(f"Unknown scheduler output type: {type(out)!r}")
    seqs = list(seqs)

    has_prefill = getattr(out, "has_prefill", None)
    has_decode = getattr(out, "has_decode", None)
    if has_prefill is None:
        has_prefill = any(bool(getattr(seq, "is_prefill", False)) for seq in seqs)
    if has_decode is None:
        has_decode = any(not bool(getattr(seq, "is_prefill", False)) for seq in seqs)
    return seqs, bool(has_prefill), bool(has_decode), type(out).__name__


class ScheduleTracer:
    def __init__(self, llm: LLM, *, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.rows: list[dict[str, Any]] = []
        self._orig_schedule = llm.scheduler.schedule

        def wrapped_schedule():
            out = self._orig_schedule()
            row = self._make_row(out)
            self.rows.append(row)
            if self.verbose:
                print(format_trace_row(row))
            return out

        llm.scheduler.schedule = wrapped_schedule

    def _make_row(self, out: Any) -> dict[str, Any]:
        seqs, api_has_prefill, api_has_decode, output_type = unpack_schedule_output(out)
        bm = getattr(self.llm.scheduler, "block_manager", None)
        free_blocks = len(getattr(bm, "free_block_ids", [])) if bm is not None else None
        used_blocks = len(getattr(bm, "used_block_ids", [])) if bm is not None else None
        total_blocks = len(getattr(bm, "blocks", [])) if bm is not None else None

        details: list[dict[str, Any]] = []
        total_scheduled = 0
        prefill_tokens = 0
        decode_tokens = 0
        has_prefill_seq = False
        has_decode_seq = False
        has_partial_prefill = False

        for seq in seqs:
            scheduled = int(getattr(seq, "num_scheduled_tokens", 0))
            cached = int(getattr(seq, "num_cached_tokens", 0))
            length = int(len(seq))
            remaining_before = max(length - cached, 0)

            # In upstream nano-vLLM the scheduler output has a global is_prefill.
            # In a mixed-batch implementation, seq.is_prefill is the per-seq state.
            seq_is_prefill = bool(getattr(seq, "is_prefill", api_has_prefill))
            if output_type == "tuple":
                seq_is_prefill = api_has_prefill

            partial = bool(seq_is_prefill and scheduled < remaining_before)
            has_partial_prefill = has_partial_prefill or partial
            has_prefill_seq = has_prefill_seq or seq_is_prefill
            has_decode_seq = has_decode_seq or (not seq_is_prefill)
            total_scheduled += scheduled
            if seq_is_prefill:
                prefill_tokens += scheduled
            else:
                decode_tokens += scheduled

            status = getattr(getattr(seq, "status", None), "name", str(getattr(seq, "status", "")))
            details.append(
                {
                    "seq_id": int(getattr(seq, "seq_id", -1)),
                    "status": status,
                    "is_prefill": seq_is_prefill,
                    "len": length,
                    "num_cached_tokens": cached,
                    "num_scheduled_tokens": scheduled,
                    "remaining_before": remaining_before,
                    "partial_prefill": partial,
                    "num_prompt_tokens": int(getattr(seq, "num_prompt_tokens", -1)),
                    "num_completion_tokens": int(getattr(seq, "num_completion_tokens", -1)),
                    "block_table_len": len(getattr(seq, "block_table", [])),
                }
            )

        return {
            "step": len(self.rows),
            "scheduler_output_type": output_type,
            "api_has_prefill": api_has_prefill,
            "api_has_decode": api_has_decode,
            "has_prefill_seq": has_prefill_seq,
            "has_decode_seq": has_decode_seq,
            "has_mixed_batch": bool(has_prefill_seq and has_decode_seq),
            "has_partial_prefill": has_partial_prefill,
            "num_seqs": len(seqs),
            "total_scheduled_tokens": total_scheduled,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "total_blocks": total_blocks,
            "details": details,
        }


def format_trace_row(row: dict[str, Any]) -> str:
    parts = []
    for d in row["details"]:
        kind = "P" if d["is_prefill"] else "D"
        star = "*" if d["partial_prefill"] else ""
        parts.append(
            f"seq{d['seq_id']}:{kind}{star} "
            f"sched={d['num_scheduled_tokens']}/{d['remaining_before']} "
            f"cached={d['num_cached_tokens']} len={d['len']} blocks={d['block_table_len']}"
        )
    return (
        f"step={row['step']:03d} mixed={row['has_mixed_batch']} "
        f"partial={row['has_partial_prefill']} "
        f"prefill_tok={row['prefill_tokens']} decode_tok={row['decode_tokens']} "
        f"used_blocks={row['used_blocks']} | "
        + " ; ".join(parts)
    )


def summarize_trace(rows: list[dict[str, Any]], block_size: int, prompt_len: int) -> dict[str, Any]:
    partial_rows = [r for r in rows if r["has_partial_prefill"]]
    mixed_rows = [r for r in rows if r["has_mixed_batch"]]
    decode_rows = [r for r in rows if r["decode_tokens"] > 0]
    prefill_rows = [r for r in rows if r["prefill_tokens"] > 0]

    first_partial_detail = None
    for row in partial_rows:
        for d in row["details"]:
            if d["partial_prefill"]:
                first_partial_detail = d
                break
        if first_partial_detail is not None:
            break

    incremental_kv_ok = None
    first_partial_block_table_len = None
    expected_blocks_until_chunk_end = None
    full_prompt_blocks = math.ceil(prompt_len / block_size)
    if first_partial_detail is not None:
        first_partial_block_table_len = first_partial_detail["block_table_len"]
        chunk_end = first_partial_detail["num_cached_tokens"] + first_partial_detail["num_scheduled_tokens"]
        expected_blocks_until_chunk_end = math.ceil(chunk_end / block_size)
        # Allow one extra block for off-by-one append/block-boundary implementations.
        incremental_kv_ok = first_partial_block_table_len <= expected_blocks_until_chunk_end + 1

    decode_steps = [r["step"] for r in decode_rows]
    max_decode_gap = None
    if len(decode_steps) >= 2:
        max_decode_gap = max(b - a for a, b in zip(decode_steps, decode_steps[1:]))

    return {
        "num_steps": len(rows),
        "num_prefill_steps": len(prefill_rows),
        "num_decode_steps": len(decode_rows),
        "num_partial_prefill_steps": len(partial_rows),
        "num_mixed_steps": len(mixed_rows),
        "max_decode_gap_steps": max_decode_gap,
        "first_partial_block_table_len": first_partial_block_table_len,
        "expected_blocks_until_first_chunk_end": expected_blocks_until_chunk_end,
        "full_prompt_blocks": full_prompt_blocks,
        "incremental_kv_looks_ok": incremental_kv_ok,
        "prefill_token_chunks": [r["prefill_tokens"] for r in prefill_rows[:20]],
    }


def save_trace(rows: list[dict[str, Any]], path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.endswith(".csv"):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "scheduler_output_type",
                    "api_has_prefill",
                    "api_has_decode",
                    "has_prefill_seq",
                    "has_decode_seq",
                    "has_mixed_batch",
                    "has_partial_prefill",
                    "num_seqs",
                    "total_scheduled_tokens",
                    "prefill_tokens",
                    "decode_tokens",
                    "free_blocks",
                    "used_blocks",
                    "total_blocks",
                    "details",
                ],
            )
            writer.writeheader()
            for r in rows:
                rr = dict(r)
                rr["details"] = json.dumps(rr["details"], ensure_ascii=False)
                writer.writerow(rr)
    else:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[trace] wrote {len(rows)} rows to {path}")


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} trace summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


def run_generate_case(
    *,
    label: str,
    model: str,
    args: argparse.Namespace,
    prompts: list[list[int]],
    sampling_params: list[SamplingParams],
    budget: int,
    chunked: bool,
) -> dict[str, Any]:
    set_seed(args.seed)
    llm, dropped = build_llm(
        model=model,
        args=args,
        max_num_batched_tokens=budget,
        chunked=chunked,
        label=label,
    )
    tracer = ScheduleTracer(llm, verbose=args.verbose_schedule)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=args.tqdm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    rows = tracer.rows
    shutdown_llm(llm)

    total_out_tokens = sum(len(o["token_ids"]) for o in outputs)
    return {
        "label": label,
        "outputs": outputs,
        "trace": rows,
        "summary": summarize_trace(rows, args.block_size, args.prompt_len),
        "elapsed_sec": elapsed,
        "total_output_tokens": total_out_tokens,
        "output_tok_per_sec": total_out_tokens / elapsed if elapsed > 0 else float("inf"),
        "peak_memory_bytes": peak_mem,
        "dropped_config_kwargs": dropped,
    }


def compare_outputs(ref: list[dict[str, Any]], test: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(ref) != len(test):
        return False, [f"output count differs: {len(ref)} vs {len(test)}"]
    for i, (a, b) in enumerate(zip(ref, test)):
        ta = list(a.get("token_ids", []))
        tb = list(b.get("token_ids", []))
        if ta != tb:
            first_diff = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            if first_diff is None:
                first_diff = min(len(ta), len(tb))
            errors.append(
                f"request {i}: token_ids differ at generated offset {first_diff}; "
                f"len(ref)={len(ta)}, len(test)={len(tb)}, "
                f"ref_token={ta[first_diff] if first_diff < len(ta) else None}, "
                f"test_token={tb[first_diff] if first_diff < len(tb) else None}"
            )
    return not errors, errors


def run_correctness(args: argparse.Namespace) -> bool:
    print("\n######## correctness test: baseline vs chunked ########")
    if args.chunk >= args.prompt_len:
        print(f"[WARN] --chunk {args.chunk} >= --prompt-len {args.prompt_len}; chunking may not happen")

    # Use a temporary tokenizer from the first LLM; this keeps the script independent
    # of model-specific tokenizer classes.
    tokenizer_probe, _ = build_llm(
        model=args.model,
        args=args,
        max_num_batched_tokens=max(args.chunk, 1),
        chunked=False,
        label="tokenizer-probe",
    )
    prompts = make_prompts(tokenizer_probe.tokenizer, args.prompt_len, args.num_prompts)
    shutdown_llm(tokenizer_probe)

    sampling_params = make_sampling_params(args.num_prompts, args.max_tokens)
    baseline_budget = args.baseline_budget or max(args.prompt_len + 16, args.chunk * 4)
    baseline_budget = max(baseline_budget, args.prompt_len)

    baseline = run_generate_case(
        label="baseline-no-chunk",
        model=args.model,
        args=args,
        prompts=prompts,
        sampling_params=sampling_params,
        budget=baseline_budget,
        chunked=False,
    )
    chunked = run_generate_case(
        label="chunked",
        model=args.model,
        args=args,
        prompts=prompts,
        sampling_params=sampling_params,
        budget=args.chunk,
        chunked=True,
    )

    print_summary("baseline", baseline["summary"])
    print(f"baseline elapsed_sec: {baseline['elapsed_sec']:.4f}")
    print(f"baseline output_tok_per_sec: {baseline['output_tok_per_sec']:.2f}")
    if baseline["peak_memory_bytes"] is not None:
        print(f"baseline peak_memory_gb: {baseline['peak_memory_bytes'] / 1e9:.3f}")

    print_summary("chunked", chunked["summary"])
    print(f"chunked elapsed_sec: {chunked['elapsed_sec']:.4f}")
    print(f"chunked output_tok_per_sec: {chunked['output_tok_per_sec']:.2f}")
    if chunked["peak_memory_bytes"] is not None:
        print(f"chunked peak_memory_gb: {chunked['peak_memory_bytes'] / 1e9:.3f}")

    ok, errors = compare_outputs(baseline["outputs"], chunked["outputs"])
    if ok:
        print("\n[PASS] baseline and chunked generated token_ids are identical under GreedySampler")
    else:
        print("\n[FAIL] baseline and chunked outputs differ")
        for e in errors[:10]:
            print("  -", e)

    if chunked["summary"]["num_partial_prefill_steps"] <= 0:
        print("[FAIL] no partial prefill step was observed; reduce --chunk or increase --prompt-len")
        ok = False
    else:
        print("[PASS] observed partial prefill steps")

    incr = chunked["summary"]["incremental_kv_looks_ok"]
    if incr is False:
        msg = (
            "[WARN] first partial prefill already has many blocks allocated: "
            f"block_table_len={chunked['summary']['first_partial_block_table_len']}, "
            f"expected_until_chunk_end≈{chunked['summary']['expected_blocks_until_first_chunk_end']}, "
            f"full_prompt_blocks={chunked['summary']['full_prompt_blocks']}. "
            "This usually means KV blocks are still allocated for the whole prompt, "
            "not incrementally per chunk."
        )
        if args.require_incremental_kv:
            print(msg.replace("[WARN]", "[FAIL]"))
            ok = False
        else:
            print(msg)
    elif incr is True:
        print("[PASS] first partial prefill looks like incremental KV allocation")

    if args.trace_out:
        root, ext = os.path.splitext(args.trace_out)
        save_trace(baseline["trace"], f"{root}.baseline{ext or '.jsonl'}")
        save_trace(chunked["trace"], f"{root}.chunked{ext or '.jsonl'}")

    return ok


def run_until_no_waiting(llm: LLM, max_steps: int = 100) -> None:
    for _ in range(max_steps):
        if not getattr(llm.scheduler, "waiting"):
            return
        llm.step()
    raise RuntimeError("waiting queue did not drain during short-request warmup")


def run_interleave(args: argparse.Namespace) -> bool:
    print("\n######## interleave test: decode-first mixed batch ########")
    set_seed(args.seed)
    llm, _ = build_llm(
        model=args.model,
        args=args,
        max_num_batched_tokens=args.chunk,
        chunked=True,
        label="interleave",
    )
    tracer = ScheduleTracer(llm, verbose=args.verbose_schedule)

    short_prompts = [make_prompt_tokens(llm.tokenizer, args.short_prompt_len, salt=1000 + i) for i in range(args.num_short)]
    short_sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.short_max_tokens)
    for p in short_prompts:
        llm.add_request(p, short_sp)

    # Let the short requests finish prefill and enter decode. Do not finish them.
    run_until_no_waiting(llm, max_steps=100)
    rows_before_long = len(tracer.rows)
    running_before = len(getattr(llm.scheduler, "running", []))
    print(f"short requests are now running: running={running_before}, trace_steps={rows_before_long}")
    if running_before == 0:
        print("[FAIL] short requests finished too early; increase --short-max-tokens")
        shutdown_llm(llm)
        return False

    long_prompt = make_prompt_tokens(llm.tokenizer, args.prompt_len, salt=9999)
    long_sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.long_max_tokens)
    llm.add_request(long_prompt, long_sp)
    print(f"added one long request: prompt_len={len(long_prompt)}, chunk={args.chunk}")

    outputs: dict[int, list[int]] = {}
    step_limit = args.interleave_steps
    for _ in range(step_limit):
        if llm.is_finished():
            break
        outs, _ = llm.step()
        for seq_id, token_ids in outs:
            outputs[seq_id] = token_ids

    inter_rows = tracer.rows[rows_before_long:]
    summary = summarize_trace(inter_rows, args.block_size, args.prompt_len)
    print_summary("interleave-after-long-added", summary)

    # Print the first few rows to make scheduling behavior obvious.
    print("\nfirst scheduling rows after long request was added:")
    for r in inter_rows[: min(len(inter_rows), args.print_rows)]:
        print(format_trace_row(r))

    ok = True
    if summary["num_partial_prefill_steps"] <= 0:
        print("[FAIL] no partial prefill observed after adding long request")
        ok = False
    else:
        print("[PASS] partial prefill observed after adding long request")

    if summary["num_decode_steps"] <= 0:
        print("[FAIL] no decode step happened while/after long request was being prefilling")
        ok = False
    else:
        print("[PASS] decode steps were observed after adding long request")

    if summary["num_mixed_steps"] <= 0:
        msg = (
            "[WARN] no mixed prefill+decode batch observed. "
            "Upstream nano-vLLM's scheduler is prefill-first, so this is expected there. "
            "A decode-first chunked-prefill implementation should normally show mixed=True rows."
        )
        if args.require_mixed:
            print(msg.replace("[WARN]", "[FAIL]"))
            ok = False
        else:
            print(msg)
    else:
        print("[PASS] mixed prefill/decode batches observed")

    if args.trace_out:
        root, ext = os.path.splitext(args.trace_out)
        save_trace(inter_rows, f"{root}.interleave{ext or '.jsonl'}")

    shutdown_llm(llm)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Chunked Prefill behavior in nano-vLLM.")
    parser.add_argument("--model", required=True, help="Local HF model path, e.g. ~/huggingface/Qwen3-0.6B")
    parser.add_argument("--mode", choices=["correctness", "interleave", "benchmark", "both", "all"], default="both")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=512, help="Small max_num_batched_tokens / prefill chunk budget")
    parser.add_argument("--baseline-budget", type=int, default=0, help="Large prefill budget for non-chunk baseline; 0 = auto")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-partial-prefills", type=int, default=1)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-patch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose-schedule", action="store_true")
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-out", default="", help="Write trace as .jsonl or .csv. For mode=both, suffixes are added.")

    # Interleave-specific knobs.
    parser.add_argument("--num-short", type=int, default=4)
    parser.add_argument("--short-prompt-len", type=int, default=64)
    parser.add_argument("--short-max-tokens", type=int, default=64)
    parser.add_argument("--long-max-tokens", type=int, default=8)
    parser.add_argument("--interleave-steps", type=int, default=200)
    parser.add_argument("--print-rows", type=int, default=20)

    # Benchmark-specific knobs.
    parser.add_argument("--bench-num-prompts", type=int, default=16, help="Number of prompts for benchmark")
    parser.add_argument("--bench-prompt-len", type=int, default=2048, help="Prompt length for benchmark")
    parser.add_argument("--bench-max-tokens", type=int, default=128, help="Max output tokens for benchmark")
    parser.add_argument("--bench-warmup", type=int, default=1, help="Number of warmup runs before timed run")
    parser.add_argument("--bench-rounds", type=int, default=3, help="Number of timed rounds for statistics")
    # Staggered scenario knobs.
    parser.add_argument("--bench-num-short", type=int, default=8, help="Staggered: number of short requests in batch A")
    parser.add_argument("--bench-short-prompt-len", type=int, default=128, help="Staggered: short prompt length")
    parser.add_argument("--bench-short-max-tokens", type=int, default=128, help="Staggered: short request max output tokens")
    parser.add_argument("--bench-num-long", type=int, default=4, help="Staggered: number of long requests in batch B")
    parser.add_argument("--bench-long-prompt-len", type=int, default=2048, help="Staggered: long prompt length")
    parser.add_argument("--bench-long-max-tokens", type=int, default=64, help="Staggered: long request max output tokens")

    # Make the script useful both as a diagnostic and as a CI-style test.
    parser.add_argument("--require-mixed", action="store_true")
    parser.add_argument("--require-incremental-kv", action="store_true")
    return parser.parse_args()


# ── Benchmark helpers ──────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Per-step timing and token counts collected during a benchmark run."""
    step: int
    wall_sec: float
    has_prefill: bool
    has_decode: bool
    num_prefill_tokens: int
    num_decode_tokens: int
    num_seqs: int
    finished: int  # number of requests that finished this step


@dataclass
class BenchResult:
    label: str
    wall_total: float
    total_prefill_tokens: int
    total_decode_tokens: int
    total_output_tokens: int
    num_steps: int
    prefill_steps: int
    decode_steps: int
    mixed_steps: int
    prefill_wall: float
    decode_wall: float
    prefill_tok_per_sec: float
    decode_tok_per_sec: float
    output_tok_per_sec: float
    ttft_per_request: list[float]  # seconds from t0 to first output token for each request
    peak_memory_bytes: int | None
    step_records: list[StepRecord]


def _run_bench_pass(
    *,
    label: str,
    model: str,
    args: argparse.Namespace,
    prompts: list[list[int]],
    max_tokens: int,
    chunked: bool,
    budget: int,
) -> BenchResult:
    set_seed(args.seed)
    llm, _ = build_llm(
        model=model,
        args=args,
        max_num_batched_tokens=budget,
        chunked=chunked,
        label=label,
    )
    sp_list = make_sampling_params(len(prompts), max_tokens)
    ordered_seq_ids: list[int] = []
    for prompt, sp in zip(prompts, sp_list):
        llm.add_request(prompt, sp)
    # Record actual seq_ids in submission order (waiting queue preserves insertion order)
    ordered_seq_ids = [seq.seq_id for seq in llm.scheduler.waiting]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Track TTFT: seq_id -> time of first output token
    ttft_map: dict[int, float] = {}
    records: list[StepRecord] = []
    t0 = time.perf_counter()
    step_idx = 0

    while not llm.is_finished():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_t0 = time.perf_counter()
        outputs, sched_out = llm.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_wall = time.perf_counter() - step_t0

        n_prefill = getattr(sched_out, "num_prefill_tokens", 0)
        n_decode = getattr(sched_out, "num_decode_tokens", 0)
        has_p = getattr(sched_out, "has_prefill", False)
        has_d = getattr(sched_out, "has_decode", False)
        now = time.perf_counter() - t0

        # Track TTFT: record when each request first produces a completion token
        # (num_completion_tokens >= 1 after postprocess means first output token is ready).
        for seq in sched_out.seqs:
            sid = getattr(seq, "seq_id", -1)
            if sid not in ttft_map and seq.num_completion_tokens >= 1:
                ttft_map[sid] = now

        records.append(StepRecord(
            step=step_idx,
            wall_sec=step_wall,
            has_prefill=has_p,
            has_decode=has_d,
            num_prefill_tokens=n_prefill,
            num_decode_tokens=n_decode,
            num_seqs=len(sched_out.seqs),
            finished=len(outputs),
        ))
        step_idx += 1

    total_wall = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    shutdown_llm(llm)

    total_prefill = sum(r.num_prefill_tokens for r in records)
    total_decode = sum(r.num_decode_tokens for r in records)
    prefill_steps = sum(1 for r in records if r.has_prefill and not r.has_decode)
    decode_steps = sum(1 for r in records if r.has_decode and not r.has_prefill)
    mixed_steps = sum(1 for r in records if r.has_prefill and r.has_decode)
    prefill_wall = sum(r.wall_sec for r in records if r.has_prefill and not r.has_decode)
    decode_wall = sum(r.wall_sec for r in records if r.has_decode and not r.has_prefill)
    mixed_wall = sum(r.wall_sec for r in records if r.has_prefill and r.has_decode)
    prefill_wall += mixed_wall  # mixed steps also do prefill work
    decode_wall += mixed_wall  # mixed steps also do decode work

    # TTFT: sorted by submission order using actual seq_ids
    ttft_list = [ttft_map.get(sid, total_wall) for sid in ordered_seq_ids]

    return BenchResult(
        label=label,
        wall_total=total_wall,
        total_prefill_tokens=total_prefill,
        total_decode_tokens=total_decode,
        total_output_tokens=total_decode,
        num_steps=len(records),
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        mixed_steps=mixed_steps,
        prefill_wall=prefill_wall,
        decode_wall=decode_wall,
        prefill_tok_per_sec=total_prefill / prefill_wall if prefill_wall > 0 else 0,
        decode_tok_per_sec=total_decode / decode_wall if decode_wall > 0 else 0,
        output_tok_per_sec=total_decode / total_wall if total_wall > 0 else 0,
        ttft_per_request=ttft_list,
        peak_memory_bytes=peak_mem,
        step_records=records,
    )


def _print_bench_result(res: BenchResult) -> None:
    print(f"\n{'='*60}")
    print(f"  {res.label}")
    print(f"{'='*60}")
    print(f"  Total wall time:       {res.wall_total:.3f} s")
    print(f"  Total steps:           {res.num_steps}")
    print(f"    Prefill-only steps:  {res.prefill_steps}")
    print(f"    Decode-only steps:   {res.decode_steps}")
    print(f"    Mixed steps:         {res.mixed_steps}")
    print(f"  Prefill tokens:        {res.total_prefill_tokens}")
    print(f"  Decode (output) tokens:{res.total_decode_tokens}")
    print(f"  Prefill throughput:    {res.prefill_tok_per_sec:.1f} tok/s")
    print(f"  Decode throughput:     {res.decode_tok_per_sec:.1f} tok/s")
    print(f"  Output tok/s (total):  {res.output_tok_per_sec:.1f} tok/s")
    if res.peak_memory_bytes is not None:
        print(f"  Peak GPU memory:       {res.peak_memory_bytes / 1e9:.3f} GB")
    if res.ttft_per_request:
        ttft = res.ttft_per_request
        print(f"  TTFT (first token):")
        print(f"    min={min(ttft):.4f}s  median={statistics.median(ttft):.4f}s  "
              f"max={max(ttft):.4f}s  mean={statistics.mean(ttft):.4f}s")


def _print_comparison(baseline: BenchResult, chunked: BenchResult) -> None:
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE COMPARISON: baseline vs chunked")
    print(f"{'='*60}")

    def _pct(a, b):
        if b == 0:
            return "n/a"
        diff = (a - b) / b * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"

    print(f"  {'Metric':<28} {'Baseline':>14} {'Chunked':>14} {'Diff':>10}")
    print(f"  {'-'*66}")
    rows = [
        ("Wall time (s)", baseline.wall_total, chunked.wall_total, False),
        ("Output tok/s", baseline.output_tok_per_sec, chunked.output_tok_per_sec, True),
        ("Prefill tok/s", baseline.prefill_tok_per_sec, chunked.prefill_tok_per_sec, True),
        ("Decode tok/s", baseline.decode_tok_per_sec, chunked.decode_tok_per_sec, True),
        ("TTFT median (s)", statistics.median(baseline.ttft_per_request) if baseline.ttft_per_request else 0,
         statistics.median(chunked.ttft_per_request) if chunked.ttft_per_request else 0, False),
        ("TTFT max (s)", max(baseline.ttft_per_request) if baseline.ttft_per_request else 0,
         max(chunked.ttft_per_request) if chunked.ttft_per_request else 0, False),
        ("Total steps", float(baseline.num_steps), float(chunked.num_steps), False),
        ("Mixed steps", float(baseline.mixed_steps), float(chunked.mixed_steps), True),
    ]
    if baseline.peak_memory_bytes is not None and chunked.peak_memory_bytes is not None:
        rows.append(("Peak mem (GB)",
                      baseline.peak_memory_bytes / 1e9,
                      chunked.peak_memory_bytes / 1e9, False))

    for name, bv, cv, higher_better in rows:
        diff = _pct(cv, bv)
        print(f"  {name:<28} {bv:>14.3f} {cv:>14.3f} {diff:>10}")

    # Interpretation
    print(f"\n  Interpretation:")
    if chunked.output_tok_per_sec > 0 and baseline.output_tok_per_sec > 0:
        ratio = chunked.output_tok_per_sec / baseline.output_tok_per_sec
        if ratio > 1.05:
            print(f"    [WIN] Chunked prefill output throughput is {ratio:.2f}x higher")
        elif ratio < 0.95:
            print(f"    [REG] Chunked prefill output throughput is {ratio:.2f}x lower")
        else:
            print(f"    [TIE] Output throughput is similar ({ratio:.2f}x)")

    if baseline.ttft_per_request and chunked.ttft_per_request:
        b_med = statistics.median(baseline.ttft_per_request)
        c_med = statistics.median(chunked.ttft_per_request)
        if b_med > 0:
            ratio = c_med / b_med
            if ratio < 0.95:
                print(f"    [WIN] Chunked prefill TTFT median is {ratio:.2f}x lower (faster)")
            elif ratio > 1.05:
                print(f"    [REG] Chunked prefill TTFT median is {ratio:.2f}x higher (slower)")
            else:
                print(f"    [TIE] TTFT median is similar ({ratio:.2f}x)")

    if chunked.mixed_steps > 0 and baseline.mixed_steps == 0:
        print(f"    [WIN] Chunked prefill enables mixed prefill+decode batching "
              f"({chunked.mixed_steps} mixed steps)")


def run_benchmark(args: argparse.Namespace) -> bool:
    print("\n######## performance benchmark: baseline vs chunked ########")

    # ── Scenario 1: Batch (all requests arrive at once) ──
    _run_batch_benchmark(args)

    # ── Scenario 2: Staggered arrival (short first, then long) ──
    _run_staggered_benchmark(args)

    return True


def _run_batch_benchmark(args: argparse.Namespace) -> None:
    """Batch scenario: all requests arrive at once. Chunked prefill usually loses here."""
    print("\n── Scenario 1: Batch arrival (all requests at once) ──")
    tokenizer_probe, _ = build_llm(
        model=args.model,
        args=args,
        max_num_batched_tokens=max(args.bench_prompt_len, 1),
        chunked=False,
        label="bench-tokenizer",
    )
    prompts = make_prompts(tokenizer_probe.tokenizer, args.bench_prompt_len, args.bench_num_prompts)
    shutdown_llm(tokenizer_probe)
    print(f"  {args.bench_num_prompts} prompts x {args.bench_prompt_len} tokens, "
          f"max_tokens={args.bench_max_tokens}, chunk={args.chunk}, rounds={args.bench_rounds}")

    baseline_budget = args.baseline_budget or max(args.bench_prompt_len + 16, args.chunk * 4)
    baseline_budget = max(baseline_budget, args.bench_prompt_len)

    # Warmup
    for w in range(args.bench_warmup):
        print(f"\n  --- warmup {w+1}/{args.bench_warmup} ---")
        _run_bench_pass(label="warmup-baseline", model=args.model, args=args,
                        prompts=prompts, max_tokens=args.bench_max_tokens,
                        chunked=False, budget=baseline_budget)
        _run_bench_pass(label="warmup-chunked", model=args.model, args=args,
                        prompts=prompts, max_tokens=args.bench_max_tokens,
                        chunked=True, budget=args.chunk)

    # Timed rounds
    baseline_results: list[BenchResult] = []
    chunked_results: list[BenchResult] = []
    for r in range(args.bench_rounds):
        print(f"\n  --- round {r+1}/{args.bench_rounds} ---")
        br = _run_bench_pass(label=f"baseline-r{r}", model=args.model, args=args,
                             prompts=prompts, max_tokens=args.bench_max_tokens,
                             chunked=False, budget=baseline_budget)
        _print_bench_result(br)
        baseline_results.append(br)
        cr = _run_bench_pass(label=f"chunked-r{r}", model=args.model, args=args,
                             prompts=prompts, max_tokens=args.bench_max_tokens,
                             chunked=True, budget=args.chunk)
        _print_bench_result(cr)
        chunked_results.append(cr)

    baseline_agg = _aggregate_results(baseline_results, "baseline-no-chunk")
    chunked_agg = _aggregate_results(chunked_results, "chunked")
    _print_comparison(baseline_agg, chunked_agg)


@dataclass
class StaggeredResult:
    label: str
    wall_total: float
    t0: float  # time origin
    inject_time: float  # when batch B was injected (relative to t0)
    # Batch A (short, arrived at t=0)
    batch_a_ttft: list[float]  # TTFT for each batch A request
    batch_a_finish_time: list[float]  # finish time for each batch A request
    # Batch B (long, arrived after batch A was in decode)
    batch_b_ttft: list[float]  # TTFT for each batch B request (relative to inject_time)
    batch_b_finish_time: list[float]  # finish time for each batch B request (relative to inject_time)
    # Decode stall: time between batch B injection and batch A's next decode token
    decode_stall_sec: float
    # Step-level stats after injection
    post_inject_steps: int
    post_inject_mixed_steps: int
    post_inject_prefill_steps: int
    post_inject_decode_steps: int
    peak_memory_bytes: int | None
    total_output_tokens: int


def _run_staggered_pass(
    *,
    label: str,
    model: str,
    args: argparse.Namespace,
    tokenizer,
    chunked: bool,
    budget: int,
) -> StaggeredResult:
    """Staggered arrival: short requests first, then long requests after they enter decode."""
    set_seed(args.seed)
    llm, _ = build_llm(model=model, args=args, max_num_batched_tokens=budget,
                       chunked=chunked, label=label)

    # Phase 1: submit short requests
    short_prompts = [make_prompt_tokens(tokenizer, args.bench_short_prompt_len, salt=2000 + i)
                     for i in range(args.bench_num_short)]
    short_sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.bench_short_max_tokens)
    for p in short_prompts:
        llm.add_request(p, short_sp)
    # Record batch A seq_ids in submission order
    batch_a_seq_ids = [seq.seq_id for seq in llm.scheduler.waiting]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    # Run until short requests finish prefill and enter decode
    for _ in range(50):
        if not getattr(llm.scheduler, "waiting", []):
            break
        llm.step()

    # A few more decode steps to ensure batch A is stable in decode
    for _ in range(5):
        if llm.is_finished():
            break
        llm.step()

    inject_time = time.perf_counter() - t0
    print(f"    [{label}] injected {args.bench_num_long} long requests at t={inject_time:.4f}s")

    # Phase 2: inject long requests
    long_prompts = [make_prompt_tokens(tokenizer, args.bench_long_prompt_len, salt=5000 + i)
                    for i in range(args.bench_num_long)]
    long_sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.bench_long_max_tokens)
    for p in long_prompts:
        llm.add_request(p, long_sp)
    # Record batch B seq_ids
    batch_b_seq_ids = [seq.seq_id for seq in llm.scheduler.waiting if seq.seq_id not in set(batch_a_seq_ids)]

    # Track TTFT and finish times
    ttft_map: dict[int, float] = {}
    finish_map: dict[int, float] = {}
    a_sid_set = set(batch_a_seq_ids)
    post_inject_steps = 0
    post_inject_mixed = 0
    post_inject_prefill = 0
    post_inject_decode = 0
    total_decode_tokens = 0
    # Decode stall: track time from injection to first post-inject decode step
    first_decode_after_inject: float | None = None

    while not llm.is_finished():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs, sched_out = llm.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter() - t0
        post_inject_steps += 1
        has_p = getattr(sched_out, "has_prefill", False)
        has_d = getattr(sched_out, "has_decode", False)
        if has_p and has_d:
            post_inject_mixed += 1
        elif has_p:
            post_inject_prefill += 1
        elif has_d:
            post_inject_decode += 1
        total_decode_tokens += getattr(sched_out, "num_decode_tokens", 0)
        # Track first decode step after injection (for decode stall)
        if has_d and first_decode_after_inject is None:
            first_decode_after_inject = now

        # Record TTFT
        for seq in sched_out.seqs:
            sid = getattr(seq, "seq_id", -1)
            if sid not in ttft_map and seq.num_completion_tokens >= 1:
                ttft_map[sid] = now
        # Record finish
        for seq_id, _ in outputs:
            if seq_id not in finish_map:
                finish_map[seq_id] = now

    total_wall = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    shutdown_llm(llm)

    # Separate batch A and batch B using recorded seq_ids
    a_sids = batch_a_seq_ids
    b_sids = batch_b_seq_ids

    batch_a_ttft = [ttft_map.get(sid, total_wall) for sid in a_sids]
    batch_a_finish = [finish_map.get(sid, total_wall) for sid in a_sids]
    batch_b_ttft = [(ttft_map.get(sid, total_wall) - inject_time) for sid in b_sids]
    batch_b_finish = [(finish_map.get(sid, total_wall) - inject_time) for sid in b_sids]

    # Decode stall: time from injection to first decode step after injection.
    # With chunked: decode runs first each step, so stall ≈ 1 decode step time.
    # With baseline: prefill blocks everything, stall ≈ entire prefill time.
    stall = (first_decode_after_inject - inject_time) if first_decode_after_inject else 0.0

    return StaggeredResult(
        label=label,
        wall_total=total_wall,
        t0=t0,
        inject_time=inject_time,
        batch_a_ttft=batch_a_ttft,
        batch_a_finish_time=batch_a_finish,
        batch_b_ttft=batch_b_ttft,
        batch_b_finish_time=batch_b_finish,
        decode_stall_sec=stall,
        post_inject_steps=post_inject_steps,
        post_inject_mixed_steps=post_inject_mixed,
        post_inject_prefill_steps=post_inject_prefill,
        post_inject_decode_steps=post_inject_decode,
        peak_memory_bytes=peak_mem,
        total_output_tokens=total_decode_tokens,
    )


def _print_staggered_result(res: StaggeredResult) -> None:
    print(f"\n{'='*60}")
    print(f"  {res.label}")
    print(f"{'='*60}")
    print(f"  Total wall time:       {res.wall_total:.3f} s")
    print(f"  Inject time:           {res.inject_time:.4f} s")
    print(f"  Post-inject steps:     {res.post_inject_steps} "
          f"(mixed={res.post_inject_mixed_steps}, "
          f"prefill={res.post_inject_prefill_steps}, "
          f"decode={res.post_inject_decode_steps})")
    if res.batch_a_ttft:
        a = res.batch_a_ttft
        print(f"  Batch A (short) TTFT:")
        print(f"    min={min(a):.4f}s  median={statistics.median(a):.4f}s  max={max(a):.4f}s")
    if res.batch_b_ttft:
        b = res.batch_b_ttft
        print(f"  Batch B (long) TTFT (from injection):")
        print(f"    min={min(b):.4f}s  median={statistics.median(b):.4f}s  max={max(b):.4f}s")
    print(f"  Decode stall:          {res.decode_stall_sec:.4f} s")
    if res.peak_memory_bytes is not None:
        print(f"  Peak GPU memory:       {res.peak_memory_bytes / 1e9:.3f} GB")


def _print_staggered_comparison(baseline: StaggeredResult, chunked: StaggeredResult) -> None:
    print(f"\n{'='*60}")
    print(f"  STAGGERED COMPARISON: baseline vs chunked")
    print(f"{'='*60}")

    def _pct(a, b):
        if b == 0:
            return "n/a"
        diff = (a - b) / b * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"

    print(f"  {'Metric':<32} {'Baseline':>14} {'Chunked':>14} {'Diff':>10}")
    print(f"  {'-'*70}")

    rows = []
    # Batch B TTFT (the key metric)
    if baseline.batch_b_ttft and chunked.batch_b_ttft:
        b_med = statistics.median(baseline.batch_b_ttft)
        c_med = statistics.median(chunked.batch_b_ttft)
        rows.append(("Batch B TTFT median (s)", b_med, c_med, False))
        rows.append(("Batch B TTFT max (s)", max(baseline.batch_b_ttft), max(chunked.batch_b_ttft), False))
        rows.append(("Batch B TTFT min (s)", min(baseline.batch_b_ttft), min(chunked.batch_b_ttft), False))
    # Batch A finish time (did chunked help short requests finish faster?)
    if baseline.batch_a_finish_time and chunked.batch_a_finish_time:
        b_med = statistics.median(baseline.batch_a_finish_time)
        c_med = statistics.median(chunked.batch_a_finish_time)
        rows.append(("Batch A finish median (s)", b_med, c_med, False))
    # Decode stall
    rows.append(("Decode stall (s)", baseline.decode_stall_sec, chunked.decode_stall_sec, False))
    # Mixed steps
    rows.append(("Post-inject mixed steps", float(baseline.post_inject_mixed_steps),
                 float(chunked.post_inject_mixed_steps), True))
    # Wall time
    rows.append(("Wall time (s)", baseline.wall_total, chunked.wall_total, False))

    for name, bv, cv, higher_better in rows:
        diff = _pct(cv, bv)
        print(f"  {name:<32} {bv:>14.4f} {cv:>14.4f} {diff:>10}")

    print(f"\n  Interpretation:")
    if baseline.batch_b_ttft and chunked.batch_b_ttft:
        b_med = statistics.median(baseline.batch_b_ttft)
        c_med = statistics.median(chunked.batch_b_ttft)
        if b_med > 0:
            ratio = c_med / b_med
            if ratio < 0.95:
                print(f"    [WIN] Chunked: Batch B TTFT median is {ratio:.2f}x lower "
                      f"(long requests get first token faster)")
            elif ratio > 1.05:
                print(f"    [REG] Chunked: Batch B TTFT median is {ratio:.2f}x higher")
            else:
                print(f"    [TIE] Batch B TTFT is similar ({ratio:.2f}x)")
    if baseline.decode_stall_sec > 0 or chunked.decode_stall_sec > 0:
        if chunked.decode_stall_sec < baseline.decode_stall_sec:
            print(f"    [WIN] Chunked: decode stall is lower "
                  f"({chunked.decode_stall_sec:.4f}s vs {baseline.decode_stall_sec:.4f}s)")
        elif chunked.decode_stall_sec > baseline.decode_stall_sec:
            print(f"    [REG] Chunked: decode stall is higher "
                  f"({chunked.decode_stall_sec:.4f}s vs {baseline.decode_stall_sec:.4f}s)")
    if chunked.post_inject_mixed_steps > 0 and baseline.post_inject_mixed_steps == 0:
        print(f"    [WIN] Chunked enables mixed prefill+decode batching "
              f"({chunked.post_inject_mixed_steps} mixed steps)")


def _run_staggered_benchmark(args: argparse.Namespace) -> None:
    """Staggered scenario: short requests first, then long requests arrive during decode."""
    print("\n── Scenario 2: Staggered arrival (short → decode → inject long) ──")
    print(f"  Batch A: {args.bench_num_short} short x {args.bench_short_prompt_len} tokens "
          f"(max_out={args.bench_short_max_tokens})")
    print(f"  Batch B: {args.bench_num_long} long x {args.bench_long_prompt_len} tokens "
          f"(max_out={args.bench_long_max_tokens})")

    # Get tokenizer
    tokenizer_probe, _ = build_llm(model=args.model, args=args,
                                   max_num_batched_tokens=max(args.bench_long_prompt_len, 1),
                                   chunked=False, label="stagger-tokenizer")
    tokenizer = tokenizer_probe.tokenizer
    shutdown_llm(tokenizer_probe)

    baseline_budget = args.baseline_budget or max(args.bench_long_prompt_len + 16, args.chunk * 4)
    baseline_budget = max(baseline_budget, args.bench_long_prompt_len)

    # Warmup
    for w in range(args.bench_warmup):
        print(f"\n  --- warmup {w+1}/{args.bench_warmup} ---")
        _run_staggered_pass(label="warmup-baseline", model=args.model, args=args,
                            tokenizer=tokenizer, chunked=False, budget=baseline_budget)
        _run_staggered_pass(label="warmup-chunked", model=args.model, args=args,
                            tokenizer=tokenizer, chunked=True, budget=args.chunk)

    # Timed rounds
    baseline_results: list[StaggeredResult] = []
    chunked_results: list[StaggeredResult] = []
    for r in range(args.bench_rounds):
        print(f"\n  --- round {r+1}/{args.bench_rounds} ---")
        br = _run_staggered_pass(label=f"baseline-r{r}", model=args.model, args=args,
                                 tokenizer=tokenizer, chunked=False, budget=baseline_budget)
        _print_staggered_result(br)
        baseline_results.append(br)
        cr = _run_staggered_pass(label=f"chunked-r{r}", model=args.model, args=args,
                                 tokenizer=tokenizer, chunked=True, budget=args.chunk)
        _print_staggered_result(cr)
        chunked_results.append(cr)

    # Show first round comparison
    _print_staggered_comparison(baseline_results[0], chunked_results[0])


def _aggregate_results(results: list[BenchResult], label: str) -> BenchResult:
    n = len(results)
    if n == 1:
        return results[0]
    avg_wall = statistics.mean(r.wall_total for r in results)
    avg_out = statistics.mean(r.output_tok_per_sec for r in results)
    avg_prefill = statistics.mean(r.prefill_tok_per_sec for r in results)
    avg_decode = statistics.mean(r.decode_tok_per_sec for r in results)
    avg_steps = int(statistics.mean(r.num_steps for r in results))
    avg_mixed = int(statistics.mean(r.mixed_steps for r in results))
    avg_prefill_steps = int(statistics.mean(r.prefill_steps for r in results))
    avg_decode_steps = int(statistics.mean(r.decode_steps for r in results))
    all_ttft = [ttft for r in results for ttft in r.ttft_per_request]
    total_prefill = int(statistics.mean(r.total_prefill_tokens for r in results))
    total_decode = int(statistics.mean(r.total_decode_tokens for r in results))
    avg_peak = None
    peaks = [r.peak_memory_bytes for r in results if r.peak_memory_bytes is not None]
    if peaks:
        avg_peak = int(statistics.mean(peaks))
    return BenchResult(
        label=f"{label} (avg of {n} rounds)",
        wall_total=avg_wall,
        total_prefill_tokens=total_prefill,
        total_decode_tokens=total_decode,
        total_output_tokens=total_decode,
        num_steps=avg_steps,
        prefill_steps=avg_prefill_steps,
        decode_steps=avg_decode_steps,
        mixed_steps=avg_mixed,
        prefill_wall=avg_wall,
        decode_wall=avg_wall,
        prefill_tok_per_sec=avg_prefill,
        decode_tok_per_sec=avg_decode,
        output_tok_per_sec=avg_out,
        ttft_per_request=all_ttft,
        peak_memory_bytes=avg_peak,
        step_records=[],
    )


def main() -> int:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    if not os.path.isdir(args.model):
        print(f"model path does not exist or is not a directory: {args.model}", file=sys.stderr)
        return 2
    if args.chunk <= 0:
        print("--chunk must be positive", file=sys.stderr)
        return 2
    if args.prompt_len <= args.chunk:
        print("[WARN] --prompt-len should be larger than --chunk to force partial prefill")

    missing_chunk_fields = sorted(
        {
            "enable_chunked_prefill",
            "enable_mixed_prefill_decode",
            "prefill_chunk_size",
            "max_num_partial_prefills",
        }
        - config_field_names()
    )
    if missing_chunk_fields:
        print(
            "[WARN] this nano-vLLM Config is missing explicit chunked-prefill fields: "
            f"{missing_chunk_fields}. The script will still use a small "
            "max_num_batched_tokens to force chunked prefill behavior."
        )

    all_ok = True
    if args.mode in ("correctness", "both", "all"):
        all_ok = run_correctness(args) and all_ok
    if args.mode in ("interleave", "both", "all"):
        all_ok = run_interleave(args) and all_ok
    if args.mode in ("benchmark", "all"):
        all_ok = run_benchmark(args) and all_ok

    print("\nFINAL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
