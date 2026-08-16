from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm7
from lm7.detection import resolve_target, synchronize, torch_device
from lm7.huggingface import _apply_quantization, _model_storage_bytes, normalize_quantization

MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"


def inputs_for(tokenizer, batch: int, sequence: int, device: torch.device):
    seed = tokenizer("The capital of France is", add_special_tokens=True)["input_ids"]
    ids = (seed * ((sequence + len(seed) - 1) // len(seed)))[:sequence]
    input_ids = torch.tensor([ids] * batch, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "use_cache": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mode = normalize_quantization(args.mode)
    target = resolve_target("nvidia")
    device = torch_device(target)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    load_ms = (time.perf_counter() - started) * 1000
    quantized_modules = 0
    quantization_ms = 0.0
    if mode != "none":
        quantization_ms, quantized_modules = _apply_quantization(model, target, mode)
    storage_bytes = _model_storage_bytes(model)
    model.to(device)
    inputs = inputs_for(tokenizer, args.batch, args.sequence, device)
    torch.cuda.reset_peak_memory_stats()
    compiled = lm7.compile(
        model,
        target="nvidia",
        backend="inductor",
        transfers="explicit",
        fallback="error",
        cache=False,
    )
    synchronize(target)
    first = time.perf_counter()
    with torch.inference_mode():
        compiled(**inputs)
    synchronize(target)
    first_call_ms = (time.perf_counter() - first) * 1000

    for _ in range(args.warmup):
        with torch.inference_mode():
            compiled(**inputs)
    synchronize(target)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            compiled(**inputs)
        synchronize(target)
        samples.append((time.perf_counter() - started) * 1000)

    report = {
        "model": MODEL_ID,
        "mode": mode,
        "batch_size": args.batch,
        "sequence_length": args.sequence,
        "matrix_rows": args.batch * args.sequence,
        "dtype": "bfloat16",
        "input_construction": "repeated token IDs for 'The capital of France is'",
        "load_ms": load_ms,
        "quantization_ms": quantization_ms,
        "quantized_modules": quantized_modules,
        "storage_bytes": storage_bytes,
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
        "latency_p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "torchao": __import__("torchao").__version__,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
