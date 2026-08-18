"""What ONNX Runtime's I/O binding is worth against the NumPy feed path it replaced.

LM7's ONNX Runtime adapter used to feed NumPy and read NumPy back, which copied
every input to the host and every output back from it on each call. It now binds
torch storage directly (see ``docs/onnxruntime.md``). That change was validated
for correctness -- tensors stay on the device and still match eager -- but never
timed, and this harness is what closes that gap.

    python benchmarks/onnx_io_binding.py \
      --provider CUDAExecutionProvider \
      --output artifacts/benchmarks/onnx-io-binding-sm89.json

Both strategies run against **the same ``InferenceSession``**, so the kernels,
the provider and the graph are identical and the only difference is how tensors
reach and leave it. Anything the two paths disagree on is copy cost.

The headline number is a property of the output, not of the model. A graph is
timed at several output widths because that is what the fetch has to move: a
classifier returning 10 logits and a causal LM returning a full vocabulary
distribution sit at opposite ends of it, and one number for "ONNX" would hide
that. ``--model smollm2`` runs the real case for the same reason.

The session is configured the way ``load_onnx`` configures it, which for a
non-CPU provider means the ``disable_cpu_ep_fallback`` entry *and*
``session.disable_fallback()``. ONNX Runtime otherwise partitions nodes onto the
CPU EP, and a graph only partly on the device is a different measurement from
the one LM7 runs.

**This harness cannot resolve a difference on SmolLM2-135M.** Its run-to-run
spread there is far wider than the gap being measured -- p10 through p90 spans
roughly 19-46 ms against a median difference of about 10 ms, and I/O binding
wins at p10 while losing at the median. The MLP sweep is stable across all three
percentiles and is the result worth reading; a real causal LM needs either a
quieter machine or many more samples than this takes.

Two things this harness does *not* claim:

* It is not an ONNX-versus-Inductor comparison. Both paths here are ONNX Runtime;
  the question is only what the adapter costs on top of the session.
* CUDA is asynchronous, so a naive timer measures the launch and not the work.
  Every timed region ends with ``torch.cuda.synchronize()`` on a CUDA provider,
  and the I/O binding path additionally calls ORT's own
  ``synchronize_inputs``/``synchronize_outputs`` as the adapter does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_BFLOAT16_ONNX_TYPE = 16


def _element_type(dtype: torch.dtype) -> Any:
    if dtype == torch.bfloat16:
        return _BFLOAT16_ONNX_TYPE
    return np.dtype(str(dtype).removeprefix("torch.")).type


class _MLP(torch.nn.Module):
    """Fixed input, parameterised output: the fetch is what the sweep varies."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], path: Path) -> Path:
    with torch.no_grad():
        exported = torch.export.export(module.eval(), args, strict=False)
    program = torch.onnx.export(
        exported, args=(), f=None, dynamo=True, external_data=False, optimize=True
    )
    program.save(path, external_data=False)
    return path


def _session(path: Path, provider: str) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    strict = provider != "CPUExecutionProvider"
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    if strict:
        # Both calls, exactly as `load_onnx` makes them. Without the second one
        # ORT still partitions nodes onto the CPU EP, and the comparison inverts:
        # a graph that is only partly on the device pays a transfer for every
        # bound tensor that crosses the boundary. See the module docstring.
        session.disable_fallback()
    return session


def _run_numpy(session: Any, feeds: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """The pre-change path: host copy in, host copy out."""
    values = {name: t.detach().cpu().contiguous().numpy() for name, t in feeds.items()}
    return tuple(torch.as_tensor(v).clone() for v in session.run(None, values))


def _run_binding(
    session: Any, feeds: dict[str, torch.Tensor], device_type: str, device_id: int
) -> tuple[torch.Tensor, ...]:
    """The current path: bind torch storage, recover outputs through dlpack."""
    binding = session.io_binding()
    bound: list[torch.Tensor] = []
    device = torch.device(device_type if device_type == "cpu" else f"{device_type}:{device_id}")
    for value in session.get_inputs():
        tensor = feeds[value.name].detach().to(device).contiguous()
        bound.append(tensor)
        binding.bind_input(
            value.name,
            device_type,
            device_id,
            _element_type(tensor.dtype),
            tuple(tensor.shape),
            tensor.data_ptr(),
        )
    for value in session.get_outputs():
        binding.bind_output(value.name, device_type, device_id)
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return tuple(torch.from_dlpack(v).clone() for v in binding.get_outputs())


def _time(fn: Any, warmup: int, runs: int, cuda: bool) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[len(samples) // 10],
        "p90_ms": samples[(len(samples) * 9) // 10],
    }


def _measure(
    session: Any,
    feeds: dict[str, torch.Tensor],
    provider: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    device_type = "cpu" if provider == "CPUExecutionProvider" else "cuda"
    cuda = device_type == "cuda"
    numpy_stats = _time(lambda: _run_numpy(session, feeds), warmup, runs, cuda)
    binding_stats = _time(lambda: _run_binding(session, feeds, device_type, 0), warmup, runs, cuda)

    reference = _run_numpy(session, feeds)
    actual = _run_binding(session, feeds, device_type, 0)
    drift = max(
        (a.cpu() - b.cpu()).abs().max().item() for a, b in zip(actual, reference, strict=True)
    )
    out_bytes = sum(t.numel() * t.element_size() for t in reference)
    in_bytes = sum(t.numel() * t.element_size() for t in feeds.values())
    return {
        "numpy": numpy_stats,
        "io_binding": binding_stats,
        "speedup": numpy_stats["median_ms"] / binding_stats["median_ms"],
        "saved_ms": numpy_stats["median_ms"] - binding_stats["median_ms"],
        "input_bytes": in_bytes,
        "output_bytes": out_bytes,
        # The two paths must agree: this is a timing harness, not a second
        # implementation, and a nonzero drift would mean it is timing the wrong
        # thing.
        "max_abs_drift": drift,
    }


def _mlp_rows(args: argparse.Namespace, tmp: Path) -> list[dict[str, Any]]:
    rows = []
    for out_features in args.output_features:
        torch.manual_seed(0)
        module = _MLP(args.in_features, out_features)
        example = torch.randn(args.batch, args.in_features)
        path = _export(module, (example,), tmp / f"mlp-{out_features}.onnx")
        session = _session(path, args.provider)
        feeds = {session.get_inputs()[0].name: example}
        row = {"model": "mlp", "batch": args.batch, "out_features": out_features}
        row.update(_measure(session, feeds, args.provider, args.warmup, args.runs))
        rows.append(row)
        print(
            f"  mlp out={out_features:<7} numpy {row['numpy']['median_ms']:7.3f} ms  "
            f"binding {row['io_binding']['median_ms']:7.3f} ms  "
            f"{row['speedup']:.2f}x  ({row['output_bytes'] / 1024:.0f} KiB out)",
            flush=True,
        )
    return rows


def _smollm2_rows(args: argparse.Namespace, tmp: Path) -> list[dict[str, Any]]:
    import transformers

    from lm7.huggingface import _LogitsOnly

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    source = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    wrapped = _LogitsOnly(source.eval()).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    example = (inputs["input_ids"], inputs["attention_mask"])
    path = _export(wrapped, example, tmp / "smollm2.onnx")
    session = _session(path, args.provider)
    feeds = dict(zip((v.name for v in session.get_inputs()), example, strict=True))
    row = {"model": model_id, "batch": 1, "out_features": None}
    row.update(_measure(session, feeds, args.provider, args.warmup, args.runs))
    print(
        f"  smollm2        numpy {row['numpy']['median_ms']:7.3f} ms  "
        f"binding {row['io_binding']['median_ms']:7.3f} ms  "
        f"{row['speedup']:.2f}x  ({row['output_bytes'] / 1024:.0f} KiB out)",
        flush=True,
    )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--model", choices=("mlp", "smollm2", "both"), default="mlp")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument(
        "--output-features",
        type=int,
        nargs="+",
        default=[10, 1000, 32000, 128000],
        help="Output widths to sweep; the fetch is what these vary.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    if args.provider not in ort.get_available_providers():
        raise SystemExit(
            f"provider {args.provider!r} is unavailable; "
            f"have {', '.join(ort.get_available_providers())}"
        )

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="onnx-io-binding-"))
    print(f"provider: {args.provider}  onnxruntime {ort.__version__}", flush=True)
    rows: list[dict[str, Any]] = []
    if args.model in ("mlp", "both"):
        rows += _mlp_rows(args, tmp)
    if args.model in ("smollm2", "both"):
        rows += _smollm2_rows(args, tmp)

    payload = {
        "provider": args.provider,
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "warmup": args.warmup,
        "runs": args.runs,
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
