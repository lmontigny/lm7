"""Ask whether this CPU's matrix hardware is doing anything for LM7.

`lm7 doctor` reports `amx_bf16`, `amx_int8` and `amx_tile` on an Intel part and
`bf16`, `i8mm`, `sve`, `sve2` on an Arm one, and [docs/cpu.md](../docs/cpu.md)
is explicit that LM7 does not act on any of them: the CPU compute dtype is FP32
everywhere. All of that hardware accelerates BF16 and INT8 matmuls and does
nothing for FP32, so the flags being present says nothing about whether any work
reaches it.

This measures the gap. For each dtype and backend it records latency, and it
separately asks oneDNN which kernels it considered -- and that second half is
where the two ISAs stop looking alike:

- **On x86 the chosen kernel is the evidence.** `ONEDNN_VERBOSE=1` names the
  implementation on every primitive, and an `amx` in a matmul's name is proof
  the tile units ran. A latency win alone would not distinguish AMX from
  AVX-512 doing BF16 conversion cheaply.
- **On AArch64 it is not.** oneDNN routes f32 *and* bf16 matmuls to the same
  `gemm:acl` -- the name of a library, Arm Compute Library, rather than of an
  instruction. Nothing in the exec log separates a BF16 matmul that used
  `BFMMLA` from one that did not, so this reports the question as unanswered
  there rather than answering it wrongly.

What does carry information on both is the *dispatch* log. `ONEDNN_VERBOSE=all`
records the implementations oneDNN rejected and why, which is how a host learns
that (for instance) its BRGEMM path was skipped for "unsupported isa" because
the part has 128-bit SVE and the kernel wants 512.

    python benchmarks/cpu_matrix_unit.py --model mlp smollm2 --repeats 30

Run it on an otherwise idle machine, or read the ratios rather than the absolute
milliseconds: CPU benchmarks are far more sensitive to a busy host than GPU ones.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HF_MODELS = {"smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct"}
MODELS = ("mlp", *HF_MODELS)
DTYPES = ("float32", "bfloat16")
PROMPT = "The capital of France is"

# The kernel flags that mean "this CPU has hardware a BF16 or INT8 matmul could
# reach". Named as the kernel names them, and matched as whole tokens -- "sve"
# is a prefix of "svebf16", so a substring test reports the wrong ISA.
X86_MATRIX_FLAGS = ("amx_tile", "amx_bf16", "amx_int8")
ARM_MATRIX_FLAGS = ("bf16", "i8mm", "sve", "sve2", "svebf16", "svei8mm")

# Primitives where reaching the matrix unit is the thing being asked about. An
# eltwise kernel can carry an ISA name while doing no matrix work at all.
MATRIX_PRIMITIVES = frozenset({"matmul", "convolution", "inner_product"})

# How many verbose lines to read before giving up on a run. Identifying the
# kernels needs a few hundred -- the same implementations repeat -- so this is
# pure protection against the pathological case: ONEDNN_VERBOSE=all on a
# 30-layer causal LM at 512 tokens emits enough to exhaust host memory.
VERBOSE_LINE_CAP = 200_000


def build(name: str, dtype_name: str, *, rows: int = 8) -> tuple[Any, tuple[Any, ...]]:
    import torch

    # Seeded, so the same weights and inputs come back for every dtype -- without
    # this each row gets fresh random weights and the FP32-versus-BF16 difference
    # measures the RNG rather than the arithmetic.
    torch.manual_seed(0)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).eval()
        return model.to(dtype=dtype), (torch.randn(rows, 1024).to(dtype=dtype),)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = HF_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    # An AMX tile is 16 rows deep, so a 5-token prompt leaves most of it idle.
    # `rows` repeats the prompt to reach a chosen sequence length, which is the
    # variable that decides whether the tile units have anything to chew on.
    encoded = tokenizer(PROMPT, return_tensors="pt")
    if rows > encoded["input_ids"].shape[1]:
        repeats = -(-rows // encoded["input_ids"].shape[1])
        encoded = {
            key: value.repeat(1, repeats)[:, :rows]
            for key, value in encoded.items()  # type: ignore[union-attr]
        }

    class TensorOut(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits

    return TensorOut(model).eval(), (encoded["input_ids"], encoded["attention_mask"])


def _cpu_facts() -> dict[str, Any]:
    import torch

    facts: dict[str, Any] = {
        "host": platform.node(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "mkldnn": torch.backends.mkldnn.is_available(),
    }
    # PyTorch's own view, which is coarser than /proc/cpuinfo but is what the
    # dispatcher actually keys on. It tops out at "AVX512" and says nothing about
    # AMX either way, so it cannot answer the question this file asks.
    try:
        facts["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:  # noqa: BLE001 - a private API that moved costs the label only
        facts["cpu_capability"] = None
    # The authoritative runtime check. AMX needs the OS to grant tile state
    # through arch_prctl, which a VM or container can decline while /proc/cpuinfo
    # still advertises the flags; `_init_amx` is what performs that request.
    # Reported as "unavailable" when the API is absent rather than as False --
    # an earlier revision defaulted it to False and reported working hardware as
    # having no AMX.
    initialize = getattr(getattr(torch, "_C", None), "_cpu", None)
    initialize = getattr(initialize, "_init_amx", None)
    facts["init_amx"] = bool(initialize()) if callable(initialize) else "unavailable"
    facts["machine"] = platform.machine().lower()
    facts["matrix_flags"] = cpu_matrix_flags()
    return facts


def cpu_matrix_flags() -> list[str]:
    """The matrix-hardware flags this host advertises, whichever ISA it is.

    x86 prints them under "flags" and AArch64 under "Features"; the two never
    co-occur, so both are read and whichever answers wins. Tokenised rather than
    substring-matched: "sve" appears inside "svebf16", and an `in` test on the
    raw text reports SVE on a part that has only the SVE forms of other
    instructions.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return []
    tokens: set[str] = set()
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"flags", "Features"}:
            tokens.update(value.split())
    return sorted(tokens & set(X86_MATRIX_FLAGS + ARM_MATRIX_FLAGS))


def measure(model: Any, inputs: tuple[Any, ...], *, backend: str, warmup: int, repeats: int) -> Any:
    import torch

    import lm7

    if backend == "eager":
        callable_model: Any = model
    else:
        callable_model = lm7.compile(
            model, target="cpu", backend=backend, fallback="error", cache=False
        )

    started = time.perf_counter()
    with torch.no_grad():
        output = callable_model(*inputs)
    first_call_ms = (time.perf_counter() - started) * 1000.0

    for _ in range(warmup):
        with torch.no_grad():
            callable_model(*inputs)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            callable_model(*inputs)
        samples.append((time.perf_counter() - started) * 1000.0)

    return {
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
    }, output.detach().float()


def matrix_matmul_kernels(implementations: dict[str, int], machine: str) -> list[str] | None:
    """Which executed matmuls prove the matrix unit ran -- where that is knowable.

    Returns a list on x86, where the implementation name carries the answer, and
    ``None`` on any ISA where it does not. ``None`` means "this log cannot say",
    which is a different claim from ``[]``, "it said no" -- conflating the two is
    how a 2.4x BF16 speedup gets recorded as the tile units never running.

    On AArch64 every matmul reads `gemm:acl` regardless of dtype, because that
    names Arm Compute Library rather than an instruction. See the dispatch log
    for what *is* knowable there.
    """
    if not machine.startswith(("x86", "amd64", "i386", "i686")):
        return None
    return sorted(
        key for key in implementations if "amx" in key and key.split(":")[0] in MATRIX_PRIMITIVES
    )


def kernels_used(model_name: str, dtype: str, backend: str, *, rows: int = 8) -> dict[str, Any]:
    """Re-run one call under ONEDNN_VERBOSE and report what oneDNN did and skipped.

    Done in a child process because oneDNN reads the variable once, at load.
    ``all`` rather than ``1`` so the same run carries both the kernels that
    executed and the ones that were rejected, which is the half that still
    carries information on an ISA whose kernel names do not.

    Streamed and line-capped rather than buffered. ``all`` on a 30-layer causal
    LM at 512 tokens emits enough verbose output to exhaust a 31 GiB host if it
    is collected in memory first, which is not a hypothetical -- it wedged the
    machine this was developed on. The cap is far above what any run needs to
    identify its kernels, since the same handful of implementations repeat.
    """
    script = (
        "import sys; sys.argv=['x'];"
        "from benchmarks.cpu_matrix_unit import build, measure;"
        f"m,i=build({model_name!r},{dtype!r},rows={rows!r});"
        f"measure(m,i,backend={backend!r},warmup=0,repeats=1)"
    )
    environment = dict(os.environ, ONEDNN_VERBOSE="all", PYTHONPATH="src:.")
    implementations: dict[str, int] = {}
    rejected: dict[str, str] = {}
    isa = None
    truncated = False
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        bufsize=1,
    )
    assert process.stdout is not None
    for count, line in enumerate(process.stdout):
        if count >= VERBOSE_LINE_CAP:
            truncated = True
            process.kill()
            break
        if not line.startswith("onednn_verbose"):
            continue
        fields = line.split(",")
        if ",info,cpu,isa:" in line:
            isa = line.split("isa:", 1)[1].strip()
            continue
        # Field offsets move between oneDNN verbose formats -- v3.12 inserts a
        # "v1" token before the operation -- so anchor on the "exec" token
        # instead of indexing from the left. Getting this wrong reports every
        # kernel as "exec:cpu" and finds AMX nowhere, which is what it did.
        if "exec" in fields:
            start = fields.index("exec")
            if len(fields) < start + 4:
                continue
            primitive, implementation = fields[start + 2], fields[start + 3]
            key = f"{primitive}:{implementation}"
            implementations[key] = implementations.get(key, 0) + 1
        elif "create:dispatch" in fields:
            start = fields.index("create:dispatch")
            # A rejected matmul reads op,engine,primitive,implementation from
            # here; a rejected reorder carries its reason immediately and has no
            # implementation at all, so the engine field tells them apart.
            if len(fields) < start + 5 or fields[start + 2] != "cpu":
                continue
            if fields[start + 1] not in MATRIX_PRIMITIVES:
                continue
            # The reason is second from the end, before the source location.
            rejected[f"{fields[start + 3]}:{fields[start + 4]}"] = fields[-2]
    process.stdout.close()
    process.wait()
    return {
        "onednn_isa": isa,
        # True when the cap cut the log short. The kernel lists stay valid --
        # they are a set, and the same implementations repeat -- but the counts
        # below them are a floor rather than a total.
        "verbose_truncated": truncated,
        "implementations": dict(sorted(implementations.items(), key=lambda kv: -kv[1])[:8]),
        # A list on x86, None where the exec log cannot answer. See
        # matrix_matmul_kernels.
        "matrix_matmul": matrix_matmul_kernels(implementations, platform.machine().lower()),
        # oneDNN names the whole x86 ISA `avx10_1_512_amx`, so an eltwise kernel
        # carries "amx" in its name while doing no tile-unit work at all. Kept
        # apart from the matmuls rather than counted with them.
        "amx_named_other": sorted(
            key
            for key in implementations
            if "amx" in key and key.split(":")[0] not in MATRIX_PRIMITIVES
        ),
        # Why oneDNN did not use the kernels it did not use. On a host whose
        # chosen kernel is uninformative this is the only evidence left.
        "rejected_matmul": dict(sorted(rejected.items())),
        "verbose_lines": sum(implementations.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", choices=MODELS, default=["mlp"])
    parser.add_argument("--dtype", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--backend", nargs="+", choices=("eager", "inductor"), default=["eager"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[8],
        help="batch rows for the MLP, sequence length for a causal LM",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    import torch

    if arguments.threads:
        torch.set_num_threads(arguments.threads)

    report: dict[str, Any] = {"cpu": _cpu_facts(), "results": []}
    print(json.dumps(report["cpu"], sort_keys=True))

    for model_name in arguments.model:
        for rows in arguments.rows:
            reference = None
            for backend in arguments.backend:
                for dtype in arguments.dtype:
                    model, inputs = build(model_name, dtype, rows=rows)
                    measured, output = measure(
                        model,
                        inputs,
                        backend=backend,
                        warmup=arguments.warmup,
                        repeats=arguments.repeats,
                    )
                    if dtype == "float32" and backend == arguments.backend[0]:
                        reference = output
                    difference = (
                        (output - reference).abs().max().item()
                        if reference is not None and output.shape == reference.shape
                        else None
                    )
                    record = {
                        "model": model_name,
                        "backend": backend,
                        "dtype": dtype,
                        "rows": rows,
                        "max_abs_diff_vs_fp32": difference,
                        **measured,
                        **kernels_used(model_name, dtype, backend, rows=rows),
                    }
                    report["results"].append(record)
                    # Three states, not two: "unknown" is what an ISA whose
                    # kernel names do not carry the answer honestly reports.
                    matrix = record["matrix_matmul"]
                    verdict = "unknown" if matrix is None else ("yes" if matrix else "no")
                    print(
                        f"{model_name:>9} {backend:<9} {dtype:<9} rows={rows:<5} "
                        f"median={record['latency_median_ms']:9.3f} ms  "
                        f"matrix_matmul={verdict:<7}"
                        + (f"  diff={difference:.3e}" if difference is not None else "")
                    )
                    del model

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"JSON: {arguments.output}")


if __name__ == "__main__":
    main()
