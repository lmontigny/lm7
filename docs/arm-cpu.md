# Arm CPU inference (Linux servers)

[CPU inference](cpu.md) covers what LM7 does on a CPU generally, and most of it
is architecture-neutral. This is the Arm-server-specific half: what a Neoverse
host needs that an x86 one does not, what LM7 reports there, and the traps that
cost time on the way in.

For Apple Silicon see [Apple Silicon](apple-mps.md) — it is also `arm64`, but a
laptop with a GPU attached is a different setup problem from a headless server.

Everything below was done on a GCP `n4a-standard-8` (Google Axion, **Arm
Neoverse N3**, 8 vCPU, 31 GiB, Debian 12 bookworm, kernel 6.1 arm64) — the host
in [tested hardware](tested-hardware.md). Version-specific findings say so.

## Setting one up

```bash
sudo apt-get install -y python3-venv python3-dev build-essential
python3 -m venv /mnt/data/venv
/mnt/data/venv/bin/pip install numpy
/mnt/data/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/mnt/data/venv/bin/pip install -e ".[dev]"
```

Four things that are not obvious, in the order they bite:

- **`python3-dev` is not optional if you intend to use Inductor.** TorchInductor
  generates C++ and compiles it at run time, so it needs `Python.h` present. A
  plain cloud image has the interpreter and not the headers, and the failure
  arrives on the first `lm7.compile(..., backend="inductor")` rather than at
  install time.
- **The boot disk is probably too small.** Cloud Arm images default to around
  10 GiB, and `torch` plus one small checkpoint will fill it. Put the virtualenv
  *and* the Hugging Face cache on a data disk — `HF_HOME` defaults to `~/.cache`
  and will quietly target the boot disk otherwise:

  ```bash
  export HF_HOME=/mnt/data/hf-cache
  ```

- **`numpy` is not pulled in by `torch`.** Without it every import prints
  `Failed to initialize NumPy: No module named 'numpy'` and continues. It is a
  warning rather than an error, which makes it easy to carry a long way.
- **The CPU wheel index is the one that was tested here.** `--index-url
  .../whl/cpu` is explicit about wanting a CPU build. Whether plain `pip install
  torch` resolves to the same wheel on aarch64 was not checked, and CUDA-on-Arm
  wheels do now exist for server parts, so do not assume the default is CPU-only
  the way it historically was.

## What LM7 reports on a Neoverse host

```
Detected targets (1):
  cpu:aarch64: Arm Neoverse N3, 31.3 GiB
```

```json
{
  "target": "cpu:aarch64",
  "name": "Arm Neoverse N3",
  "capabilities": {
    "isa_extensions": ["asimd", "asimddp", "asimdhp", "bf16", "i8mm", "sve", "sve2"],
    "logical_cores": 8,
    "physical_cores": 8,
    "vendor_id": null
  }
}
```

The core is named from its `CPU implementer` and `CPU part` numbers, because
AArch64 publishes no `model name`; `physical_cores` comes from sysfs topology,
because AArch64 publishes no `core id` either; `vendor_id` stays `null` because
nothing publishes one. All three are explained in
[CPU inference](cpu.md#on-aarch64-the-kernel-prints-less).

## What the toolchain says about itself

Worth capturing on a new host, because three different layers each answer a
different question and only the last one is about matrix hardware:

| | on the N3 |
| --- | --- |
| `torch.backends.cpu.get_cpu_capability()` | `SVE128` |
| `torch.__config__.show()` | `USE_MKLDNN=1`, `USE_OPENMP=ON`, `BLAS_INFO=open` |
| oneDNN (`ONEDNN_VERBOSE=1`) | `v3.12.0`, `isa:AArch64 SVE (128 bits)` |

- **`get_cpu_capability()` reports the SVE vector length, not a matrix-unit
  answer.** `SVE128` says the vectors are 128 bits wide. It says nothing about
  whether `bf16` or `i8mm` instructions are reached, exactly as its `AVX512`
  answer on an Intel host says nothing about AMX.
- **BLAS is OpenBLAS, and oneDNN is present and reaches Arm Compute Library.**
  A matmul dispatches to `gemm:acl` — at *every* dtype, which is why it cannot
  be used as evidence that BF16 instructions ran. See
  [the Arm dtype section](cpu.md#the-same-question-on-arm-where-the-logs-cannot-answer-it).
- **Vector width decides which oneDNN kernels are even eligible.** On this part
  `ONEDNN_VERBOSE=all` shows `matmul:brg:sve_512` rejected for `unsupported
  isa`: oneDNN's Arm BRGEMM path wants 512-bit vectors and this core has 128.
  A wider-SVE part (Graviton 3/4, Neoverse V-series) may take a different path,
  so do not carry a Neoverse N-series result onto a V-series one.

Threading defaults to one thread per logical core (8 here). Neoverse server
cores are one thread per core, so that is also the physical count — but LM7 now
reads that from sysfs rather than assuming it.

## Which extras resolve on aarch64 Linux

`pip install --dry-run` against the `[project.optional-dependencies]` names, on
the host above:

| Extra | aarch64 Linux wheel | Run on the N3 |
| --- | --- |
| `openvino`, `nncf` | resolves | `tests/test_openvino_integration.py`: 11 passed, 2 NPU skips; IR path [measured 4.16x over eager](openvino-evaluation.md#arm-neoverse-n3-linux-aarch64) |
| `onnxruntime` | resolves | `tests/test_onnxruntime_integration.py`: 3 passed, 2 skips |
| `executorch` | resolves | `tests/test_executorch_integration.py`: 6 passed |
| `tvm` | **run** — suite passes, artifact validated, see [TVM](tvm.md#linux-aarch64-neoverse-servers) | `tests/test_tvm_integration.py`: 15 passed |
| `torchao` | resolves | Used by the Arm INT8 measurement, not a backend by itself |
| `litert` | resolves | Not run here; LiteRT Torch caps PyTorch below 2.13, so it needs a separate environment |
| `iree-vulkan` | resolves | Not useful on this VM: GCP Axion exposes no Mali/Vulkan GPU |
| `serve` (FastAPI, uvicorn) | resolves | `lm7 model serve` ran; see below |
| `zentorch` | **no wheel** — x86-64 Linux only, by construction | N/A |

> [!IMPORTANT]
> **"Resolves" still means only that a wheel exists.** The right column is the
> extra step that turns a packaging fact into a backend fact. It was run in an
> isolated checkout and virtualenv at `/mnt/data/codex-arm-smoke-lm7` on the same
> `n4a-standard-8`, with `torch 2.13.0+cpu`, so it did not reuse the long-lived
> benchmark environment. `torch-tensorrt` also resolves, and is for NVIDIA GPUs.

`zentorch` is the one certainty, and it is a deliberate one — it is AMD's ZenDNN
extension and ships x86-64 Linux wheels only, so `lm7 doctor` reports it
unavailable on Arm with that reason. It is the AMD-CPU counterpart to
[OpenVINO](openvino-evaluation.md) on Intel; the Arm equivalent of neither
exists as an LM7 backend, and Arm Compute Library is reached through oneDNN
rather than through a backend of its own.

## Before you time anything on one

Two mistakes made on this host, both of which produced numbers that looked
plausible and were wrong:

- **A benchmark process can outlive the SSH command that started it.** A
  `gcloud compute ssh --command` whose *local* side times out does not kill the
  remote process. One left running alongside the next sweep inflated its
  medians by up to 10x, and neither run announced anything was wrong. Check the
  host is idle (`uptime`, `pgrep`) before trusting a timing, and treat two runs
  that disagree by more than a few percent as contention until proven otherwise.
- **`ONEDNN_VERBOSE=all` is not free.** On a 30-layer causal LM at 512 tokens it
  emits enough output to exhaust 31 GiB if something buffers it in memory, which
  wedged this machine hard enough to need a reset.
  [`benchmarks/cpu_matrix_unit.py`](../benchmarks/cpu_matrix_unit.py) now streams
  and line-caps it; anything else reading that variable should too.

Otherwise the usual CPU-benchmarking advice in
[`benchmarks/cpu_matrix_unit.py`](../benchmarks/cpu_matrix_unit.py) applies —
read ratios rather than absolute milliseconds, and run on an idle host.

## What has actually been measured here

- Eager against Inductor on the FP32 MLP, batch 1–512 — and
  [why compiling wins nothing on it](cpu.md#latency-on-a-neoverse-n3).
- FP32 against BF16 on that MLP and on SmolLM2-135M, where
  [the two workloads disagree on BF16's sign](cpu.md#the-same-question-on-arm-where-the-logs-cannot-answer-it).

- [`lm7 model serve`](serving.md#on-linux-arm-arm-neoverse-n3), where the target
  string a client sees is `cpu:aarch64` rather than the `cpu:arm64` every other
  Arm row on that page reports.
- [CPU AOTInductor artifacts](aot-artifact-compatibility.md#cpu-packages-are-architecture-bound-too),
  which carry a native `aarch64` shared object and were not architecture-gated
  until this host demonstrated it.
- ONNX Runtime, OpenVINO, TVM and ExecuTorch integration suites in a fresh
  aarch64 virtualenv, which turns their wheel availability into a real import,
  export/load and numerical-agreement check for the small models those suites
  cover.
- [INT8 and `i8mm`](quantization.md#i8mm-does-not-rescue-it-either-and-here-is-the-kernel-proving-why),
  where the INT8 matrix instructions this part advertises are never reached,
  because weight-only quantization issues no INT8 GEMM for them to accelerate.

- [TVM](tvm.md#linux-aarch64-neoverse-servers), where the whole integration
  suite passes and an exported `.so` really does carry the host's `aarch64`
  triple, as [artifact compatibility](aot-artifact-compatibility.md#cpu-packages-are-architecture-bound-too)
  had assumed from x86 alone.
- [Sparse MoE](limitations.md#what-torchcompile-actually-does-to-a-sparse-moe), where zero graph
  breaks and the near-1.0x CPU speedup both reproduce from a second ISA, and
  where a tiny MoE compiles 1.4-1.5x faster on the same host that gets nothing
  from compiling the FP32 MLP.

- [OpenVINO](openvino-evaluation.md#arm-neoverse-n3-linux-aarch64), whose CPU
  plugin loads on Arm and whose IR path is the fastest thing measured on this
  host — 4.16x over eager, more than it manages on Intel or Apple — while its
  INT8 route is the one that does *not* transfer from Intel.

Not measured on Arm: LiteRT conversion in its older-Torch environment, Mali
Vulkan execution, wider-SVE server parts, 512-token prompts on the N3, and
artifact portability for ONNX Runtime or ExecuTorch built on Arm. Mixtral-8x7B
does not fit: ~93 GB at BF16 against 31 GB of RAM. See
[limitations](limitations.md).
