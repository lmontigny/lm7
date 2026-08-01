# CPU inference

LM7 supports CPU inference without an additional runtime:

- `eager` executes with standard PyTorch.
- `inductor` uses `torch.compile` and generates optimized CPU kernels.
- `aot_inductor` creates the current persistent CPU `.pt2` prototype.

Install a CPU-capable PyTorch build and LM7:

```bash
uv venv --python 3.12
uv pip install torch --torch-backend=cpu
uv pip install -e ".[dev]"
source .venv/bin/activate
```

`--torch-backend=cpu` pins the CPU-only wheel index, which keeps the download
small on a machine that also has a GPU. With pip instead of uv, the equivalent is
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## What LM7 knows about the host CPU

`lm7 doctor` describes the CPU the way it describes an accelerator — by name,
with its memory and its capabilities:

```
Detected targets (1):
  cpu:x86_64: AMD EPYC 7B13, 62.8 GiB
```

`lm7 doctor --json` carries the rest under the target's `capabilities`:

| Key | Meaning |
| --- | --- |
| `vendor_id` | `AuthenticAMD`, `GenuineIntel`, or absent |
| `physical_cores` | Distinct (socket, core) pairs — SMT siblings folded together |
| `logical_cores` | CPUs the OS exposes, SMT siblings included |
| `isa_extensions` | Vector and matrix extensions, named as the kernel names them |

`isa_extensions` is the interesting one. It records only the flags that change
how LM7 should compile or quantize — `avx2`, the AVX-512 family, `avx512_vnni`,
`avx512_bf16`, and the AMX trio on x86; `asimd`, `sve`, `bf16`, and `i8mm` on
AArch64 — because those decide whether BF16 arithmetic is native or emulated and
whether an INT8 GEMM has a dot-product instruction behind it.

Two things this does *not* mean:

- **LM7 does not yet act on these flags.** They are reported, not consulted. The
  CPU compute dtype is still pinned to FP32 for every x86 host regardless of what
  is detected — see [quantization](quantization.md).
- **An absent flag means "not reported", not "not supported".** The source is
  `/proc/cpuinfo`, so on a host without `/proc` the list is empty and the core
  counts fall back to what Python can see. Treat empty as unknown.

## Threads

Thread count is the dominant CPU latency knob, and `lm7.benchmark` takes one:

```python
lm7.benchmark(model, args=(x,), target="cpu", backend="inductor", threads=8)
```

`benchmarks/local.py` sweeps it:

```bash
python benchmarks/local.py --target cpu --backend eager inductor \
  --threads 1 2 4 8 16 --batch-size 64 --warmup 5 --repeats 30
```

On an AMD EPYC 7B13 — 8 physical cores, 16 logical, AVX2 — that gives:

| Threads | eager | inductor | vs 1 thread |
| --- | --- | --- | --- |
| 1 | 15.49 ms | 15.27 ms | 1.00x |
| 2 | 8.82 ms | 9.25 ms | 1.71x |
| 4 | 5.27 ms | 5.33 ms | 2.90x |
| 8 | **3.47 ms** | **3.39 ms** | **4.50x** |
| 16 | 3.80 ms | 3.71 ms | 4.15x |

Two things to take from it:

- **Scaling stops at the physical core count.** Going from 8 threads to 16 —
  one per logical CPU — is about 9% *slower*, not faster. The two SMT siblings
  of a core share one vector unit, so a compute-bound GEMM gains nothing from
  the second and pays for the contention. A larger MLP on the same host showed
  the same shape at 31%, so treat the size of the penalty as workload-dependent
  and the direction as reliable.
- **Torch's default is usually already right.** It picks the physical core
  count, which is the optimum above, so passing `threads` is for measuring the
  curve or for hosts where that default guesses wrong — a cgroup CPU limit, for
  instance, which `nproc` reports but `/proc/cpuinfo` does not.

`threads` is a benchmark parameter, not a `compile` one. `torch.set_num_threads`
is process-global, so a compiled module cannot own a thread count without
silently changing every other module in the process. `lm7.benchmark` pins it for
the duration of one measurement and restores it afterwards, which is what keeps
a sweep from reporting its first number five times.

Every CPU result records the host it came from — `device_name`, `vendor_id`,
`physical_cpu_count`, `isa_extensions`, `total_memory_bytes`, and the
`torch_threads` actually used — so a saved run can be compared against one from
another machine without guessing at what produced it.

## Validate CPU and GPU locally

The correctness example runs identical weights and inputs through CPU
TorchInductor and, when available, NVIDIA or Apple Silicon TorchInductor:

```bash
python examples/local_targets.py
python examples/local_targets.py --require-nvidia
python examples/local_targets.py --require-apple
```

Run the real CPU integration test directly:

```bash
python -m pytest tests/test_cpu_integration.py -q
```

Compare first-call compilation cost and steady-state inference:

```bash
python benchmarks/local.py \
  --target cpu nvidia \
  --backend eager inductor \
  --batch-size 8 \
  --warmup 5 \
  --repeats 30
```

The benchmark uses FP32 on both targets so the numbers are directly
comparable. Production GPU inference will often use FP16 or BF16 instead.

## Is OpenVINO required?

No. TorchInductor already provides the generic CPU compiler path used by LM7,
and PyTorch can generate optimized C++ CPU kernels without OpenVINO.

OpenVINO is now available as an **opt-in** backend for Intel CPU deployment and
for OpenVINO IR artifacts:

```bash
uv pip install -e ".[openvino]"
```

```python
model = lm7.compile(model, target="cpu", backend="openvino")
```

It is not an automatic choice. It ranks below Inductor and AOTInductor, so
`backend="auto"` never selects it — the evaluation established a latency win on
Intel but not broad operator coverage, and pulling a large optional runtime into
generic CPU support is not needed for correctness.

Reach for it when you want an artifact that runs without PyTorch, or when you
are deploying to Intel hardware specifically. See the
[OpenVINO evaluation](openvino-evaluation.md) for the measurements behind that
and for the backend's documented limits.

## Shrinking a model on CPU

`lm7 model run` can quantize a Hugging Face causal LM's weights to INT8 on CPU,
which is the one weight-only mode measured off NVIDIA:

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --quantize int8
```

That cuts SmolLM2-135M from 513 MiB to 210 MiB with the same next token on every
prompt tried. Compute stays FP32, because x86-64 without AVX-512 has no native
BF16 path.

> [!TIP]
> On Intel CPU, `--backend openvino --quantize int8` is the faster INT8 route and
> uses a different mechanism (NNCF on the IR, not TorchAO on the modules). Measured
> on SmolLM2-135M it is 1.83x faster than FP32 on an AVX2 part and 2.53x faster on
> an AVX-512 + VNNI Xeon, where the TorchAO path below is *slower* than FP32 on
> both. See [quantization](quantization.md).

The footprint saving is reliable; the latency effect is not, and depends mostly on
model size. Measured at sequence length 16, INT8 was 1.5x slower for
SmolLM2-135M and 2.3x slower for Llama-3.2-1B on an AVX2-only part — and
**re-measuring on an AVX-512 + VNNI Xeon did not recover the 1B regression**,
because weight-only quantization leaves activations in FP32 and so never issues an
INT8 GEMM for `vpdpbusd` to accelerate. Measure it for your model rather than
assuming a newer CPU will help. See [quantization](quantization.md).

For an artifact to ship to an edge device rather than a process to run here, the
ExecuTorch backend has a separate calibrated INT8 export flow — see
[ExecuTorch](executorch.md).

## Shipping a smaller artifact

`lm7 model run --quantize int8` shrinks a model inside the current process. To
produce something to deploy, export an OpenVINO IR with NNCF weight compression:

```bash
uv pip install -e ".[hf,openvino]"
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct out.lm7 \
  --backend openvino --target cpu --quantize int8
```

That writes a 135 MB IR against 538 MB for FP32 — 3.98x smaller, because NNCF
compresses the embedding and vocabulary projection that the TorchAO runtime path
leaves alone. It loads without PyTorch, and on this host it was the one
quantization path measured *faster* than its FP32 baseline. See
[quantization](quantization.md) for the numbers and the per-model gate.
