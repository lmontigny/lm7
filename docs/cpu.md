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
| `physical_cores` | Distinct (socket, core) pairs — SMT siblings folded together. From `/proc/cpuinfo`, or from sysfs topology where that prints none |
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

### On AArch64, the kernel prints less

Everything above is an x86 host. An Arm one fills in less of the table, because
`/proc/cpuinfo` carries less. Captured on the `ubuntu-24.04-arm` CI runner
(Azure Cobalt 100, Arm Neoverse N2, 4 vCPU, 15.6 GiB):

```
Detected targets (1):
  cpu:aarch64: Arm Neoverse N2, 15.6 GiB
```

- **The target string is `cpu:aarch64`, not `cpu:arm64`.** Both come from
  `platform.machine()`, which says `arm64` on macOS and `aarch64` on Linux — so
  the same core can arrive under either name depending on the OS. The CPU
  architecture is a free-form qualifier that no backend dispatches on, so this
  is one target family with two spellings rather than two code paths.
- **The name is reconstructed rather than read.** AArch64 publishes no
  `model name` — it prints numeric `CPU implementer` and `CPU part` identifiers,
  which LM7 maps back to a core name. That table covers Arm's own Neoverse
  server cores, which is where Graviton, Azure Cobalt, Grace and Ampere Altra
  all land, since those vendors ship Neoverse designs. An implementer or part
  outside it degrades to the vendor and the raw part number, or to nothing,
  rather than guessing.
- **`vendor_id` is absent and stays absent.** AArch64 prints no `vendor_id` at
  all, and nothing else publishes one.
- **`physical_cores` comes from sysfs here, not from `/proc/cpuinfo`.** AArch64
  prints no `physical id` or `core id` either, so the file-based count is always
  `None` on Arm. `/sys/devices/system/cpu/cpu*/topology/` carries the same
  (socket, core) pairs on every architecture, so LM7 falls back to it and counts
  distinct pairs — which folds SMT siblings together and stays right on a
  multi-socket host, exactly as the `/proc/cpuinfo` path does. x86 is unchanged:
  it still answers from the file, and sysfs only fills a gap. Note the pairs are
  what matter and not their values — a GCP Axion reports all eight of its cores
  in package `148`.
- **`isa_extensions` populates.** `bf16` and `i8mm` are both present on this
  part — the Arm analogues of `amx_bf16` and `amx_int8`. The SVE forms of the
  same instructions (`svebf16`, `svei8mm`) are on the kernel's `Features` line
  but are deliberately not recorded, because nothing here has established
  whether oneDNN reaches for those or the NEON variants — and, as the section
  below finds, oneDNN's own logs decline to say.

### What consulting the matrix-unit flags would be worth

Measured on an Intel Xeon Platinum 8559C (Emerald Rapids, 8 physical cores, 8
threads) with `torch 2.13.0+cu130` and oneDNN v3.12, through
[`benchmarks/cpu_matrix_unit.py`](../benchmarks/cpu_matrix_unit.py). AMX
accelerates BF16 and INT8 matmuls and does nothing for FP32, so with the compute
dtype pinned to FP32 none of that hardware is reached today.

Switching the same models to BF16 does reach it — oneDNN dispatches every matmul
to `brg_matmul:avx10_1_512_amx` — and what it buys depends entirely on the shape
of the GEMM. SmolLM2-135M, median latency:

| prompt | eager FP32 | eager BF16 | inductor FP32 | inductor BF16 |
| --- | --- | --- | --- | --- |
| 5 tokens | 23.27 ms | 23.16 ms | 15.85 ms | 13.32 ms |
| 64 tokens | 55.31 ms | 39.46 ms | 41.92 ms | 34.47 ms |
| 512 tokens | 205.83 ms | 92.70 ms | 158.66 ms | **81.59 ms** |

**An AMX tile is 16 rows deep, so a short prompt leaves it idle.** At 5 tokens
BF16 is worth nothing in eager (1.00x); at 512 it is worth 2.22x. The synthetic
MLP isolates the mechanism — same model, only the batch changes:

| rows | FP32 | BF16 (AMX) | |
| --- | --- | --- | --- |
| 1 | 0.358 ms | 0.466 ms | **0.77x — slower** |
| 8 | 0.916 ms | 0.413 ms | 2.22x |
| 64 | 3.278 ms | 0.673 ms | **4.87x** |
| 512 | 11.238 ms | 3.558 ms | 3.16x |

At one row the tile units have nothing to fill them and the BF16 conversion
costs more than they save. So "the host reports `amx_bf16`, therefore use BF16"
would be a regression for single-sequence decode and a large win for prefill,
which is why the flags stay reported rather than consulted until there is a
policy that can tell those apart.

The same switch on a host *without* AMX is a straight loss: on an AVX2-only
Core i7-8086K the identical MLP at 8 rows goes from 2.229 ms at FP32 to 2.673 ms
at BF16, with `amx_flags` empty and no matmul reaching a BRGEMM kernel. A dtype
policy would have to be per-host, not per-model.

Two things to know before repeating this:

- **`torch.backends.cpu.get_cpu_capability()` cannot answer the question.** It
  returns `AVX512` on this machine and never mentions AMX. Only oneDNN's own
  kernel choice does, which is why the benchmark re-runs each case under
  `ONEDNN_VERBOSE=all` and looks for a matmul reaching a BRGEMM implementation.
  oneDNN names the whole ISA `avx10_1_512_amx`, so an *eltwise* kernel carries
  `amx` in its name while doing no tile-unit work at all.
- **BF16 is a numerics change, not a speed setting.** SmolLM2's logits move by
  1.9 absolute in eager and roughly 0.4 under Inductor. That is a
  [quantization](quantization.md)-shaped decision, with the validation that
  implies, rather than a free switch.

### The same question on Arm, where the logs cannot answer it

Repeated on a GCP `n4a-standard-8` (Google Axion, Arm Neoverse N3, 8 vCPU,
`torch 2.13.0+cpu`, oneDNN v3.12) with the same benchmark. Two things came out
of it, and the second one undoes the first.

**On the synthetic MLP, BF16 wins everywhere — including where AMX loses.**
Eager, median latency:

| rows | FP32 | BF16 | | x86 AMX, for contrast |
| --- | --- | --- | --- | --- |
| 1 | 1.442 ms | 1.264 ms | 1.14x | **0.77x — slower** |
| 8 | 1.324 ms | 0.546 ms | 2.42x | 2.22x |
| 64 | 3.851 ms | 2.119 ms | 1.82x | 4.87x |
| 512 | 25.728 ms | 8.136 ms | 3.16x | 3.16x |

The AMX result above turns on a tile being 16 rows deep, so a single row leaves
it idle and BF16 costs more than it saves. Nothing on this Arm part behaves that
way: one row is already 1.14x. Read on its own, this says the x86 caveat about
single-sequence decode does not apply here.

**On a real causal LM it reverses.** SmolLM2-135M, same host, same benchmark:

| prompt | eager FP32 | eager BF16 | inductor FP32 | inductor BF16 |
| --- | --- | --- | --- | --- |
| 5 tokens | 43.59 ms | 40.36 ms | 32.01 ms | **27.46 ms** |
| 64 tokens | 94.28 ms | 302.52 ms | **76.95 ms** | 281.02 ms |

At 64 tokens BF16 is **3.2x slower** in eager and 3.7x slower under Inductor —
the opposite sign to the MLP at the same row count, where BF16 was 1.82x
*faster*. So on this part the synthetic MLP does not predict the model, and the
x86 section's habit of using it to "isolate the mechanism" does not carry over.
Which of the two generalises to other Arm parts is not established by one core.

**Neither oneDNN log can say whether the matrix instructions ran.** On x86 an
`amx` in a matmul's implementation name is the proof. Here every matmul reads
`gemm:acl` at *both* dtypes, because that names Arm Compute Library rather than
an instruction, and the dispatch log is identical for FP32 and BF16 too. The
benchmark therefore reports `matrix_matmul=unknown` on AArch64 rather than
`no` — recording "the matrix unit never ran" beside a 2.42x BF16 speedup would
be plainly false. The latency is the only evidence available, and it is
circumstantial.

What the dispatch log *does* answer is what was rejected and why:

| implementation | why oneDNN skipped it |
| --- | --- |
| `matmul:brg:sve_512` | `unsupported isa` |
| `matmul:lowp_gemm:acl` | `unsupported datatype combination` |
| `matmul:lowp_gemm_sq:acl` | `unsupported datatype combination` |

oneDNN reports the ISA as `AArch64 SVE (128 bits)`, and its Arm BRGEMM path
wants 512-bit vectors — so **that path is unreachable on this core**, and a
part with wider SVE (Graviton 3/4, Neoverse V-series) could land somewhere else
entirely. The two `lowp_gemm` rows are the INT8 kernels, declined here only
because nothing offered them INT8 operands.

Two limits on the above:

- **512-token prompts were not measured.** Each BF16 call there runs into
  seconds, and the sweep was cut for cost. The x86 table's best BF16 case is its
  512-token row, so the Arm comparison stops one length short of where AMX looks
  best.
- **Absolute latencies are not comparable to the x86 table.** Different host,
  different `transformers`, and this repo's rule that behaviour is a property of
  (model, library version) applies. The FP32-versus-BF16 ratios within each host
  are the comparable part.

The conclusion the x86 half reached — that a dtype policy has to be per-host,
not per-model — survives, and gets stronger: the policy that would be right on
Emerald Rapids (BF16 above 16 rows) is wrong on Neoverse N3 in both directions
at once, winning at one row and losing at 64 tokens of a real model. That is
still an argument for leaving the flags reported and not consulted.

## Latency on a Neoverse N3

The detection above came from a 4-vCPU shared CI runner, which is not a machine
to time anything on. These are the first latency numbers this project has taken
on Linux Arm: a GCP `n4a-standard-8` (Google Axion, **Arm Neoverse N3**, 8
dedicated vCPU, 31 GiB, Debian 12, kernel 6.1), `torch 2.13.0+cpu` on 8 threads.
Note this is a *different* Neoverse generation from the N2 in CI, so the two
hosts are two data points rather than one repeated.

[`benchmarks/local.py`](../benchmarks/local.py), FP32 MLP, median of 100 calls
after 10 warmups:

| batch | eager | inductor | |
| --- | --- | --- | --- |
| 1 | 1.548 ms | 1.616 ms | 0.96x |
| 8 | 1.306 ms | 1.374 ms | 0.95x |
| 64 | 4.079 ms | 4.025 ms | 1.01x |
| 512 | 25.598 ms | 25.311 ms | 1.01x |

**Compiling this workload buys nothing on this part, at any batch size tried.**
Every ratio is within run-to-run spread — repeating the sweep moves the batch-1
eager median between 1.44 ms and 1.58 ms on its own, which is wider than the gap
to Inductor.

That is the arithmetic working out rather than a backend falling back. The
compiled path is real: Inductor emits five `.cpp`/`.so` pairs using `at::vec` and
OpenMP, and none of the timings above come from an eager fallback. But Inductor
fuses pointwise work and does not write GEMM kernels, so both paths hand
`Linear` to the same BLAS — and this MLP is almost entirely `Linear`.
[`benchmarks/cpu_fusion_headroom.py`](../benchmarks/cpu_fusion_headroom.py)
times the layers separately and puts a ceiling on what fusion could recover:

| batch | GEMM share | what fusion can win |
| --- | --- | --- |
| 1 | 98.7% | 1.3% |
| 8 | 98.1% | 1.9% |
| 64 | 97.3% | 2.7% |
| 512 | 97.1% | 2.9% |

A 1.0x result is what a 2% ceiling predicts. The honest reading is not "Inductor
is useless on Arm" but "this workload cannot answer that question on any CPU with
a competent BLAS" — and the same caveat applies to the CPU rows of every other
`local.py` comparison in these docs.

Two things this does not establish:

- **Nothing here is a causal LM.** A 30-layer decoder is launch-bound in a way a
  2-layer MLP is not, which is exactly where fusion earns its keep. That the MLP
  shows no win predicts nothing about SmolLM2 on this part.
- **FP32 only, so the matrix unit never came into it.** `torch` reports this core
  as `SVE128`, and oneDNN does reach Arm Compute Library — a BF16 matmul
  dispatches to `gemm:acl` — but the section above still has no Arm half, because
  FP32 does not go there.

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

## Is there an AMD equivalent?

For AMD CPUs generally — what PyTorch already uses there, whether Intel's MKL
costs you anything, and the one thread-affinity setting that does — see
[AMD CPU inference](amd-cpu.md). The short version of the backend question:

Yes — `zentorch`, AMD's ZenDNN PyTorch extension, on the same opt-in terms:

```bash
uv pip install -e ".[zentorch]"
```

```python
model = lm7.compile(model, target="cpu", backend="zentorch")
```

Like OpenVINO it ranks below Inductor, so `backend="auto"` never picks it. On an
EPYC 7B13 at FP32 it ran SmolLM2-135M 1.11x faster than Inductor, tied on
Llama-3.2-1B, and lost by 1.07x on a synthetic transformer block — worth trying
on AMD hardware, not worth making the default. See [zentorch](zentorch.md).

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
