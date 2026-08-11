# AOTInductor artifacts across a process boundary

What an `lm7.export(backend="aot_inductor")` artifact costs to reload in a
process that never compiled it, what it refuses to load on, and what it turns
out not to care about.

Measured through
[`benchmarks/aot_artifact_lifecycle.py`](../benchmarks/aot_artifact_lifecycle.py)
on three machines, all `torch 2.13.0+cu130` / CUDA 13.0:

| | GPU | host | used for |
| --- | --- | --- | --- |
| **B1** | RTX PRO 6000 Blackwell (`sm120`), driver 580.126.20 | Lightning studio, 48 vCPU, network filesystem | the MLP |
| **B2** | the same card, same driver | a second Lightning studio, 16 vCPU, local overlay | SmolLM2, Llama-3.2-1B |
| **A** | RTX 4070 SUPER (`sm89`), driver 595.71 | local WSL2 box | the foreign-architecture and cross-version artifacts |
| **H** | H100 80GB HBM3 (`sm90`), driver 580.173.02 | Lightning studio, 24 vCPU Xeon 8470 | [the Hopper rows](#hopper-sm90-a-third-architecture) |

Timings are never compared across those rows: B1 has three times the cores of
B2, and A was shared with another workload throughout. Where a comparison
matters it is between two measurements taken back to back on one machine.

## Why a separate process

The [backend matrix](nvidia-blackwell.md#the-backend-compatibility-matrix)
already has an `aot_inductor` export/reload row, and it exports and reloads
inside one interpreter. That is a weaker claim than it reads as: the compiling
process still holds Inductor's caches, the model's source library is still
imported, the CUDA context is warm, and nothing has had to interpret the
manifest as a stranger would.

So the stages here do not share an interpreter. `export` writes the artifact and
exits; `load` starts a fresh interpreter, optionally drops the artifact's pages
from the page cache, reloads it, and checks the result against a reference the
export process saved; `mismatch` breaks exactly one thing and records how the
failure reads.

```console
$ python benchmarks/aot_artifact_lifecycle.py run --model smollm2 \
    --results-dir artifacts/aoti --other-python /path/to/torch-2.12/bin/python
```

## On a 1B model, loading the artifact costs what compiling it costs

Machine **B2**, FP16, fresh interpreter, cold page cache. Time from process
start to a finished first inference:

| path | SmolLM2-135M | Llama-3.2-1B |
| --- | --- | --- |
| `inductor` JIT, cold Inductor cache | 20.25 s | 14.69 s |
| `inductor` JIT, warm Inductor cache | 7.65 s | 7.58 s |
| artifact via `lm7.load_artifact` | 6.31 s | **14.42 s** |
| artifact via `aoti_load_package` | 3.65 s | **5.69 s** |

Read the Llama column. **An artifact loaded through LM7's own API reaches its
first token no sooner than compiling the model from scratch** — 14.42 s against
14.69 s. The entire point of shipping a precompiled artifact is given back at
load time. Through PyTorch's own loader the same artifact starts in 5.69 s, 2.6x
faster than the JIT, so the artifact is doing its job and the API in front of it
is not.

The reload itself, same conditions:

| model | artifact | `lm7.load_artifact` | `aoti_load_package` | ratio |
| --- | --- | --- | --- | --- |
| SmolLM2-135M | 0.55 GB | 4.86 s | 2.41 s | 2.0x |
| Llama-3.2-1B | 4.95 GB | 10.71 s | 4.28 s | 2.5x |

Where the extra 2.44 s and 6.43 s go, each component timed in its own process:

| | SmolLM2 | Llama-3.2-1B |
| --- | --- | --- |
| SHA-256, both files | 0.37 s | 3.18 s |
| — of which the program half | 0.19 s | ~1.6 s |
| `torch.export.load` of the program | 2.67 s | 3.16 s |

Checksums are not free at scale: SHA-256 runs at about 1.5 GB/s here, so a
4.95 GB artifact costs 3.18 s to verify — as much as parsing the program does.

Both costs trace to one structural fact. The artifact is **half source program**:

| | payload | program |
| --- | --- | --- |
| SmolLM2-135M | 273 MB | 274 MB |
| Llama-3.2-1B | 2475 MB | 2474 MB |

So an AOTInductor caller hashes 2.47 GB it will not run, then spends 3.16 s
parsing it into an `ExportedProgram` it will not call — roughly 4.8 s of the
6.43 s gap. The remaining ~1.6 s is hashing the payload itself, which is the
integrity check doing its job. The program earns its place in the artifact —
inspection, rebuilds, `backend="export"` fallback and the bundle story all need
it — but nothing here needed it today.

The fix is a lazy `exported_program` on `ExportArtifact`, which would skip both
its hash and its parse until something asks for it. That is a public dataclass
field today, so it is an API decision rather than a cleanup, and it is not made
here. On the 35 MB MLP the same overhead is 3–8%, which is exactly how one small
model hides a regression this size.

### What it costs to build, and what it buys per call

Build time on **B2**, and the steady-state call it produces:

| model | capture | Inductor | artifact | steady, artifact | steady, JIT |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 2.42 s | 33.55 s | 0.55 GB | 2.323 ms | 4.065 ms |
| Llama-3.2-1B | 1.45 s | 40.18 s | 4.95 GB | 2.750 ms | 3.183 ms |

Compile time tracks the graph, not the parameter count: SmolLM2-135M has 30
layers against Llama-3.2-1B's 16, and takes *longer* to JIT-compile (20.25 s vs
14.69 s cold) while being nine times smaller.

The packaged artifact is also faster per call than the JIT it came from — 1.75x
on SmolLM2, 1.16x on Llama — the same effect the matrix records for TensorRT,
whose serialized engine beat its in-process compile by 1.48x. The advantage
shrinks as the model grows, which is what you would expect if it is per-call
framework overhead being amortized.

Numerics match the existing matrix exactly: Llama-3.2-1B reloads to a
`max_abs_diff` of `2.441e-02` against eager, which is the figure
[nvidia-blackwell.md](nvidia-blackwell.md#the-backend-compatibility-matrix)
already records for the in-process `aot_inductor` row. SmolLM2 lands at
`2.109e-01`. Both agree with eager on the greedy next token.

## Reload on `sm120`, small model

Machine **B1**. The 8.4 M-parameter MLP (`8x1024 -> 4096 -> 1024`, FP16), median
of 20 after 5 warmup calls.

| stage | wall | reload | to first inference | steady |
| --- | --- | --- | --- | --- |
| export (process A) | 16.33 s | — | — | — |
| `lm7.load_artifact`, cold | 3.63 s | 1.779 s | 2.90 s | 0.0367 ms |
| `lm7.load_artifact`, warm | 3.58 s | 1.746 s | 2.80 s | 0.0367 ms |
| `aoti_load_package`, cold | 3.33 s | 1.720 s | 2.75 s | 0.0358 ms |
| `aoti_load_package`, warm | 3.14 s | 1.612 s | 2.57 s | 0.0357 ms |
| `inductor` JIT, cold cache | 5.80 s | — | 3.43 s | 0.0711 ms |
| `inductor` JIT, warm cache | 5.34 s | — | 2.72 s | 0.0677 ms |

Build was 13.13 s — 0.83 s of `torch.export` capture, 12.29 s of Inductor. The
artifact is 35.2 MB. Every reload matched eager exactly (`max_abs_diff` 0.0), and
none imported `transformers` or `torchvision`.

The per-call win is 1.84x here (0.0367 ms against 0.0677 ms), the largest of the
three models and consistent with it being framework overhead: the smaller the
model, the more of the call it is.

**Time to first inference barely favours the artifact at this size**, 2.90 s
against 3.43 s, because this MLP compiles in 2.2 s and there is almost nothing
to save. That is the honest reason to measure something bigger, and why the
Llama result above is the one that matters.

### Where the 1.78 s goes

Phases of the cold `lm7.load_artifact` process on `sm120`:

| phase | ms |
| --- | --- |
| `import torch` | 741 |
| `import lm7` | 16 |
| CUDA context init | 133 |
| reload the artifact | 1779 |
| first call | 86 |
| *second reload, same process* | *45* |

**Reloading costs more than importing PyTorch.** At this size LM7's extra work is
59 ms cold and 134 ms warm — 3.3% and 7.7% of the reload — so on a small artifact
the cost really is inside `aoti_load_package`, unpacking the `.pt2` and
`dlopen`ing the wrapper. Only once the `ExportedProgram` is hundreds of
megabytes does the API in front of it start to dominate.

**A second reload in the same process is 40x cheaper** (45 ms, or 9 ms through
the torch API), so reload cost is per-process, not per-model. That still holds
at scale, but much less dramatically: on Llama-3.2-1B a second
`lm7.load_artifact` is 7.13 s against 10.71 s, because the checksums and the
`ExportedProgram` are read again from scratch.

**Cold and warm are within noise at every size measured** — 1.779 s vs 1.746 s
for the 35 MB MLP, 4.86 s vs 4.49 s for SmolLM2, 10.71 s vs 9.20 s for the 4.95
GB Llama artifact. The pages really were evicted (`POSIX_FADV_DONTNEED` after a
sync, and the harness records whether the call was available); reload is bound by
decompressing and linking, not by reading. Storage speed is not where this cost
lives.

## Hopper (`sm90`), a third architecture

Machine **H**, `torch 2.13.0+cu130`, CUDA 13.0, Python 3.12.3. Same harness, same
three models, every stage a separate process.

| model | capture | Inductor | artifact | `lm7.load_artifact` cold | warm | `aoti_load_package` cold | ratio | to first inference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 1.34 s | 22.56 s | 35 MB | 3.222 s | 3.289 s | 3.216 s | 1.00x | 5.30 s |
| SmolLM2-135M | 3.23 s | 41.70 s | 0.55 GB | 6.332 s | 6.165 s | 3.725 s | 1.70x | 9.60 s |
| Llama-3.2-1B | 2.00 s | 55.22 s | 4.95 GB | 13.144 s | 12.272 s | 5.179 s | 2.54x | 37.30 s |

Every reload agreed with the in-process reference (`max_abs_diff` 0.0 on the MLP,
FP16-level on the two causal LMs), and none imported `transformers` or
`torchvision`.

**Artifact bytes are identical across architectures.** 0.55 GB and 4.95 GB here
match machine B2 exactly, and the MLP's 35 MB matches B1's 35.2 MB. The payload
is the same size whichever card built it — only the time to produce it moves.

**Hopper builds and reloads roughly 1.8x slower than Blackwell.** The MLP takes
23.9 s to build here against 13.13 s on B1 (Inductor 22.56 s against 12.29 s),
and reloads in 3.222 s against 1.779 s. That is worth stating because
"datacenter part" reads as "faster": on this workload the RTX PRO 6000 wins, the
same way it does on the small launch-bound benchmarks in
[NVIDIA H100](nvidia-h100.md).

**The `lm7.load_artifact` overhead reproduces.** The 2.5x ratio B2 measured on
Llama-3.2-1B lands at 2.54x here, and SmolLM2's 2.0x at 1.70x. That is the
checksum-and-`torch.export.load` cost [already broken down above](#on-a-1b-model-loading-the-artifact-costs-what-compiling-it-costs),
confirmed on a second architecture rather than a new finding.

Phases of the cold `lm7.load_artifact` process:

| phase | MLP | SmolLM2 | Llama-3.2-1B |
| --- | --- | --- | --- |
| `import torch` | 1266 ms | 1352 ms | 1669 ms |
| `import lm7` | 28 ms | 28 ms | 45 ms |
| CUDA context init | 129 ms | 126 ms | 170 ms |
| reload the artifact | 3222 ms | 6332 ms | 13144 ms |

> [!NOTE]
> **Llama-3.2-1B's cold time to first inference (37.30 s) is not reproducible and
> should not be quoted.** Warm is 13.98 s for the same artifact, while the raw
> `aoti_load_package` path moves only 7.17 s → 6.78 s. A 4.95 GB payload read for
> the first time off a Lightning studio's filesystem is the obvious explanation
> and it was not isolated. The `load_ms` column above is the number to compare;
> it is stable cold-to-warm (13.144 s against 12.272 s).

### The architecture guard on a third card

All six mismatch cases were checked against all three models — **18 of 18
rejected, every one with a clear message**:

```text
its aot_inductor payload was built for nvidia:sm89, but this machine is
nvidia:sm90
```

That is the guard refusing an `sm89`-labelled artifact on Hopper, which
previously had only been shown as `sm89` refused on `sm120`. Three architectures
make it a property rather than a coincidence between two cards.

The seventh case, `torch-version`, needs a second interpreter via
`--other-python` and was **not** run here; its result is the `sm120` one.

## What an artifact refuses

Each case takes a valid artifact, changes one thing, and loads it in a fresh
process. The six byte-and-metadata cases ran against all four artifacts (MLP,
SmolLM2, Llama-3.2-1B, and the foreign `sm89` build) with identical outcomes;
the PyTorch-version case needs a second interpreter and ran on **B1** and **A**.

| case | outcome | what the user sees |
| --- | --- | --- |
| architecture claims another GPU | rejected | `its aot_inductor payload was built for nvidia:sm89, but this machine is nvidia:sm120 ... Re-export on a matching machine, or ship a bundle` |
| architecture claims another CPU | rejected | `its aot_inductor payload was built for cpu:aarch64, but this machine is cpu:x86_64 ...` — the same gate, now reaching CPU packages, whose payload is a native `.so` |
| `format_version` bumped | rejected | `Unsupported LM7 artifact format 2; this LM7 version supports format 1` |
| payload byte flipped | rejected | `compiled package checksum does not match the manifest` |
| program byte flipped | rejected | `program checksum does not match the manifest` |
| payload deleted | rejected | `compiled_model.pt2 is missing` |
| payload corrupt, checksum updated to match | rejected | `Failed to initialize zip archive ... The artifact was built with PyTorch 2.13.0+cu130, CUDA runtime 13.0, GPU architecture sm89, which is what this process has, so the package or its dependencies are at fault` |
| **different PyTorch (2.13.0 → 2.12.1)** | **loaded and ran** | — |

Five of the seven are caught by metadata before PyTorch is asked to do anything.
The sixth is the only case where PyTorch has to be the one to refuse, and its own
error (`failed finding central directory`) says nothing about where the artifact
came from. LM7 now appends the build environment to that failure — and when the
environment matches, says so, because telling someone to re-export a package
that was built right here sends them to the wrong place. When something has
genuinely moved, the same message names it:

```
The artifact was built with PyTorch 2.13.0+cu130, CUDA runtime 13.0, GPU
architecture sm120, and this process differs: GPU architecture sm120 -> sm89.
An AOTInductor package holds kernels compiled for one architecture and a
wrapper linked against one CUDA runtime, so re-export the model on this machine.
```

### The architecture guard is load-bearing

The rejection above is LM7 reading a manifest, so it proves the check fires — not
that the check is *needed*. PTX is forward-compatible, and if an AOTInductor
package shipped PTX the driver could JIT it onto a newer card and the guard would
be refusing work that would have succeeded.

It does not. A real `sm89` package, built on an RTX 4070 SUPER, carried to the
Blackwell card and loaded through PyTorch directly — no LM7, no manifest:

```console
$ python benchmarks/aot_artifact_lifecycle.py load --api torch \
    --artifact foreign-sm89/mlp.aot.lm7
W torch/export/pt2_archive/_package.py:1059] Device information mismatch for
  AOTI_COMPUTE_CAPABILITY: 120 vs 89. This could cause some issues when loading
  the AOTInductor compiled artifacts.
RuntimeError: CUDA driver error: no kernel image is available for execution on
  the device
```

The package holds cubins for one architecture and nothing to fall back to. Two
things follow. The guard is necessary, not defensive. And **PyTorch's own check
is a warning** — it notices the capability mismatch, says it "could cause some
issues", and proceeds into a driver error; LM7 turns that into a refusal that
names both architectures before anything is loaded.

## Two things the measurement changed

**A PyTorch version guard would have been wrong.** The manifest has always
recorded `torch_version`, and rejecting a mismatch looked like obvious
hardening. It is not: a `2.13.0+cu130` package loaded and ran under
`2.12.1+cu130`, bit-identical, on `sm120` and again on `sm89`. LM7 still does
not enforce the version — it records it, and uses it only to explain a failure
that happened for some other reason.

The bounds matter: **newer-built loaded on older**, one minor version apart, same
CUDA major, two models. The reverse direction is not measured, nor a CUDA-major
change, nor a wider gap. It is evidence that a strict equality guard would
reject working artifacts, not a promise that any two PyTorch versions
interoperate.

**An NVIDIA artifact could not say what it was built against.** It recorded the
compute capability under `target.architecture` and nothing else — no CUDA
version, no card. A TensorRT artifact has recorded all three since it was added,
and AOTInductor is bound to the architecture in the same way and refused on the
same grounds. Now:

```json
"runtime_requirements": {
  "api_status": "beta",
  "compute_capability": "sm120",
  "cuda": "13.0",
  "device": "nvidia",
  "device_bound": true,
  "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "torch": "2.13.0+cu130"
}
```

`lm7 artifact inspect` already had a branch for a device-bound AOTInductor
package that could never fire, because nothing set `device_bound` for this
backend. It fires now.

### CPU packages are architecture-bound too

Characterized on a GCP `n4a-standard-8` (Arm Neoverse N3, Debian 12), which is
the first Linux Arm host this project could export from. A
`backend="aot_inductor"`, `target="cpu"` artifact contains:

```
compiled_model/data/aotinductor/model/<hash>.wrapper.so
  ELF 64-bit LSB shared object, ARM aarch64, version 1 (GNU/Linux)
```

That is 1.58 MB of natively compiled code, and no x86-64 host can `dlopen` it.
So a CPU AOTInductor package is bound to its architecture in exactly the way a
GPU one is bound to its compute capability — the guess this file previously
declined to make now has a measurement behind it, and `aot_inductor` is gated
for `cpu` as well as for `nvidia` and `amd`.

Two things had to change together, because the gate alone would not have fired:

- **The architecture was not being recorded.** `target="cpu"` parsed to a spec
  with `architecture=None` while `target="auto"` on the same machine resolved to
  `aarch64`, and the check is silent without a recorded architecture. Naming the
  target explicitly therefore *disabled* the guard — including for `tvm`, whose
  CPU gate this file already documents, and for `nvidia` when written without an
  `sm` qualifier.
- **Only bound backends get it.** The architecture is part of an artifact's
  identity inside a bundle, so recording `cpu:aarch64` on a portable `export`
  payload would stop it answering a request for plain `cpu`. Portable backends
  still record the vendor alone.

Apple artifacts still record nothing new. An `apple` target's architecture is
`metal`, which describes the GPU rather than the CPU the payload was compiled
for, so the same reasoning does not transfer without its own measurement.

## Scope

- **Three models, one prompt, one shape, one dtype.** A 5-token prefill at batch
  1 is close to the most launch-overhead-dominated point on the curve, which
  flatters every per-call comparison here. Larger batches would shrink the
  artifact-versus-JIT steady-state gap and would not touch the load-time
  results, which is where the finding is.
- **B2 was shared with another benchmark** for the duration. Each pair being
  compared ran back to back under the same load, and the load-time ratios are
  large (2.0x, 2.5x) relative to any plausible noise, but absolute seconds on
  that machine should be read as approximate.
- **Numerics are checked against eager on the same card** — exact for the MLP,
  `2.441e-02` for Llama-3.2-1B and `2.109e-01` for SmolLM2 at FP16, all agreeing
  on the greedy next token. That last check is as weak here as everywhere else.
- **The cross-architecture result is one direction.** An `sm89` package fails on
  `sm120`; the reverse — a Blackwell package on Ada — is not measured, and there
  is no reason to expect it to fare better.
