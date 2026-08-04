# AOTInductor artifacts across a process boundary

What an `lm7.export(backend="aot_inductor")` artifact costs to reload in a
process that never compiled it, what it refuses to load on, and what it turns
out not to care about.

Measured on an RTX PRO 6000 Blackwell Server Edition (`sm120`, driver
580.126.20) and an RTX 4070 SUPER (`sm89`), both `torch 2.13.0+cu130` / CUDA
13.0, through
[`benchmarks/aot_artifact_lifecycle.py`](../benchmarks/aot_artifact_lifecycle.py).

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

## The reload cost nobody was paying attention to

SmolLM2-135M, FP16, `sm89`, 546 MB artifact. The two rows load the same payload
and produce the same logits; they differ only in which API asked.

| API | reload | to first inference | second load, same process |
| --- | --- | --- | --- |
| `lm7.load_artifact` | 12.11 s | 15.67 s | 6.97 s |
| `torch._inductor.aoti_load_package` | 2.58 s | 4.91 s | 0.36 s |

**LM7's own API is 4.7x slower to reload the same artifact**, and the gap is not
validation overhead in any meaningful sense:

| component of `load_artifact` | seconds |
| --- | --- |
| SHA-256 over both payloads (546 MB) | 1.85 |
| `torch.export.load` of `exported_program.pt2` | 7.50 |
| `aoti_load_package` of `compiled_model.pt2` | 6.88 |

`load_artifact` eagerly loads the `ExportedProgram` **that an AOTInductor
consumer never executes**. It is carried for inspection, rebuilds, and non-AOTI
fallback, and it costs about as much to load as the compiled payload itself —
because it is about the same size (273.4 MB of program next to 272.6 MB of
kernels). Half the artifact, and half the reload, is for a path this caller did
not take.

That suggests a lazy `exported_program` on `ExportArtifact` — it is a public
dataclass field today, so it is a deliberate API change rather than something to
slip in, and it is not made here. On the 35 MB MLP below the same overhead is
3–8%, which is why one model would have hidden this entirely.

## Reload on `sm120`

The 8.4 M-parameter MLP (`8x1024 -> 4096 -> 1024`, FP16), median of 20 after 5
warmup calls. One model — the Blackwell studio is interruptible-priced and was
preempted before the larger models ran, and is off at the time of writing.

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

**The reloaded artifact is 1.84x faster per call than the JIT it came from** —
0.0367 ms against 0.0677 ms — on a model far too small to hide framework
overhead. The matrix records the same effect for TensorRT, whose serialized
engine beat its in-process compile by 1.48x. Whatever `torch.compile` keeps
doing per call, the packaged wrapper does not.

**Time to first inference barely favours the artifact here**, 2.90 s against
3.43 s, because this MLP compiles in 2.2 s and there is almost nothing to save.
The 135M model is where that gap becomes 4.91 s against a JIT path that never
finished a comparable measurement on a shared card.

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
`dlopen`ing the wrapper. The SmolLM2 table above is what the same comparison
looks like once the `ExportedProgram` is large.

**A second reload in the same process is 40x cheaper** (45 ms, or 9 ms through
the torch API), so reload cost is per-process, not per-model.

**Cold and warm are within noise at 35 MB** (1.779 s vs 1.746 s). The pages were
really evicted — `POSIX_FADV_DONTNEED` after a sync, and the harness records
whether the call was available — the read is simply not the bottleneck at this
size. At 546 MB on `sm89` the two are also close (12.11 s vs 11.85 s), which
says the same thing about a decompress-and-link-bound reload.

## What an artifact refuses

Each case takes a valid artifact, changes one thing, and loads it in a fresh
process. Every row was run on both cards, with identical outcomes.

| case | outcome | what the user sees |
| --- | --- | --- |
| architecture claims another GPU | rejected | `its aot_inductor payload was built for nvidia:sm120, but this machine is nvidia:sm89 ... Re-export on this GPU, or ship a bundle` |
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

CPU and Apple artifacts record nothing new. Their payload is bound to a host
toolchain too, but LM7 has not characterized how, and a guess in a manifest is
worse than a gap.

## Scope

- **One model on `sm120`.** The MLP is the smallest thing in the harness and the
  least favourable case for an artifact. SmolLM2-135M and Llama-3.2-1B on
  Blackwell are the rows that would matter and are not here: the studio was
  preempted mid-run and is currently off. The harness runs unchanged when it
  returns.
- **The `sm89` card was shared with another workload.** That is fine for
  correctness, sizes, and the reload-API ratio measured under identical
  conditions; it is not fine for absolute milliseconds. The JIT comparison on
  that box is omitted rather than published, because its warm-cache compile was
  no faster than its cold one and that is not explained.
- **Numerics are checked against eager on the same card** — exact for the MLP,
  `5.86e-2` max absolute difference for SmolLM2 at FP16, agreeing on the greedy
  next token. That last check is as weak here as everywhere else.
- **A foreign-architecture payload is refused, never executed.** Whether the
  guard is *necessary* on Blackwell — whether an `sm89` package would genuinely
  fail there rather than being JIT-compiled forward from its PTX — is untested.
  Real `sm89` artifacts exist for the harness to answer it with; the answer
  needs the card back.
