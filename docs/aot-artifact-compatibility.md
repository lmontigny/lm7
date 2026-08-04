# AOTInductor artifacts across a process boundary

What an `lm7.export(backend="aot_inductor")` artifact costs to reload, what it
refuses to load on, and what it turns out not to care about. Measured on an RTX
PRO 6000 Blackwell Server Edition (`sm120`, driver 580.126.20) and an RTX 4070
SUPER (`sm89`), both `torch 2.13.0+cu130`.

Everything here comes from
[`benchmarks/aot_artifact_lifecycle.py`](../benchmarks/aot_artifact_lifecycle.py),
which runs each stage as its own interpreter.

## Why a separate process

The [backend matrix](nvidia-blackwell.md#the-backend-compatibility-matrix)
already has an `aot_inductor` export/reload row, and it exports and reloads
inside one process. That is a weaker claim than it looks: the compiling process
still holds Inductor's caches, the model's source library is still imported, the
CUDA context is already warm, and nothing has been asked to interpret the
manifest as a stranger would.

So the stages here do not share an interpreter:

```console
$ python benchmarks/aot_artifact_lifecycle.py run --model smollm2 \
    --results-dir artifacts/aoti --other-python /path/to/other/venv/bin/python
```

`export` writes the artifact and exits. `load` starts a fresh interpreter,
optionally drops the artifact's pages from the page cache first, reloads it,
and checks the answer against a reference saved by the export process.
`mismatch` breaks exactly one thing and records how the failure reads.

## Reload on `sm120`

The 4 M-parameter MLP (`8x1024 -> 4096 -> 1024`, FP16), median of 20 after 5
warmup calls. One model — see [scope](#scope).

| stage | wall | reload | to first inference | steady |
| --- | --- | --- | --- | --- |
| export (process A) | 16.33 s | — | — | — |
| `lm7.load_artifact`, cold | 3.63 s | 1.779 s | 2.90 s | 0.037 ms |
| `lm7.load_artifact`, warm | 3.58 s | 1.746 s | 2.80 s | 0.037 ms |
| `aoti_load_package`, cold | 3.33 s | 1.720 s | 2.75 s | 0.036 ms |
| `aoti_load_package`, warm | 3.14 s | 1.612 s | 2.57 s | 0.036 ms |
| `inductor` JIT, cold cache | 5.80 s | — | 3.43 s | 0.071 ms |
| `inductor` JIT, warm cache | 5.34 s | — | 2.72 s | 0.068 ms |

Build was 13.13 s, of which `torch.export` capture was 0.83 s and Inductor 12.29
s. The artifact is 35.2 MB. Every reload agreed with eager exactly
(`max_abs_diff` 0.0), and none of them imported `transformers` or `torchvision`.

**The reloaded artifact is 1.9x faster per call than the JIT it came from** —
0.037 ms against 0.068 ms — on a model far too small to hide framework overhead.
This is the same effect the matrix records for TensorRT, where the serialized
engine beat the in-process compile by 1.48x. Whatever `torch.compile` keeps
doing per call, the packaged wrapper does not.

**Time to first inference barely favours the artifact here** (2.90 s vs 3.43 s
cold) and the reason is not flattering to either: the MLP compiles in about 2 s,
so there is little for the artifact to save. The gap widens with model size,
which is exactly what the bigger rows are for.

### Where the 1.78 s goes

Phase breakdown of the cold `lm7.load_artifact` process:

| phase | ms |
| --- | --- |
| `import torch` | 741 |
| `import lm7` | 16 |
| CUDA context init | 133 |
| load the artifact | 1779 |
| first call | 86 |
| *second load, same process* | *45* |

Two things worth taking from this.

**Reloading costs more than importing PyTorch, and LM7 is not why.** The
`--api torch` rows call `torch._inductor.aoti_load_package` on the payload
alone, skipping the manifest, both SHA-256 checks and the `ExportedProgram`
entirely; they land within 60–170 ms of the full API. LM7's validation is
roughly 4% of a reload at this size. The cost is inside `aoti_load_package` —
unpacking the `.pt2` and `dlopen`ing the wrapper.

**A second load in the same process is 40x cheaper** (45 ms, or 9 ms through the
torch API). Reload cost is per-process, not per-model, so a server that reloads
between requests is paying for the process, not the artifact.

**Cold and warm are within noise** (1.779 s vs 1.746 s). The pages were really
evicted — `POSIX_FADV_DONTNEED` after an `fsync`, and the harness records
whether the call was available — but at 35 MB the read is not the bottleneck.
This number is a placeholder until a multi-GB artifact goes through the same
path; do not read it as "storage does not matter".

**Nearly half the artifact never runs.** `compiled_model.pt2` is 18.4 MB and
`exported_program.pt2` is 16.8 MB. The AOTInductor runtime executes the former;
the latter is carried for inspection, rebuilds and non-AOTI fallback. It is also
why the matrix shows the export paths peaking higher in VRAM than the JIT ones.

## What an artifact refuses

Each case takes a valid artifact, changes one thing, and loads it in a fresh
process. Run on both cards.

| case | outcome | error |
| --- | --- | --- |
| architecture claims another GPU | rejected | `its aot_inductor payload was built for nvidia:sm120, but this machine is nvidia:sm89 ... Re-export on this GPU, or ship a bundle` |
| `format_version` bumped | rejected | `Unsupported LM7 artifact format 2; this LM7 version supports format 1` |
| payload byte flipped | rejected | `compiled package checksum does not match the manifest` |
| program byte flipped | rejected | `program checksum does not match the manifest` |
| payload deleted | rejected | `compiled_model.pt2 is missing` |
| payload corrupt, checksum updated to match | rejected | `Failed to initialize zip archive ... The artifact was built with PyTorch 2.13.0+cu130, CUDA runtime 13.0, GPU architecture sm89, which is what this process has, so the package or its dependencies are at fault` |
| **different PyTorch (2.13.0 → 2.12.1)** | **loaded and ran** | — |

Five of the seven are caught by metadata before PyTorch is asked to do anything.
The sixth is the interesting one, and the seventh changed the code.

**A corrupt package that passes the checksum is the only case where PyTorch has
to be the one to refuse**, and its own error (`failed finding central
directory`) says nothing about where the artifact came from. LM7 now appends the
build environment to that failure — and, when the environment matches, says so,
because telling someone to re-export a package that was built right here sends
them to the wrong place. When something has genuinely moved the same message
names the field:

```
The artifact was built with PyTorch 2.13.0+cu130, CUDA runtime 13.0, GPU
architecture sm120, and this process differs: GPU architecture sm120 -> sm89.
An AOTInductor package holds kernels compiled for one architecture and a
wrapper linked against one CUDA runtime, so re-export the model on this machine.
```

## Two things the measurement changed

**A PyTorch version guard would have been wrong.** The manifest has always
recorded `torch_version`, and rejecting a mismatch looked like an obvious
hardening. It is not: a `2.13.0+cu130` package loaded and ran under
`2.12.1+cu130`, bit-identical to eager, on `sm120` and again on `sm89`. That was
measured before anything was written, and the result is that LM7 still does not
enforce the version — it records it, and uses it only to explain a failure that
happened for some other reason.

The claim is bounded, and the bounds matter: **newer-built loaded on older**,
one minor version apart, same CUDA major (`cu130`), on the two models below. The
reverse direction is not measured, nor is a CUDA-major change, nor a gap wider
than one release. It is evidence that a strict equality guard would reject
working artifacts — not a promise that any two PyTorch versions interoperate.

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

CPU and Apple artifacts record nothing new. Their payload is bound to a host
toolchain too, but LM7 has not characterized how, and a guess in a manifest is
worse than a gap.

## Scope

- **One model on `sm120` so far.** The MLP is the smallest thing in the harness
  and the least favourable to an artifact: it compiles in about 2 s, so the
  reload-versus-recompile comparison has almost nothing to weigh. SmolLM2-135M
  and Llama-3.2-1B are the rows that matter and are not measured here — the
  Blackwell studio is interruptible-priced and was preempted mid-run.
- **Cold and warm are indistinguishable at 35 MB**, and the `sm120` artifact sat
  on Lightning's network filesystem. A multi-GB artifact on local NVMe is the
  case where "reload time" needs two numbers.
- **The `sm89` latencies are noisy.** That card was shared with another workload
  while these ran, which is fine for correctness and rejection behaviour and not
  fine for milliseconds.
- **A foreign-architecture payload is refused, never executed.** That the guard
  is *necessary* on Blackwell — that an `sm89` package would actually fail there
  rather than being JIT-compiled forward from PTX — is untested. LM7 ships real
  `sm89` artifacts through the harness to answer it; the answer needs the card.
