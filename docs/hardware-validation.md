# Hardware validation procedure

Use this when adding a physical machine to [tested hardware](tested-hardware.md).
The goal is to record evidence that LM7 really ran on the device, without
turning every rented host into a long benchmark campaign.

## Scope

For a new CPU host, collect three levels of evidence:

- detection and backend planning;
- at least one real compile path with `fallback="error"`;
- one backend-specific integration test or bounded benchmark for the runtime
  being claimed.

If the docs mention an LLM, run an LLM. A synthetic MLP validates the compiler
path, not language-model coverage.

A GPU host needs the same three, plus two the CPU procedure has no equivalent
of, and both come from the machine being **metered**:

- **The order is chosen so that losing the box early still leaves a result.**
  Identity and detection first, because they cost seconds and are what a
  hardware row is actually made of; weights last, because a checkpoint download
  can consume the whole rental. See [GPU hosts](#gpu-hosts).
- **Getting the JSON off the box is a step, not an afterthought.** It is the one
  failure that wastes the entire rental, and `rsync` is not present on these
  images.

`fallback="error"` matters more on a GPU than on a CPU: a silent fall-back to
eager still produces plausible latencies on an accelerator, and reads as a
successful compile.

## Setup

Start from the exact branch you plan to document:

```bash
git rev-parse --short HEAD
uv venv --python 3.12
uv pip install torch --torch-backend=auto
uv pip install -e ".[dev]"
source .venv/bin/activate
```

On minimal Linux images, TorchInductor CPU needs a C++ compiler:

```bash
c++ --version || sudo apt-get update
c++ --version || sudo apt-get install -y g++
```

Install only the extras needed for the validation claim. For Intel CPU with
OpenVINO and Hugging Face:

```bash
uv pip install -e ".[openvino,hf]" pytest numpy
```

## Record the host

Capture the machine and runtime identity:

```bash
uname -a
lscpu | sed -n '1,35p'
lm7 targets
lm7 backends
lm7 explain --target cpu
```

For OpenVINO:

```bash
python - <<'PY'
import openvino as ov

core = ov.Core()
print(ov.__version__)
print(core.available_devices)
if "CPU" in core.available_devices:
    print(core.get_property("CPU", "FULL_DEVICE_NAME"))
PY
```

## CPU smoke test

This proves LM7 compiles and does not silently fall back:

```bash
python - <<'PY'
import torch
import lm7

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
).eval()
x = torch.randn(3, 4)
ref = model(x)
compiled = lm7.compile(model, target="cpu", backend="auto", fallback="error")
out = compiled(x)
print(compiled.target)
print(compiled.selected_backend)
print(torch.max(torch.abs(ref - out)).item())
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("LM7_CPU_COMPILE_OK")
PY
```

## OpenVINO CPU validation

Run the non-NPU integration subset:

```bash
python -m pytest -q tests/test_openvino_integration.py -k "not npu"
```

Then run bounded benchmarks. Keep repeats low unless you are writing a full
evaluation page:

```bash
python benchmarks/openvino_eval.py \
  --path eager inductor openvino openvino_ir \
  --model mlp --batch-size 8 --warmup 10 --repeats 10 \
  --device CPU --require-all \
  --output artifacts/benchmarks/openvino-mlp-b8.json

python benchmarks/openvino_eval.py \
  --path eager inductor openvino_ir \
  --model mlp --batch-size 32 --warmup 10 --repeats 10 \
  --device CPU --require-all \
  --output artifacts/benchmarks/openvino-mlp-b32.json
```

## LLM validation

Run a configuration preflight first:

```bash
lm7 model compatibility hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target cpu --backend auto
```

Then run one bounded causal-LM prefill benchmark:

```bash
python benchmarks/openvino_eval.py \
  --path eager inductor openvino_ir \
  --model smollm2 --batch-size 1 --warmup 2 --repeats 5 \
  --device CPU --require-all \
  --output artifacts/benchmarks/openvino-smollm2-b1.json
```

Finally run the public LM7 HF command:

```bash
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" --target cpu --json
```

Do not describe this as generation validation. `model run` checks the forward
path and next-token logits. Use `lm7 model generate` or the generation
benchmarks if the claim is about decode.

## Decode, if the claim is about generation

The forward-pass benchmarks above say nothing about the memory-bound half of
generation, and compiling is worth a different amount to each:

```bash
python benchmarks/decode.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --target cpu --sequence-length 128 512 --batch-size 1 \
  --decode-steps 32 --warmup-steps 4 \
  --output artifacts/benchmarks/decode.json
```

Read `recompiled` and `same tokens` before reading the latencies. A recompile
during the steady loop invalidates the number rather than being jitter.

## What the matrix unit is doing, on a part that has one

Any x86 host advertising `amx_*` or an Arm host advertising `bf16`/`i8mm` should
get this, because the flags alone say nothing — LM7 pins the CPU compute dtype to
FP32, so the hardware may be entirely idle:

```bash
python benchmarks/cpu_matrix_unit.py --model mlp --backend eager \
  --rows 1 8 64 512 --repeats 30 \
  --output artifacts/benchmarks/matrix-mlp.json

python benchmarks/cpu_matrix_unit.py --model smollm2 --backend eager inductor \
  --rows 5 64 --repeats 20 \
  --output artifacts/benchmarks/matrix-smollm2.json
```

Split the run by model and row count rather than sweeping everything at once —
each cell re-runs under `ONEDNN_VERBOSE=all` in a child process, and a 512-token
causal LM is the expensive case. Three things to read out of the JSON:

- `matrix_matmul` — `yes`/`no` on x86, where the kernel name carries the proof,
  and `unknown` on AArch64, where it does not. Do not write `no` for `unknown`.
- `onednn_isa` — `null` means oneDNN never ran for that cell at all, which is a
  real outcome (ATen/MKL served the linears) and a limit on every later
  kernel-level question about that model on that host.
- `rejected_matmul` — what oneDNN declined and why, which is the only evidence
  left when the chosen kernel's name is uninformative.

## Artifact lifecycle, if the host will export

One process cannot see the failures that matter for a shipped artifact, so this
runs export, reload and the mismatch cases as separate processes:

```bash
python benchmarks/aot_artifact_lifecycle.py run --model smollm2 \
  --target cpu --results-dir artifacts/aoti-<host> --repeats 15
```

Every mismatch case should be `rejected` with a message naming the cause. A case
that loads is the finding.

## GPU hosts

Everything above applies. What follows is the GPU-specific ordering, written for
a rented, metered box where the budget is wall-clock rather than effort.

### Before the clock starts

Land the code locally. A rented GPU is a data-collection window, not a
development window — `CLAUDE.md` puts it as *"collect JSON into `artifacts/` on
the box and author docs locally afterwards; don't write prose on a rented GPU."*
Anything that needs a compiler, a test run, or a second opinion should already be
on a branch before the first SSH.

### Prove the numbers are yours, before you collect any

`git rev-parse` below records which commit you *checked out*. On a shared or
reused box that is not the same question as which code ran, and the gap is
silent — every hazard here produces plausible numbers rather than an error.

**An editable install pins `import lm7` to one path, whichever directory you
run from.** `pip install -e` writes a `.pth` naming an absolute `src`, so a
second checkout does not get used by cd-ing into it:

```bash
python -c "import lm7; print(lm7.__file__)"   # the only answer that counts
```

If that is not the tree you think you are measuring, fix it before anything
else. `PYTHONPATH=<checkout>/src` takes precedence over the `.pth` and is enough;
re-check the line above rather than assuming it worked.

**The working tree may not match the commit.** These boxes get reused across
sessions, and uncommitted work in `src/` is measured exactly like committed work:

```bash
git status --short                      # expect empty
git diff --stat origin/main -- src/ benchmarks/
```

Docs-only drift is harmless. Anything under `src/` or `benchmarks/` means the
run describes code that is not on any branch. Clone to a fresh directory at a
known commit and point `PYTHONPATH` at it rather than checking out over someone
else's work.

**Another process can hold the GPU without using it.** An inference server
parked at idle still reserves its memory pool — vLLM defaults to 90% of VRAM —
so latency measured beside one is contended, and neither run says so:

```bash
rocm-smi --showmemuse | grep VRAM%      # or: nvidia-smi --query-gpu=memory.used --format=csv
```

Expect roughly zero before a timing run, and check again afterwards. This is the
one hazard that also damages *someone else's* numbers, so on a shared box find
out what is running before ending it.

### Identity, first, because it is cheap and it is the row

```bash
nvidia-smi || rocminfo | grep gfx     # whichever vendor this is
git rev-parse --short HEAD
lm7 doctor --json  > artifacts/<host>/doctor.json
lm7 targets --json > artifacts/<host>/targets.json
lm7 explain --target <vendor> --json
```

`--json` is where the detail lives: `lm7 targets` prints the generation and
precision line, and the JSON additionally carries `capabilities` (`hip`,
`gcn_arch_name`, `compute_capability`, `fp8_format`) and `cuda_build`, which
answers whether the installed wheel has kernels for this exact chip. Save both
files — they are what a hardware row is written from weeks later.

`lm7 explain` is a real check, not a formality: it resolves the target against
detected hardware, so on a host without that vendor it exits 2 with
`Requested target ... was not found locally`. Its success is evidence.

### One real compile, then the vendor's integration tests

```bash
python -m pytest -m cuda -q       # or -m rocm, -m mps, -m tpu
```

### The matrix, which is where comparable numbers come from

[`benchmarks/nvidia_matrix.py`](../benchmarks/nvidia_matrix.py) is the per-GPU
suite, and despite the name it takes `--target amd` as well — see [the NVIDIA
validation suite](nvidia-validation.md#it-describes-an-amd-gpu-too-which-is-the-point-of-having-it).
Use it rather than `benchmarks/gpu.py` or `benchmarks/moe.py` for anything that
will be compared against another card: **the harnesses disagree by 2.3x** on the
same card and model, so mixing them can invert a conclusion. Whatever you
report, name the harness that produced it.

```bash
python benchmarks/nvidia_matrix.py --plan core | while read -r args; do
  python benchmarks/nvidia_matrix.py $args --target <vendor> --results-dir artifacts/<host>
done
python benchmarks/nvidia_matrix.py --summarize artifacts/<host>
```

One cell per process is deliberate — a compiler can abort the interpreter and a
large model can poison the device context for everything after it — and each
cell writes its JSON immediately, so a box reclaimed mid-sweep costs one cell
rather than the run. Cells that fail record `works: false` with a traceback;
cells for a path this vendor never had record `skipped` with a reason. Do not
read the second as the first.

Other plans, in rough order of cost: `artifacts`, `quant`, `large`, `moe`.

### When a harness dies mid-sweep, isolate before you debug

One cell per process is the matrix's design; on a new vendor it is also the
first thing to try on any *other* harness that will not finish. Most multi-arm
benchmarks — `benchmarks/decode.py` is the worked example — run every arm in one
process, which means one arm can take the run down with it.

Split the loop the driver would have run, one process per cell, and let a
crashing cell cost only itself:

```bash
for arm in eager inductor cudagraphs decode-only; do
  python benchmarks/decode.py --target <vendor> --arm "$arm" \
    --sequence-length 512 --batch-size 1 \
    --output artifacts/<host>/cells/$arm-s512-b1.json
  echo "$arm rc=$?"
done
```

Two things this buys beyond finishing. It tells you whether the failure is a
shape, an arm, or an *ordering* — on the MI300X the answer was ordering, and the
first two diagnoses (a sequence length, then graph capture at that length) were
both wrong because a crash that only sometimes happens looks shape-dependent.
And it turns "the sweep died" into a bounded set of results plus a defect.

Two things it costs, and both belong in whatever you write:

- **Isolation is a different execution mode.** A fresh allocator and cold caches
  per cell is not obviously worth nothing, so numbers taken this way are not
  strictly row-for-row against a sweep that ran arms in one process.
- **Cross-arm assertions stop working.** A harness that compares each arm to a
  reference arm compares each cell to *itself* once isolated, so fields like
  "same tokens as reference" become vacuously true. Recover the check by
  comparing the recorded outputs across cells afterwards, and say that you did.

A crash worth reporting needs the signature, not the symptom: capture whether it
is a Python exception, a non-zero exit, or a fault, and for a fault check
`dmesg | tail` for the faulting binary. `segfault ... in python3.12` and three
different glibc messages across runs is one heap-corruption bug, not three bugs
— and it is a different conversation from an OOM, which `rocm-smi` and `free -g`
rule in or out in a second.

### The artifact lifecycle, with the GPU's own target

[Above](#artifact-lifecycle-if-the-host-will-export) runs on `--target cpu`.
Point it at the GPU instead — the cross-process part is the whole point, and on
a GPU it also exercises the architecture guard, which is the check that a
package built for one chip refuses to load on another:

```bash
python benchmarks/aot_artifact_lifecycle.py run --model smollm2 \
  --dtype bfloat16 --target <vendor> \
  --results-dir artifacts/aoti-<host> --repeats 15
```

Every mismatch case should be `rejected` with a message naming the cause; a case
that loads is the finding. On a first-of-its-kind card the guard message is
itself worth pasting into the hardware doc, because it is the first time that
code path has named this architecture.

Run it against the vendor's *own* loader as well as LM7's where the harness
offers both. The MI300X session found a torch bug that way — `aoti_load_package`
uses `torch._inductor.codecache` without importing it — which was visible only
because the arm calling PyTorch directly failed where the arm calling
`lm7.load_artifact` succeeded.

### Weights last, and queued in the background from minute zero

Downloads are usually the binding constraint, and they are the one thing that
can run while everything above is happening. Queue them smallest-first so a
short session still lands the small ladder, and treat anything above ~15 GB as
optional. Sizing traps worth knowing before you plan around a name: Qwen3-1.7B
is 2.03B parameters, Mixtral-8x7B is ~93 GB at BF16, and the Llama checkpoints
need the `unsloth/` mirrors because the Meta repos are gated and a rented box
has no HF token.

### Get the results off the box

```bash
tar czf - artifacts/<host> | ssh <local> 'cat > <host>-artifacts.tar.gz'
```

`rsync` is not installed on these images. Do this before the clock runs out, not
after the last benchmark — a sweep that finished and was never retrieved is
worth exactly as much as one that never ran.

## What to write down

Record:

- instance type, CPU/GPU/NPU name, core count, memory, OS, and notable ISA or
  accelerator capability flags;
- exact LM7 branch/commit;
- installed runtime versions, for example PyTorch and OpenVINO;
- tests and commands run;
- median latencies only with command shape, model, batch, warmup, repeats, dtype,
  **which harness produced them**, and whether the run was a bounded smoke
  benchmark or a full sweep;
- skipped paths and why;
- **which tree actually ran** — the `lm7.__file__` the process resolved and
  whether `src/` was clean at that commit — and, if the box was shared, that the
  GPU was idle when timing started;
- **whether cells ran isolated or in one process**, if the harness supports both,
  and which cross-arm checks were recovered afterwards rather than asserted
  during the run.

Do not add a hardware row for a failed setup. Record the blocker in the relevant
backend or hardware doc instead.
