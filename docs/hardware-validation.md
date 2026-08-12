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

## What to write down

Record:

- instance type, CPU/GPU/NPU name, core count, memory, OS, and notable ISA or
  accelerator capability flags;
- exact LM7 branch/commit;
- installed runtime versions, for example PyTorch and OpenVINO;
- tests and commands run;
- median latencies only with command shape, model, batch, warmup, repeats, dtype,
  and whether the run was a bounded smoke benchmark or a full sweep;
- skipped paths and why.

Do not add a hardware row for a failed setup. Record the blocker in the relevant
backend or hardware doc instead.
