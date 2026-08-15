# MI300X session runbook

A two-hour rented AMD Instinct MI300X (`gfx942`, CDNA 3, 192 GiB), on an image
that already has a ROCm PyTorch build. Weights download inside the billed
window, which is the constraint everything below is ordered around.

The general procedure is
[docs/hardware-validation.md](../docs/hardware-validation.md#gpu-hosts). This is
the session-specific half: what to run, in what order, and what to skip when the
clock is against you. Written before the session, so anything here that turns
out to be wrong is a finding.

Why this box is worth two hours rather than one: it closes four separate gaps
that no other machine available to this project can.

| gap | where it is claimed today |
| --- | --- |
| AMD ROCm has never run on real hardware | `docs/tested-hardware.md`, `README.md`, `docs/limitations.md`, `CLAUDE.md` — four places, same words |
| No AMD `.lm7` AOT artifact exists | `docs/limitations.md` per-backend table, `docs/aot-artifact-compatibility.md#scope` |
| Quantization has "no path at all" off NVIDIA and CPU | `docs/limitations.md#quantization` |
| Mistral-7B "needs a rented card"; Qwen3-30B-A3B has run nowhere | `docs/limitations.md#model-coverage` |

## Preflight — three imports decide what is worth doing

Run these before budgeting time on anything. Each one gates a whole track.

```bash
rocminfo | grep gfx
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)"
python -c "import torchao; print(torchao.__version__)"
python -c "import migraphx; print(migraphx.__file__)"
ls -d /opt/rocm
```

- **`torchao` gates the entire quantization track.** Its PyPI wheels carry
  CUDA-built extensions, and `pip install -e ".[torchao]"` may drag a CUDA torch
  over the ROCm one. Install with `--no-deps` and re-check
  `torch.version.hip` after every install that touches torch. Measured: the
  pinned `torchao==0.17.0` installs and quantizes fine on ROCm, but prints
  `Skipping import of cpp extensions due to incompatible torch version` against
  the container's torch 2.10 — so the quantized path runs unfused in pure
  PyTorch, which is a confound on every latency number it produces and has to be
  recorded beside them.
- **`migraphx` gates the MIGraphX track, and is the single most likely way to
  lose 40 minutes.** The native bindings are not on PyPI at all and
  `torch_migraphx` JIT-builds a C++ extension on first import — see
  [docs/amd-migraphx.md](../docs/amd-migraphx.md). AMD's ROCm images usually
  ship it. **If the import fails, skip the track. Do not `apt install` on the
  clock.**
- **`/opt/rocm` gates the AOTInductor track.** The PyTorch ROCm wheel links
  against it rather than bundling it; LM7 checks for it before packaging and
  refuses by name if it is absent.

**On the AMD Developer Cloud the ROCm PyTorch is a pre-pulled Docker image, not
a system install.** The host's `python3` has no torch at all, `migraphx` imports
only on the host, and everything below runs inside the container:

```bash
docker images | grep rocm/pytorch
docker run -d --name lm7 \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --ipc=host --shm-size 32G --network host \
  -v /root/session:/session -w /session/lm7 -e HF_HOME=/session/hf-cache \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 sleep infinity
docker exec lm7 python -c "import torch;print(torch.__version__, torch.version.hip)"
```

Mount a host directory for `artifacts/` and the HF cache, as above — it is what
survives the container and what gets tarred off the box at the end.

Then, without replacing torch. `uv` is often absent on a vendor ROCm image, so
plain `pip` is the fallback rather than something to go install:

```bash
uv pip install -e ".[dev,hf]" || python -m pip install -e ".[dev,hf]"
python -c "import torch; assert torch.version.hip, 'the install replaced the ROCm torch'"
```

Put the Hugging Face cache on the data disk before downloading anything. The
list below is ~54 GB to the cut line and ~210 GB in full, and a container boot
disk is usually 20–100 GB:

```bash
df -h /
export HF_HOME=/workspace/hf-cache      # wherever the large volume is mounted
mkdir -p "$HF_HOME"
```

## The download queue, started at minute zero

Runs in the background while every cheap tier proceeds. Smallest-first, so a
short session still lands the small ladder.

| # | model | ~BF16 | why this one |
| --- | --- | --- | --- |
| 1 | SmolLM2-135M | 0.3 GB | the fast smoke test; 30 layers makes it launch-bound |
| 2 | LFM2.5-230M / 350M | 1.2 GB | 350M is on the unmeasured ladder |
| 3 | Qwen3.5-0.8B | 1.6 GB | architecture variety |
| 4 | Llama-3.2-1B | 2.5 GB | the quantization reference model |
| 5 | Qwen3-1.7B | 4 GB | unmeasured ladder — and it is **2.03B**, not 1.7B |
| 6 | OLMoE-1B-7B | 14 GB | the MoE whose *export* was abandoned on a 62 GB host |
| 7 | Mistral-7B-v0.3 | 14.5 GB | unmeasured ladder; the first dense 7B |
| 8 | Llama-3.1-8B | 16 GB | smallest model where GEMM time dominates |
| | *≈54 GB to here — everything below is optional* | | |
| 9 | Qwen3-30B-A3B | 61 GB | has never run anywhere in this repo |
| 10 | Mixtral-8x7B | 93 GB | used 93.8 of 95.0 GiB on the Blackwell; here it has headroom |

Llama needs the `unsloth/` mirrors — the Meta repos are gated and a rented box
has no HF token. Nothing else in the list is gated, `mistralai/` included.

**Filter the download or pay for it twice.** Many of these repos carry the
legacy `pytorch_model*.bin` beside the `.safetensors` for the same weights, plus
an `onnx/` tree of quantized variants; `mistralai/` adds a `consolidated*` copy
of the whole model. A bare `hf download` takes all of it — which turns
Mixtral-8x7B from ~93 GB into ~186 GB and would consume the entire rental.

**`--include` and `--exclude` take one pattern each, and extra patterns are
silently parsed as positional filenames.** This is the trap, and it fails
quietly: `hf download REPO --include "*.safetensors" "*.json"` uses only the
first as a filter, treats `*.json` as a literal filename to fetch, downloads
essentially nothing, and still exits reporting success. Measured on the first
attempt of the MI300X session: eight repos "OK" in nine seconds, 70 MB on disk.
Repeat the flag instead:

```bash
cd "$HF_HOME"
cat > queue.txt <<'EOF'
HuggingFaceTB/SmolLM2-135M-Instruct
LiquidAI/LFM2.5-230M
Qwen/Qwen3.5-0.8B
unsloth/Llama-3.2-1B-Instruct
deepseek-ai/deepseek-coder-1.3b-instruct
allenai/OLMoE-1B-7B-0924-Instruct
mistralai/Mistral-7B-Instruct-v0.3
unsloth/Llama-3.1-8B-Instruct
EOF
nohup bash -c 'while read -r repo; do
  echo "### $repo start $(date +%T)"
  hf download "$repo" --exclude "*.bin" --exclude "onnx/*" --exclude "consolidated*"
  echo "### $repo done $(date +%T)"
done < queue.txt' > download.log 2>&1 &
```

That list came down in **under three minutes for 51 GB** on the AMD Developer
Cloud box, so the ordering matters less than it was written to; check the real
link speed before planning tiers around a download.

`hf` is the current name of the CLI (`huggingface-cli` still works, and is what
older notes use). Check progress with `tail -f "$HF_HOME/download.log"` and
`du -sh "$HF_HOME"`; a tier that needs a model not yet on disk should be
skipped and returned to, not waited on.

Add `Qwen/Qwen3-30B-A3B` and `mistralai/Mixtral-8x7B-Instruct-v0.1` to the end
of the queue only once the first nine are down and there is disk for them.

## Tiers

Independently useful, in priority order. `artifacts/mi300x/` throughout.

### 1 — identity and detection (~12 min)

```bash
lm7 doctor --json  > artifacts/mi300x/doctor.json
lm7 targets --json > artifacts/mi300x/targets.json
lm7 explain --target amd --json
lm7 explain --target amd --backend aot_inductor --json
python examples/rocm_mlp.py
python -m pytest -m rocm -q
```

This tier is the hardware row on its own. It also confirms or corrects
everything the `gfx` tables predict — generation `CDNA 3`, `fp8` native,
`fp8_format: fnuz`, `fp4` absent — none of which has been seen on hardware.
Check `cuda_build.native_kernels`: on ROCm a missing `gfx` target is a hard
failure at load, not a slow start.

### 2 — the matrix, small ladder (~30 min)

The primary comparable numbers. Not `benchmarks/gpu.py` — the harnesses disagree
by 2.3x.

```bash
python benchmarks/nvidia_matrix.py --plan core | while read -r args; do
  python benchmarks/nvidia_matrix.py $args --target amd --results-dir artifacts/mi300x
done
python benchmarks/nvidia_matrix.py --summarize artifacts/mi300x
```

`tensorrt` and `onnxruntime` cells will record `skipped`, which is correct.
Worth a look on its own: the `inductor-cudagraphs` arm reports
`cudagraphs_requested` / `cudagraph_skips` / `cudagraphs_active` on AMD from the
same Dynamo counter, and nothing has ever checked whether HIP graph capture
behaves like CUDA's.

### 3 — quantization (~20 min), which needs no code change

`benchmarks/quantization.py` calls `_apply_quantization` directly and never
`_validate_quantization`, so the NVIDIA-and-CPU vendor gate does not apply to
it, and a mode this silicon refuses is recorded as an `unsupported` row rather
than ending the run.

```bash
python benchmarks/quantization.py --model llama32-1b --target amd \
  --mode none int8 fp8 nvfp4 fp8-dynamic fp8-dynamic-rowwise nvfp4-dynamic \
  --output artifacts/mi300x/quant-llama32-1b.json
python benchmarks/fp8_kernel_check.py --output artifacts/mi300x/fp8-kernels.json
```

Expect `nvfp4*` to come back unsupported — no FP4 on CDNA 3 — and that is a
result. The interesting cells are `fp8` and `fp8-dynamic-rowwise` on the first
non-NVIDIA tensor cores this project has touched, and whether torchao's
`Float8*Config` emits an `fnuz` GEMM or refuses. `fp8_kernel_check.py` is what
says which kernel actually ran.

Flipping `_QUANTIZATION_VENDORS` is a **later** PR carrying these numbers, not
something to do on the box. Whoever does it must also add `"amd"` to
`_QUANTIZED_COMPUTE_DTYPE`, or `huggingface.py`'s bare subscript raises
`KeyError` instead of an `LM7Error`.

### 4 — the AOTInductor artifact (~25 min)

Needs the branch from #173 checked out.

```bash
python -m pytest tests/test_amd_aot_integration.py -q -m rocm
python benchmarks/aot_artifact_lifecycle.py run --model smollm2 \
  --target amd --results-dir artifacts/mi300x/aoti --repeats 15
lm7 artifact inspect artifacts/mi300x/aoti/smollm2.aot.lm7 --json
```

**Whether the wrapper links against ROCm is the open question of that PR.** If
it does not, that is the finding and the PR does not merge — record the blocker
in `docs/amd-rocm.md` instead. Every mismatch case should be `rejected` with a
message naming the cause; a case that loads is the finding.

### 5 — capacity, with whatever downloaded (~20 min)

```bash
python benchmarks/moe.py --model olmoe-1b-7b --target amd --dtype bfloat16 \
  --output artifacts/mi300x/moe-olmoe.json
lm7 model run hf://mistralai/Mistral-7B-Instruct-v0.3 --target amd --json
python benchmarks/quantization.py --model llama31-8b --target amd --mode none int8 \
  --output artifacts/mi300x/quant-llama31-8b.json
```

If Qwen3-30B-A3B or Mixtral-8x7B arrived, run `moe.py` against them. Exporting
OLMoE is also newly reachable — the attempt was abandoned after destabilizing a
62 GB host twice during weight loading, and this box has 192 GiB of HBM.

### 6 — stretch: two handovers that have never been run

```bash
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target amd --dry-run --json
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target amd --backend vllm --dry-run --json
```

`serve/vllm.py` maps `amd` → `rocm` and `docs/limitations.md` says the ROCm
handover has never been run. `--dry-run` proves the argv translation for free; a
real start needs a ROCm vLLM wheel and is out of budget unless the image has one.

## At T+1:55, before anything else

```bash
tar czf - artifacts/mi300x | ssh <local> 'cat > mi300x-artifacts.tar.gz'
```

`rsync` is not on these images. A sweep that finished and was never retrieved is
worth exactly as much as one that never ran.

## Traps

- **Do not let pip replace the ROCm torch.** Re-check `torch.version.hip` after
  every install.
- **Name the harness beside every number.** `nvidia_matrix.py` and `moe.py`
  disagree by 2.3x on the same card and model.
- **Use `amd:gfx942`, not `amd:mi300x`.** They parse to different specs and
  neither normalizes into the other — `mi300x` becomes `model`, not
  `architecture`, and the artifact gate compares `architecture`.
- **`--plan core` cells are crash-safe** (one process each, JSON written
  immediately); `quantization.py` and `moe.py` serialize once at the end, so
  they are the ones a reclaimed box loses.
- **Do not write prose on the box.** Collect JSON; author the docs locally.

## Afterwards

Doc PRs authored locally from the JSON, one question each, following the H100
campaign's shape — its first PR touched exactly two files, a new hardware doc
and one line in `tested-hardware.md`, with no code. The four "never run on real
hardware" claims are a single load-bearing edit across
`docs/tested-hardware.md`, `README.md`, `docs/limitations.md` and `CLAUDE.md`.
