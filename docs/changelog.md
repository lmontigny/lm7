# Merged work

One line per merged pull request, so the shape of the project is readable
without walking the git log. Grouped by what the change was about rather than
by date; the number links the full reasoning, which is where the detail lives.

Add a line here when a pull request merges. Titles are copied verbatim so this
file stays a index rather than a second, drifting description.

## Backends

| PR | Title |
| --- | --- |
| [#2](https://github.com/lmontigny/lm7/pull/2) | Add an opt-in OpenVINO backend for Intel CPU |
| [#7](https://github.com/lmontigny/lm7/pull/7) | AOTInductor export for NVIDIA |
| [#11](https://github.com/lmontigny/lm7/pull/11) | Add stablehlo: a PyTorch-free, vendor-neutral export backend |
| [#12](https://github.com/lmontigny/lm7/pull/12) | Add experimental IREE Vulkan export backend |
| [#14](https://github.com/lmontigny/lm7/pull/14) | Add Tenstorrent support through the tt-xla PJRT plugin |
| [#15](https://github.com/lmontigny/lm7/pull/15) | Add ONNX Runtime backend |
| [#16](https://github.com/lmontigny/lm7/pull/16) | Add edge and mobile export through ExecuTorch |
| [#17](https://github.com/lmontigny/lm7/pull/17) | Add LiteRT export backend |
| [#18](https://github.com/lmontigny/lm7/pull/18) | Add an initial Apache TVM backend through Relax |
| [#30](https://github.com/lmontigny/lm7/pull/30) | Add the Intel NPU as a target through the OpenVINO NPU plugin |
| [#44](https://github.com/lmontigny/lm7/pull/44) | Add zentorch, AMD's ZenDNN compiler, as a CPU backend |
| [#51](https://github.com/lmontigny/lm7/pull/51) | Add ExecuTorch QNN backend |
| [#126](https://github.com/lmontigny/lm7/pull/126) | Make Arm Mali expressible as a target, without pretending it runs |

## Quantization

| PR | Title |
| --- | --- |
| [#4](https://github.com/lmontigny/lm7/pull/4) | Validate weight-only quantization on sm89 and fail loudly on no-op filters |
| [#21](https://github.com/lmontigny/lm7/pull/21) | Add NVFP4 weight-only quantization and a unified `--quantize` flag |
| [#22](https://github.com/lmontigny/lm7/pull/22) | Add ExecuTorch XNNPACK INT8 export |
| [#23](https://github.com/lmontigny/lm7/pull/23) | Enable INT8 weight-only quantization on CPU |
| [#27](https://github.com/lmontigny/lm7/pull/27) | Add NNCF INT8 weight compression to OpenVINO export |
| [#37](https://github.com/lmontigny/lm7/pull/37) | Re-examine OpenVINO INT8 on VNNI, and keep full PTQ rejected |
| [#38](https://github.com/lmontigny/lm7/pull/38) | Make OpenVINO INT8 reachable from `lm7 model run` |
| [#39](https://github.com/lmontigny/lm7/pull/39) | Refuse weight-only quantization on pre-Ampere NVIDIA |
| [#56](https://github.com/lmontigny/lm7/pull/56) | Fix INT8 export for models that index with integer arithmetic |
| [#106](https://github.com/lmontigny/lm7/pull/106) | Add Hopper FP8 dynamic activation and weight quantization |
| [#140](https://github.com/lmontigny/lm7/pull/140) | Reach for the Arm INT8 instructions, and watch nothing reach them |
| [#145](https://github.com/lmontigny/lm7/pull/145) | Find out whether "OpenVINO is faster" was about OpenVINO or about Intel |
| [#184](https://github.com/lmontigny/lm7/pull/184) | Reach the FP8 tensor cores that are not NVIDIA's |

## Artifacts, CLI, and generation

| PR | Title |
| --- | --- |
| [#3](https://github.com/lmontigny/lm7/pull/3) | Export OpenVINO IR as an LM7 artifact |
| [#5](https://github.com/lmontigny/lm7/pull/5) | Add `lm7 model export` for Hugging Face models |
| [#8](https://github.com/lmontigny/lm7/pull/8) | Capture a dynamic sequence length for causal-LM artifacts |
| [#9](https://github.com/lmontigny/lm7/pull/9) | Add compiled KV-cache generation |
| [#32](https://github.com/lmontigny/lm7/pull/32) | Serialize TensorRT engines so a second process need not rebuild them |
| [#34](https://github.com/lmontigny/lm7/pull/34) | Report the decode backend that actually ran, and correct its docs |
| [#40](https://github.com/lmontigny/lm7/pull/40) | Refuse an artifact built for a different GPU architecture |
| [#47](https://github.com/lmontigny/lm7/pull/47) | Add Hugging Face model compatibility preflight |
| [#54](https://github.com/lmontigny/lm7/pull/54) | Add artifact inspection command |
| [#112](https://github.com/lmontigny/lm7/pull/112) | Add separate compiled prefill and KV-cache decode graphs |
| [#116](https://github.com/lmontigny/lm7/pull/116) | Add `lm7 model serve`: an OpenAI-compatible endpoint over the compiled decode loop |
| [#118](https://github.com/lmontigny/lm7/pull/118) | Add a built-in chat page, and give serving its own README section |
| [#120](https://github.com/lmontigny/lm7/pull/120) | Serve a model from a local directory, not only from the Hub |
| [#121](https://github.com/lmontigny/lm7/pull/121) | Add CORS, bearer auth, quantization and a cache-length alias to serve |
| [#122](https://github.com/lmontigny/lm7/pull/122) | Ask for whatever fits, so a growing conversation stops hitting a wall |
| [#124](https://github.com/lmontigny/lm7/pull/124) | Refuse --quantize on a local directory by name, not by leaking a path |
| [#128](https://github.com/lmontigny/lm7/pull/128) | Serve quantized weights in the dtype they were quantized for |
| [#130](https://github.com/lmontigny/lm7/pull/130) | Make the vLLM handover start on CUDA, where it never had |
| [#132](https://github.com/lmontigny/lm7/pull/132) | Say what the model is made of, and what it is occupying |
| [#110](https://github.com/lmontigny/lm7/pull/110) | Add experimental TensorRT-LLM serving backend for NVIDIA |
| [#138](https://github.com/lmontigny/lm7/pull/138) | Gate the CPU artifact that was as architecture-bound as the GPU one |
| [#160](https://github.com/lmontigny/lm7/pull/160) | Gate the decode path's backend selection, and correct the Apple claim |
| [#161](https://github.com/lmontigny/lm7/pull/161) | Export a KV-cache decode step |
| [#162](https://github.com/lmontigny/lm7/pull/162) | Prefill a whole prompt in one call, from the same exported graph |
| [#164](https://github.com/lmontigny/lm7/pull/164) | Record what an artifact was built from, and add `lm7 artifact generate` |
| [#173](https://github.com/lmontigny/lm7/pull/173) | Give AMD an AOTInductor package, and record what built it |
| [#178](https://github.com/lmontigny/lm7/pull/178) | Give the lifecycle benchmark the import LM7 already has |
| [#182](https://github.com/lmontigny/lm7/pull/182) | Stop telling an AMD artifact to find a CUDA runtime |
| [#187](https://github.com/lmontigny/lm7/pull/187) | Drive both AMD servers with the client the other rows used |

## On-device validation

| PR | Title |
| --- | --- |
| [#49](https://github.com/lmontigny/lm7/pull/49) | Validate exported artifacts on a real Android device, up to SmolLM2-135M |
| [#53](https://github.com/lmontigny/lm7/pull/53) | Run SmolLM2-135M on the device, and add the runner that makes it possible |
| [#55](https://github.com/lmontigny/lm7/pull/55) | Validate the LiteRT export on the device, CPU and Adreno GPU |
| [#123](https://github.com/lmontigny/lm7/pull/123) | Validate the vLLM handover on Apple Silicon through vllm-metal |
| [#131](https://github.com/lmontigny/lm7/pull/131) | Serve on the other kind of CPU, where INT8 stops being worth it |
| [#139](https://github.com/lmontigny/lm7/pull/139) | Serve on the Arm that is not a Mac, and find the spelling that leaks |
| [#177](https://github.com/lmontigny/lm7/pull/177) | Run LM7 on a GPU that is not NVIDIA's |
| [#181](https://github.com/lmontigny/lm7/pull/181) | Run the example the AMD guide opens with |

## Model coverage and measurement

| PR | Title |
| --- | --- |
| [#6](https://github.com/lmontigny/lm7/pull/6) | Validate TensorRT on NVIDIA Ada |
| [#33](https://github.com/lmontigny/lm7/pull/33) | Validate DeepSeek-Coder-1.3B across ten backends |
| [#35](https://github.com/lmontigny/lm7/pull/35) | Pin decode correctness with a test, and measure what recompiles |
| [#36](https://github.com/lmontigny/lm7/pull/36) | Correct the CPU INT8 latency story with a VNNI measurement |
| [#42](https://github.com/lmontigny/lm7/pull/42) | Let a CPU benchmark pin its thread count, and record the host |
| [#43](https://github.com/lmontigny/lm7/pull/43) | Validate Llama-3.1-8B INT8 on CPU, where the weights fit |
| [#48](https://github.com/lmontigny/lm7/pull/48) | Correct the MoE export claim, and cover a second MoE architecture |
| [#142](https://github.com/lmontigny/lm7/pull/142) | Validate optional CPU backends on Arm |
| [#146](https://github.com/lmontigny/lm7/pull/146) | Record LiteRT's aarch64 packaging gap |
| [#143](https://github.com/lmontigny/lm7/pull/143) | Put the MoE claims to a second CPU ISA, including the 6.92B one |
| [#103](https://github.com/lmontigny/lm7/pull/103) | Measure LM7 on the GPU production inference actually runs on |
| [#104](https://github.com/lmontigny/lm7/pull/104) | Run the workloads an LLM serving engine will not take |
| [#105](https://github.com/lmontigny/lm7/pull/105) | Find where the H100's flat-latency regime ends, and where compiling stops paying |
| [#134](https://github.com/lmontigny/lm7/pull/134) | Measure Arm latency at last, and find nothing to compile |
| [#136](https://github.com/lmontigny/lm7/pull/136) | Put the dtype question to a second ISA, and get a different answer |
| [#165](https://github.com/lmontigny/lm7/pull/165) | Record GCP Intel C4 CPU validation |
| [#166](https://github.com/lmontigny/lm7/pull/166) | Name a dense validation ladder, and say plainly that it is unmeasured |
| [#172](https://github.com/lmontigny/lm7/pull/172) | Let the GPU matrix describe a GPU that is not NVIDIA |
| [#180](https://github.com/lmontigny/lm7/pull/180) | Answer the three questions the first MI300X session left open |
| [#183](https://github.com/lmontigny/lm7/pull/183) | Measure the memory-bound half of generation on the MI300X |

## Detection and diagnostics

| PR | Title |
| --- | --- |
| [#31](https://github.com/lmontigny/lm7/pull/31) | Preflight Triton's Intel GPU backend instead of falling back silently |
| [#41](https://github.com/lmontigny/lm7/pull/41) | Describe the host CPU instead of guessing at it |
| [#57](https://github.com/lmontigny/lm7/pull/57) | Add Hexagon toolchain diagnostics |
| [#135](https://github.com/lmontigny/lm7/pull/135) | Count Arm cores from the place the kernel actually publishes them |
| [#171](https://github.com/lmontigny/lm7/pull/171) | Say what an AMD GPU is, before one has been seen |
| [#186](https://github.com/lmontigny/lm7/pull/186) | Give CDNA 4 the FP4 it ships, and say why FP4 on CDNA 3 is slow |

## CI

| PR | Title |
| --- | --- |
| [#20](https://github.com/lmontigny/lm7/pull/20) | Add TorchBench `torch.compile` CI |
| [#24](https://github.com/lmontigny/lm7/pull/24) | Add BERT/ViT coverage to TorchBench CI |
| [#25](https://github.com/lmontigny/lm7/pull/25) | Add mypy type checking to CI |
| [#26](https://github.com/lmontigny/lm7/pull/26) | Add sparse MoE (Mixtral) model to TorchBench CI |
| [#52](https://github.com/lmontigny/lm7/pull/52) | Run the ExecuTorch export on ARM64 in CI |
| [#117](https://github.com/lmontigny/lm7/pull/117) | Stop downgrading torch under ExecuTorch, which broke the Core ML CI job |
| [#125](https://github.com/lmontigny/lm7/pull/125) | Make CI load a real model, so the serve load path stops being untested |
| [#141](https://github.com/lmontigny/lm7/pull/141) | Run the portable suite on the Arm that a deployment actually is |
| [#147](https://github.com/lmontigny/lm7/pull/147) | Prove the architecture gate on two machines instead of one |

## Evaluations that did not become backends

| PR | Title |
| --- | --- |
| [#1](https://github.com/lmontigny/lm7/pull/1) | Record the OpenVINO evaluation on an Intel CPU host |
| [#10](https://github.com/lmontigny/lm7/pull/10) | Evaluate `torch.export` → StableHLO → PJRT for LM7 artifacts |
| [#13](https://github.com/lmontigny/lm7/pull/13) | Evaluate torch-mlir as the StableHLO lowering path |
| [#185](https://github.com/lmontigny/lm7/pull/185) | Measure MIGraphX on an MI300X, and decline to adopt it |

## Docs and examples

| PR | Title |
| --- | --- |
| [#19](https://github.com/lmontigny/lm7/pull/19) | Restructure the README and move limitations into the docs |
| [#28](https://github.com/lmontigny/lm7/pull/28) | Add sparse MoE example outside CI |
| [#29](https://github.com/lmontigny/lm7/pull/29) | Add tested model coverage section to README |
| [#45](https://github.com/lmontigny/lm7/pull/45) | Document TorchInductor compile options |
| [#46](https://github.com/lmontigny/lm7/pull/46) | Document what is actually different about AMD CPUs |
| [#119](https://github.com/lmontigny/lm7/pull/119) | Draft an external one-page summary of LM7, and correct the README's CI claim |
| [#127](https://github.com/lmontigny/lm7/pull/127) | Record that Mali is what IREE's Vulkan path is actually aimed at |
| [#144](https://github.com/lmontigny/lm7/pull/144) | Build a TVM artifact on the architecture its guard was written about |
| [#137](https://github.com/lmontigny/lm7/pull/137) | Document Arm server CPU setup for LM7 |
| [#174](https://github.com/lmontigny/lm7/pull/174) | Write the validation procedure for a GPU, not only a CPU |
| [#175](https://github.com/lmontigny/lm7/pull/175) | Write the MI300X session down before renting it, not after |
| [#176](https://github.com/lmontigny/lm7/pull/176) | Give the MI300X runbook the three commands it was missing |
| [#179](https://github.com/lmontigny/lm7/pull/179) | Add the changelog lines for the AMD campaign |
| [#188](https://github.com/lmontigny/lm7/pull/188) | Refresh AMD ROCm validation docs |
