# Testing LM7 artifacts on a real Android device

[ExecuTorch](executorch.md) is LM7's export path to phones, and its manifest
claims `device_bound: false` — the same `.pte` bytes should run on the build
host and on an ARM64 handset. Until now LM7 had never checked the second half of
that claim: validation was host XNNPACK on x86-64.

`benchmarks/android_device.py` checks it. It exports with LM7 on the host, runs
the artifact on an adb-reachable phone through the ExecuTorch C++ runtime, and
compares the device outputs against host eager.

The device never sees LM7, PyTorch, or Python. It sees a `.pte`, its inputs as
raw tensor bytes, and `lm7_runner` — the small C++ binary in
`tools/android_runner`. That is the same deployment surface a phone app has.

## Measured on this project

Snapdragon 8 Elite (SM8750), Android 16, arm64-v8a, 8 Oryon cores, reached
through Qualcomm Device Cloud. Host export: ExecuTorch 1.3.1, PyTorch
2.12.1+cpu, x86-64.

| | 3-layer MLP | SmolLM2-135M, float32 |
| --- | --- | --- |
| Max abs diff vs host eager | 1.2e-07 | 4.6e-05 |
| Max abs diff vs host ExecuTorch runtime | 3.7e-08 | 5.5e-05 |
| Next token matches host | n/a | yes (`' Paris'`), top-5 overlap 5/5 |
| On-device forward, 8 cores | 0.0023 ms | 13.75 ms |
| Host forward, same `.pte` | — | 132.5 ms |
| `.pte` size | 5,520 B | 651,823,488 B |
| Delegate coverage | 1 / 2 | 155 / 1389 |
| Host export | 8 s | 222 s |
| Push over the tunnel | 2.2 s | 54 s (~12 MB/s) |

The SmolLM2 `.pte` size and its **155 delegated calls** match the numbers
already in [executorch.md](executorch.md), which is independent confirmation
that this is the same export the docs describe. The total call count differs
(1389 against 1970) because the harness returns last-token logits and captures
with eager attention.

**Comparing against the host ExecuTorch runtime, not just eager, is the point of
the harness.** Eager mixes export error with architecture error; the same `.pte`
on x86-64 isolates the second. On the MLP the two agree to 3.7e-08, and INT8
agrees *exactly* — the device and the host produce bit-identical bytes.

### The latency gap is threading, not architecture

A phone completing a prefill in 13.75 ms against 132.5 ms on an x86-64 host
invites the conclusion that the phone is ten times faster. It is not.

| | SmolLM2 forward |
| --- | --- |
| Device, 8 cores | ~13.5 ms |
| Device, pinned to one core (`taskset 1`) | ~110 ms |
| Host, same `.pte` | ~132 ms |

Single-threaded, the Oryon core and the host core land within about 20% of each
other. Nearly all of the apparent 9.6x is XNNPACK's threadpool using eight cores
on the device while the host path runs effectively single-threaded. Any
device-versus-host latency claim that does not state a thread count is
describing parallelism and calling it architecture.

### Only 11% of the graph is delegated

1,234 of SmolLM2's 1,389 calls fall back to ExecuTorch's portable kernels rather
than XNNPACK, and the model still prefills in 13.75 ms. `executorch.md` calls a
low delegation ratio "the signal to look at what did not partition"; this is the
first measurement of what that ratio costs in wall-clock on a phone.

> [!WARNING]
> One device, one prompt, one model of each kind. Latency is a prefill of a
> five-token prompt with no KV cache and no decode loop, so it is not a
> token-generation rate. The MLP figures at ~0.002 ms are far too small to be
> meaningful as timings and are reported only to show the path works.

## INT8 does not work for language models

`--quantize int8` succeeds on the MLP and **fails on every causal LM**:

```
ExecuTorch INT8 prepare/calibrate/convert failed: tensors used as indices
must be long, int, byte or bool tensors.
```

LM7 applies `XNNPACKQuantizer().set_global(...)`, so `prepare_pt2e` observes
every input including `input_ids` and converts it to float, which breaks the
embedding lookup. This affects the `lm7 model export ... --quantize int8`
command as well as the Python API. Until it is fixed, the INT8 device numbers
above exist only for the MLP.

## Connect a device

Any adb-reachable device works — USB, network, or a cloud phone.

### Qualcomm Device Cloud

QDC gives free minutes on real Snapdragon hardware and reaches the device by
forwarding its adb server over SSH. After reserving a device, run the tunnel
command it shows and leave it open:

```bash
ssh -i ~/qdc_id.pem -L 5037:<session-host>:5037 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 sshtunnel@ssh.qdc.qualcomm.com
```

`5037` is the adb server port, so once the tunnel is up an ordinary `adb devices`
finds the cloud phone.

Three things make this fail in practice:

- **The key must be mode `600`.** If it is on a Windows filesystem — including
  `/mnt/c` under WSL — `chmod` silently does nothing and ssh refuses the key.
  Copy it onto a Linux filesystem first.
- **Port 5037 must be free before the tunnel starts.** A local adb server
  already holding it means the forward silently does nothing without
  `ExitOnForwardFailure=yes`.
- **Never run `adb kill-server` while the tunnel is up.** It reaches through the
  forward and kills the cloud-side server, ending the session.

Use a current `adb`. If the local client and the remote server disagree on
version, the client tries to restart the server through the tunnel.

## Build the runner

The device needs `tools/android_runner/lm7_runner.cpp` cross-compiled for
arm64. The ExecuTorch checkout must match the installed wheel — a `.pte` is
versioned against its runtime.

```bash
git clone --depth 1 --branch v1.3.1 --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/executorch.git
curl -O https://dl.google.com/android/repository/android-ndk-r27c-linux.zip
unzip android-ndk-r27c-linux.zip
```

Build ExecuTorch for `arm64-v8a`:

```bash
export ANDROID_NDK=$PWD/android-ndk-r27c
cd executorch
cmake -B cmake-android-out -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON .
cmake --build cmake-android-out --target install -j"$(nproc)"
```

Then the runner against it:

```bash
cmake -B cmake-android-out/lm7_runner -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -Dexecutorch_DIR="$PWD/cmake-android-out/lib/cmake/ExecuTorch" \
  -Dgflags_DIR="$PWD/cmake-android-out/third-party/gflags" \
  -DCMAKE_BUILD_TYPE=Release /path/to/lm7/tools/android_runner
cmake --build cmake-android-out/lm7_runner -j"$(nproc)"
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip \
  -o lm7_runner_arm64 cmake-android-out/lm7_runner/lm7_runner
```

Three flags are not optional and are each fatal in a way that does not name the
real problem:

- `EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON` — configure fails without it
  once `EXTENSION_MODULE` is on.
- `CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH` — the NDK toolchain otherwise
  restricts `find_package` to the NDK sysroot, and the runner cannot see the
  ExecuTorch just installed.
- Link the **aggregate** targets (`executorch_backends`, `executorch_kernels`,
  `executorch_extensions`), which `tools/android_runner/CMakeLists.txt` does.
  Delegates register from static initializers and need whole-archive linking;
  the installed config already sets that on the targets that need it. Naming the
  individual libraries instead produces either hundreds of duplicate symbols or
  a runtime `Backend XnnpackBackend is not registered`.

Stripping matters: the binary is 66 MB with debug info and 5 MB without, which
is the difference between a fast push and a slow one over a cloud tunnel.

## Run

The export half needs the ExecuTorch environment from
[executorch.md](executorch.md), because it runs LM7 on the host.

```bash
python benchmarks/android_device.py --runner ./lm7_runner_arm64 --serial <serial>
```

```bash
python benchmarks/android_device.py --runner ./lm7_runner_arm64 --serial <serial> --model HuggingFaceTB/SmolLM2-135M-Instruct --dtype float32 --output artifacts/benchmarks/android-sm8750.json
```

The exit status is non-zero if any configuration exceeds its tolerance, so this
works as a gate. Tolerances are `1e-4` for float32 — matching the host test —
and `2e-2` for INT8.

Pass `--dtype` deliberately. Transformers 5 defaults `from_pretrained` to the
checkpoint's dtype, and SmolLM2's checkpoint is bfloat16, so omitting it exports
a different model than the float32 baseline in `executorch.md`.

## Why not ExecuTorch's example_runner

ExecuTorch ships `example_runner`, which validates a BundledProgram — a `.pte`
rewrapped with its inputs and expected outputs travelling inside the file, so
the device can check itself. That is the better tool for a small model, and it
is what produced this project's first device result.

It does not reach a real one. BundledProgram serialization goes `.pte` → JSON →
`flatc`, and a 622 MB SmolLM2 produces a JSON intermediate that aborts `flatc`
with SIGABRT after eighteen minutes. Separately, `example_runner` reports
outputs by logging one line per element, and a transformer's logits are hundreds
of thousands of values.

`lm7_runner` reads inputs and writes outputs as raw bytes instead, which removes
both limits and improves two things:

- **Precision.** `ET_LOG` prints `%f`, roughly six decimals. Reading raw bytes
  moved the MLP's measured float32 difference from 4.6e-07 to 1.2e-07 and the
  INT8 device-versus-host difference from 4.5e-07 to exactly zero. The earlier
  figures were partly measuring the printer.
- **Latency.** `example_runner` has no iteration flag, so repeats are separate
  process launches — over a cloud tunnel that is ~1.5 s of network around a
  0.03 ms inference. `lm7_runner` repeats in-process.

## Limits

- **Not in CI.** This needs a device and a cross-compiled runner. It is a manual
  gate. The `ExecuTorch ARM64` CI job covers the cheaper half — the same
  lowering on ARM64 Linux — on every commit.
- **One device class.** Snapdragon 8 Elite only. Other SoCs, and iOS, are
  unmeasured.
- **CPU only.** XNNPACK is a CPU delegate, so this exercises the Oryon cores.
  The Hexagon NPU is untouched — see [qualcomm-hexagon.md](qualcomm-hexagon.md).
  A cloud Snapdragon is, however, exactly the hardware that plan was blocked on.
- **INT8 is MLP-only**, for the reason above.
- **Static shapes and positional tensors**, inherited from the artifact.
- **Single output.** `lm7_runner` writes output 0; a model with several outputs
  needs the runner extended.
