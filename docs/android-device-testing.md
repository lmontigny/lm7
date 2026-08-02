# Testing LM7 artifacts on a real Android device

[ExecuTorch](executorch.md) is LM7's export path to phones, and its manifest
claims `device_bound: false` — the same `.pte` bytes should run on the build
host and on an ARM64 handset. Until now LM7 had never checked the second half of
that claim: validation was host XNNPACK on x86-64.

`benchmarks/android_device.py` checks it. It exports with LM7 on the host, runs
the artifact on an adb-reachable phone through the ExecuTorch C++ runtime, and
compares the device outputs against host eager.

The device never sees LM7, PyTorch, or Python. It sees two files: a
cross-compiled `example_runner` and a `.bpte` — the LM7 `.pte` rewrapped as an
ExecuTorch BundledProgram, which carries the example inputs and the expected
outputs inside the file. That is the same deployment surface a phone app has.

## Measured on this project

Snapdragon 8 Elite (SM8750), Android 16, arm64-v8a, via Qualcomm Device Cloud.
Host export: ExecuTorch 1.3.1, PyTorch 2.12.1+cpu, x86-64. The model is the
3-layer MLP `tests/test_executorch_integration.py` already exports, so the host
and device results describe the same subject.

| | float32 | INT8 |
| --- | --- | --- |
| Max abs diff vs host eager | 4.6e-07 | 6.5e-03 |
| Max abs diff vs host ExecuTorch runtime | 5.1e-07 | 4.5e-07 |
| Bundled strict verification (rtol 1e-3, atol 1e-5) | passed | not applicable |
| On-device `Method::execute` | 0.047–0.050 ms | 0.029–0.037 ms |
| On-device `Method::init` | ~0.54 ms | ~0.60 ms |
| `.pte` size | 5,520 B | 5,776 B |

The difference columns are reproducible to the digits shown — they were
identical across repeated runs. The timings are not: each figure is one
untimed-average ETDump sample, and the range shown is what three runs produced.

**The two difference columns are the point of the harness.** Comparing against
host eager mixes two unrelated sources of error; comparing against the host
ExecuTorch runtime isolates the architecture. INT8 deviates from eager by 6.5e-03
but from the same `.pte` on x86-64 by 4.5e-07 — so essentially all of the INT8
deviation is single-sample PTQ calibration, and almost none of it is ARM. That is
the answer LM7 could not previously give.

The float32 agreement at 4.6e-07 is also stronger than the 1e-4 tolerance the
host test uses. On this model, on this device, the portability claim holds.

The device operators recorded in the ETDump confirm the kernels are the intended
ones rather than a silent fallback:

```
float32:  Fully Connected (NC, F32) GEMM #1, GELU (NC) #1, Fully Connected (NC, F32) GEMM #2
INT8:     Convert (NC) #1, Fully Connected (NC, QS8, QC8W) GEMM #1, Invalid Unary Op #1,
          Fully Connected (NC, QS8, QC8W) GEMM #2, Convert (NC) #2
```

`QS8, QC8W` is XNNPACK's per-channel INT8 weight GEMM — the quantized path
really executed on the phone. (`Invalid Unary Op` is XNNPACK's display name for
the quantized GELU, not an error.)

> [!WARNING]
> These numbers come from one 3-layer MLP. They establish that the path works
> and that the ARM/x86 gap is small for these operators. They are not a
> statement about a language model: `Method::execute` at ~0.03 ms is too small
> to be a meaningful latency measurement — the INT8 figure overlaps float32
> across runs, so do not read a speedup into it — and neither operator coverage
> nor INT8 accuracy generalizes from an MLP to a transformer.

## Connect a device

Any adb-reachable device works — USB, network, or a cloud phone.

### Qualcomm Device Cloud

QDC gives free minutes on real Snapdragon hardware and reaches the device by
forwarding its adb server over SSH. After reserving a device, QDC shows a tunnel
command; run it and leave it open:

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

The device needs an ARM64 `example_runner`, which is ExecuTorch's bundled-program
runner. It is not shipped as a binary, so it is cross-compiled once.

The ExecuTorch checkout must match the installed wheel — a `.pte` is versioned
against its runtime.

```bash
git clone --depth 1 --branch v1.3.1 --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/executorch.git
curl -O https://dl.google.com/android/repository/android-ndk-r27c-linux.zip
unzip android-ndk-r27c-linux.zip
```

Build ExecuTorch for `arm64-v8a`, then the runner against it:

```bash
export ANDROID_NDK=$PWD/android-ndk-r27c
cd executorch
cmake -B cmake-android-out -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON .
cmake --build cmake-android-out --target install -j"$(nproc)"
```

```bash
cmake -B cmake-android-out/examples/devtools -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -Dexecutorch_DIR="$PWD/cmake-android-out/lib/cmake/ExecuTorch" \
  -Dgflags_DIR="$PWD/cmake-android-out/third-party/gflags" \
  -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_XNNPACK=ON examples/devtools
cmake --build cmake-android-out/examples/devtools -j"$(nproc)"
```

Two flags are not optional and are easy to miss:

- `EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON` — the configure step fails
  without it once `EXTENSION_MODULE` is on.
- `CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH` — the NDK toolchain otherwise
  restricts `find_package` to the NDK sysroot, and the second stage cannot see
  the ExecuTorch it just installed.

The binary is 66 MB with debug info. Strip it before pushing, which matters over
a cloud tunnel:

```bash
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o example_runner_arm64 cmake-android-out/examples/devtools/example_runner
```

## Run

The export half needs the ExecuTorch environment from
[executorch.md](executorch.md), because it runs LM7 on the host.

```bash
python benchmarks/android_device.py --runner ./example_runner_arm64 \
  --serial <device-serial> --quantize none int8 \
  --output artifacts/benchmarks/android-sm8750.json
```

The exit status is non-zero if any configuration exceeds its tolerance, so this
works as a gate. Tolerances are `1e-4` for float32 — matching the host test —
and `2e-2` for INT8.

## How correctness is checked

Two independent checks, because each alone is insufficient.

**Strict bundled verification** (`--output_verification`) compares on-device
against the expected outputs embedded in the `.bpte` and aborts on mismatch. It
is the stronger check, but its tolerance is hardcoded in `example_runner.cpp` at
`rtol=1e-3, atol=1e-5`, which no INT8 model will pass. The harness enables it
only for float32.

**Printed-output comparison** parses the values the runner logs and compares them
in Python. This works at any tolerance and produces the actual deviation rather
than a pass/fail, which is what makes the eager-vs-runtime split above possible.
It costs precision — `ET_LOG` prints `%f`, so roughly six decimals.

Latency comes from the ETDump the runner writes, pulled back and read with
ExecuTorch's `Inspector`. Wall-clock around `adb shell` is reported too, but only
as a round-trip: at ~1.5 s against a cloud device it is network and process
startup, roughly 50,000x the actual inference.

## Limits

- **Not in CI.** This needs a device and a cross-compiled runner, neither of
  which the GitHub runners have. It is a manual gate.
- **One device class.** Snapdragon 8 Elite only. Other SoCs, and iOS, are
  unmeasured.
- **CPU only.** XNNPACK is a CPU delegate, so this exercises the Oryon cores.
  The Hexagon NPU is untouched — see [qualcomm-hexagon.md](qualcomm-hexagon.md).
  A cloud Snapdragon is, however, exactly the hardware that plan was blocked on.
- **Static shapes and positional tensors**, inherited from the artifact.
- **float32 outputs.** `example_runner` prints outputs assuming fp32 tensors, so
  a model with non-float outputs needs the strict-verification path instead.
