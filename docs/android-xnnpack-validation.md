# Android XNNPACK validation

LM7's ExecuTorch backend exports a portable `.pte` for XNNPACK, but normal CI
executes that file only through the x86-64 host runtime. This manual harness
prepares the same program for strict correctness validation on an ARM64 Android
device. It does not add a device benchmark or claim measured phone performance.

```text
LM7 export -> XNNPACK .pte -> BundledProgram .bpte -> ARM64 example_runner
```

The BundledProgram embeds one deterministic input and its eager-PyTorch expected
output. The Android device needs only the `.bpte` and ExecuTorch's C++
`example_runner`; it does not need Python, PyTorch, or LM7.

## Prepare without a device

Use the version-matched ExecuTorch environment from [executorch.md](executorch.md):

```bash
python benchmarks/android_xnnpack.py \
  --prepare-only \
  --output-dir artifacts/android-xnnpack
```

This performs all host-side work without invoking `adb`:

1. Export the deterministic float32 MLP through LM7 and XNNPACK.
2. Execute the `.pte` through the host ExecuTorch runtime and compare it with
   eager PyTorch at `rtol=1e-4`, `atol=1e-4`.
3. Write `model.bpte`, containing the program, input, and expected output.
4. Write `report.json` with `device_validation: "not-run"`.

## Build the ARM64 runner

The ExecuTorch source checkout and installed runtime must use the same release.
The following build configuration targets ExecuTorch 1.3.1 and Android NDK r27c for ExecuTorch 1.3.1 and Android NDK
r27c; adjust paths without mixing ExecuTorch versions.

```bash
export EXECUTORCH_ROOT=/path/to/executorch
export ANDROID_NDK_ROOT=/path/to/android-ndk-r27c
cd "$EXECUTORCH_ROOT"

cmake -B cmake-android-out -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX="$PWD/cmake-android-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON .
cmake --build cmake-android-out --target install -j"$(nproc)"

cmake -B cmake-android-out/examples/devtools -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -Dexecutorch_DIR="$PWD/cmake-android-out/lib/cmake/ExecuTorch" \
  -Dgflags_DIR="$PWD/cmake-android-out/third-party/gflags" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  examples/devtools
cmake --build cmake-android-out/examples/devtools -j"$(nproc)"
```

Strip the runner before sending it through a cloud tunnel:

```bash
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
  -o example_runner_arm64 \
  cmake-android-out/examples/devtools/example_runner
```

## Qualcomm Device Cloud

QDC forwards a remote adb server over SSH. Use a local port that does not
conflict with another adb server or tunnel, and keep the SSH process separate
from the harness:

```bash
ssh -i /path/to/qdc-key.pem \
  -L 5039:<session-host>:5037 \
  -N -o ExitOnForwardFailure=yes \
  sshtunnel@ssh.qdc.qualcomm.com
```

Keep private keys outside the repository and set their mode to `600`. Do not run
`adb kill-server` through this tunnel: it targets the cloud-side adb server.

Confirm the forwarded endpoint without using the default local adb server:

```bash
adb -H 127.0.0.1 -P 5039 devices -l
```

## Validate on the device later

Once the device is idle, run:

```bash
python benchmarks/android_xnnpack.py \
  --runner /path/to/example_runner_arm64 \
  --adb /path/to/adb \
  --adb-host 127.0.0.1 \
  --adb-port 5039 \
  --serial <device-serial> \
  --output-dir artifacts/android-xnnpack
```

The harness selects exactly one ready device unless `--serial` is supplied,
pushes files under `/data/local/tmp/lm7-xnnpack`, and invokes:

```text
example_runner --bundled_program_path=.../model.bpte --output_verification
```

A nonzero runner exit becomes a harness failure. A passing report records the
Android model, SoC, platform, version, API level, and ABI. It deliberately does
not report adb wall time as inference latency.

## Limits

- This is a manual correctness gate, not physical-device CI.
- It validates a small float32 MLP; it does not establish transformer coverage.
- XNNPACK uses the Snapdragon CPU. Adreno GPU and Hexagon HTP require separate
  Vulkan or QNN work.
- It does not start, stop, or discover SSH tunnels, and it never handles keys.
- Static positional tensor inputs are inherited from the ExecuTorch artifact.
