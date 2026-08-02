# Hexagon-MLIR toolchain diagnostics

Hexagon-MLIR is built from source and combines several version-coupled
components. Check an environment before lowering a model:

```bash
lm7 hexagon doctor
lm7 hexagon doctor --mode simulator
lm7 hexagon doctor --mode device --json
```

`--mode` selects which readiness result controls the exit status:

| Mode | Required readiness |
| --- | --- |
| `compile` | Host, target architecture, SDK roots, Python modules, and compiler tools |
| `simulator` | Compilation requirements plus `hexagon-sim` |
| `device` | Compilation requirements plus `adb`, `ANDROID_HOST`, and `ANDROID_SERIAL` |

The default is `compile`. Exit status is zero when the selected mode is ready
and one when a requirement is missing, invalid, or unsupported.

## What is checked

- x86-64 Linux and a Python version supported by current upstream environments;
- the recommended Ubuntu version, reported as advice rather than a blocker;
- `HEXAGON_ARCH_VERSION` (`73`, `75`, `79`, or `81`);
- `HEXAGON_MLIR_ROOT`, `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS`, and `HEXKL_ROOT`,
  including expected files below each root;
- `torch_mlir`, the patched Triton Hexagon backend, and its launcher;
- `linalg-hexagon-opt`, `linalg-hexagon-translate`, `hexagon-clang++`,
  `hexagon-sim`, and `adb`, including version output when available;
- optional build/runtime variables normally populated by
  `scripts/set_local_env.sh`;
- device environment variables without attempting a connection.

The JSON form includes every check's category, status, value, affected modes,
explanation, and remediation, plus separate `compile_ready`,
`simulator_ready`, and `device_ready` booleans. This makes it suitable for a
setup script or CI preflight.

## Safety boundary

The command is read-only. It does not import torch-mlir or the Qualcomm Triton
plugin, invoke a compiler, run `adb`, open an SSH tunnel, or contact Qualcomm
Device Cloud. Executable `--version` output is the only subprocess activity.
Device readiness therefore means that the local prerequisites and connection
parameters are present; it does not claim that the device is reachable.

For installation and evaluation commands, see the
[Qualcomm Hexagon NPU evaluation plan](qualcomm-hexagon.md).
