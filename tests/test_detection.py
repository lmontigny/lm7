import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from lm7 import detection
from lm7.detection import (
    _detect_tenstorrent_targets,
    _detect_tpu_targets,
    amd_fp8_format,
    amd_generation,
    compute_capability,
    cuda_build_targets,
    detect_cpu_target,
    detect_targets,
    nvidia_generation,
    parse_cpu_info,
    parse_total_memory_bytes,
    precision_support,
    read_physical_cores,
    resolve_target,
    torch_device,
)
from lm7.targets import DeviceInfo, TargetSpec


def test_cpu_is_always_detected():
    assert any(device.target.vendor == "cpu" for device in detect_targets())


def test_explicit_cpu_resolves():
    assert resolve_target("cpu").vendor == "cpu"


def test_rocm_device_reports_normalized_gfx_architecture(monkeypatch):
    properties = SimpleNamespace(
        name="AMD Radeon Test GPU",
        total_memory=16 * 1024**3,
        gcnArchName="gfx1100:sramecc+:xnack-",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda ordinal: properties)
    monkeypatch.setattr(torch.version, "hip", "7.0-test")

    device = next(item for item in detect_targets() if item.target.vendor == "amd")

    assert device.target.architecture == "gfx1100"
    assert device.name == "AMD Radeon Test GPU"
    assert device.capabilities == {
        "hip": "7.0-test",
        "gcn_arch_name": "gfx1100:sramecc+:xnack-",
        "generation": "RDNA 3",
        "precision": {
            "fp32": "native",
            "fp16": "native",
            "bf16": "native",
            "int8": "native",
            "fp8": "absent",
            "fp4": "absent",
        },
    }
    # RDNA 3 has no FP8, so the qualifier is absent rather than reported as some
    # default encoding.
    assert "fp8_format" not in device.capabilities


def test_mi300x_detection_reports_cdna3_and_the_fnuz_fp8_qualifier(monkeypatch):
    """The part this was written for, and the one the mocked case above is not.

    Everything asserted here started as an AMD ISA-documentation prediction. The
    first MI300X run confirmed this mocked gfx942 report unmodified.
    """
    properties = SimpleNamespace(
        name="AMD Instinct MI300X",
        total_memory=192 * 1024**3,
        gcnArchName="gfx942:sramecc+:xnack-",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda ordinal: properties)
    monkeypatch.setattr(torch.version, "hip", "7.0-test")

    device = next(item for item in detect_targets() if item.target.vendor == "amd")

    assert str(device.target) == "amd:gfx942"
    assert device.capabilities["generation"] == "CDNA 3"
    assert device.capabilities["precision"]["fp8"] == "native"
    assert device.capabilities["precision"]["fp4"] == "absent"
    # The whole point of reporting this: "fp8 native" on gfx942 and on sm90 name
    # different encodings, so the two numbers are not interchangeable.
    assert device.capabilities["fp8_format"] == "fnuz"


def test_tpu_detection_uses_pjrt_runtime(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    torch_xla = SimpleNamespace(__version__="2.9-test")
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        addressable_device_count=lambda: 2,
        global_runtime_device_attributes=lambda: [{"device_kind": "TPU v5e"}],
    )
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime if name == "torch_xla.runtime" else torch_xla,
    )

    devices = _detect_tpu_targets()

    assert [device.target.ordinal for device in devices] == [0, 1]
    assert all(device.target.vendor == "tpu" for device in devices)
    assert all(device.target.model == "v5e" for device in devices)
    assert devices[0].name == "TPU v5e"
    assert devices[0].capabilities["pjrt_device"] == "TPU"


def test_tpu_detection_names_the_generation_when_attributes_do_not(monkeypatch):
    """A real TPU VM reports no device_kind, so the generation comes from the env.

    Measured on a v6e: global_runtime_device_attributes() returns coords,
    core_on_chip, num_cores and a name, and nothing that identifies the silicon.
    Without the fallback every TPU detects as an unqualified "Google TPU".
    """
    detection_module = importlib.import_module("lm7.detection")
    torch_xla = SimpleNamespace(__version__="2.9-test")
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        addressable_device_count=lambda: 1,
        global_runtime_device_attributes=lambda: [
            {"coords": [0, 0, 0], "core_on_chip": 0, "num_cores": 1, "name": "TPU:0"}
        ],
    )
    tpu_env = SimpleNamespace(
        get_tpu_env=lambda: {"ACCELERATOR_TYPE": "v6e-1", "CONSUMER_PROJECT_ID": "secret"}
    )
    modules = {
        "torch_xla": torch_xla,
        "torch_xla.runtime": runtime,
        "torch_xla._internal.tpu": tpu_env,
    }
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(detection_module.importlib, "import_module", lambda name: modules[name])

    devices = _detect_tpu_targets()

    assert devices[0].name == "Google TPU v6e-1"
    assert devices[0].target.model == "v6e"
    assert devices[0].capabilities["accelerator_type"] == "v6e-1"
    # The same mapping carries project and node identifiers; only the one key
    # is read, so none of the rest can leak into a device record.
    assert "secret" not in repr(devices[0].capabilities)


def test_tpu_accelerator_type_is_absent_without_the_private_module(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("no torch_xla")),
    )

    assert detection_module.tpu_accelerator_type() is None


def test_tpu_detection_ignores_xla_cpu_runtime(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    runtime = SimpleNamespace(device_type=lambda: "CPU")
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime,
    )

    assert _detect_tpu_targets() == []


def test_auto_prefers_detected_tpu_over_cpu(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(
        detection_module,
        "detect_targets",
        lambda: [
            DeviceInfo(TargetSpec("tpu", "accelerator", model="v5e"), "TPU v5e"),
            DeviceInfo(TargetSpec("cpu", "cpu"), "CPU"),
        ],
    )

    assert resolve_target("auto").vendor == "tpu"
    assert torch_device(TargetSpec("tpu", "accelerator", ordinal=1)) == torch.device("xla:1")


@pytest.mark.parametrize(
    ("target", "expects_inference_mode"),
    [
        (TargetSpec("cpu", "cpu"), True),
        (TargetSpec("nvidia", "gpu"), True),
        (TargetSpec("apple", "gpu", architecture="metal"), True),
        (None, True),
        # Both PyTorch/XLA devices need the version counters inference mode
        # disables, or a call fails with "Cannot set version_counter for
        # inference tensor" partway through.
        (TargetSpec("tpu", "accelerator"), False),
        (TargetSpec("tenstorrent", "accelerator"), False),
    ],
)
def test_inference_context_avoids_inference_mode_on_xla(target, expects_inference_mode):
    with detection.inference_context(target):
        assert torch.is_inference_mode_enabled() is expects_inference_mode
        # Whichever context is chosen, gradients stay off.
        assert torch.is_grad_enabled() is False


def test_tpu_synchronize_waits_for_the_device_not_just_the_dispatch(monkeypatch):
    calls = {}

    class FakeTorchXla:
        @staticmethod
        def sync(*, wait):
            calls["wait"] = wait

    class FakeXlaModel:
        @staticmethod
        def wait_device_ops():
            calls["waited_for_device"] = True

    modules = {"torch_xla": FakeTorchXla, "torch_xla.core.xla_model": FakeXlaModel}
    monkeypatch.setattr(
        "lm7.detection.importlib.import_module",
        lambda name: modules[name],
    )

    detection.synchronize(TargetSpec("tpu", "accelerator"))

    # sync() alone returns once the work is dispatched, so timing anything with
    # it measures the host. The device barrier is the load-bearing half.
    assert calls == {"wait": True, "waited_for_device": True}


def _patch_tenstorrent_runtime(monkeypatch, runtime, *, torch_xla=None) -> None:
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime if name == "torch_xla.runtime" else (torch_xla or SimpleNamespace()),
    )


def test_tenstorrent_detection_selects_the_tt_pjrt_device(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    selected = {}
    runtime = SimpleNamespace(
        device_type=lambda: selected.get("device_type", "CPU"),
        set_device_type=lambda value: selected.update(device_type=value),
        addressable_device_count=lambda: 2,
        global_runtime_device_attributes=lambda: [{"device_kind": "Blackhole p150"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime, torch_xla=SimpleNamespace(__version__="2.9"))
    monkeypatch.setattr(detection_module, "tenstorrent_device_nodes", lambda: ["0", "1"])

    devices = _detect_tenstorrent_targets()

    assert [device.target.ordinal for device in devices] == [0, 1]
    assert all(device.target.vendor == "tenstorrent" for device in devices)
    assert all(device.target.kind == "accelerator" for device in devices)
    assert all(device.target.architecture == "blackhole" for device in devices)
    assert devices[0].name == "Blackhole p150"
    assert devices[0].capabilities["pjrt_device"] == "TT"
    assert devices[0].capabilities["device_nodes"] == ["0", "1"]


def test_tenstorrent_detection_requires_the_plugin(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(detection_module.importlib.util, "find_spec", lambda name: None)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_never_hijacks_a_tpu_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        set_device_type=lambda value: pytest.fail("must not reassign a live PJRT runtime"),
        addressable_device_count=lambda: 4,
        global_runtime_device_attributes=lambda: [{"device_kind": "TPU v5e"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_honours_an_explicit_pjrt_device(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "CUDA",
        set_device_type=lambda value: pytest.fail("must not override an explicit PJRT_DEVICE"),
        addressable_device_count=lambda: 1,
        global_runtime_device_attributes=lambda: [{"device_kind": "Wormhole n300"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)
    monkeypatch.setenv("PJRT_DEVICE", "CUDA")

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_ignores_a_card_less_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TT",
        set_device_type=lambda value: None,
        addressable_device_count=lambda: 0,
        global_runtime_device_attributes=list,
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_uses_the_xla_device(monkeypatch):
    target = TargetSpec("tenstorrent", "accelerator", architecture="wormhole", ordinal=1)

    assert torch_device(target) == torch.device("xla:1")


# Captured from an AMD EPYC 7B13 (Zen 3), trimmed to two of its sixteen logical
# CPUs: one SMT sibling pair, which is what makes the physical-core count a real
# assertion rather than a restatement of the block count.
EPYC_CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
cpu family\t: 25
model name\t: AMD EPYC 7B13
physical id\t: 0
siblings\t: 16
core id\t: 0
cpu cores\t: 8
flags\t\t: fpu vme de pse tsc msr pae sse2 ht syscall lm constant_tsc pni ssse3 fma cx16 sse4_1 sse4_2 popcnt aes xsave avx f16c rdrand hypervisor bmi1 avx2 bmi2 erms rdseed adx clflushopt clwb sha_ni xsaveopt vaes vpclmulqdq rdpid fsrm
bogomips\t: 4899.99

processor\t: 1
vendor_id\t: AuthenticAMD
cpu family\t: 25
model name\t: AMD EPYC 7B13
physical id\t: 0
siblings\t: 16
core id\t: 0
cpu cores\t: 8
flags\t\t: fpu vme de pse tsc msr pae sse2 ht syscall lm constant_tsc pni ssse3 fma cx16 sse4_1 sse4_2 popcnt aes xsave avx f16c rdrand hypervisor bmi1 avx2 bmi2 erms rdseed adx clflushopt clwb sha_ni xsaveopt vaes vpclmulqdq rdpid fsrm
bogomips\t: 4899.99
"""

# A Sapphire Rapids Xeon, which is the case LM7 cannot reach on any host it owns:
# AMX and AVX-512 BF16 are exactly the flags that decide whether CPU BF16 is
# native, so they are covered by fixture or not at all.
SAPPHIRE_RAPIDS_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480+
physical id\t: 0
core id\t: 0
cpu cores\t: 56
flags\t\t: fpu vme de pse tsc msr avx avx2 f16c avx512f avx512bw avx512vl avx512_vnni avx512_bf16 avx512_fp16 amx_bf16 amx_tile amx_int8

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480+
physical id\t: 1
core id\t: 0
cpu cores\t: 56
flags\t\t: fpu vme de pse tsc msr avx avx2 f16c avx512f avx512bw avx512vl avx512_vnni avx512_bf16 avx512_fp16 amx_bf16 amx_tile amx_int8
"""

# AArch64 prints "Features", not "flags", and names nothing the x86 vocabulary
# knows. Graviton and Apple Silicon are both LM7 targets, so the parser has to
# come back with a vector ISA here rather than an empty tuple.
#
# Captured verbatim from the ubuntu-24.04-arm CI runner (Azure Cobalt 100), so
# this is a real kernel's output rather than a plausible one: implementer 0x41
# part 0xd49 is an Arm Neoverse N2, and it carries the bf16 and i8mm flags that
# decide the CPU dtype question on Arm the way the AMX trio does on x86.
AARCH64_CPUINFO = """processor\t: 0
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0xd49
CPU revision\t: 0
"""


def test_cpu_info_reports_amd_vendor_topology_and_isa():
    info = parse_cpu_info(EPYC_CPUINFO)

    assert info["vendor_id"] == "AuthenticAMD"
    assert info["model_name"] == "AMD EPYC 7B13"
    assert info["logical_cores"] == 2
    # Both blocks are SMT siblings of one core, so the physical count is 1.
    assert info["physical_cores"] == 1
    assert info["isa_extensions"] == ("avx", "avx2", "f16c")


def test_cpu_info_reports_avx512_and_amx_on_sapphire_rapids():
    info = parse_cpu_info(SAPPHIRE_RAPIDS_CPUINFO)

    assert info["vendor_id"] == "GenuineIntel"
    assert info["physical_cores"] == 2
    assert set(info["isa_extensions"]) >= {
        "avx512_bf16",
        "avx512_vnni",
        "amx_bf16",
        "amx_int8",
        "amx_tile",
    }


def test_cpu_info_reads_the_aarch64_features_line():
    info = parse_cpu_info(AARCH64_CPUINFO)

    assert info["isa_extensions"] == ("asimd", "asimddp", "asimdhp", "bf16", "i8mm", "sve", "sve2")
    # The SVE forms of the same instructions -- svebf16, svei8mm -- are on the
    # Features line above and deliberately not recorded: nothing here has
    # established whether oneDNN reaches for them or the NEON variants.
    assert "svebf16" not in info["isa_extensions"]
    # AArch64 publishes no vendor_id and no topology fields, so those degrade
    # rather than inventing a value.
    assert info["vendor_id"] is None
    assert info["physical_cores"] is None
    assert info["logical_cores"] == 1


def test_cpu_info_names_an_arm_core_from_its_part_number():
    # Without this the name is platform.machine() -- the string "aarch64",
    # which identifies no chip and cannot be looked up.
    assert parse_cpu_info(AARCH64_CPUINFO)["model_name"] == "Arm Neoverse N2"


def test_cpu_info_names_the_arm_vendor_when_the_part_is_unknown():
    # A part number this table does not carry still beats "aarch64": the vendor
    # is named and the raw identifier is preserved for looking up.
    cpuinfo = "processor\t: 0\nCPU implementer\t: 0xc0\nCPU part\t: 0xac3\n"

    assert parse_cpu_info(cpuinfo)["model_name"] == "Ampere 0xac3"


def test_cpu_info_reads_a_part_number_only_as_the_implementer_that_issued_it():
    # 0xd49 is a Neoverse N2 only because implementer 0x41 is Arm itself.
    # Another vendor's designs number their own parts.
    cpuinfo = "processor\t: 0\nCPU implementer\t: 0x61\nCPU part\t: 0xd49\n"

    assert parse_cpu_info(cpuinfo)["model_name"] == "Apple 0xd49"


@pytest.mark.parametrize(
    "cpuinfo",
    [
        # An implementer outside the table, and the pre-5.x kernels that print
        # an implementer without a part: no name is better than a wrong one.
        "processor\t: 0\nCPU implementer\t: 0x99\nCPU part\t: 0xd49\n",
        "processor\t: 0\nCPU implementer\t: 0x41\n",
        "processor\t: 0\nFeatures\t: fp asimd\n",
    ],
)
def test_cpu_info_declines_to_name_an_unidentifiable_arm_cpu(cpuinfo):
    assert parse_cpu_info(cpuinfo)["model_name"] is None


def test_total_memory_is_read_in_bytes():
    assert (
        parse_total_memory_bytes("MemTotal:       65841440 kB\nMemFree: 1 kB\n") == 65841440 * 1024
    )


def test_total_memory_is_absent_when_meminfo_has_no_total():
    assert parse_total_memory_bytes("MemFree:  123 kB\n") is None


def test_cpu_target_carries_the_detected_description(monkeypatch, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(EPYC_CPUINFO)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65841440 kB\n")
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(detection_module, "CPU_INFO_PATH", cpuinfo)
    monkeypatch.setattr(detection_module, "MEMORY_INFO_PATH", meminfo)

    device = detect_cpu_target()

    assert device.target.vendor == "cpu"
    assert device.name == "AMD EPYC 7B13"
    assert device.total_memory_bytes == 65841440 * 1024
    assert device.capabilities["vendor_id"] == "AuthenticAMD"
    assert device.capabilities["physical_cores"] == 1
    assert device.capabilities["isa_extensions"] == ("avx", "avx2", "f16c")


def test_cpu_target_degrades_without_proc(monkeypatch, tmp_path):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(detection_module, "CPU_INFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(detection_module, "MEMORY_INFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(detection_module, "CPU_TOPOLOGY_PATH", tmp_path / "absent")

    device = detect_cpu_target()

    # The CPU is LM7's fallback target, so an unreadable /proc must leave a
    # resolvable device behind rather than removing it.
    assert device.target.vendor == "cpu"
    assert device.name
    assert device.total_memory_bytes is None
    assert device.capabilities["vendor_id"] is None
    assert device.capabilities["isa_extensions"] == ()
    assert device.capabilities["physical_cores"] is None
    assert device.capabilities["logical_cores"] == os.cpu_count()


def write_topology(root, pairs):
    """Build the part of /sys/devices/system/cpu that read_physical_cores reads.

    `pairs` is one (package, core) per logical CPU, in cpu0..cpuN order; a None
    entry stands for an offline CPU, which has no topology directory at all.
    """
    for index, pair in enumerate(pairs):
        if pair is None:
            (root / f"cpu{index}").mkdir(parents=True)
            continue
        package, core = pair
        topology = root / f"cpu{index}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")
    return root


def test_physical_cores_come_from_sysfs_when_cpuinfo_has_no_topology(tmp_path):
    # The GCP Axion n4a-standard-8 this was captured from: 8 cores, one thread
    # each, and every one of them in package 148 rather than package 0.
    root = write_topology(tmp_path, [(148, core) for core in range(8)])

    assert read_physical_cores(root) == 8


def test_physical_cores_fold_smt_siblings_together(tmp_path):
    # Two logical CPUs sharing a core id are one physical core, which is the
    # whole reason this counts pairs instead of directories.
    root = write_topology(tmp_path, [(0, 0), (0, 0), (0, 1), (0, 1)])

    assert read_physical_cores(root) == 2


def test_physical_cores_separate_sockets_that_reuse_core_ids(tmp_path):
    # Core 0 exists in both sockets and is two different cores.
    root = write_topology(tmp_path, [(0, 0), (0, 1), (1, 0), (1, 1)])

    assert read_physical_cores(root) == 4


def test_physical_cores_skip_a_cpu_with_no_topology(tmp_path):
    # An offline CPU publishes no topology directory. The online ones still
    # answer the question, so it is skipped rather than abandoning the count.
    root = write_topology(tmp_path, [(0, 0), None, (0, 1)])

    assert read_physical_cores(root) == 2


def test_physical_cores_are_absent_without_sysfs(tmp_path):
    # Every non-Linux host. None means "unknown", not "zero cores".
    assert read_physical_cores(tmp_path / "absent") is None


def test_physical_cores_are_absent_when_sysfs_lists_no_cpus(tmp_path):
    assert read_physical_cores(tmp_path) is None


def test_cpu_target_fills_aarch64_cores_from_sysfs(monkeypatch, tmp_path):
    """The gap this closes: AArch64 /proc/cpuinfo prints no topology at all, so
    physical_cores was always None on Arm however many cores the host had."""
    detection_module = importlib.import_module("lm7.detection")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(AARCH64_CPUINFO)
    monkeypatch.setattr(detection_module, "CPU_INFO_PATH", cpuinfo)
    monkeypatch.setattr(detection_module, "MEMORY_INFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(
        detection_module,
        "CPU_TOPOLOGY_PATH",
        write_topology(tmp_path / "sys", [(148, core) for core in range(8)]),
    )

    device = detect_cpu_target()

    assert parse_cpu_info(AARCH64_CPUINFO)["physical_cores"] is None
    assert device.capabilities["physical_cores"] == 8


def test_cpu_target_prefers_cpuinfo_topology_over_sysfs(monkeypatch, tmp_path):
    """x86 already answers this from /proc/cpuinfo, and keeps doing so -- sysfs
    is the fallback, so this change cannot move an x86 count."""
    detection_module = importlib.import_module("lm7.detection")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(EPYC_CPUINFO)
    monkeypatch.setattr(detection_module, "CPU_INFO_PATH", cpuinfo)
    monkeypatch.setattr(detection_module, "MEMORY_INFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(
        detection_module,
        "CPU_TOPOLOGY_PATH",
        write_topology(tmp_path / "sys", [(0, core) for core in range(64)]),
    )

    device = detect_cpu_target()

    assert parse_cpu_info(EPYC_CPUINFO)["physical_cores"] == 1
    assert device.capabilities["physical_cores"] == 1


def test_explicit_remote_target_does_not_require_local_detection(monkeypatch):
    monkeypatch.setattr(
        detection,
        "detect_targets",
        lambda: pytest.fail("remote target must not inspect local devices"),
    )

    target = resolve_target("qualcomm:sm8750")

    assert str(target) == "qualcomm:sm8750"
    assert target.remote is True


def nvidia(architecture: str | None) -> TargetSpec:
    return TargetSpec("nvidia", "gpu", architecture=architecture)


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("sm75", "Turing"),
        ("sm80", "Ampere"),
        ("sm86", "Ampere"),
        ("sm89", "Ada Lovelace"),
        ("sm90", "Hopper"),
        ("sm100", "Blackwell"),
        ("sm120", "Blackwell"),
    ],
)
def test_nvidia_generation_names_the_silicon(architecture, expected):
    """`sm120` means nothing to a reader who has not memorized the table."""
    assert nvidia_generation(nvidia(architecture)) == expected


def test_nvidia_generation_declines_rather_than_guessing():
    """A capability newer than the table costs the label and nothing else."""
    assert nvidia_generation(nvidia(None)) is None
    assert nvidia_generation(nvidia("gfx942")) is None
    assert nvidia_generation(TargetSpec("amd", "gpu", architecture="gfx942")) is None
    assert nvidia_generation(TargetSpec("cpu", "cpu", architecture="x86_64")) is None


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("gfx908", "CDNA 1"),
        ("gfx90a", "CDNA 2"),
        ("gfx942", "CDNA 3"),
        ("gfx950", "CDNA 4"),
        ("gfx1100", "RDNA 3"),
        ("gfx1201", "RDNA 4"),
    ],
)
def test_amd_generation_names_the_isa_family(architecture, expected):
    assert amd_generation(TargetSpec("amd", "gpu", architecture=architecture)) == expected


def test_amd_generation_is_an_exact_map_because_gfx_numbers_do_not_order():
    """The property that makes this table a dict and not a threshold list.

    `sm120 > sm89` and the larger number is the more capable part. `gfx1100 >
    gfx942` and the larger number is a consumer RDNA 3 chip with no FP8 at all,
    while the smaller is a datacenter CDNA 3 one that has it. A descending
    threshold table like `_NVIDIA_GENERATIONS` would report the wrong family for
    every Instinct part.
    """
    assert amd_generation(TargetSpec("amd", "gpu", architecture="gfx1100")) == "RDNA 3"
    assert amd_generation(TargetSpec("amd", "gpu", architecture="gfx942")) == "CDNA 3"
    rdna3 = precision_support(TargetSpec("amd", "gpu", architecture="gfx1100"))
    cdna3 = precision_support(TargetSpec("amd", "gpu", architecture="gfx942"))
    assert rdna3["fp8"] == "absent"
    assert cdna3["fp8"] == "native"

    # And it declines rather than guessing, exactly as the NVIDIA table does.
    assert amd_generation(TargetSpec("amd", "gpu", architecture="gfx1250")) is None
    assert amd_generation(TargetSpec("amd", "gpu")) is None
    assert amd_generation(nvidia("sm90")) is None


def test_amd_fp8_format_separates_the_two_encodings():
    """Reporting fp8 as native is not one claim but two. CDNA 3 implements the `fnuz` variants --
    no infinities, one NaN, a different exponent bias -- so `torch.float8_e4m3fnuz`
    is the dtype that exists there, while sm89+ and CDNA 4 use the OCP `e4m3`.
    Without this qualifier an FP8 number from a MI300X would compare directly
    against one from an H100, and they were not produced in the same format."""
    assert amd_fp8_format(TargetSpec("amd", "gpu", architecture="gfx942")) == "fnuz"
    assert amd_fp8_format(TargetSpec("amd", "gpu", architecture="gfx950")) == "ocp"
    assert amd_fp8_format(TargetSpec("amd", "gpu", architecture="gfx1201")) == "ocp"
    # No FP8 silicon means no encoding to name.
    assert amd_fp8_format(TargetSpec("amd", "gpu", architecture="gfx90a")) is None
    assert amd_fp8_format(TargetSpec("amd", "gpu", architecture="gfx1100")) is None
    assert amd_fp8_format(nvidia("sm90")) is None


def test_amd_precision_never_reports_emulated():
    """`emulated` exists because a Tesla T4 fakes BF16 and reports success. No
    equivalent case is known on any gfx part in the table, and inventing one
    would be the guess this file refuses to make -- so a format is native or it
    is absent."""
    for architecture in ("gfx906", "gfx908", "gfx90a", "gfx942", "gfx1100"):
        precision = precision_support(TargetSpec("amd", "gpu", architecture=architecture))
        assert "emulated" not in precision.values()
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx906"))["bf16"] == "absent"


def test_amd_and_nvidia_precision_answer_the_same_keys():
    """A row from one card has to compare against a row from another, and
    `benchmarks/nvidia_matrix.py` writes both into the same `supported_precisions`
    field of one environment.json."""
    assert set(precision_support(TargetSpec("amd", "gpu", architecture="gfx942"))) == set(
        precision_support(nvidia("sm90"))
    )


def test_compute_capability_orders_blackwell_above_ada():
    """Every gate in LM7 compares these as plain integers, which is only correct
    because CUDA capabilities sort that way as concatenated digits. This is the
    property that let sm120 work with no special case anywhere."""
    assert compute_capability(nvidia("sm120")) == 120
    numbers = [compute_capability(nvidia(name)) for name in ("sm75", "sm80", "sm89", "sm90")]
    assert numbers == [75, 80, 89, 90]
    assert compute_capability(nvidia("sm120")) > compute_capability(nvidia("sm89"))
    assert compute_capability(nvidia("sm100")) > compute_capability(nvidia("sm90"))


def test_turing_reports_bfloat16_as_emulated():
    """The case that motivated this report. A T4 answers True to
    torch.cuda.is_bf16_supported() and then emulates BF16, which measured 3.4x
    slower than FP16 on the same model. "native" and "emulated" have to be
    distinguishable or that slowdown has no explanation."""
    precision = precision_support(nvidia("sm75"))

    assert precision["bf16"] == "emulated"
    assert precision["fp16"] == "native"
    assert precision["fp8"] == "absent"
    assert precision["fp4"] == "absent"


def test_fp4_is_native_only_on_blackwell():
    assert precision_support(nvidia("sm89"))["fp4"] == "absent"
    assert precision_support(nvidia("sm90"))["fp4"] == "absent"
    assert precision_support(nvidia("sm100"))["fp4"] == "native"
    assert precision_support(nvidia("sm120"))["fp4"] == "native"


def test_amd_fp4_arrives_with_cdna_4():
    """CDNA 3 stops at FP8 and CDNA 4 adds block-scaled MXFP4, so these two rows
    are the difference between an emulated format and a computed one. An `nvfp4`
    run on the MI300X measured 1.96x *slower* than BF16 while every FP8 mode
    landed near 1.2x, which is what emulation costs; reporting `gfx950` as absent
    would have made that indistinguishable from silicon that simply lacks it."""
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx950"))["fp4"] == "native"
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx942"))["fp4"] == "absent"
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx90a"))["fp4"] == "absent"
    # RDNA 4 has the OCP FP8 encoding and no FP4 -- FP4 is a CDNA 4 addition.
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx1201"))["fp4"] == "absent"


def test_blackwell_reports_every_format_as_native():
    assert set(precision_support(nvidia("sm120")).values()) == {"native"}


def test_precision_support_is_empty_when_it_cannot_be_known():
    """An unqualified target has no architecture until it resolves, and no vendor
    outside NVIDIA and AMD has a capability string LM7 knows. Reporting "native"
    for a CPU whose AVX-512 BF16 support was never probed would be the same
    unmeasured claim this report exists to prevent."""
    assert precision_support(nvidia(None)) == {}
    assert precision_support(TargetSpec("amd", "gpu")) == {}
    assert precision_support(TargetSpec("cpu", "cpu", architecture="x86_64")) == {}
    # A gfx string outside the table declines for the same reason a capability
    # newer than _NVIDIA_GENERATIONS does.
    assert precision_support(TargetSpec("amd", "gpu", architecture="gfx1250")) == {}


def test_cuda_build_targets_is_gpu_only():
    """The arch list describes a GPU build. ROCm answers through the same API, so
    an AMD target gets a report; a CPU has no arch list to describe."""
    assert cuda_build_targets(TargetSpec("cpu", "cpu", architecture="x86_64")) is None
    assert cuda_build_targets(TargetSpec("apple", "gpu")) is None


def test_cuda_build_targets_answers_the_gfx_question_on_rocm(monkeypatch):
    """The question this function exists for is sharper on AMD than on NVIDIA: a
    missing `sm_` target still runs by JIT-ing PTX, and a missing `gfx` target is
    a hard "no kernel image is available" at load. Predicted, not measured."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["gfx906", "gfx90a", "gfx942"])
    report = cuda_build_targets(TargetSpec("amd", "gpu", architecture="gfx942"))
    assert report is not None
    assert report["native_kernels"] is True
    # ROCm carries the equivalent of sm_90a as `:sramecc+:xnack-` suffixes on the
    # target string, not as a separate architecture, so None means "this vendor
    # does not answer that question" rather than "no".
    assert report["architecture_specific"] is None

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["gfx906", "gfx90a"])
    missing = cuda_build_targets(TargetSpec("amd", "gpu", architecture="gfx942"))
    assert missing["native_kernels"] is False


def test_cuda_build_targets_separates_sm90_from_sm90a(monkeypatch):
    """From compute capability 9.0 NVIDIA splits the target: `sm_90` is portable,
    `sm_90a` carries the architecture-specific instructions (`wgmma`, TMA) and is
    not forward compatible. A wheel with only `sm_90` runs correctly on an H100
    and cannot reach them, which is exactly the case measured on a real H100 --
    torch 2.13.0+cu130 ships no `a` variant for any architecture."""
    monkeypatch.setattr(
        torch.cuda,
        "get_arch_list",
        lambda: ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
    )
    report = cuda_build_targets(nvidia("sm90"))
    assert report is not None
    assert report["native_kernels"] is True
    assert report["architecture_specific"] is False

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_90a"])
    assert cuda_build_targets(nvidia("sm90"))["architecture_specific"] is True


def test_cuda_build_targets_flags_an_architecture_without_kernels(monkeypatch):
    """A GPU missing from the arch list still runs -- CUDA JITs it from PTX -- so
    this is a warm-up and feature-reach caveat, not a failure. It is worth
    reporting precisely because nothing else surfaces it."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_86"])
    report = cuda_build_targets(nvidia("sm120"))
    assert report["native_kernels"] is False
    assert report["arch_list"] == ["sm_80", "sm_86"]


def test_cuda_build_targets_without_an_architecture_reports_only_the_list(monkeypatch):
    """An unqualified `nvidia` target has no capability until it resolves against
    hardware, so there is nothing to compare the list against."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    report = cuda_build_targets(nvidia(None))
    assert report["arch_list"] == ["sm_90"]
    assert report["native_kernels"] is None
    assert report["architecture_specific"] is None
