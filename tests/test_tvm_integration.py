from __future__ import annotations

import importlib.util

import pytest
import torch

import lm7
from lm7.backends.tvm import TVMBackend
from lm7.errors import CompilationError


def _tvm_available() -> bool:
    try:
        if importlib.util.find_spec("tvm") is None:
            return False
        importlib.import_module("tvm.relax.frontend.torch")
        return True
    except (ImportError, AttributeError, ValueError):
        return False


pytestmark = [
    pytest.mark.tvm,
    pytest.mark.skipif(not _tvm_available(), reason="Apache TVM Relax is unavailable"),
]


def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_compiled_output_matches_eager():
    source = model()
    example = torch.randn(8, 16)
    with torch.no_grad():
        expected = source(example)

    compiled = lm7.compile(source, target="cpu", backend="tvm", fallback="error")
    actual = compiled(example)

    assert compiled.selected_backend == "tvm"
    assert compiled.artifact.metadata["frontend"] == "relax.from_exported_program"
    assert compiled.artifact.metadata["tvm_target"] == "llvm"
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_auto_never_selects_tvm():
    """TVM sits at priority 0 because its untuned codegen loses badly to Inductor."""
    compiled = lm7.compile(model(), target="cpu", fallback="error")
    compiled(torch.randn(8, 16))

    assert compiled.selected_backend != "tvm"


def test_embedding_compiles():
    """The reason this backend uses torch.export rather than relax_dynamo().

    TVM's from_fx translator rejects `embedding`, so the dynamo path cannot
    compile any causal LM. The exported-program frontend handles it.
    """

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.out = torch.nn.Linear(8, 4)

        def forward(self, ids):
            return self.out(self.embedding(ids))

    source = Embed().eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        expected = source(ids)

    compiled = lm7.compile(source, target="cpu", backend="tvm", fallback="error")
    torch.testing.assert_close(compiled(ids), expected, rtol=1e-4, atol=1e-4)


def test_multiple_input_signatures_compile_separately():
    source = model()
    compiled = lm7.compile(source, target="cpu", backend="tvm", fallback="error")

    for batch in (4, 8):
        example = torch.randn(batch, 16)
        with torch.no_grad():
            expected = source(example)
        torch.testing.assert_close(compiled(example), expected, rtol=1e-4, atol=1e-4)


def test_target_option_accepts_the_json_dict_form():
    """TVM 0.25 dropped the CLI-string target form -- see docs/tvm.md -- so
    the dict form is the only way to reach architecture-specific codegen. No
    `mcpu` here: valid values are architecture-specific, and this test runs on
    both x86-64 (CI) and arm64 (this project's own validation host)."""
    source = model()
    example = torch.randn(8, 16)
    with torch.no_grad():
        expected = source(example)

    compiled = lm7.compile(
        source,
        target="cpu",
        backend="tvm",
        fallback="error",
        options={"target": {"kind": "llvm"}},
    )
    actual = compiled(example)

    assert compiled.artifact.metadata["tvm_target"] == {"kind": "llvm"}
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_target_option_rejects_the_old_cli_string_form():
    """Regression guard: LM7's own docs used to show `"llvm -mcpu=..."`, which
    real TVM 0.25 rejects. The mocked unit test cannot catch this -- its fake
    Target does not replicate TVM's parser -- so this runs against the real
    thing."""
    with pytest.raises(CompilationError, match="no longer supported"):
        lm7.compile(
            model(),
            target="cpu",
            backend="tvm",
            fallback="error",
            options={"target": "llvm -mcpu=x"},
        )(torch.randn(8, 16))


def test_non_cpu_target_is_rejected():
    """Asked of the backend directly: resolve_target would raise first on a
    machine with no GPU, which would make this assert nothing."""
    from lm7.backends.base import CompileRequest
    from lm7.targets import parse_target

    support = TVMBackend().supports(
        CompileRequest(
            model=model(),
            target=parse_target("nvidia"),
            mode="lazy",
            transfers="automatic",
            fallback="error",
        )
    )

    assert not support.supported
    assert "CPU (LLVM) targets only" in support.reason


def test_keyword_inputs_are_rejected():
    backend = TVMBackend()

    with pytest.raises(CompilationError, match="positional inputs only"):
        from lm7.backends.base import CompileRequest
        from lm7.targets import parse_target

        backend.compile(
            CompileRequest(
                model=model(),
                target=parse_target("cpu"),
                mode="lazy",
                transfers="automatic",
                fallback="error",
            ),
            (),
            {"x": torch.randn(8, 16)},
        )
