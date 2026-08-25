from __future__ import annotations

import tomllib
from pathlib import Path


def test_base_install_includes_numpy_for_torch_import() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dependencies = pyproject["project"]["dependencies"]

    assert "torch>=2.0" in dependencies
    assert "numpy>=1.26,<2.5" in dependencies
