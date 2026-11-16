"""Plotting backends load on demand and missing extras give install guidance."""

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_import_does_not_load_plotting_backends():
    result = subprocess.run(
        [sys.executable, "-c", "import rfx, sys; print(sorted(m for m in ('matplotlib.pyplot','pyvista','plotly') if m in sys.modules))"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "pyvista"])
def test_missing_backend_install_guidance(monkeypatch, backend):
    from rfx._plotting import _pyplot
    from rfx.visualize import _plotly
    from rfx.visualize3d import _pyvista

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == backend:
            raise ImportError("blocked backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    loaders = {"matplotlib": _pyplot, "plotly": _plotly,
               "pyvista": lambda: _pyvista(required=True)}
    with pytest.raises(ImportError, match=r'pip install "rfx-fdtd\[visualization\]"'):
        loaders[backend]()
    if backend == "pyvista":
        assert _pyvista() is None
