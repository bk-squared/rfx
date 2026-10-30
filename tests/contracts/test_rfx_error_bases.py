"""Public refusal bases retain the builtin catches used by existing callers."""

import ast
from pathlib import Path

import jax
import numpy as np
import pytest

import rfx
from rfx._ringdown_pencil import IdentificationMemoryRefused
from rfx._traced_refusal import TracedStructureRefusal
from rfx.api._ad_memory_types import ADMemoryUnavailable
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.errors import RfxError, RfxRefusal, RfxUnsupported
from rfx.farfield import NTFFBoxPlacementError
from rfx.geometry.rasterize_grid import PadFillShortfall, TracedShapeDimensionError
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.model.graded_ports import TracedWireLiveCellsError
from rfx.model.thin_conductors import DCFilmAdmissionError
from rfx.preflight._common import PreflightConfigError


def test_public_error_mros_and_exports():
    assert RfxError.__mro__ == (RfxError, Exception, BaseException, object)
    assert RfxRefusal.__mro__ == (
        RfxRefusal, RfxError, ValueError, Exception, BaseException, object,
    )
    assert RfxUnsupported.__mro__ == (
        RfxUnsupported, RfxRefusal, RfxError, ValueError,
        NotImplementedError, RuntimeError, Exception, BaseException, object,
    )
    assert rfx.RfxError is RfxError
    assert rfx.RfxRefusal is RfxRefusal
    assert rfx.RfxUnsupported is RfxUnsupported
    assert {"RfxError", "RfxRefusal", "RfxUnsupported"} <= set(rfx.__all__)


@pytest.mark.parametrize("cls, old_base", [
    (IdentificationMemoryRefused, ValueError),
    (ADMemoryUnavailable, ValueError),
    (NTFFBoxPlacementError, ValueError),
    (PadFillShortfall, ValueError),
    (UnsupportedDesignFeature, ValueError),
    (DCFilmAdmissionError, ValueError),
    (PreflightConfigError, ValueError),
    (TracedStructureRefusal, NotImplementedError),
    (TracedShapeDimensionError, NotImplementedError),
    (TracedShapeDimensionError, jax.errors.ConcretizationTypeError),
    (TracedWireLiveCellsError, NotImplementedError),
    (TracedWireLiveCellsError, jax.errors.TracerArrayConversionError),
])
def test_existing_exception_bases(cls, old_base):
    assert issubclass(cls, RfxError)
    assert issubclass(cls, old_base)


@pytest.mark.source_scan
def test_refusal_sites_use_public_bases():
    package = Path(__file__).resolve().parents[2] / "rfx"
    violations = set()
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            values = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "refuse" in node.name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise):
                        values.append(child.exc)
                    elif isinstance(child, ast.Return):
                        values.append(child.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "diagnostic_refusal"):
                values.extend(node.args[:1])
            for value in values:
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id in ("ValueError", "NotImplementedError")):
                    violations.add(
                        f"{path.relative_to(package.parent)}:{value.lineno}: {value.func.id}"
                    )
    assert not violations, "Builtin refusal constructors remain:\n" + "\n".join(sorted(violations))


def test_waveguide_pmc_refusal_through_public_run():
    sim = rfx.Simulation(
        freq_max=12e9, domain=(.040, .020, .004), dx=.001,
        boundary=BoundarySpec(x="cpml", y=Boundary("pec", "pmc"), z="pec"),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        .010, direction="+x", mode=(1, 0), freqs=np.array([10e9]),
        reference_plane=.014,
    )
    with pytest.raises(RfxRefusal, match="y_hi.*waveguide-port.*magnetic aperture") as caught:
        sim.run(n_steps=2, skip_preflight=True)
    assert isinstance(caught.value, NotImplementedError)
