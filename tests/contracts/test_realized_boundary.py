"""Per-face field comparisons with B1's declared planes; CPU, always on."""

from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch

import jax
import pytest

from tests.contracts.boundary_compare import BoundaryDeparture, classify, departures
from tests.contracts.boundary_fields import measure, measured_fields


ROOT = Path(__file__).resolve().parents[2]
BASELINE = json.loads((ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.json").read_text())
CELLS = BASELINE["cells"]


@lru_cache(maxsize=None)
def measured(case, entry):
    sim, grid, _, records = measure(case, entry)
    assert records, "entry point produced no observed field scan"
    fields, psi, reference = measured_fields(records, grid, entry)
    return departures(sim.boundary_model(), grid, classify(fields, grid, psi, reference), entry)


def key(departure):
    return departure["face"], departure["code"]


def compare_departures(problems):
    if problems:
        raise BoundaryDeparture(str(problems))


@pytest.mark.parametrize("cell", [pytest.param(c, id=f"{c['case']}--{c['entry']}",
                                              marks=pytest.mark.xdist_group(f"{c['case']}--{c['entry']}")) for c in CELLS])
def test_no_unlisted_departures(cell):
    if cell["status"] == "REFUSED":
        exception_type = {"ValueError": ValueError, "NotImplementedError": NotImplementedError}[cell["exception"]]
        with pytest.raises(exception_type) as exc:
            measured(cell["case"], cell["entry"])
        assert str(exc.value) == cell["message"]
        return
    expected = {key(d) for d in cell["departures"]}
    compare_departures([d for d in measured(cell["case"], cell["entry"]) if key(d) not in expected])


EXPECTED = [pytest.param(cell["case"], cell["entry"], departure["face"], departure["code"],
                         id=f"{cell['case']}--{cell['entry']}--{departure['face']}--{departure['code']}",
                         marks=[pytest.mark.xdist_group(f"{cell['case']}--{cell['entry']}"),
                                pytest.mark.xfail(strict=True, raises=BoundaryDeparture,
                                                  reason=f"{departure['code']}; {departure['detail']}; fixed in {departure['step']}")])
            for cell in CELLS if cell["status"] == "MEASURED" for departure in cell["departures"]]


@pytest.mark.parametrize("case,entry,face,code", EXPECTED)
def test_declared_face(case, entry, face, code):
    compare_departures([d for d in measured(case, entry) if key(d) == (face, code)])


def test_wrapped_wall_keeps_field_classification():
    import rfx.simulation as simulation
    original = simulation.apply_pec_faces
    baseline = measured("pec", "run")

    def renamed_operation(*args, **kwargs):
        return original(*args, **kwargs)

    jax.clear_caches()
    with patch.object(simulation, "apply_pec_faces", renamed_operation):
        sim, grid, _, records = measure("pec", "run")
    fields, psi, reference = measured_fields(records, grid, "run")
    assert departures(sim.boundary_model(), grid, classify(fields, grid, psi, reference), "run") == baseline
    jax.clear_caches()


@pytest.mark.parametrize("case,expected", [("pec", True), ("pmc-pec", False)])
def test_cpu_gpu_query_dispatch(case, expected):
    jax.clear_caches()
    _, _, _, records = measure(case, "gpu-query")
    assert any(r["fast_trace_calls"] > 0 for r in records) == expected
    jax.clear_caches()
