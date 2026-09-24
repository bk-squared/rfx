"""Per-face field comparisons with B1's declared planes; CPU, always on."""

from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch

import jax
import pytest

from tests.contracts.boundary_compare import BoundaryDeparture, classify, compare_values, departures
from tests.contracts.boundary_fields import measure, measured_fields


ROOT = Path(__file__).resolve().parents[2]
BASELINE = json.loads((ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.json").read_text())
CELLS = BASELINE["cells"]
BY_CELL = {(c["case"], c["entry"]): c for c in CELLS}


@lru_cache(maxsize=None)
def measured(case, entry):
    sim, grid, _, records = measure(case, entry)
    assert records, "entry point produced no observed field scan"
    fields, psi, reference = measured_fields(records, grid, entry)
    faces = classify(fields, grid, psi, reference)
    compare_values(sim.boundary_model(), grid, faces, BY_CELL[case, entry]["faces"])
    return departures(sim.boundary_model(), grid, faces, entry)


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
        assert type(exc.value) is exception_type
        assert str(exc.value).startswith(cell["message_prefix"])
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
    calls = []

    def renamed_operation(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    jax.clear_caches()
    with patch.object(simulation, "apply_pec_faces", renamed_operation):
        sim, grid, _, records = measure("pec", "run")
    fields, psi, reference = measured_fields(records, grid, "run")
    assert calls, "the wrapped wall did not run"
    assert departures(sim.boundary_model(), grid, classify(fields, grid, psi, reference), "run") == baseline
    jax.clear_caches()


def test_worse_magnetic_wall_value_is_rejected():
    import rfx.boundaries.pmc as pmc
    original = pmc.apply_pmc_faces
    calls = []

    def worse(state, faces):
        calls.append(1)
        state = original(state, faces)
        values = {name: getattr(state, name) for name in ("hx", "hy", "hz")}
        for face in faces:
            axis = "xyz".index(face[0])
            selection = [slice(None)] * 3
            selection[axis] = 1 if face.endswith("lo") else -3
            for component, name in enumerate(values):
                if component != axis:
                    values[name] = values[name].at[tuple(selection)].set(0.0)
        return state._replace(**values)

    jax.clear_caches()
    try:
        with patch.object(pmc, "apply_pmc_faces", worse):
            sim, grid, _, records = measure("pmc-pec", "run")
        fields, psi, reference = measured_fields(records, grid, "run")
        assert calls, "the original magnetic wall helper must still run"
        faces = classify(fields, grid, psi, reference)
        baseline = BY_CELL["pmc-pec", "run"]
        assert {key(d) for d in departures(sim.boundary_model(), grid, faces, "run")} == {
            key(d) for d in baseline["departures"]}
        with pytest.raises(AssertionError, match="away from the declared plane") as exc:
            compare_values(sim.boundary_model(), grid, faces, baseline["faces"])
        assert type(exc.value) is AssertionError
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("case,expected", [("pec", True), ("pmc-pec", False)])
def test_cpu_gpu_query_dispatch(case, expected):
    jax.clear_caches()
    _, _, _, records = measure(case, "gpu-query")
    assert any(r["fast_trace_calls"] > 0 for r in records) == expected
    jax.clear_caches()
