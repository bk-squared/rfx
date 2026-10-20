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
# PR3a supersedes the explicit-guide departure rows without rewriting B1's record.
WALLED_DEFAULT_ENTRIES = {"run", "forward", "distributed", "sweep", "gpu-query", "subgridded"}
# Addendum 5f (2026-10-11) supersedes the graded explicit-guide rows; B1's record stays immutable.
# The graded default stays refused (5d), so it has no walled-default twin below.
GUIDE_REFUSAL_ENTRIES = WALLED_DEFAULT_ENTRIES | {"nonuniform"}
CELLS = [dict(c, status="REFUSED_AT_DISPATCH", departures=[])
         if c["case"] == "waveguide-cpml" and c["entry"] in GUIDE_REFUSAL_ENTRIES else c
         for c in BASELINE["cells"]]


def backed(cell):
    """Tracker 1549 supersedes B1's "electric backing required" rows (code f, "fixed in B3") without rewriting
    B1's record: forward() and the multi-device uniform run now apply the conducting plane behind every absorbing
    face from the boundary record, as run() does, and so do the waveguide entries. Where B1 recorded the backing as
    missing, the expected record becomes "present at the plane B1 said it was required at"."""
    if cell["status"] != "MEASURED" or not (cell["entry"] in ("forward", "distributed")
                                             or cell["case"] == "waveguide-pec"):
        return cell
    # A face a feature wraps (TF/SF, Floquet: code "feature") is not an absorbing face in the run; it gets no
    # conducting plane and its "f" row stays listed.
    wrapped = {d["face"] for d in cell["departures"] if d["code"] == "feature"}
    missing = {d["face"] for d in cell["departures"] if d["code"] == "f"} - wrapped
    faces = {name: dict(face, e_zero=True, e_zero_planes_m=[face["e_plane_m"]]) if name in missing else face
             for name, face in cell["faces"].items()}
    return dict(cell, faces=faces,
                departures=[d for d in cell["departures"] if d["code"] != "f" or d["face"] in wrapped])


def magnetic_distributed(cell):
    """#1221 B4 supersedes B1's refused distributed PMC cells with their single-device run records."""
    if cell["case"].startswith("pmc-") and cell["entry"] == "distributed" and cell["status"] == "REFUSED":
        reference = next(c for c in CELLS if c["case"] == cell["case"] and c["entry"] == "run")
        return dict(cell, **{key: reference[key] for key in ("status", "faces", "departures")})
    return cell


def periodic_graded(cell):
    """#1221 (b) supersedes B1's refused graded periodic cell with the uniform run's record: the graded single-device
    lane now carries periodic axes, so it must realize the same faces as run() on the uniform mesh."""
    if cell["case"] == "periodic-xy" and cell["entry"] == "nonuniform" and cell["status"] == "REFUSED":
        reference = next(c for c in CELLS if c["case"] == cell["case"] and c["entry"] == "run")
        return dict(cell, **{key: reference[key] for key in ("status", "faces", "departures")})
    return cell


CELLS = [backed(magnetic_distributed(periodic_graded(c))) for c in CELLS]
BY_CELL = {(c["case"], c["entry"]): c for c in CELLS}


def adi_absorber_cell(case, entry):
    return entry == "adi" and case not in ("pec", "pmc-pec", "upml")


@lru_cache(maxsize=None)
def measured(case, entry):
    sim, grid, _, records = measure(case, entry)
    assert records, "entry point produced no observed field scan"
    fields, psi, reference = measured_fields(records, grid, entry)
    faces = classify(fields, grid, psi, reference)
    compare_values(sim.boundary_model(), grid, faces, BY_CELL[case, entry]["faces"])
    return departures(sim.boundary_model(), grid, faces, entry)


def floquet_refused(entry):
    """#1944 supersedes B1's floquet cells: they record explicit CPML lateral faces
    silently made periodic, which a Floquet port now refuses by name. The graded-z
    and ADI refusals fire first, at registration and construction. Keep B1 immutable."""
    match = {"nonuniform": "do not support non-uniform z mesh",
             "adi": "refuses absorbing boundaries"}.get(entry, "requires periodic lateral faces")
    with pytest.raises(ValueError, match=match):
        measured("floquet", entry)


def key(departure):
    return departure["face"], departure["code"]


def compare_departures(problems):
    if problems:
        raise BoundaryDeparture(str(problems))


@pytest.mark.parametrize("cell", [pytest.param(c, id=f"{c['case']}--{c['entry']}",
                                              marks=pytest.mark.xdist_group(f"{c['case']}--{c['entry']}")) for c in CELLS])
def test_no_unlisted_departures(cell):
    if cell["case"] == "floquet":
        return floquet_refused(cell["entry"])
    if cell["status"] == "REFUSED_AT_DISPATCH":
        with pytest.raises(ValueError) as caught:
            measured(cell["case"], cell["entry"])
        message = str(caught.value)
        assert "full-aperture waveguide requires PEC" in message
        assert all(face in message for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
        assert "declare boundary=" in message and "'y': 'pec'" in message and "'z': 'pec'" in message
        return
    # The public sponge refusal supersedes B1; retain the historical record.
    if adi_absorber_cell(cell["case"], cell["entry"]):
        with pytest.raises(ValueError, match="unmatched graded-conductivity sponge"):
            measured(cell["case"], cell["entry"])
        return
    # #1465 supersedes B1's subgrid refusal reason; the two construction /
    # preflight refusals still occur before lane admission. Keep B1 immutable.
    if cell["entry"] == "subgridded" and cell["case"] not in ("upml", "waveguide-pmc"):
        with pytest.raises(NotImplementedError, match="subgridded lane is unstable and unverified"):
            measured(cell["case"], cell["entry"])
        return
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
            for cell in CELLS if cell["status"] == "MEASURED"
            and not adi_absorber_cell(cell["case"], cell["entry"])
            and cell["case"] != "floquet"  # refused since #1944; pinned in test_no_unlisted_departures
            for departure in cell["departures"]]
# PMC images are checked at the declared plane on both device counts.
EXPECTED += [pytest.param(case, entry, face, code,
                          id=f"{case}--{entry}--{face}--{code}")
             for case in ("pmc-pec", "pmc-cpml")
             for entry in ("run", "forward", "nonuniform", "wire-fast", "gpu-query", "distributed")
             for face in ("x_lo", "x_hi") for code in ("b1", "h")]
# Waveguide PMC now refuses; test_no_unlisted_departures pins that status.
EXPECTED += [pytest.param("periodic-xy", entry, face, "e",
                          id=f"periodic-xy--{entry}--{face}--e",
                          marks=pytest.mark.xdist_group(f"periodic-xy--{entry}"))
             for entry in ("run", "forward", "sweep", "gpu-query")
             for face in ("x_lo", "x_hi", "y_lo", "y_hi")]


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

    def worse(state, faces, **kwargs):
        calls.append(1)
        state = original(state, faces, **kwargs)
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


@pytest.mark.parametrize("entry", sorted(WALLED_DEFAULT_ENTRIES))
def test_default_waveguide_twin_records_its_walls(entry, monkeypatch):
    import warnings
    from rfx.boundaries.depths import Kind
    from rfx.boundaries.features import WaveguideBoundaryWarning
    from rfx.boundaries.model import realize
    from tests.contracts.boundary_cases import build
    from tests.contracts.boundary_fields import execute

    class Admitted(Exception):
        pass

    def stop(*args, **kwargs):
        raise Admitted

    sim, _ = build("waveguide-default", "run" if entry == "gpu-query" else entry)
    grid = sim._build_grid()
    assert tuple(f.kind for f in grid.boundary_depths) == (
        Kind.ABSORBER, Kind.ABSORBER, Kind.PEC, Kind.PEC, Kind.PEC, Kind.PEC)
    assert tuple(f.realized for f in grid.boundary_depths) == (8, 8, 0, 0, 0, 0)
    assert tuple(f.kind for f in realize(sim.boundary_model(), grid).faces) == tuple(
        f.kind for f in grid.boundary_depths)
    monkeypatch.setattr(jax.lax, "scan", stop)
    if entry == "gpu-query":
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Each sweep invocation constructs a distinct Simulation copy.
        for _ in range(1 if entry == "sweep" else 2):
            if entry == "subgridded":
                with pytest.raises(NotImplementedError, match="subgridded lane is unstable and unverified"):
                    execute(sim, entry)
            else:
                with pytest.raises(Admitted):
                    execute(sim, "run" if entry == "gpu-query" else entry)
    defaults = [w for w in caught if issubclass(w.category, WaveguideBoundaryWarning)]
    assert len(defaults) == 1
    assert all(face in str(defaults[0].message) for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
