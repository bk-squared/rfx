"""NTFF exterior admission: literal outcomes at public execution entries."""
import gc

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


EXTERIOR = (
    "an NTFF box radiates its surface currents into free space, but {faces}: "
    "the far field would miss their images (a symmetry or ground plane, or a "
    "periodic wrap, is not reconstructed). Declare an absorbing boundary on "
    "every face (draw a finite ground plane as geometry), or use "
    "closed_box=True for a plane wave on a finite scatterer."
)
PLANE = (
    "an NTFF box needs a scattered-field region around it, but this plane "
    "wave's total-field region spans the whole cross-section; use "
    "add_tfsf_source(..., closed_box=True) for a finite scatterer."
)
UPML = (
    "boundary='upml' does not support the non-uniform {entry}() lane: the "
    "non-uniform runner implements CPML only (rfx/nonuniform.py dispatches on "
    "cpml_layers > 0 and never reads the boundary type), so this configuration "
    "used to run CPML while sim._boundary kept reporting 'upml' (issue #680). "
    "Pass boundary='cpml' to run the absorber that is actually implemented "
    "here, or drop the mesh profile(s) to use the uniform lane, which does "
    "implement UPML."
)
CLOSED_GRADED = (
    "CustomWaveform and closed_box TFSF require a uniform, single-device, "
    "3D second-order Yee run/forward with no subgrids"
)

# Literal expected table: no expected outcomes are derived from production.
# Existing UPML refusal: rfx/api/_execute.py:1154 (_reject_upml_on_nonuniform).
# Existing graded closed-box refusal: rfx/api/_dispatch.py:115.
EXPECTED_TABLE = (
    ("uniform", "cpml", "admitted", None),
    ("uniform", "upml", "admitted", None),
    ("uniform", "pmc", "refused", EXTERIOR.format(faces="y_lo is pmc")),
    ("uniform", "pec", "refused", EXTERIOR.format(faces="y_lo is pec")),
    ("uniform", "periodic", "refused", EXTERIOR.format(faces="x_hi is periodic; x_lo is periodic")),
    ("uniform", "zero", "refused", EXTERIOR.format(faces="y_lo is zero-depth absorber")),
    ("uniform", "plane", "refused", PLANE),
    ("uniform", "closed", "admitted", None),
    ("graded", "cpml", "admitted", None),
    ("graded", "upml", "refused", UPML),
    ("graded", "pmc", "refused", EXTERIOR.format(faces="y_lo is pmc")),
    ("graded", "pec", "refused", EXTERIOR.format(faces="y_lo is pec")),
    ("graded", "periodic", "refused", EXTERIOR.format(faces="x_hi is periodic; x_lo is periodic")),
    ("graded", "zero", "refused", EXTERIOR.format(faces="y_lo is zero-depth absorber")),
    ("graded", "plane", "refused", PLANE),
    ("graded", "closed", "refused", CLOSED_GRADED),
)


def scene(mesh, case):
    boundary = "upml" if case == "upml" else "cpml"
    if case in ("pmc", "pec"):
        boundary = BoundarySpec(x="cpml", y=Boundary(lo=case, hi="cpml"), z="cpml")
    elif case == "zero":
        boundary = BoundarySpec(x="cpml", y=Boundary(lo="cpml", hi="cpml", lo_thickness=0), z="cpml")
    elif case == "periodic":
        boundary = BoundarySpec(x="periodic", y="cpml", z="cpml")
    profiles = {"dz_profile": np.array([.0015, .0025] * 5)} if mesh == "graded" else {}
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.002,
                     cpml_layers=2, boundary=boundary, **profiles)
    if case in ("plane", "closed"):
        sim.add_tfsf_source(f0=5e9, margin=3, closed_box=case == "closed")
    else:
        sim.add_source((.01, .01, .01), "ez", waveform=GaussianPulse(f0=5e9),
                       amplitude_kind="current")
    sim.add_ntff_box((.002, .002, .002), (.018, .018, .018), freqs=[5e9])
    return sim


@pytest.fixture(autouse=True)
def release_fields():
    yield
    jax.clear_caches()
    gc.collect()


def forbid_field_loops(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("NTFF refusal reached a field loop")

    for name in ("scan", "fori_loop", "while_loop"):
        monkeypatch.setattr(jax.lax, name, reached)


@pytest.mark.parametrize("entry", ("run", "run_skip", "forward"))
@pytest.mark.parametrize("mesh,case,outcome,message", EXPECTED_TABLE,
                         ids=[f"{row[0]}-{row[1]}" for row in EXPECTED_TABLE])
def test_ntff_boundary_table(mesh, case, outcome, message, entry, monkeypatch):
    sim = scene(mesh, case)
    method = "run" if entry == "run_skip" else entry
    kwargs = {"n_steps": 1}
    if entry == "run_skip":
        kwargs["skip_preflight"] = True
    if outcome == "admitted":
        result = getattr(sim, method)(**kwargs)
        assert result.ntff_data is not None
        return
    forbid_field_loops(monkeypatch)
    if case == "upml":
        expected = message.format(entry=method)
    elif case == "closed":
        expected = message
    else:
        expected = f"{method} dispatch: {message}"
    with pytest.raises(NotImplementedError if case == "closed" else ValueError) as caught:
        getattr(sim, method)(**kwargs)
    assert str(caught.value) == expected


@pytest.mark.parametrize("case,detail", (("pmc", EXTERIOR.format(faces="y_lo is pmc")),
                                        ("plane", PLANE)))
def test_direct_runner_admission(case, detail, monkeypatch):
    from rfx.runners._admission import admit

    forbid_field_loops(monkeypatch)
    with pytest.raises(ValueError) as caught:
        admit(scene("uniform", case), "run_uniform")
    assert str(caught.value) == f"run_uniform: {detail}"
