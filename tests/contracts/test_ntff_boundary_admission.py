"""NTFF exterior admission: literal outcomes at public execution entries."""
import gc

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


EXTERIOR = (
    "an NTFF box radiates its surface currents into free space, but {faces}: "
    "the far field would miss their images (a symmetry or ground plane is not "
    "reconstructed). Every face of the realized model must be absorbing: draw a "
    "finite ground plane as geometry, and note that a full-aperture waveguide port "
    "with an undeclared boundary on the uniform mesh solves its transverse faces "
    "as PEC walls; a symmetry plane is not supported "
    "with NTFF yet."
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
    "CustomWaveform and closed_box TFSF require a uniform, "
    "3D second-order Yee run/forward with no subgrids"
)


# Literal expected table: no expected outcomes are derived from production.
# Existing UPML refusal: rfx/runners/_admission.py (_reject_upml_on_nonuniform).
# Existing graded closed-box refusal: at declaration, add_tfsf_source ->
# rfx/sources/tfsf_discrete_plane_wave.py validate_declaration (same text as
# the run-time refusal in rfx/api/_dispatch.py, which it now precedes).
# Periodic axes are carried by both single-device Yee lanes (#1221).
EXPECTED_TABLE = (
    ("uniform", "cpml", "admitted", None),
    ("uniform", "upml", "admitted", None),
    ("uniform", "pmc", "refused", EXTERIOR.format(faces="y_lo is pmc")),
    ("uniform", "pec", "refused", EXTERIOR.format(faces="y_lo is pec")),
    ("uniform", "periodic", "admitted", None),  # not decided by this rule
    ("uniform", "zero", "refused", EXTERIOR.format(faces="y_lo is zero-depth absorber")),
    ("uniform", "closed", "admitted", None),
    ("graded", "cpml", "admitted", None),
    ("graded", "upml", "refused", UPML),
    ("graded", "pmc", "refused", EXTERIOR.format(faces="y_lo is pmc")),
    ("graded", "pec", "refused", EXTERIOR.format(faces="y_lo is pec")),
    ("graded", "periodic", "admitted", None),
    ("graded", "zero", "refused", EXTERIOR.format(faces="y_lo is zero-depth absorber")),
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
    if case == "closed":
        sim.add_tfsf_source(f0=5e9, margin=3, closed_box=True)
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
    method = "run" if entry == "run_skip" else entry
    kwargs = {"n_steps": 1}
    if entry == "run_skip":
        kwargs["skip_preflight"] = True
    if outcome == "admitted":
        result = getattr(scene(mesh, case), method)(**kwargs)
        assert result.ntff_data is not None
        return
    forbid_field_loops(monkeypatch)
    if case in ("upml", "periodic"):
        expected = message.format(entry=method)
    elif case == "closed":
        expected = message
    else:
        expected = f"{method} dispatch: {message}"
    # The scene is built inside the check: a graded closed box is refused when
    # it is declared, before any entry runs.
    with pytest.raises(NotImplementedError if case in ("closed", "periodic") else ValueError) as caught:
        getattr(scene(mesh, case), method)(**kwargs)
    assert str(caught.value) == expected


@pytest.mark.parametrize("case,detail", (("pmc", EXTERIOR.format(faces="y_lo is pmc")),
                                        ("zero", EXTERIOR.format(faces="y_lo is zero-depth absorber"))))
def test_direct_runner_admission(case, detail, monkeypatch):
    from rfx.runners._admission import admit

    forbid_field_loops(monkeypatch)
    with pytest.raises(ValueError) as caught:
        admit(scene("uniform", case), "run_uniform")
    assert str(caught.value) == f"run_uniform: {detail}"


RCS_TABLE = (
    (None, "admitted", None),
    ("pec", "refused", EXTERIOR.format(
        faces="x_hi is pec; x_lo is pec; y_hi is pec; y_lo is pec; z_hi is pec; z_lo is pec")),
    ("periodic", "refused", EXTERIOR.format(
        faces="x_hi is pec; x_lo is pec; y_hi is pec; y_lo is pec; z_hi is pec; z_lo is pec")),
)


@pytest.mark.parametrize("boundary,outcome,message", RCS_TABLE, ids=("default", "pec", "periodic"))
def test_rcs_boundary_table(boundary, outcome, message, monkeypatch):
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.rcs import compute_rcs

    grid = Grid(freq_max=15e9, domain=(.054, .054, .054), dx=.003, cpml_layers=6)
    kwargs = {} if boundary is None else {"boundary": boundary}
    if outcome == "refused":
        forbid_field_loops(monkeypatch)

    def run_calculator():
        return compute_rcs(grid, init_materials(grid.shape), 4, f0=10e9,
                           cpml_layers=6, tfsf_margin=3, freqs=np.array([1e10]),
                           theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.]), **kwargs)

    if outcome == "admitted":
        assert np.all(np.isfinite(np.asarray(run_calculator().monostatic_rcs)))
    else:
        with pytest.raises(ValueError) as caught:
            run_calculator()
        assert str(caught.value) == f"compute_rcs: {message}"


def test_waveguide_calculator_warns_and_drops_ntff(monkeypatch):
    """The #704 calculator forms no far field from its PEC-walled guide."""
    from tests.unit.farfield.test_ntff_smatrix_drop_warning import (
        _GridBuildIntercept, _add_box, _intercept_grid_build, _wr90,
    )

    sim = _add_box(_wr90())
    _intercept_grid_build(monkeypatch)
    with pytest.warns(UserWarning, match=r"add_ntff_box.*far-field"):
        with pytest.raises(_GridBuildIntercept):
            sim.compute_waveguide_s_matrix(normalize=True)


def test_material_fit_forms_no_far_field(monkeypatch):
    from tests.unit.autodiff.test_differentiable_material_fit_inputs import (
        _acceptance_fixture, _FDTDReached, _fit_to_scan,
    )

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = _acceptance_fixture(eps_inf, debye_poles, lorentz_poles, boundary="pec")
        sim.add_ntff_box((.002, .002, .002), (.010, .008, .006), freqs=[3e9])
        return sim

    assert isinstance(_fit_to_scan(monkeypatch, factory), _FDTDReached)


def test_graded_waveguide_drops_ntff_before_runner(monkeypatch):
    from tests.unit.autodiff.test_waveguide_nu_flux_ad import _wr90_nu_sim

    sim, domain_x = _wr90_nu_sim()
    sim.add_ntff_box((.002, .002, .002), (domain_x - .002, .02, .008), freqs=[10e9])
    registered = sim._ntff

    class RunnerReached(Exception):
        pass

    def reached(device, **kwargs):
        assert device is not sim
        assert device._ntff is None
        assert sim._ntff is registered
        raise RunnerReached

    monkeypatch.setattr("rfx.runners.nonuniform.run_nonuniform_path", reached)
    with pytest.warns(UserWarning, match=r"add_ntff_box.*far-field"):
        with pytest.raises(RunnerReached):
            sim.compute_waveguide_s_matrix(normalize=True, n_steps=2)
    assert sim._ntff is registered


@pytest.mark.parametrize("entry", ("run", "forward"))
def test_adi_ntff_refusal_precedes_exterior(entry, monkeypatch):
    sim = Simulation(freq_max=10e9, domain=(.02, .02, .02), dx=.002,
                     boundary="pec", solver="adi")
    sim.add_ntff_box((.002, .002, .002), (.018, .018, .018), freqs=[5e9])
    forbid_field_loops(monkeypatch)
    with pytest.raises(ValueError) as caught:
        getattr(sim, entry)(n_steps=1)
    assert str(caught.value) == "solver='adi' does not support NTFF accumulation yet"


WAVEGUIDE_TABLE = (
    ("small", "default", None),
    ("small", "cpml", None),
    ("full", "default", "y_hi is pec; y_lo is pec; z_hi is pec; z_lo is pec"),
    ("full", "cpml", None),
)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("aperture,boundary,faces", WAVEGUIDE_TABLE,
                         ids=[f"{r[0]}-{r[1]}" for r in WAVEGUIDE_TABLE])
def test_waveguide_realized_exterior(aperture, boundary, faces, entry, monkeypatch):
    kwargs = {} if boundary == "default" else {"boundary": "cpml"}
    sim = Simulation(freq_max=15e9, domain=(.024, .012, .012), dx=.001, **kwargs)
    port = {} if aperture == "full" else {"y_range": (.002, .010), "z_range": (.002, .010)}
    sim.add_waveguide_port(.006, direction="+x", f0=7.5e9, **port)
    if aperture == "small":
        for lo, hi in (
            ((0., .001, .001), (.016, .002, .011)),
            ((0., .010, .001), (.016, .011, .011)),
            ((0., .001, .001), (.016, .011, .002)),
            ((0., .001, .010), (.016, .011, .011)),
        ):
            sim.add(Box(lo, hi), material="pec")
    sim.add_ntff_box((.004, .004, .004), (.020, .008, .008), n_freqs=3)
    if aperture == "small":
        result = getattr(sim, entry)(n_steps=1)
        assert result is not None
        return
    forbid_field_loops(monkeypatch)
    expected = (f"{entry} dispatch: " + EXTERIOR.format(faces=faces) if faces else
                f"{entry} dispatch: full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; "
                "declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis).")
    with pytest.raises(ValueError) as caught:
        getattr(sim, entry)(n_steps=1)
    assert str(caught.value) == expected


@pytest.mark.parametrize("case", ("xy_pads", "pec_z"))
def test_oblique_rcs_wrapped_z_is_not_refused(case, monkeypatch):
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.rcs import compute_rcs

    kwargs = {"cpml_axes": "xy"} if case == "xy_pads" else {"pec_faces": {"z_lo", "z_hi"}}
    grid = Grid(freq_max=15e9, domain=(.054, .054, .054), dx=.003, cpml_layers=6, **kwargs)

    class FieldLoopReached(Exception):
        pass

    def reached(*args, **kwargs):
        assert kwargs["periodic"] == (False, False, True)
        assert kwargs["cpml_axes"] == "xy"
        raise FieldLoopReached

    monkeypatch.setattr("rfx.rcs.run", reached)
    with pytest.raises(FieldLoopReached):
        compute_rcs(grid, init_materials(grid.shape), 4, f0=10e9, theta_inc=20.,
                    cpml_layers=6, tfsf_margin=3, freqs=np.array([1e10]),
                    theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.]))
