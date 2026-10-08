"""Addendum 5c admissions and provenance, without a field update."""
import gc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.features import WaveguideBoundaryWarning
from rfx.boundaries.tfsf import UnjudgedTFSFInvarianceWarning
from tests.contracts.test_tfsf_boundary_admission import plane
from tests.contracts.test_waveguide_boundary_admission import guide


class Admitted(Exception):
    pass


@pytest.fixture(autouse=True)
def no_field_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("admission judge reached a field scan")
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    yield
    gc.collect()


def stop_kernel(monkeypatch, *, periodic=None):
    import rfx.simulation as kernel

    def stop(*args, **kwargs):
        if periodic is not None:
            assert args[0].periodic == periodic
        raise Admitted

    monkeypatch.setattr(kernel, "make_core_step", stop)


@pytest.mark.parametrize("skip", [False, True])
def test_declared_walls_refuse_a_finite_box(skip):
    sim = plane(finite=True, boundary={"x": "cpml", "y": "pmc", "z": "pec"})
    with pytest.raises(ValueError, match=r"y_lo, y_hi.*eps_r.*closed_box"):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_declared_invariant_walls_keep_the_reported_legacy_wrap(skip, monkeypatch):
    sim = plane(boundary={"x": "cpml", "y": "pmc", "z": "pec"})
    findings = [p for p in sim.preflight(check_ntff=False)
                if p.code == "tfsf_transverse_periodic"]
    assert len(findings) == 1
    assert all(face in str(findings[0]) for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
    stop_kernel(monkeypatch, periodic=(False, True, True))
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_localized_updates_are_not_invariant(skip):
    sim = plane()
    # Two series elements use a localized ADE update, not conductivity folding.
    sim.add_lumped_rlc((.01, .005, .005), component="ez", R=50., C=1e-12)
    with pytest.raises(ValueError, match=r"y_lo, y_hi: _lumped_rlc"):
        sim.run(n_steps=1, skip_preflight=skip)


def test_point_source_keeps_its_existing_registration_fence():
    sim = plane()
    with pytest.raises(ValueError, match="Lumped ports are not supported together with the TFSF"):
        sim.add_port((.01, .005, .005), component="ez", impedance=0)


@pytest.mark.parametrize("skip", [False, True])
def test_upml_remains_refused(skip):
    with pytest.raises(ValueError, match="upml"):
        plane(boundary="upml").run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
def test_propagation_axis_wall_keeps_main_admission(skip, monkeypatch):
    sim = plane(boundary={"x": {"lo": "cpml", "hi": "pec"}, "y": "cpml", "z": "cpml"})
    stop_kernel(monkeypatch, periodic=(False, True, True))
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("slot", [False, True])
def test_only_the_duplicated_end_edge_is_exempt(skip, slot, monkeypatch):
    sim = plane(snap="declared")
    ranges = [(-1., .004), (.006, 1.)] if slot else [(-1., 1.)]
    for lo, hi in ranges:
        sim.add_thin_conductor(Box((.014, lo, -1.), (.014, hi, 1.)), sigma_bulk=5.8e7)
    stop_kernel(monkeypatch)
    if slot:
        with pytest.raises(ValueError, match=r"y_lo, y_hi.*pec_edge_masks"):
            sim.run(n_steps=1, skip_preflight=skip)
    else:
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("transform", ["jit", "jit_grad", "vmap"])
@pytest.mark.parametrize("finite", [False, True])
def test_traced_override_is_admitted_with_unjudged_finding(skip, transform, finite, monkeypatch):
    sim = plane(finite=finite)
    shape = sim._build_grid().shape
    stop_kernel(monkeypatch)

    def loss(value):
        result = sim.forward(n_steps=1, eps_override=jnp.ones(shape) * value,
                             skip_preflight=skip)
        return jnp.sum(result.time_series)

    fn = jax.jit(jax.grad(loss)) if transform == "jit_grad" else getattr(jax, transform)(loss)
    argument = jnp.array([2.]) if transform == "vmap" else jnp.array(2.)
    with pytest.warns(UnjudgedTFSFInvarianceWarning, match="invariance was not judged"):
        with pytest.raises(Admitted):
            fn(argument)



@pytest.mark.parametrize("skip", [False, True])
def test_concrete_override_replaces_finite_base_for_admission(skip, monkeypatch):
    sim = plane(finite=True)
    stop_kernel(monkeypatch, periodic=(False, True, True))
    with pytest.raises(Admitted):
        sim.forward(n_steps=1, eps_override=jnp.full(sim._build_grid().shape, 2.),
                    skip_preflight=skip)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("component", ["sigma", "mu_r"])
def test_override_does_not_hide_an_unoverridden_material(skip, component):
    sim = plane(finite=True)
    sim.add_material("other", eps_r=2.5, **{component: 2.})
    sim._geometry.clear()
    sim.add(Box((.008, .003, .003), (.012, .007, .007)), material="other")
    shape = sim._build_grid().shape

    @jax.jit
    def solve(value):
        return sim.forward(n_steps=1, eps_override=jnp.ones(shape) * value,
                           skip_preflight=skip).time_series

    with pytest.raises(ValueError, match="TF/SF y_lo, y_hi: .*" + component):
        solve(jnp.array(2.))


def test_default_provenance_survives_convergence_factory(monkeypatch):
    import rfx.convergence as convergence

    def build(sim_factory, **kwargs):
        return sim_factory(.001)

    monkeypatch.setattr(convergence, "convergence_study", build)
    rebuilt = convergence.quick_convergence(guide(), dx_factors=[1.], n_steps=1)
    assert rebuilt._boundary_explicit is False
    assert tuple(f.kind.value for f in rebuilt._build_grid().boundary_depths) == (
        "ABSORBER", "ABSORBER", "PEC", "PEC", "PEC", "PEC")


def test_default_provenance_survives_waveguide_reference_factory(monkeypatch):
    import rfx.sparams.waveguide as waveguide
    original = waveguide._empty_waveguide_reference

    def capture(sim):
        rebuilt = original(sim)
        assert rebuilt._boundary_explicit is False
        assert rebuilt._boundary == "cpml"
        raise Admitted

    monkeypatch.setattr(waveguide, "_empty_waveguide_reference", capture)
    sim = guide(dz_profile=np.full(10, .001))
    sim.add_waveguide_port(.03, direction="-x", f0=8e9, probe_offset=2, ref_offset=1)
    with pytest.raises(Admitted):
        waveguide._empty_waveguide_reference(sim)
    with pytest.raises(ValueError, match="graded mesh keeps absorbers"):
        sim.compute_waveguide_s_matrix(n_steps=8, normalize=True)


def test_realized_kind_agrees_with_default_grid_record():
    from rfx.boundaries.model import realize
    sim = guide()
    actual = realize(sim.boundary_model(), sim._build_grid())
    assert tuple(face.kind.value for face in actual.faces) == (
        "ABSORBER", "ABSORBER", "PEC", "PEC", "PEC", "PEC")


def test_nonuniform_without_stored_kinds_keeps_declared_wall_kinds():
    from rfx.boundaries.model import realize
    sim = Simulation(freq_max=10e9, domain=(.04, .02, .01), dx=.001,
                     cpml_layers=4, dz_profile=np.full(10, .001),
                     boundary={"x": "cpml", "y": "pmc", "z": "pec"})
    actual = realize(sim.boundary_model(), sim._build_nonuniform_grid())
    assert tuple(face.kind.value for face in actual.faces) == (
        "ABSORBER", "ABSORBER", "PMC", "PMC", "PEC", "PEC")


def test_guide_relabel_does_not_relabel_the_floquet_declaration():
    from rfx.boundaries.model import realize
    sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001,
                     cpml_layers=8, boundary="cpml")
    sim.add_floquet_port(.003, freqs=np.array([10e9]), f0=10e9)
    grid = sim._build_grid()
    assert tuple(face.kind.value for face in grid.boundary_depths) == (
        "PERIODIC", "PERIODIC", "PERIODIC", "PERIODIC", "ABSORBER", "ABSORBER")
    actual = realize(sim.boundary_model(), grid)
    assert tuple(face.kind.value for face in actual.faces) == (
        "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER", "ABSORBER")
    assert actual.periods == ()


@pytest.mark.parametrize("skip", [False, True])
def test_main_document_without_provenance_keeps_its_realized_walls(skip, monkeypatch):
    from rfx.interop import design_to_dict, simulation_from_design
    document = design_to_dict(guide(boundary="cpml"))
    assert document["boundary"] == {
        "spec": {a: {"lo": "cpml", "hi": "cpml"} for a in "xyz"},
        "legacy": {"boundary": "cpml", "cpml_layers": 4, "cpml_kappa_max": 1.,
                   "pec_faces": [], "periodic_axes": ""}}
    rebuilt = simulation_from_design(document)
    assert design_to_dict(rebuilt) == document
    assert rebuilt._build_grid().shape == (49, 21, 11)
    stop_kernel(monkeypatch)
    with pytest.raises(Admitted):
        rebuilt.run(n_steps=1, skip_preflight=skip)


def test_default_export_does_not_extend_the_frozen_schema():
    from rfx.interop import design_to_dict
    sim = Simulation(freq_max=10e9, domain=(.04, .02, .01), dx=.001, cpml_layers=4)
    boundary = design_to_dict(sim)["boundary"]
    assert set(boundary) == {"spec", "legacy"}
    assert boundary["spec"] == {a: {"lo": "cpml", "hi": "cpml"} for a in "xyz"}


def test_warning_names_the_user_helper_frame(monkeypatch):
    from pathlib import Path
    from tests.contracts import boundary_warning_helper
    stop_kernel(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Admitted):
            boundary_warning_helper.run(guide())
    default = [w for w in caught if issubclass(w.category, WaveguideBoundaryWarning)]
    assert len(default) == 1
    assert Path(default[0].filename).name == "boundary_warning_helper.py"
    assert default[0].lineno == 5


@pytest.mark.parametrize("skip", [False, True])
def test_nonuniform_explicit_absorbers_keep_real_pads_without_default_warning(skip, monkeypatch):
    import rfx.runners.nonuniform as kernel
    sim = guide("cpml", dz_profile=np.full(10, .001))
    def stop(grid, *args, **kwargs):
        assert grid.shape == (49, 29, 19)
        assert tuple(getattr(grid, f"pad_{a}_{side}") for a in "xyz" for side in ("lo", "hi")) == (4,) * 6
        raise Admitted
    monkeypatch.setattr(kernel, "run_nonuniform", stop)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=skip)
    assert not any(issubclass(w.category, WaveguideBoundaryWarning) for w in caught)
