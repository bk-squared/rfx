"""Addendum 5d: default graded guides refuse; declared walls keep the guide."""
import gc
import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.features import WaveguideBoundaryWarning


@pytest.fixture(autouse=True)
def release_models():
    yield
    jax.clear_caches()
    gc.collect()


def guide(*, graded=True, boundary=None, dielectric=False):
    kwargs = {} if boundary is None else {"boundary": boundary}
    if graded:
        kwargs["dz_profile"] = np.full(5, .002)
    sim = Simulation(12e9, (.06, .02, .01), dx=.002, cpml_layers=8, **kwargs)
    if dielectric:
        sim.add_material("slab", eps_r=2.)
        sim.add(Box((.028, 0., 0.), (.032, .02, .01)), material="slab")
    for position, direction in ((.010, "+x"), (.050, "-x")):
        sim.add_waveguide_port(position, direction=direction, freqs=np.array([9e9, 9.5e9, 10e9, 10.5e9, 11e9, 11.5e9, 12e9, 12.5e9, 13e9]),
                               f0=11e9, bandwidth=.6, probe_offset=2, ref_offset=1)
    return sim


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("entry", ["run", "forward", "distributed_run", "distributed_forward", "s_matrix"])
def test_default_graded_refuses_before_field_step(skip, entry, monkeypatch):
    def field_step(*args, **kwargs):
        pytest.fail("default graded guide reached a field step")
    monkeypatch.setattr(jax.lax, "scan", field_step)
    sim = guide()
    with pytest.raises(ValueError) as caught:
        if entry == "s_matrix":
            # This calculator has no skip_preflight argument or distributed API.
            if not skip:
                sim.preflight()
            sim.compute_waveguide_s_matrix(n_steps=1, normalize=True)
        else:
            method = "forward" if "forward" in entry else "run"
            kwargs = ({"distributed": True} if method == "forward" else
                      {"devices": jax.devices()}) if entry.startswith("distributed") else {}
            getattr(sim, method)(n_steps=1, skip_preflight=skip, **kwargs)
    text = str(caught.value)
    assert all(face in text for face in ("y_lo", "y_hi", "z_lo", "z_hi"))
    assert "graded mesh keeps absorbers" in text
    assert "port would not be in a guide" in text
    assert "boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}" in text


def test_declared_walls_match_uniform_guide():
    uniform = guide(graded=False, dielectric=True)
    graded = guide(boundary={"x": "cpml", "y": "pec", "z": "pec"}, dielectric=True)
    grids = [sim._build_realized_grid() for sim in (uniform, graded)]
    assert grids[0].shape == grids[1].shape
    assert grids[0].shape[1:] == (11, 6)
    assert [tuple(getattr(grid, f"pad_{a}_{s}") for a in "xyz" for s in ("lo", "hi"))
            for grid in grids] == [(8, 8, 0, 0, 0, 0)] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WaveguideBoundaryWarning)
        expected = np.asarray(uniform.compute_waveguide_s_matrix(n_steps=600, normalize=True).s_params)
        actual = np.asarray(graded.compute_waveguide_s_matrix(n_steps=600, normalize=True).s_params)
    assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    expected_db = 20 * np.log10(np.abs(expected[:, 0]))
    actual_db = 20 * np.log10(np.abs(actual[:, 0]))
    deep_null_below_minus_40_db = expected_db[0] < -40
    reflection_bins = ~deep_null_below_minus_40_db
    assert np.any(reflection_bins)
    reflection = np.abs(actual_db[0] - expected_db[0])
    transmission = np.abs(actual_db[1] - expected_db[1])
    phase = np.abs(np.angle(actual[1, 0] / expected[1, 0], deg=True))
    print("5d bins GHz", [9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13], "S11 dB differences", reflection.tolist(),
          "S21 dB differences", transmission.tolist(), "S21 degree differences", phase.tolist(),
          "excluded S11 bins GHz", np.array([9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13])[deep_null_below_minus_40_db].tolist())
    assert np.all(reflection[reflection_bins] < .1)
    assert np.all(transmission < .1)
    assert np.all(phase < .1)


@pytest.mark.parametrize("method", ["run", "forward"])
@pytest.mark.parametrize("boundary", ["cpml", {"x": "cpml", "y": "pec", "z": "pec"}])
def test_explicit_graded_dispatch_reaches_distributed_nu(method, boundary, monkeypatch):
    sim = guide(boundary=boundary)
    original = sim._dispatch_plan
    reached = []

    class Admitted(Exception):
        pass

    def stop(**kwargs):
        plan = original(**kwargs)
        reached.append(plan.lane)
        raise Admitted

    monkeypatch.setattr(sim, "_dispatch_plan", stop)
    kwargs = {"devices": jax.devices()} if method == "run" else {"distributed": True}
    assert len(jax.devices()) >= 2, "run this judge with two CPU devices"
    if method == "forward":
        with pytest.raises(NotImplementedError, match="Waveguide ports are not supported on the distributed forward path"):
            sim.forward(n_steps=1, skip_preflight=True, **kwargs)
        assert reached == []
        return
    with pytest.raises(Admitted):
        getattr(sim, method)(n_steps=1, skip_preflight=True, **kwargs)
    assert reached == ["run_distributed_nu" if method == "run" else "fwd_distributed_nu"]


@pytest.mark.parametrize("provenance", [None, "document"])
def test_legacy_graded_document_keeps_main_admission(provenance, monkeypatch):
    from rfx.interop import design_to_dict, simulation_from_design
    sim = guide(boundary="cpml")
    if provenance == "document":
        document = design_to_dict(sim)
        sim = simulation_from_design(document)
        assert design_to_dict(sim) == document
    else:
        sim._boundary_explicit = None
    original = sim._dispatch_plan

    class Admitted(Exception):
        pass

    def stop(**kwargs):
        plan = original(**kwargs)
        assert plan.lane == "run_nonuniform"
        assert sim._build_realized_grid().shape == (47, 27, 22)
        raise Admitted

    monkeypatch.setattr(sim, "_dispatch_plan", stop)
    with pytest.raises(Admitted):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("axis", ["y", "z"])
def test_refusal_substitutes_port_axis(axis):
    sim = Simulation(12e9, (.02,) * 3, dx=.002, cpml_layers=8,
                     dz_profile=np.full(10, .002))
    sim.add_waveguide_port(.006, direction="+" + axis, f0=11e9,
                           probe_offset=2, ref_offset=1)
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=1, skip_preflight=True)
    expected = {a: "cpml" if a == axis else "pec" for a in "xyz"}
    assert f"boundary={expected!r}" in str(caught.value)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_explicit_aperture_ranges_on_graded_mesh(full, distributed, skip, monkeypatch):
    sim = Simulation(12e9, (.06, .02, .01), dx=.002, cpml_layers=8,
                     dz_profile=np.full(5, .002))
    sim.add_waveguide_port(.010, direction="+x", y_range=(0., .02 if full else .01),
                           z_range=(0., .01), f0=11e9, probe_offset=2, ref_offset=1)
    original = sim._dispatch_plan

    class Admitted(Exception):
        pass

    def stop(**kwargs):
        plan = original(**kwargs)
        assert plan.lane == ("run_distributed_nu" if distributed else "run_nonuniform")
        grid = sim._build_realized_grid()
        # Main 4acbd7925 build: both aperture declarations keep these pads.
        assert grid.shape == (47, 27, 22)
        assert tuple(getattr(grid, f"pad_{a}_{s}") for a in "xyz" for s in ("lo", "hi")) == (8,) * 6
        raise Admitted

    monkeypatch.setattr(sim, "_dispatch_plan", stop)
    if distributed:
        assert len(jax.devices()) >= 2, "run this judge with two CPU devices"
    kwargs = {"devices": jax.devices()} if distributed else {}
    if full:
        with pytest.raises(ValueError, match=r"y_lo.*z_hi.*graded mesh keeps absorbers.*boundary="):
            sim.run(n_steps=1, skip_preflight=skip, **kwargs)
    else:
        with pytest.raises(Admitted):
            sim.run(n_steps=1, skip_preflight=skip, **kwargs)
