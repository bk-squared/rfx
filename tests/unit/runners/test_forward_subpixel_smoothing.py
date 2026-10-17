"""The other public entry supplies the smoothed field reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation, Sphere
from rfx.sources.sources import GaussianPulse


def sphere_sim(graded=False, radius=.0023, *, debye=False, chi3=0.0, solver="yee"):
    profile = np.array([.001] * 4 + [.0008] * 5 + [.001] * 4)
    sim = Simulation(freq_max=20e9, domain=(.012, .012, .012), dx=.001,
                     dz_profile=profile if graded else None,
                     boundary="pec" if solver == "adi" else "cpml",
                     cpml_layers=0 if solver == "adi" else 2, solver=solver)
    sim.add_material("sphere", eps_r=4.0, chi3=chi3,
                     debye_poles=[DebyePole(delta_eps=1., tau=1e-11)] if debye else None)
    sim.add(Sphere((.00537, .00613, .00573), radius), material="sphere")
    sim.add_source((.0031, .0042, .0043), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8),
                   amplitude_kind="current")
    sim.add_probe((.0072, .0051, .0064), "ez")
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_smoothed_forward_matches_run_and_changes_fields(graded, smoothing):
    if graded and smoothing == "kottke_pec":
        pytest.skip("Stage 2 is refused on graded meshes")
    kwargs = dict(n_steps=40, skip_preflight=True, subpixel_smoothing=smoothing)
    expected = np.asarray(sphere_sim(graded).run(**kwargs).time_series)
    actual = np.asarray(sphere_sim(graded).forward(checkpoint=False, **kwargs).time_series)
    plain = np.asarray(sphere_sim(graded).forward(n_steps=40, skip_preflight=True,
                                                 checkpoint=False).time_series)
    # The contract compares common fields exactly.
    np.testing.assert_array_equal(actual, expected)
    difference = float(np.max(np.abs(actual - plain)))
    print(f"smoothing measurement: graded={graded}, mode={smoothing}, max_abs_diff={difference}")
    assert difference > 0


@pytest.mark.parametrize("body", ["sphere", "box"])
def test_kottke_pec_forward_keeps_fenced_pec_edges(body):
    # A PEC body next to the dielectric: run() keeps partly filled PEC
    # edges fractional, so forward() must not treat the Kottke tensor as
    # the smooth occupancy operator. Probes sit outside the metal.
    def build():
        sim = sphere_sim()
        if body == "sphere":
            sim.add(Sphere((.0085, .0043, .0068), .0016), material="pec")
        else:
            sim.add(Box((.0071, .0032, .0059), (.0094, .0057, .0083)), material="pec")
        sim.add_probe((.0063, .0081, .0047), "ez")
        return sim

    kwargs = dict(n_steps=60, skip_preflight=True, subpixel_smoothing="kottke_pec")
    expected = np.asarray(build().run(**kwargs).time_series)
    actual = np.asarray(build().forward(checkpoint=False, **kwargs).time_series)
    np.testing.assert_array_equal(actual, expected)


def no_step(*args, **kwargs):
    pytest.fail("a refused smoothing combination reached the first step")


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("override", ["eps_override", "pec_occupancy_override", "design_occupancy_override"])
def test_override_conflicts_refuse_before_stepping(monkeypatch, graded, override):
    import rfx.simulation
    import rfx.runners.nonuniform
    monkeypatch.setattr(rfx.simulation, "run", no_step)
    monkeypatch.setattr(rfx.runners.nonuniform, "run_nonuniform", no_step)
    with pytest.raises(NotImplementedError, match="subpixel_smoothing"):
        sphere_sim(graded).forward(n_steps=1, skip_preflight=True,
                                   subpixel_smoothing=True, **{override: jnp.ones((2, 2, 2))})


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_debye_refuses_before_stepping(monkeypatch, graded):
    import rfx.simulation
    import rfx.runners.nonuniform
    monkeypatch.setattr(rfx.simulation, "run", no_step)
    monkeypatch.setattr(rfx.runners.nonuniform, "run_nonuniform", no_step)
    with pytest.raises(NotImplementedError, match="subpixel_smoothing"):
        sphere_sim(graded, debye=True).forward(n_steps=1, skip_preflight=True,
                                              subpixel_smoothing=True)


def traced_run_sim(amplitude, graded=False, eps=None):
    sim = Simulation(freq_max=15e9, domain=(.014, .004, .004), dx=.001,
                     boundary="cpml", cpml_layers=1,
                     **(dict(dz_profile=np.full(4, .001)) if graded else {}))
    sim.add_source((.0063, .0021, .0019), "ez", amplitude_kind="field",
                   waveform=lambda t: amplitude * jnp.cos(t * 2e10))
    sim.add_probe((.0082, .0023, .0022), "ez")
    if eps is not None:
        sim.add_material("d", eps_r=eps)
        sim.add(Box((.003, .001, .001), (.005, .003, .003)), material="d")
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("transform", ["grad", "jit", "vmap"])
def test_run_traced_source_amplitude(graded, transform):
    def objective(amplitude):
        result = traced_run_sim(amplitude, graded).run(n_steps=30, skip_preflight=True)
        return jnp.sum(jnp.asarray(result.time_series) ** 2)
    plain = objective(1.0)
    if transform == "grad":
        actual, expected = jax.grad(objective)(1.0), 2 * plain
    elif transform == "jit":
        actual, expected = jax.jit(objective)(1.0), plain
    else:
        actual = jax.vmap(objective)(jnp.array([1.0, 2.0]))
        expected = jnp.array([plain, 4 * plain])
    print(f"traced source: graded={graded}, transform={transform}, actual={actual}, expected={expected}")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_run_traced_material_matches_forward(graded):
    def objective(eps, entry):
        result = getattr(traced_run_sim(1.0, graded, eps), entry)(
            n_steps=30, skip_preflight=True)
        return jnp.sum(jnp.asarray(result.time_series) ** 2)
    actual = jax.grad(lambda eps: objective(eps, "run"))(4.0)
    expected = jax.grad(lambda eps: objective(eps, "forward"))(4.0)
    print(f"traced material: graded={graded}, run={actual}, forward={expected}")
    assert float(expected) != 0
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)




@pytest.mark.parametrize("graded", [False, True])
def test_distributed_smoothing_refused(graded):
    messages = []
    for entry in ("run", "forward"):
        options = dict(devices=jax.devices()[:2]) if entry == "run" else dict(distributed=True)
        with pytest.raises(NotImplementedError, match="subpixel_smoothing") as info:
            getattr(sphere_sim(graded), entry)(n_steps=1, skip_preflight=True,
                                              subpixel_smoothing=True, **options)
        messages.append(str(info.value))
    assert messages[0] == messages[1]


def test_adi_smoothing_refused():
    with pytest.raises(NotImplementedError, match="subpixel_smoothing"):
        sphere_sim(solver="adi").forward(n_steps=1, subpixel_smoothing=True)


def test_kerr_smoothing_refused():
    with pytest.raises(NotImplementedError, match="subpixel_smoothing.*kerr_chi3"):
        sphere_sim(chi3=1e-20).forward(n_steps=1, subpixel_smoothing=True, skip_preflight=True)


def test_kerr_smoothing_refusal_matches_run():
    messages = []
    for entry in ("run", "forward"):
        with pytest.raises(NotImplementedError, match="subpixel_smoothing.*kerr_chi3") as info:
            getattr(sphere_sim(chi3=1e-20), entry)(
                n_steps=10, subpixel_smoothing=True, skip_preflight=True)
        messages.append(str(info.value))
    assert messages[0] == messages[1]


def test_graded_stage2_refusal_matches_run():
    messages = []
    for entry in ("run", "forward"):
        with pytest.raises(NotImplementedError) as info:
            getattr(sphere_sim(True), entry)(n_steps=1, subpixel_smoothing="kottke_pec", skip_preflight=True)
        messages.append(str(info.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_design_box_smoothing_reaches_tensor_refusal(monkeypatch, graded):
    import rfx.simulation
    import rfx.stepping.uniform_loop
    monkeypatch.setattr(rfx.simulation, "make_core_step", no_step)
    monkeypatch.setattr(rfx.stepping.uniform_loop, "make_core_step", no_step)
    with pytest.raises(NotImplementedError, match=r"design-box.*aniso_eps") as info:
        sphere_sim(graded).forward(n_steps=1, skip_preflight=True,
            subpixel_smoothing=True, design_box=((.004,) * 3, (.006,) * 3),
            design_eps_override=jnp.ones((3, 3, 3)) * 3)
    print(f"design_box measurement: graded={graded}, {info.value}")


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_sigma_override_is_carried_with_smoothing(monkeypatch, graded):
    import rfx.simulation
    import rfx.runners.nonuniform
    module = rfx.runners.nonuniform if graded else rfx.simulation
    name = "run_nonuniform" if graded else "run"
    original = getattr(module, name)
    observed = []

    def capture(grid, materials, *args, **kwargs):
        np.testing.assert_array_equal(materials.sigma, np.full(grid.shape, .01, dtype=np.float32))
        assert kwargs["aniso_eps"] is not None
        observed.append(True)
        return original(grid, materials, *args, **kwargs)

    monkeypatch.setattr(module, name, capture)
    sim = sphere_sim(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    result = sim.forward(n_steps=2, skip_preflight=True, subpixel_smoothing=True,
                          sigma_override=jnp.full(grid.shape, .01))
    assert observed == [True]
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_rlc_override_with_smoothing_matches_run(smoothing):
    expected_sim, actual_sim = sphere_sim(), sphere_sim()
    expected_sim.add_lumped_rlc((.007, .005, .006), "ez", C=2e-14, topology="parallel")
    actual_sim.add_lumped_rlc((.007, .005, .006), "ez", C=1e-14, topology="parallel")
    kwargs = dict(n_steps=12, skip_preflight=True, subpixel_smoothing=smoothing)
    expected = expected_sim.run(**kwargs).time_series
    actual = actual_sim.forward(rlc_values_override={0: {"C": 2e-14}},
                                checkpoint=False, **kwargs).time_series
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_monitor_overrides_are_carried_with_smoothing(monkeypatch):
    import rfx.current_moments
    original = rfx.current_moments.monitor_for_simulation
    observed = []

    def capture(sim, grid, *args, overrides=None, **kwargs):
        observed.append(overrides["sigma"])
        return original(sim, grid, *args, overrides=overrides, **kwargs)

    monkeypatch.setattr(rfx.current_moments, "monitor_for_simulation", capture)
    sim = sphere_sim()
    sigma = jnp.full(sim._build_grid().shape, .01)
    sim.forward(n_steps=2, skip_preflight=True, subpixel_smoothing=True, sigma_override=sigma)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], sigma)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_subgridded_smoothing_is_refused(entry):
    sim = Simulation(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0)
    sim.add_refinement((.008, .012), ratio=2, validation="research")
    with pytest.raises(NotImplementedError, match="subgrid|refinement") as info:
        getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing=True)
    print(f"subgridded {entry} smoothing refusal: {info.value}")


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_surface_impedance_smoothing_refusal_matches_run(monkeypatch, smoothing):
    import rfx.simulation
    monkeypatch.setattr(rfx.simulation, "run", no_step)
    messages = []
    for entry in ("run", "forward"):
        sim = sphere_sim()
        sim.add_thin_conductor(Box((.002, .002, .009), (.010, .010, .009)),
                               sigma_bulk=1e4, surface_impedance_f0=10e9)
        with pytest.raises(ValueError, match="subpixel_smoothing") as info:
            getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing=smoothing)
        messages.append(str(info.value))
    assert messages[0] == messages[1]
