"""A fixed-radius probe between PEC plates, against the radial TEM Green function.

The radius is fixed in metres on both meshes. The historical wire instead
gains about 15 ohms per halving at 11.3 GHz. The oracle assertions have their
own negative control, so deleting the assertion path does not turn this test
silently green. No stored solver response is used as the physics oracle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import hankel2

from rfx import Box, DebyePole, GaussianPulse, LorentzPole, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import (
    component_h_materials, init_materials, init_state, precompute_coeffs,
    update_e, update_h, update_h_fast, update_h_nu, update_he_fast,
)
from rfx.sources.sources import WirePort, setup_wire_port
from rfx.sources.wire_radius import MAX_RADIUS_RATIO
from tests._x64_compat import enable_x64


C0 = 299792458.0
ETA0 = 376.730313668
HEIGHT = 1.5e-3
RADIUS = 0.05e-3
FREQS = np.array([8e9, 11.3e9, 14e9])


def _plates(dx, lane, *, radius=RADIUS, length=8e-3):
    n = int(round(length / dx))
    profiles = {} if lane == "uniform" else dict(
        dx_profile=np.full(n, dx), dy_profile=np.full(n, dx))
    sim = Simulation(freq_max=30e9, domain=(length, length, HEIGHT), dx=dx,
                     boundary=BoundarySpec(x="cpml", y="cpml", z="pec"),
                     cpml_layers=12, **profiles)
    sim.add_port((length/2, length/2, 0.), "ez", extent=HEIGHT,
                 radius=radius, waveform=GaussianPulse(f0=11e9, bandwidth=.9))
    sim.add_probe((length/2+dx, length/2, HEIGHT/2), "ez")
    return sim


def _oracle():
    k = 2*np.pi*FREQS/C0
    return ETA0*k*HEIGHT/4 * hankel2(0, k*RADIUS)


def _assert_oracle(z):
    expected = _oracle()
    np.testing.assert_array_less(np.abs(z-expected)/np.abs(expected), .01)
    radiation = ETA0*(2*np.pi*FREQS/C0)*HEIGHT/4
    np.testing.assert_array_less(np.abs(z.real/radiation-1), .01)
    assert abs(z[1, 1].imag-z[0, 1].imag) <= 1.0


def _impedance(sim):
    grid = sim._build_realized_grid()
    result = sim.run(n_steps=int(np.ceil(.65e-9/grid.dt)), skip_preflight=True,
                     compute_s_params=True, s_param_freqs=FREQS)
    s = np.asarray(result.s_params).reshape(-1)
    return 50*(1+s)/(1-s)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_fixed_radius_parallel_plate_oracle(lane):
    z = np.stack([_impedance(_plates(dx, lane)) for dx in (.5e-3, .25e-3)])
    _assert_oracle(z)


@pytest.mark.parametrize("failure", ["complex_error", "resistance", "drift"])
def test_oracle_assertion_path_is_live(failure):
    wrong = np.stack([_oracle(), _oracle()])
    if failure == "complex_error":
        wrong += 15j
    elif failure == "resistance":
        wrong += .02*wrong.real
    else:
        wrong[0] -= .7j
        wrong[1] += .7j
    with pytest.raises(AssertionError):
        _assert_oracle(wrong)


@pytest.mark.parametrize("radius", [0, -1e-6, np.nan, np.inf])
def test_bad_radius_is_refused(radius):
    with pytest.raises(ValueError, match="finite positive"):
        _plates(.5e-3, "uniform", radius=radius)


@pytest.mark.parametrize("ratio", [MAX_RADIUS_RATIO+.001, .5, 1.])
def test_radius_requires_extent_and_resolved_pin_above_bound(ratio):
    sim = Simulation(freq_max=30e9, domain=(.006,)*3, dx=.0005)
    with pytest.raises(ValueError, match="extent"):
        sim.add_port((.003,)*3, radius=1e-5)
    with pytest.raises(ValueError, match="resolve the pin geometrically"):
        _impedance(_plates(.5e-3, "uniform", radius=.5e-3*ratio))


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_radius_none_full_field_history_is_bit_identical(lane, monkeypatch):
    """Record every field, every cell, every step on the actual run() path.

    The control bypasses the new H material owner with the old cell-owned
    permeability. The separate worktree-vs-4725b748 measurement is recorded
    in REPORT.md; this test remains portable across JAX backends/dtypes.
    """
    import rfx.simulation as uniform
    import rfx.nonuniform as nu
    captured = []

    def record(state):
        captured.append(np.stack([np.asarray(v) for v in state[:6]]))

    def wrap(step):
        def body(*args):
            result = step(*args)
            jax.debug.callback(record, result[0]["fdtd"], ordered=True)
            return result
        return body

    if lane == "uniform":
        original = uniform.make_core_step
        monkeypatch.setattr(uniform, "make_core_step", lambda ctx: wrap(original(ctx)))
    else:
        original = nu._build_nu_scan

        def build(*args, **kwargs):
            setup = original(*args, **kwargs)
            return setup._replace(step_fn=wrap(setup.step_fn))
        monkeypatch.setattr(nu, "_build_nu_scan", build)

    def solve():
        captured.clear()
        profiles = {} if lane == "uniform" else dict(
            dx_profile=np.full(6, .001),
            dz_profile=np.array([.001, .001, .0009, .0011, .001, .001]))
        sim = Simulation(freq_max=30e9, domain=(.006,)*3, dx=.001,
                         boundary="pec", cpml_layers=0, **profiles)
        sim.add_port((.003, .003, .001), "ez", extent=.003, radius=None,
                     waveform=GaussianPulse(f0=15e9))
        sim.run(n_steps=80, compute_s_params=False, skip_preflight=True)
        jax.effects_barrier()
        return np.stack(captured)

    current = solve()
    monkeypatch.setattr("rfx.core.yee.component_h_materials", lambda m, periodic=(False, False, False): (m.mu_r,)*3)
    update_h.clear_cache()
    legacy = solve()
    assert np.max(np.abs(current)) > 0
    assert current.tobytes() == legacy.tobytes()


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_radius_forward_material_gradient(lane):
    profiles = {} if lane == "uniform" else dict(dx_profile=np.full(12, .5e-3))
    sim = Simulation(freq_max=30e9, domain=(.006, .006, HEIGHT), dx=.5e-3,
                     boundary="pec", cpml_layers=0, **profiles)
    sim.add_port((.003, .003, 0.), "ez", extent=HEIGHT, radius=RADIUS,
                 waveform=GaussianPulse(f0=11e9, bandwidth=.9))
    sim.add_probe((.004, .003, HEIGHT/2), "ez")
    shape = sim._build_realized_grid().shape

    def objective(eps):
        result = sim.forward(eps_override=jnp.ones(shape)*eps, n_steps=160,
                             skip_preflight=True)
        return jnp.log(jnp.mean(result.time_series**2))

    value, ad = jax.jit(jax.value_and_grad(objective))(jnp.array(1.4))
    step = .005
    fd = (objective(1.4+step)-objective(1.4-step))/(2*step)
    print(f"{lane}: objective={float(value):.9g}, AD={float(ad):.9g}, "
          f"FD={float(fd):.9g}, relative_error={float(abs((ad-fd)/fd)):.6g}")
    assert np.isfinite(float(value)) and abs(float(fd)) > 1e-3
    assert float(ad) == pytest.approx(float(fd), rel=1e-3)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_shared_h_owner_all_axes_and_x64(axis):
    from rfx.grid import Grid
    with enable_x64():
        grid = Grid(freq_max=30e9, domain=(.008,)*3, dx=.001, cpml_layers=0)
        start = [.004]*3
        end = start.copy()
        end[axis] += .002
        background = init_materials(grid.shape)._replace(
            mu_r=jnp.ones(grid.shape, dtype=jnp.float64))
        mats = setup_wire_port(grid, WirePort(tuple(start), tuple(end),
                               ("ex", "ey", "ez")[axis], radius=.0001),
                               background)
        mu = component_h_materials(mats)
        assert all(m.dtype == jnp.float64 for m in mu)
        assert np.array_equal(mu[axis], mats.mu_r)
        assert sum(np.count_nonzero(np.asarray(v-mats.mu_r)) for v in mu) == 8
        state = init_state(grid.shape, field_dtype=jnp.float64)
        impulse = jnp.zeros(grid.shape, dtype=jnp.float64).at[4, 4, 4].set(1)
        state = state._replace(**{("ex", "ey", "ez")[axis]: impulse})
        uniform = update_h(state, mats, grid.dt, grid.dx)
        inv = jnp.ones(grid.shape[0], dtype=jnp.float64)/grid.dx
        nu = update_h_nu(state, mats, grid.dt, inv, inv, inv)
        for a, b in zip(uniform[3:6], nu[3:6]):
            assert a.dtype == jnp.float64
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-20)
        coeffs = precompute_coeffs(mats, grid.dt, grid.dx)
        assert all(c.dtype == jnp.float64 for c in coeffs.ch)
        fast = update_h_fast(state, coeffs.ch)
        for a, b in zip(uniform[3:6], fast[3:6]):
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-20)
        full = update_e(uniform, mats, grid.dt, grid.dx)
        baked = update_he_fast(state, coeffs)
        for a, b in zip(full[:6], baked[:6]):
            np.testing.assert_allclose(a, b, rtol=3e-7, atol=1e-12)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_radius_refuses_dispersive_board_before_step(lane, kind, monkeypatch):
    sim = _plates(.5e-3, lane)
    poles = (dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-11)]) if kind == "debye"
             else dict(lorentz_poles=[LorentzPole(omega_0=2*np.pi*20e9,
                                                 delta=1e9, kappa=(2*np.pi*20e9)**2)]))
    sim.add_material("dispersive", eps_r=2., **poles)
    sim.add(Box((.001, .001, .0005), (.002, .002, .001)), material="dispersive")
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped before refusal"))
    with pytest.raises(NotImplementedError, match="radius"):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("entry", ["adi_run", "adi_forward", "subgridded",
                                   "distributed_v2", "distributed_nu", "upml"])
def test_unsupported_radius_path_refuses_before_first_step(entry, monkeypatch):
    from rfx.runners.distributed_v2 import run_distributed
    options = dict(solver="adi") if entry.startswith("adi") else {}
    if entry == "distributed_nu":
        options["dx_profile"] = np.full(12, .0005)
    sim = Simulation(freq_max=30e9, domain=(.006,)*3, dx=.0005,
                     boundary="upml" if entry == "upml" else "pec", **options)
    sim.add_port((.003, .003, .001), "ez", extent=.003, radius=.00005)
    if entry == "subgridded":
        sim.add_refinement(z_range=(.001, .004), ratio=2)
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped before refusal"))
    with pytest.raises((NotImplementedError, ValueError),
                       match="(?i)(radius|soft sources|wire|extended)"):
        if entry == "distributed_v2":
            run_distributed(sim, n_steps=1, devices=jax.devices("cpu")[:2])
        elif entry == "distributed_nu":
            sim.forward(n_steps=1, distributed=True, devices=jax.devices("cpu")[:2],
                        skip_preflight=True)
        elif entry == "adi_forward":
            sim.forward(n_steps=1, skip_preflight=True)
        else:
            sim.run(n_steps=1, skip_preflight=True)


def test_low_level_distributed_nu_refuses_stamped_radius(monkeypatch):
    from rfx.runners.distributed_nu import run_nonuniform_distributed_pec
    mats = init_materials((3,)*3)
    mats = mats._replace(mu_r_wire=(jnp.zeros_like(mats.mu_r), None, None))
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped before refusal"))
    with pytest.raises(NotImplementedError, match="radius.*distributed_nu"):
        run_nonuniform_distributed_pec(None, mats, None, 1, n_devices=2)


def test_radius_material_records_survive_checkpoint(tmp_path):
    from rfx.checkpoint import load_materials, save_materials
    from rfx.core.yee import component_e_materials
    from rfx.grid import Grid
    grid = Grid(freq_max=30e9, domain=(.008,)*3, dx=.001, cpml_layers=0)
    port = WirePort((.004, .004, .002), (.004, .004, .005), "ez", radius=.0001)
    mats = setup_wire_port(grid, port, init_materials(grid.shape))
    save_materials(tmp_path/"radius.h5", mats)
    loaded = load_materials(tmp_path/"radius.h5")
    before = (*component_h_materials(mats), *component_e_materials(mats)[0])
    after = (*component_h_materials(loaded), *component_e_materials(loaded)[0])
    for a, b in zip(before, after):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_declared_radius_roundtrips_without_changing_legacy_records():
    from rfx.artifacts import build_scene_artifact
    from rfx.interop import design_to_dict, simulation_from_design, UnsupportedDesignFeature
    from rfx.interop.emitters import plan_openems_projection
    legacy = _plates(.5e-3, "uniform", radius=None)
    doc = design_to_dict(legacy)
    assert "radius" not in doc["excitations"]["lumped_ports"][0]
    assert "radius" not in build_scene_artifact(legacy)["ports"][0]
    assert design_to_dict(simulation_from_design(doc)) == doc

    sim = _plates(.5e-3, "uniform")
    doc = design_to_dict(sim)
    assert doc["excitations"]["lumped_ports"][0]["radius"] == RADIUS
    restored = simulation_from_design(doc)
    assert restored._ports[0].radius == RADIUS
    assert design_to_dict(restored) == doc
    restored._probes.clear()  # openEMS's independent point-probe refusal is earlier.
    with pytest.raises(UnsupportedDesignFeature, match="radius"):
        plan_openems_projection(design_to_dict(restored))


def test_radius_refuses_local_transverse_grading():
    sim = Simulation(freq_max=30e9, domain=(.006,)*3, dx=.0005,
                     boundary="pec", dx_profile=np.array([.0005]+[.0004]*5+[.0006]*5+[.0005]))
    sim.add_port((.0025, .003, .001), "ez", extent=.003, radius=.00005)
    with pytest.raises(NotImplementedError, match="locally uniform square"):
        sim.run(n_steps=1, skip_preflight=True)


def test_radius_refuses_axial_grading_before_step(monkeypatch):
    d = .25e-3
    sim = Simulation(freq_max=30e9, domain=(.006, .006, HEIGHT), dx=d,
                     boundary="pec", cpml_layers=0,
                     dz_profile=d*np.array([1., .84, 1.16, 1.16, .84, 1.]))
    sim.add_port((.003, .003, 0.), "ez", extent=HEIGHT, radius=RADIUS)
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped before refusal"))
    with pytest.raises(NotImplementedError, match="uniform spacing along the port"):
        sim.run(n_steps=1, skip_preflight=True)


def test_radius_refuses_stamping_in_axial_cpml():
    from rfx.grid import Grid
    grid = Grid(freq_max=30e9, domain=(.006,)*3, dx=.0005, cpml_layers=4)
    port = WirePort((.003, .003, -.0005), (.003, .003, .001), "ez", radius=RADIUS)
    with pytest.raises(ValueError, match="outside axial CPML"):
        setup_wire_port(grid, port, init_materials(grid.shape))
