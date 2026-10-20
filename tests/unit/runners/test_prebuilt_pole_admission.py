"""Low-level runners judge the ADE coefficients they actually receive (#1934)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx.core.yee import init_materials
from rfx.materials.lorentz import drude_pole, init_lorentz
from rfx.simulation import run
from tests.unit.materials.test_pole_stability_admission import _box


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("decay", [False, True])
def test_prebuilt_lorentz_is_judged(graded, decay):
    sim = _box(.25, graded=graded)
    grid = sim._build_realized_grid()
    materials = init_materials(grid.shape)
    if graded:
        from rfx.nonuniform import run_nonuniform, run_nonuniform_until_decay
        runner = run_nonuniform_until_decay if decay else run_nonuniform
    elif decay:
        from rfx.stepping.uniform_loop import run_until_decay
        runner = run_until_decay
    else:
        runner = run
    kwargs = dict(max_steps=2, min_steps=1, check_interval=1) if decay else dict(n_steps=2)
    stable = init_lorentz([drude_pole(.25 / grid.dt, 0.)], materials, grid.dt)
    runner(grid, materials, lorentz=stable, **kwargs)
    unstable = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        runner(grid, materials, lorentz=unstable, **kwargs)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
def test_prebuilt_runner_transforms_refuse(transform):
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    coeffs, state = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt)
    def loss(scale):
        scaled = coeffs._replace(c=tuple(v * scale for v in coeffs.c))
        result = run(grid, materials, n_steps=1, lorentz=(scaled, state))
        return jnp.sum(result.state.ez ** 2)
    with pytest.raises(Exception, match="unstable at this model's time step"):
        if transform == "jit":
            result = jax.jit(loss)(1.)
        elif transform == "vmap":
            result = jax.vmap(loss)(jnp.ones(2))
        elif transform == "jvp":
            result = jax.jvp(loss, (1.,), (1.,))
        else:
            result = jax.grad(loss)(1.)
        jax.block_until_ready(result)


def _coarse_edge_scene():
    from rfx import Simulation, Box, GaussianPulse
    dz = np.concatenate([np.full(8, 1e-3), np.full(8, .25e-3)])
    sim = Simulation(freq_max=10e9, domain=(.012, .012, float(dz.sum())),
                     dx=1e-3, boundary="pec", dz_profile=dz)
    dt = float(sim._build_realized_grid().dt)
    sim.add_material("m", eps_r=1., lorentz_poles=[drude_pole(1.5 / dt, 0.)])
    sim.add(Box((0, 0, 0), (.012, .012, .007)), material="m")
    sim.add_source((.005, .006, .003), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.007, .005, .002), "ez")
    return sim


@pytest.mark.parametrize("n_steps", [2, 1000])
def test_assembled_coarse_edge_uses_cell_judge(monkeypatch, n_steps):
    import rfx.model.pole_stability as stability
    def unexpected_prebuilt(*args, **kwargs):
        raise AssertionError("assembled coefficient set was already judged")
    monkeypatch.setattr(stability, "judge_prebuilt_poles", unexpected_prebuilt)
    series = np.asarray(_coarse_edge_scene().run(
        n_steps=n_steps, compute_s_params=False, skip_preflight=True).time_series)
    assert np.all(np.isfinite(series))
    if n_steps == 1000:
        assert np.max(abs(series[500:])) < 2 * np.max(abs(series[:500]))


def test_prebuilt_occupied_cells_do_not_include_edge_extension():
    from rfx.model.pole_stability import judge_prebuilt_poles
    sim = _coarse_edge_scene()
    grid = sim._build_realized_grid()
    materials = init_materials(grid.shape)
    mask = np.zeros(grid.shape, bool)
    mask[:, :, :7] = True
    poles = init_lorentz([drude_pole(1.5 / grid.dt, 0.)], materials, grid.dt, mask=mask)
    assert not getattr(poles[0], "pole_sets_judged", False)
    judge_prebuilt_poles(grid, poles)


@pytest.mark.parametrize("n,levels", [(16, 50), (20, None)])
def test_prebuilt_cost_reduces_per_pole_set(monkeypatch, n, levels):
    from rfx.grid import Grid
    import rfx.model.pole_stability as stability
    grid = Grid(freq_max=10e9, domain=(n * 1e-3,) * 3, dx=1e-3, cpml_layers=0)
    rng = np.random.default_rng(0)
    eps = (1 + 3 * rng.random(grid.shape) if levels is None else
           1 + rng.integers(0, levels, grid.shape) * (3. / (levels - 1)))
    materials = init_materials(grid.shape)._replace(eps_r=jnp.asarray(eps, jnp.float32))
    poles = init_lorentz([drude_pole(.2 / grid.dt, 1e10)], materials, grid.dt)
    calls = []
    original = stability.largest_growth
    def measured(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(stability, "largest_growth", measured)
    stability.judge_prebuilt_poles(grid, poles)
    assert len(calls) == 2  # One pole set, two conductivity extremes.


def test_prebuilt_binary_masks_recover_cell_patterns(monkeypatch):
    from rfx.grid import Grid
    import rfx.model.pole_stability as stability
    grid = Grid(freq_max=10e9, domain=(.004,) * 3, dx=.001, cpml_layers=0)
    rng = np.random.default_rng(42)
    mask = rng.random(grid.shape) > .5
    materials = init_materials(grid.shape)
    lorentz = init_lorentz([drude_pole(.123 / grid.dt, 0.)], materials, grid.dt, mask=mask)
    recorded = []
    original = stability.judge_pole_sets
    def measured(records, *args, **kwargs):
        recorded.extend(records)
        return original(records, *args, **kwargs)
    monkeypatch.setattr(stability, "judge_pole_sets", measured)
    stability.judge_prebuilt_poles(grid, lorentz)
    assert len(recorded) == 1
    assert recorded[0].count == np.count_nonzero(mask)
    for axis, actual in enumerate(recorded[0].occupied):
        expected = np.any(mask, axis=tuple(a for a in range(3) if a != axis))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
def test_assembled_admission_mark_survives_transforms(transform):
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    def coefficients(eps):
        cells = materials._replace(eps_r=materials.eps_r * eps)
        return init_lorentz([drude_pole(.2 / grid.dt, 0.)], cells, grid.dt, grid=grid)[0]
    if transform == "jit":
        result = jax.jit(coefficients)(1.)
    elif transform == "vmap":
        result = jax.vmap(coefficients)(jnp.ones(1))
    elif transform == "jvp":
        result = jax.jvp(coefficients, (1.,), (1.,))[0]
    else:
        def loss(eps):
            coeffs = coefficients(eps)
            assert coeffs.pole_sets_judged
            return jnp.sum(coeffs.ca[0])
        assert np.isfinite(jax.grad(loss)(1.))
        return
    jax.block_until_ready(result)
    assert result.pole_sets_judged
    assert result._fields == ("ca", "cb", "a", "b", "c", "cc")


@pytest.mark.parametrize("missing", ["cells", "dt"])
def test_prebuilt_missing_metrics_refuses_by_name(missing):
    from types import SimpleNamespace
    from rfx.model.pole_stability import judge_prebuilt_poles
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    lorentz = init_lorentz([drude_pole(.2 / grid.dt, 0.)], materials, grid.dt)
    incomplete = SimpleNamespace(cells=grid.cells, dt=grid.dt)
    delattr(incomplete, missing)
    with pytest.raises(ValueError, match="require cell sizes and dt for pole stability admission"):
        judge_prebuilt_poles(incomplete, lorentz)
