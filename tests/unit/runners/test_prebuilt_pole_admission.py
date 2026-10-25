"""Low-level runners judge the ADE coefficients they actually receive (#1934)."""

from rfx.geometry.rasterize_grid import unpadded_high_walls
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx.core.yee import init_materials
from rfx.materials.lorentz import LorentzCoeffs, drude_pole, init_lorentz, mark_judged, pole_sets_judged
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
    stable = init_lorentz([drude_pole(.25 / grid.dt, 0.)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    runner(grid, materials, lorentz=stable, **kwargs)
    unstable = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        runner(grid, materials, lorentz=unstable, **kwargs)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
def test_prebuilt_runner_transforms_refuse(transform):
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    coeffs, state = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
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
    poles = init_lorentz([drude_pole(1.5 / grid.dt, 0.)], materials, grid.dt, mask=mask, high_wall=unpadded_high_walls(grid))
    assert not pole_sets_judged(poles[0])
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
    poles = init_lorentz([drude_pole(.2 / grid.dt, 1e10)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    calls = []
    original = stability.largest_growth
    def measured(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(stability, "largest_growth", measured)
    stability.judge_prebuilt_poles(grid, poles)
    assert len(calls) == 2  # One pole set, two conductivity extremes.


def test_prebuilt_binary_masks_recover_cell_patterns(monkeypatch):
    """The inverse recovers the REALIZED cell pattern, not the stored one.

    On an axis with an unpadded high wall the last stored row lies beyond the
    high-wall node. No edge of the realized model reads it: the wall edges take
    the last inside row, as the low-wall mirror replicates row 0 (#1966). The
    stored row entered the model only through the earlier per-writer repair, so
    the expected pattern replicates the last inside row on every flagged axis.
    """
    from rfx.grid import Grid
    import rfx.model.pole_stability as stability
    grid = Grid(freq_max=10e9, domain=(.004,) * 3, dx=.001, cpml_layers=0)
    rng = np.random.default_rng(42)
    mask = rng.random(grid.shape) > .5
    high_wall = unpadded_high_walls(grid)
    assert high_wall == (True, True, True)
    materials = init_materials(grid.shape)
    lorentz = init_lorentz([drude_pole(.123 / grid.dt, 0.)], materials, grid.dt, mask=mask, high_wall=high_wall)
    realized = mask.copy()
    for axis, flagged in enumerate(high_wall):
        if flagged:
            inside = np.take(realized, [-2], axis=axis)
            index = [slice(None)] * 3
            index[axis] = slice(-1, None)
            realized[tuple(index)] = inside
    mask = realized
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
def test_assembled_admission_mark_requires_original_arrays_through_transforms(transform):
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    def coefficients(eps):
        cells = materials._replace(eps_r=materials.eps_r * eps)
        coeffs = init_lorentz([drude_pole(.2 / grid.dt, 0.)], cells, grid.dt, grid=grid)[0]
        assert pole_sets_judged(coeffs)
        return coeffs
    if transform == "jit":
        result = jax.jit(coefficients)(1.)
    elif transform == "vmap":
        result = jax.vmap(coefficients)(jnp.ones(1))
    elif transform == "jvp":
        result = jax.jvp(coefficients, (1.,), (1.,))[0]
    else:
        def loss(eps):
            coeffs = coefficients(eps)
            assert pole_sets_judged(coeffs)
            return jnp.sum(coeffs.ca[0])
        assert np.isfinite(jax.grad(loss)(1.))
        return
    jax.block_until_ready(result)
    assert not pole_sets_judged(result)  # returned arrays replace the assembly-time tracers
    assert result._fields == ("ca", "cb", "a", "b", "c", "cc")


@pytest.mark.parametrize("missing", ["cells", "dt"])
def test_prebuilt_missing_metrics_refuses_by_name(missing):
    from types import SimpleNamespace
    from rfx.model.pole_stability import judge_prebuilt_poles
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    lorentz = init_lorentz([drude_pole(.2 / grid.dt, 0.)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    incomplete = SimpleNamespace(cells=grid.cells, dt=grid.dt)
    delattr(incomplete, missing)
    with pytest.raises(ValueError, match="require cell sizes and dt for pole stability admission"):
        judge_prebuilt_poles(incomplete, lorentz)


def _periodic_prebuilt(periodic, wp):
    from rfx.grid import Grid
    grid = Grid(freq_max=10e9, domain=(.012,) * 3, dx=.001, cpml_layers=0)
    eps = np.ones(grid.shape)
    eps[:, :, 6:] = 3.
    materials = init_materials(grid.shape)._replace(eps_r=jnp.asarray(eps, jnp.float32))
    mask = np.zeros(grid.shape, bool)
    mask[:3, 2:9, 2:9] = mask[-2:, 2:9, 2:9] = True
    lorentz = init_lorentz([drude_pole(wp / grid.dt, 0.)], materials, grid.dt,
                           mask=[jnp.asarray(mask)], periodic=periodic, high_wall=unpadded_high_walls(grid))
    return grid, materials, lorentz, mask


@pytest.mark.parametrize("periodic", [(False,) * 3, (True, False, False), (True,) * 3],
                         ids=["none", "x", "all"])
def test_periodic_prebuilt_recovers_cells_and_refuses_unstable(periodic, monkeypatch):
    import rfx.model.pole_stability as stability
    grid, materials, lorentz, mask = _periodic_prebuilt(periodic, .45)
    recorded = []
    original = stability.judge_pole_sets
    def observe(records, *args, **kwargs):
        recorded.extend(records)
        return original(records, *args, **kwargs)
    monkeypatch.setattr(stability, "judge_pole_sets", observe)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        run(grid, materials, 1, lorentz=lorentz, periodic=periodic)
    assert len(recorded) == 1
    assert recorded[0].count == np.count_nonzero(mask) == 245
    assert recorded[0].eps_low == pytest.approx(1., rel=1e-5)


@pytest.mark.parametrize("periodic", [(False,) * 3, (True, False, False), (True,) * 3],
                         ids=["none", "x", "all"])
def test_periodic_prebuilt_stable_medium_is_admitted(periodic):
    grid, materials, lorentz, _ = _periodic_prebuilt(periodic, .25)
    result = run(grid, materials, 1, lorentz=lorentz, periodic=periodic)
    assert np.all(np.isfinite(result.state.ez))


@pytest.mark.parametrize("periodic", [(True, False, False), (True,) * 3], ids=["x", "all"])
def test_even_periodic_prebuilt_stable_medium_refuses_by_name(periodic):
    from rfx.grid import Grid
    grid = Grid(freq_max=10e9, domain=(.011,) * 3, dx=.001, cpml_layers=0)
    assert grid.shape == (12,) * 3
    eps = np.ones(grid.shape)
    eps[:, :, 6:] = 3.
    materials = init_materials(grid.shape)._replace(eps_r=jnp.asarray(eps, jnp.float32))
    mask = np.zeros(grid.shape, bool)
    mask[:3, 2:9, 2:9] = mask[-2:, 2:9, 2:9] = True
    lorentz = init_lorentz([drude_pole(.25 / grid.dt, 0.)], materials, grid.dt,
                           mask=[jnp.asarray(mask)], periodic=periodic, high_wall=unpadded_high_walls(grid))
    message = (r"cannot judge prebuilt Lorentz coefficients: periodic edge mean on even axis [012] "
               r"does not determine unique cells")
    with pytest.raises(ValueError, match=message):
        run(grid, materials, 1, lorentz=lorentz, periodic=periodic)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
def test_periodic_prebuilt_transforms_refuse_unstable(transform):
    periodic = (True,) * 3
    grid, materials, (coeffs, state), _ = _periodic_prebuilt(periodic, .45)
    def loss(scale):
        scaled = coeffs._replace(c=tuple(v * scale for v in coeffs.c))
        return jnp.sum(run(grid, materials, 1, lorentz=(scaled, state), periodic=periodic).state.ez ** 2)
    with pytest.raises(Exception, match="unstable at this model's time step"):
        if transform == "jit":
            result = jax.jit(loss)(1.)
        elif transform == "vmap":
            result = jax.vmap(loss)(jnp.ones(1))
        elif transform == "jvp":
            result = jax.jvp(loss, (1.,), (1.,))
        else:
            result = jax.grad(loss)(1.)
        jax.block_until_ready(result)


@pytest.mark.parametrize("graded", [False, True])
def test_prebuilt_setup_passes_run_periodic_axes(graded, monkeypatch):
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    import rfx.model.pole_stability as stability
    sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.001,
                     boundary=BoundarySpec(x="periodic", y="pec", z="pec"),
                     **(dict(dz_profile=np.full(12, .001)) if graded else {}))
    grid = sim._build_realized_grid()
    materials = init_materials(grid.shape)
    periodic = (True, False, False)
    lorentz = init_lorentz([drude_pole(.2 / grid.dt, 0.)], materials, grid.dt, periodic=periodic, high_wall=unpadded_high_walls(grid))
    seen = []
    def observe(*args, **kwargs):
        seen.append(kwargs["periodic"])
    monkeypatch.setattr(stability, "judge_prebuilt_poles", observe)
    if graded:
        from rfx.nonuniform import run_nonuniform
        run_nonuniform(grid, materials, 1, lorentz=lorentz)
    else:
        run(grid, materials, 1, lorentz=lorentz, periodic=periodic)
    assert seen == [periodic]


@pytest.mark.parametrize("graded", [False, True])
def test_rescaled_assembled_coefficients_are_judged_again(graded):
    grid = _box(.25, graded=graded)._build_realized_grid()
    materials = init_materials(grid.shape)
    coeffs, state = init_lorentz([drude_pole(.2 / grid.dt, 0.)], materials, grid.dt, grid=grid)
    assert pole_sets_judged(coeffs)
    scaled = coeffs._replace(c=tuple(v * (.45 / .2) ** 2 for v in coeffs.c))
    assert not pole_sets_judged(scaled)
    if graded:
        from rfx.nonuniform import run_nonuniform as runner
    else:
        runner = run
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        runner(grid, materials, 1, lorentz=(scaled, state))


@pytest.mark.parametrize("field", ["ca", "cb", "a", "b", "c", "cc"])
def test_assembly_mark_is_tied_to_every_array_identity(field):
    grid = _box(.25)._build_realized_grid()
    coeffs, _ = init_lorentz([drude_pole(.2 / grid.dt, 0.)], init_materials(grid.shape), grid.dt, grid=grid)
    assert pole_sets_judged(coeffs._replace())
    rebuilt = jax.tree.map(lambda a: a, coeffs)
    assert pole_sets_judged(rebuilt)  # same array leaves are the judged values
    assert jax.tree.structure(coeffs) == jax.tree.structure(rebuilt)
    assert pole_sets_judged(LorentzCoeffs(*coeffs))
    copied = jax.tree.map(lambda a: a.copy(), getattr(coeffs, field))
    assert not pole_sets_judged(coeffs._replace(**{field: copied}))


def test_marked_lorentz_coefficients_keep_plain_tree_structure():
    # The expected node is a plain LorentzCoeffs constructed independently of
    # marking/assembly helpers, with the original six field positions.
    fields = [jnp.zeros(2) for _ in range(6)]
    expected = jax.tree.structure(LorentzCoeffs(*fields))
    coeffs = LorentzCoeffs(*[jnp.ones(2) for _ in range(6)])
    assert not pole_sets_judged(coeffs)
    marked = mark_judged(coeffs)
    assert jax.tree.structure(marked) == expected
    assert marked is coeffs
    assert type(marked) is LorentzCoeffs
    assert pole_sets_judged(marked)


@pytest.mark.parametrize("bad", ["zero_eps", "nan_eps", "negative_c"])
def test_invalid_prebuilt_cells_refuse_by_name(bad):
    import rfx.model.pole_stability as stability
    grid, _, (coeffs, state), _ = _periodic_prebuilt((False,) * 3, .25)
    if bad == "negative_c":
        coeffs = coeffs._replace(c=tuple(-v for v in coeffs.c))
    else:
        value = -1. if bad == "zero_eps" else jnp.nan
        coeffs = coeffs._replace(ca=tuple(jnp.full_like(v, value) for v in coeffs.ca))
    with pytest.raises(ValueError, match="cannot judge prebuilt Lorentz coefficients:"):
        stability.judge_prebuilt_poles(grid, (coeffs, state))


def test_inconsistent_periodic_edges_refuse_by_name():
    from rfx.grid import Grid
    from rfx.model.pole_stability import judge_prebuilt_poles
    grid = Grid(freq_max=10e9, domain=(.003,) * 3, dx=.001, cpml_layers=0)
    materials = init_materials(grid.shape)
    periodic = (True,) * 3
    coeffs, state = init_lorentz([drude_pole(.2 / grid.dt, 0.)], materials, grid.dt, periodic=periodic, high_wall=unpadded_high_walls(grid))
    factor = jnp.where(jnp.indices(grid.shape)[1] % 2, 2., 1.)
    coeffs = coeffs._replace(cc=(coeffs.cc[0] / factor, *coeffs.cc[1:]))
    with pytest.raises(ValueError, match="cannot judge prebuilt Lorentz coefficients: recovered cells do not reproduce"):
        judge_prebuilt_poles(grid, (coeffs, state), periodic=periodic)
