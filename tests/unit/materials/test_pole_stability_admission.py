"""A Lorentz/Drude medium above its own time-step limit is refused with the way through (tracker 1590).

The pole update is explicit, so the medium has a step limit on top of the grid's. For a Drude pole without
damping it is ``(omega_p dt)^2 + x <= 4 eps_inf`` for every ``0 <= x <= 4 s^2`` (``s`` the Courant fraction);
above it a filled box goes non-finite within a few hundred steps. Expected values here come from that closed
form and from runs, not from the module's predicate.
"""
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.materials.lorentz import LorentzPole, drude_pole, lorentz_pole
from rfx.model.pole_stability import largest_growth, largest_stable_step

N, DX = 12, 1e-3
C0 = 299792458.0


def _box(omega_p_dt, *, eps_inf=1.0, gamma_dt=0.0, graded=False, dt=None, extra=()):
    kwargs = dict(dz_profile=np.full(N, DX)) if graded else {}
    if dt is not None:
        kwargs["dt"] = dt
    sim = Simulation(freq_max=10e9, domain=(N * DX,) * 3, dx=DX, boundary="pec", **kwargs)
    step = _default_step()
    sim.add_material("m", eps_r=eps_inf,
                     lorentz_poles=[drude_pole(omega_p_dt / step, gamma_dt / step), *extra])
    sim.add(Box((0, 0, 0), (N * DX,) * 3), material="m")
    sim.add_source((.005, .006, .006), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.007, .005, .006), "ez")
    return sim


def _default_step():
    return float(Simulation(freq_max=10e9, domain=(N * DX,) * 3, dx=DX, boundary="pec")._build_grid().dt)


def _run(sim, n_steps=400):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(sim.run(n_steps=n_steps, compute_s_params=False, skip_preflight=True).time_series)


S2 = (C0 * _default_step()) ** 2 * 3 / DX ** 2          # Courant fraction squared, 0.99^2
BULK_LIMIT = 2 * np.sqrt(1.0 - S2)                       # omega_p dt, eps_inf = 1


@pytest.mark.parametrize("eps_inf", [1.0, 2.0, 4.0])
def test_growth_factor_matches_the_closed_form_for_an_undamped_drude_pole(eps_inf):
    dt = _default_step()
    limit = 2 * np.sqrt(eps_inf - S2)
    for factor, stable in ((0.5, True), (0.98, True), (1.02, False), (1.5, False)):
        wp = factor * limit / dt
        growth = largest_growth(eps_inf, 0.0, [(0.0, 0.0, wp * wp)], dt, 4 * S2)
        assert (growth <= 1 + 1e-6) == stable, (eps_inf, factor, growth)
        if not stable:
            # the unstable root of eps (z-1)^2 + (x + wp^2 dt^2) z at x = 4 s^2, by hand
            q = (4 * S2 + (factor * limit) ** 2) / eps_inf
            assert growth == pytest.approx((q - 2 + np.sqrt(q * (q - 4))) / 2, rel=1e-6)


@pytest.mark.parametrize("eps_inf", [1.0, 2.0, 4.0])
def test_largest_stable_step_is_the_closed_form(eps_inf):
    dt = _default_step()
    wp = 1.2 * 2 * np.sqrt(eps_inf - S2) / dt
    budget = C0 * C0 * 3 / DX ** 2
    want = 2 * np.sqrt(eps_inf) / np.sqrt(wp * wp + 4 * budget)             # (wp dt)^2 + 4 dt^2 budget = 4, eps_inf = 1
    assert largest_stable_step(eps_inf, 0.0, [(0.0, 0.0, wp * wp)], dt, budget) == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_medium_below_the_limit_runs_and_stays_finite(graded):
    assert np.all(np.isfinite(_run(_box(0.9 * BULK_LIMIT, graded=graded))))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("gamma_dt", [0.0, 0.5])
def test_medium_above_the_limit_is_refused_with_the_stable_step(graded, gamma_dt):
    with pytest.raises(ValueError, match="unstable at this model's time step") as err:
        _run(_box(0.45, graded=graded, gamma_dt=gamma_dt), n_steps=5)
    message = str(err.value)
    assert "largest stable step" in message and "eps_low=" in message and "kappa=" in message
    assert ("pass dt=" in message) == graded            # dt= is named only where it is accepted


def test_forward_refuses_the_same_medium():
    sim = _box(0.45)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.forward(n_steps=5, skip_preflight=True)


def test_the_stable_step_from_the_message_makes_the_graded_model_run():
    """The way through the refusal names: the same medium at the stated step is admitted and stays finite
    over a record on which the default step is non-finite (step 314)."""
    import re
    with pytest.raises(ValueError) as error:
        _run(_box(0.45, graded=True), n_steps=5)
    stable = float(re.search(r"pass dt=([0-9.e+-]+)", str(error.value))[1])
    series = _run(_box(0.45, graded=True, dt=stable), n_steps=1500)
    assert np.all(np.isfinite(series)) and np.max(np.abs(series)) > 0


def test_a_lorentz_pole_and_a_second_pole_are_judged_together():
    dt = _default_step()
    lorentz = lorentz_pole(1.0, 0.5 / dt, 0.01 / dt)
    # measured on this box (rfx-archive records/20261010-pole-stability-limit): bounded at 0.97 of 1.97, non-finite at 1.03
    assert np.all(np.isfinite(_run(_box(0.85 * 1.95, eps_inf=2.0, extra=(lorentz,)))))
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        _run(_box(1.15 * 1.97, eps_inf=2.0, extra=(lorentz,)), n_steps=5)


def test_what_the_refusal_prevents():
    """Without the check the same medium returns non-finite fields: the refusal is not over-cautious here."""
    import rfx.model.pole_stability as module
    original = module.refuse_unstable_cells
    module.refuse_unstable_cells = lambda *args, **kwargs: None
    try:
        import rfx.model.materials  # the hook imports the name at call time
        series = _run(_box(0.45), n_steps=500)
    finally:
        module.refuse_unstable_cells = original
    assert not np.all(np.isfinite(series))


@pytest.mark.parametrize("case", ["two_lorentz", "three_lorentz", "zero_drude"])
def test_stable_clustered_poles_run_for_2000_steps(case):
    sim = _box(0.0, eps_inf=4.0 if case == "zero_drude" else 2.0)
    if case == "two_lorentz":
        poles = [lorentz_pole(de, 2 * np.pi * f, 2 * np.pi * f / 100)
                 for de, f in ((2, 2.5e9), (1, 7.5e9))]
    elif case == "three_lorentz":
        poles = [lorentz_pole(1, 2 * np.pi * f, 2 * np.pi * f / 20)
                 for f in (1e9, 2e9, 3e9)]
    else:
        poles = [drude_pole(0, 0)]
    sim.add_material("m", eps_r=4.0 if case == "zero_drude" else 2.0, lorentz_poles=poles)
    series = _run(sim, n_steps=2000)
    assert np.all(np.isfinite(series)) and np.max(np.abs(series)) > 0
    assert np.max(np.abs(series[-200:])) <= np.max(np.abs(series[:200]))


@pytest.mark.parametrize("count", [3, 4])
def test_identical_damped_lorentz_poles_have_unit_or_lower_growth(count):
    dt = _default_step()
    pole = lorentz_pole(1, 0.05 / dt, 0.0025 / dt)
    assert largest_growth(2, 0, [pole] * count, dt, 4 * S2) <= 1 + 1e-6


def test_identical_drude_poles_have_unit_or_lower_growth():
    dt = _default_step()
    assert largest_growth(4, 0, [drude_pole(0.3 / dt, 0)] * 3, dt, 4 * S2) <= 1 + 1e-6


@pytest.mark.parametrize("w0_dt", [0.01, 0.003, 0.001])
def test_small_undamped_lorentz_pole_has_unit_or_lower_growth(w0_dt):
    dt = _default_step()
    assert largest_growth(2, 0, [lorentz_pole(1, w0_dt / dt, 0)], dt, 4 * S2) <= 1 + 1e-6


@pytest.mark.parametrize("partial", [False, True])
def test_overlapping_drude_media_are_judged_together(partial):
    limit = 2 * np.sqrt(4 - S2)
    sim = _box(0.90 * limit, eps_inf=4)
    sim.add_material("second", eps_r=4, lorentz_poles=[drude_pole(0.85 * limit / _default_step(), 0)])
    sim.add(Box((.003,) * 3, (.009,) * 3) if partial else Box((0,) * 3, (.012,) * 3), material="second")
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        _run(sim, n_steps=5)


def test_plain_block_over_drude_uses_realized_epsilon():
    sim = _box(3.0, eps_inf=4)
    sim.add_material("plain", eps_r=1)
    sim.add(Box((.003,) * 3, (.009,) * 3), material="plain")
    with pytest.raises(ValueError, match="eps_low=1"):
        _run(sim, n_steps=5)


def test_forward_override_uses_realized_epsilon():
    import jax.numpy as jnp
    sim = _box(3.0, eps_inf=4)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        sim.forward(n_steps=5, eps_override=jnp.ones(sim._build_grid().shape), skip_preflight=True)


@pytest.mark.parametrize("parameter", ["kappa", "eps_inf"])
def test_traced_material_is_refused(parameter):
    import jax
    import jax.numpy as jnp
    kappa = (0.45 / _default_step()) ** 2
    def loss(value):
        sim = _box(0.45)
        sim.add_material("m", eps_r=value if parameter == "eps_inf" else 1.,
                         lorentz_poles=[LorentzPole(0., 0., value if parameter == "kappa" else kappa)])
        return jnp.sum(sim.forward(n_steps=5, skip_preflight=True).time_series ** 2)
    with pytest.raises(Exception, match="unstable at this model's time step"):
        jax.block_until_ready(jax.value_and_grad(loss)(1. if parameter == "eps_inf" else kappa))


@pytest.mark.parametrize("factor", [0.97, 1.03])
def test_drude_with_conductivity(factor):
    dt = _default_step()
    sim = _box(factor * 2.036, eps_inf=2)
    sim.add_material("m", eps_r=2, sigma=0.4 * 8.8541878128e-12 / dt,
                     lorentz_poles=[drude_pole(factor * 2.036 / dt, 0)])
    if factor < 1:
        assert np.all(np.isfinite(_run(sim, n_steps=2000)))
    else:
        with pytest.raises(ValueError, match="unstable at this model's time step"):
            _run(sim, n_steps=5)


def test_two_drude_poles_in_one_medium():
    dt = _default_step()
    limit = 2 * np.sqrt(4 - S2)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        _run(_box(.90 * limit, eps_inf=4, extra=[drude_pole(.85 * limit / dt, 0)]), n_steps=5)


@pytest.mark.parametrize("wp_dt", [0.25, 0.45])
def test_two_dimensional_tmz(wp_dt):
    sim = Simulation(freq_max=10e9, domain=(.012, .012, DX), dx=DX, boundary="pec", mode="2d_tmz")
    dt = float(sim._build_grid().dt)
    sim.add_material("m", eps_r=1, lorentz_poles=[drude_pole(wp_dt / dt, 0)])
    sim.add(Box((0, 0, 0), (.012, .012, DX)), material="m")
    sim.add_source((.005, .006, 0), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.007, .005, 0), "ez")
    if wp_dt < .3:
        series = _run(sim, n_steps=3000)
        assert np.all(np.isfinite(series)) and np.max(np.abs(series)) > 0
    else:
        with pytest.raises(ValueError, match="unstable at this model's time step"):
            _run(sim, n_steps=5)


def test_growth_includes_conductivity():
    # At x=0 one Drude pair solves (eps+beta) z^2 +
    # (kappa*dt^2 - 2 eps) z + eps-beta = 0, plus neutral roots.
    roots = np.roots([2.2, 9. - 4., 1.8])
    assert largest_growth(2, .4 * 8.8541878128e-12, [(0, 0, 9)], 1., 0.) == pytest.approx(max(abs(roots)))


def test_growth_just_above_the_bar_is_refused():
    from rfx.core.yee import MaterialArrays
    from rfx.model.pole_stability import refuse_unstable_cells
    # z=-1.005; reciprocal root yields q=2+1.005+1/1.005.
    q = 2 + 1.005 + 1 / 1.005
    cells = MaterialArrays(np.ones((1, 1, 1)), np.zeros((1, 1, 1)), None)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        refuse_unstable_cells(cells, ([(0, 0, q)], None), 1., 0., grid_kind="Grid")


def _coarse_cell_medium(omega_p_dt_of_this_grid):
    """Drude medium on the 1 mm cells of a mesh that also has 0.25 mm cells (which set the time step)."""
    dz = np.concatenate([np.full(8, 1e-3), np.full(8, 0.25e-3)])
    sim = Simulation(freq_max=10e9, domain=(N * DX, N * DX, float(dz.sum())), dx=DX, boundary="pec", dz_profile=dz)
    dt = float(sim._build_realized_grid().dt)
    sim.add_material("m", eps_r=1.0, lorentz_poles=[drude_pole(omega_p_dt_of_this_grid / dt, 0.0)])
    sim.add(Box((0, 0, 0), (N * DX, N * DX, 6e-3)), material="m")
    sim.add_source((.005, .006, .003), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.007, .005, .002), "ez")
    return sim


def test_a_medium_on_the_coarse_cells_of_a_graded_mesh_is_judged_by_its_own_cells():
    """The fine cells elsewhere set the step but do not bound a medium that sits on coarse cells: with the
    whole grid's smallest cell the limit would read 0.28; on its own 1 mm cells it is 1.83 (measured: bounded
    at 1.8, non-finite at 1.9, step 155)."""
    assert np.all(np.isfinite(_run(_coarse_cell_medium(1.5), n_steps=3000)))
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        _run(_coarse_cell_medium(2.05), n_steps=5)


def test_assembly_alone_refuses():
    """The check at the end of model assembly, without any run."""
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _box(0.45).realized_geometry()


def test_the_distributed_drive_model_refuses():
    """The layer function the two-device lanes build their drive materials with."""
    import jax.numpy as jnp
    from rfx.core.yee import MaterialArrays
    from rfx.model.source_coefficients import dispersive_drive_model

    sim = _box(0.25)
    grid = sim._build_grid()
    shape = tuple(int(n) for n in grid.shape)
    materials = MaterialArrays(eps_r=jnp.ones(shape), sigma=jnp.zeros(shape), mu_r=jnp.ones(shape))
    mask = np.ones(shape, dtype=bool)
    dispersive_drive_model(materials, None, ([drude_pole(0.25 / float(grid.dt), 0.0)], [mask]), stability_grid=grid)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        dispersive_drive_model(materials, None, ([drude_pole(0.45 / float(grid.dt), 0.0)], [mask]),
                               stability_grid=grid)


def test_the_unstable_pole_of_two_disjoint_media_is_the_one_named():
    """Two media on different cells, only the second above its limit: the refusal names that pole and its cells."""
    from types import SimpleNamespace
    from rfx.model.pole_stability import CourantBudget, refuse_unstable_cells

    dt = _default_step()
    shape = (6, 6, 6)
    left = np.zeros(shape, dtype=bool)
    left[:2] = True
    right = np.zeros(shape, dtype=bool)
    right[4:] = True
    materials = SimpleNamespace(eps_r=np.ones(shape, np.float32), sigma=np.zeros(shape, np.float32))
    budget = CourantBudget([np.full(6, DX)] * 3)
    stable, unstable = (0.0, 0.0, (0.1 / dt) ** 2), (0.0, 0.0, (0.45 / dt) ** 2)
    refuse_unstable_cells(materials, ([stable, stable], [left, right]), dt, budget, grid_kind="Grid")
    import re
    named = re.escape(f"kappa={unstable[2]:.6g})")
    with pytest.raises(ValueError, match=named + f" on {int(right.sum())} cells"):
        refuse_unstable_cells(materials, ([stable, unstable], [left, right]), dt, budget, grid_kind="Grid")
    with pytest.raises(ValueError, match=named + f" on {int(left.sum())} cells"):
        refuse_unstable_cells(materials, ([unstable, stable], [left, right]), dt, budget, grid_kind="Grid")
