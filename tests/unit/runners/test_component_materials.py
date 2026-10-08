"""Realization operands, shared consumption, and one build per main-path run."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core import yee
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz
from rfx.model import materials as model


def assert_tree_equal(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("periodic", [(False, False, False), (True, False, True)])
@pytest.mark.parametrize("graded", [False, True])
def test_realized_volume_stamps_poles_and_metric(periodic, graded):
    shape = (4, 3, 5)
    a = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    stamp = jnp.zeros(shape).at[1, 1, 2].set(3.)
    cells = yee.MaterialArrays(1 + a / 17 + stamp, a / 71 + 2 * stamp,
                              1 + a / 13,
                              sigma_lumped=(None, 2 * stamp, None),
                              eps_r_lumped=(stamp, None, None),
                              mu_r_wire=(None, None, stamp))
    grid = SimpleNamespace(dx_arr=jnp.array([.9, 1.1, .8]),
                           dy_arr=jnp.array([.7, 1.3]),
                           dz=jnp.array([1., 1.1, .9, 1.2]), shape=shape, dt=1e-12) if graded else None
    d = DebyePole(1.2, 2e-11)
    lorentz_pole = LorentzPole(2e10, 1e9, 3e20)
    specs = model.ComponentCells(cells, ([d], [a > 20]), ([lorentz_pole], [a < 30]))
    result = model.realize_components(specs, grid, periodic=periodic)
    e_widths = model.electric_cell_sizes(grid)
    assert_tree_equal((result.eps_update, result.sigma_update),
                      yee.component_e_materials(cells, periodic, cell_sizes=e_widths))
    widths = (grid.dx_arr, grid.dy_arr, grid.dz) if graded else None
    assert_tree_equal(result.mu_update, yee.component_h_materials(
        cells, periodic, cell_sizes=widths))
    assert_tree_equal(result.eps, yee.edge_mean_components(cells.eps_r - stamp, periodic, cell_sizes=e_widths))
    assert_tree_equal(result.sigma, yee.edge_mean_components(cells.sigma - 2 * stamp, periodic, cell_sizes=e_widths))
    assert_tree_equal((result.upml_eps, result.upml_sigma), yee.cell_owned_component_materials(cells))
    dw = (yee.edge_mean_components((a > 20).astype(jnp.float32), periodic, cell_sizes=e_widths),)
    lw = (yee.edge_mean_components((a < 30).astype(jnp.float32), periodic, cell_sizes=e_widths),)
    if graded:
        from rfx.materials.debye import debye_pole_coeffs
        from rfx.materials.lorentz import lorentz_pole_coeffs
        assert result.debye.weights is result.lorentz.weights is None
        assert_tree_equal(result.debye.coefficients, debye_pole_coeffs([d], grid.dt, shape, dw))
        assert_tree_equal(result.lorentz.coefficients, lorentz_pole_coeffs([lorentz_pole], grid.dt, shape, lw))
    else:
        assert_tree_equal(result.debye.weights, dw)
        assert_tree_equal(result.lorentz.weights, lw)
    assert result.cells_view.components is None
    with pytest.raises(FrozenInstanceError):
        result.eps = ()


def test_main_kernels_and_ade_read_stored_operands(monkeypatch):
    cells = yee.init_materials((4, 3, 5))
    d = DebyePole(1.2, 2e-11)
    lorentz_pole = LorentzPole(2e10, 1e9, 3e20)
    materials = model.with_components(cells, None, periodic=(False,) * 3,
                                     debye_spec=([d], None), lorentz_spec=([lorentz_pole], None))
    def forbidden(*args, **kwargs):
        raise AssertionError("repeated material realization/averaging")
    monkeypatch.setattr(model, "realize_components", forbidden)
    monkeypatch.setattr(yee, "component_e_materials", forbidden)
    monkeypatch.setattr(yee, "component_h_materials", forbidden)
    jax.clear_caches()
    state = yee.init_state(cells.eps_r.shape)
    state = yee.update_h(state, materials, 1e-12, .001)
    state = yee.update_e(state, materials, 1e-12, .001)
    inv = [jnp.ones(n) * 1000 for n in cells.eps_r.shape]
    state = yee.update_h_nu(state, materials, 1e-12, *inv)
    state = yee.update_e_nu(state, materials, 1e-12, *inv)
    yee.precompute_coeffs(materials, 1e-12, .001)
    init_debye([d], materials, 1e-12)
    init_lorentz([lorentz_pole], materials, 1e-12)
    jax.block_until_ready(state)
    jax.clear_caches()


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_one_realization_for_a_main_path_with_both_poles(monkeypatch, graded, entry):
    count = []
    original = model.realize_components
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        count.append(result)
        return result
    monkeypatch.setattr(model, "realize_components", observe)
    kwargs = {"dx_profile": np.array([.001] * 3 + [.0009, .0011] + [.001] * 3)} if graded else {}
    sim = Simulation(freq_max=8e9, domain=(.008, .007, .009), dx=.001,
                     boundary="cpml", cpml_layers=2, **kwargs)
    sim.add_material("block", eps_r=3., sigma=.03, mu_r=2.,
                     debye_poles=[DebyePole(1.2, 2e-11)],
                     lorentz_poles=[LorentzPole(2e10, 1e9, 3e20)])
    sim.add(Box((.002, .002, .003), (.006, .005, .007)), material="block")
    sim.add_source((.004, .003, .005), "ez", waveform=lambda t: jnp.ones_like(t),
                   amplitude_kind="field")
    sim.add_probe((.004, .004, .005), "ez")
    result = getattr(sim, entry)(n_steps=3, skip_preflight=True,
                                **({"compute_s_params": False} if entry == "run" else {}))
    jax.block_until_ready(result.time_series)
    assert len(count) == 1
    assert len(count[0].debye.poles) == len(count[0].lorentz.poles) == 1
    jax.clear_caches()


@pytest.mark.parametrize("quantity", ["eps_r", "sigma", "mu_r"])
def test_scan_value_and_gradient_match_cell_path_peak_tolerance(quantity):
    shape = (4, 3, 5)
    a = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    cells = yee.MaterialArrays(1 + a / 17, a / 71, 1 + a / 13)
    initial = yee.init_state(shape)._replace(ez=jnp.sin(a))
    def objective(values, realized):
        m = cells._replace(**{quantity: values})
        if realized:
            m = model.with_components(m, None, periodic=(False,) * 3)
        def step(state, _):
            state = yee.update_h(state, m, 1e-12, .001)
            return yee.update_e(state, m, 1e-12, .001), None
        state, _ = jax.lax.scan(step, initial, None, length=4)
        return jnp.sum(state.ex**2 + state.ey**2 + state.ez**2)
    values = getattr(cells, quantity)
    raw = jax.value_and_grad(lambda v: objective(v, False))(values)
    realized = jax.value_and_grad(lambda v: objective(v, True))(values)
    for reference, actual in zip(jax.tree.leaves(raw), jax.tree.leaves(realized)):
        peak = np.max(np.abs(np.asarray(reference)))
        assert np.max(np.abs(np.asarray(actual - reference))) <= 1e-4 * peak
    jax.clear_caches()


def test_graded_forward_design_gradient_matches_finite_difference():
    """AD follows the realized edge rule through a graded public forward call."""
    sim = Simulation(freq_max=8e9, domain=(.008, .007, .009), dx=.001,
                     boundary="cpml", cpml_layers=2,
                     dx_profile=np.array([.001] * 3 + [.0009, .0011] + [.001] * 3))
    sim.add_material("block", eps_r=3., sigma=.03, mu_r=2.)
    sim.add(Box((.002, .002, .003), (.006, .005, .007)), material="block")
    sim.add_source((.004, .003, .005), "ez", waveform=lambda t: jnp.ones_like(t),
                   amplitude_kind="field")
    sim.add_probe((.004, .004, .005), "ez")
    grid = sim._build_nonuniform_grid()
    cells = sim._assemble_materials_nu(grid)[0]
    def objective(multiplier):
        result = sim.forward(eps_override=cells.eps_r * multiplier,
                             n_steps=8, skip_preflight=True)
        return jnp.sum(result.time_series**2)
    objective = jax.jit(objective)
    x = jnp.float32(1.)
    gradient = float(jax.grad(objective)(x))
    for delta in (1e-3, 5e-4):
        finite_difference = float((objective(x + delta) - objective(x - delta)) / (2 * delta))
        relative = abs(gradient - finite_difference) / abs(gradient)
        print(f"delta={delta} AD={gradient:.12g} FD={finite_difference:.12g} relative={relative:.12g}")
        assert relative < 1e-3
    jax.clear_caches()


@pytest.mark.parametrize("quantity", ["eps_r", "sigma", "mu_r"])
@pytest.mark.parametrize("entry", ["step", "debye", "lorentz"])
def test_stale_realization_is_rejected(quantity, entry):
    from dataclasses import MISSING, fields
    from rfx.simulation import _StepContext, core_step_invariants
    cells = yee.init_materials((4, 3, 5))
    materials = model.with_components(cells, None, periodic=(False,) * 3,
        debye_spec=([DebyePole(1.2, 2e-11)], None),
        lorentz_spec=([LorentzPole(2e10, 1e9, 3e20)], None))
    materials = materials._replace(**{quantity: getattr(materials, quantity) * 2})
    with pytest.raises(ValueError, match="stale realized material"):
        if entry == "step":
            kwargs = {f.name: False for f in fields(_StepContext)
                      if f.default is MISSING and f.default_factory is MISSING}
            kwargs.update(grid=None, materials=materials, dt=1e-12, dx=.001,
                          periodic=(False,) * 3, pec_axes="", stencil_order=2)
            core_step_invariants(_StepContext(**kwargs))
        elif entry == "debye":
            init_debye([DebyePole(1.2, 2e-11)], materials, 1e-12)
        else:
            init_lorentz([LorentzPole(2e10, 1e9, 3e20)], materials, 1e-12)


def test_realization_reuse_checks_periodic_and_grid():
    cells = yee.init_materials((4, 3, 5))
    grid = SimpleNamespace(shape=cells.eps_r.shape)
    materials = model.with_components(cells, grid, periodic=(False,) * 3)
    assert model.with_components(materials, grid, periodic=(False,) * 3) is materials
    with pytest.raises(ValueError, match="periodic"):
        model.with_components(materials, grid, periodic=(True, False, False))
    with pytest.raises(ValueError, match="grid"):
        model.with_components(materials, SimpleNamespace(shape=(5, 3, 5)), periodic=(False,) * 3)


def test_step_materials_have_no_cell_or_pole_arrays():
    cells = yee.init_materials((4, 3, 5))
    materials = model.with_components(cells, None, periodic=(False,) * 3)
    slim = model.kernel_materials(materials)
    assert slim.eps_r.shape == ()  # dtype token for CPML, not the cell epsilon
    assert slim.sigma is slim.mu_r is None
    assert slim.components._fields == ("eps_update", "sigma_update", "mu_update")
    assert slim.components.eps_update is materials.components.eps_update
    assert slim.components.mu_update is materials.components.mu_update


def test_raw_cpml_h_does_not_realize_electric_materials(monkeypatch):
    from rfx.boundaries.cpml import apply_cpml_h, init_cpml
    sim = Simulation(freq_max=8e9, domain=(.006, .005, .007), dx=.001,
                     boundary="cpml", cpml_layers=2)
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    params, state = init_cpml(grid)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPML H must not realize E or poles")
    monkeypatch.setattr(model, "realize_components", forbidden)
    monkeypatch.setattr(model, "edge_averaged_materials", forbidden)
    result = apply_cpml_h(yee.init_state(grid.shape), params, state, grid,
                         "xyz", materials=materials)
    jax.block_until_ready(result)
    jax.clear_caches()


@pytest.mark.parametrize("mode", ["2d_tmz", "2d_tez"])
def test_two_dimensional_forward_resolves_realization_periodic_flags(mode):
    sim = Simulation(freq_max=8e9, domain=(.006, .005, .001), dx=.001,
                     boundary="cpml", cpml_layers=2, mode=mode)
    component = "ez" if mode == "2d_tmz" else "ex"
    sim.add_material("block", eps_r=3., debye_poles=[DebyePole(1.2, 2e-11)])
    sim.add(Box((.002, .001, 0), (.004, .004, .001)), material="block")
    sim.add_source((.003, .002, 0), component, waveform=lambda t: jnp.ones_like(t),
                   amplitude_kind="field")
    sim.add_probe((.003, .003, 0), component)
    run = sim.run(n_steps=3, skip_preflight=True, compute_s_params=False)
    forward = sim.forward(n_steps=3, skip_preflight=True)
    np.testing.assert_array_equal(run.time_series, forward.time_series)
    jax.clear_caches()


def test_upml_reuses_explicit_kernel_periodic_flags():
    from rfx.boundaries.upml import init_upml
    sim = Simulation(freq_max=8e9, domain=(.006, .005, .007), dx=.001,
                     boundary="upml", cpml_layers=2)
    grid = sim._build_grid()
    periodic = (True, False, False)
    materials = model.with_components(sim._assemble_materials(grid)[0], grid,
                                     periodic=periodic)
    coeffs = init_upml(grid, materials, periodic=periodic)
    jax.block_until_ready(coeffs)
    jax.clear_caches()


@pytest.mark.parametrize("family", ["debye", "lorentz"])
@pytest.mark.parametrize("mismatch", ["dt", "pole"])
def test_cached_pole_coefficients_reject_stale_parameters(family, mismatch):
    cells = yee.init_materials((4, 3, 5))
    grid = SimpleNamespace(shape=cells.eps_r.shape, dt=1e-12)
    pole = DebyePole(1.2, 2e-11) if family == "debye" else LorentzPole(2e10, 1e9, 3e20)
    materials = model.with_components(cells, grid, periodic=(False,) * 3,
                                     **{family + "_spec": ([pole], None)})
    init = init_debye if family == "debye" else init_lorentz
    init([type(pole)(*pole)], materials, grid.dt)  # equal scalar parameters are reusable
    if mismatch == "pole":
        pole = pole._replace(**{pole._fields[0]: pole[0] * 2})
    with pytest.raises(ValueError, match="realized ADE coefficients"):
        init([pole], materials, grid.dt * (2 if mismatch == "dt" else 1))
    jax.clear_caches()
