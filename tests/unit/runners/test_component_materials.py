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
                           dz=jnp.array([1., 1.1, .9, 1.2])) if graded else None
    d = DebyePole(1.2, 2e-11)
    lorentz_pole = LorentzPole(2e10, 1e9, 3e20)
    specs = model.ComponentCells(cells, ([d], [a > 20]), ([lorentz_pole], [a < 30]))
    result = model.realize_components(specs, grid, periodic=periodic)
    assert_tree_equal((result.eps_update, result.sigma_update),
                      yee.component_e_materials(cells, periodic))
    widths = (grid.dx_arr, grid.dy_arr, grid.dz) if graded else None
    assert_tree_equal(result.mu_update, yee.component_h_materials(
        cells, periodic, cell_sizes=widths))
    assert_tree_equal(result.eps, yee.edge_mean_components(cells.eps_r - stamp, periodic))
    assert_tree_equal(result.sigma, yee.edge_mean_components(cells.sigma - 2 * stamp, periodic))
    assert_tree_equal((result.upml_eps, result.upml_sigma), yee.cell_owned_component_materials(cells))
    assert_tree_equal(result.debye[0].weights, yee.edge_mean_components((a > 20).astype(jnp.float32), periodic))
    assert_tree_equal(result.lorentz[0].weights, yee.edge_mean_components((a < 30).astype(jnp.float32), periodic))
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
    assert len(count[0].debye) == len(count[0].lorentz) == 1
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
