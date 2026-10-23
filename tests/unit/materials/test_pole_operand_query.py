"""The second operand-query caller preserves cell admission and owns its policy."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays
from rfx.model.operands import OperandKind, OperandRecord, realized_operands
from rfx.model.pole_stability import (
    OPERAND_RULES, POLE_TIME_STEP_REASONS, pole_set_extremes, refuse_unstable_cells,
)


def operands(*, kappa=3., mu=1., mask=None):
    cells = MaterialArrays(np.ones((4, 2, 2)), np.zeros((4, 2, 2)),
                           np.full((4, 2, 2), mu))
    return realized_operands(None, cells, {
        "lorentz_spec": ([(0., 0., kappa)], [np.ones(cells.eps_r.shape, bool) if mask is None else mask])}, ())


def test_lorentz_drude_record_is_read():
    records = operands(kappa=9.)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        refuse_unstable_cells(records, dt=1., budget=0., grid_kind="Grid")


def test_magnetic_pole_cell_is_admitted():
    records = operands(mu=4.)
    refuse_unstable_cells(records, dt=1., budget=0., grid_kind="Grid")
    assert pole_set_extremes(records)[0].poles == ((0., 0., 3.),)


@pytest.mark.parametrize("kind", list(OperandKind))
def test_policy_has_a_reason_and_admits_existing_terms(kind):
    assert OPERAND_RULES["pole_time_step"][kind] == "accept"
    assert POLE_TIME_STEP_REASONS[kind]
    records = operands()
    if kind in {OperandKind.EPS, OperandKind.SIGMA, OperandKind.DEBYE,
                OperandKind.LORENTZ, OperandKind.DRUDE, OperandKind.CHI3}:
        return
    extra = OperandRecord("witness", frozenset({kind}), np.ones((4, 2, 2)),
                          0, kind.name, None, None, 0, None)
    refuse_unstable_cells((*records, extra), dt=1., budget=0., grid_kind="Grid")


def test_kerr_beside_pole_medium_assembles():
    """A Kerr box next to a Lorentz box was admitted before this check read operands; it still is."""
    from rfx import Box, GaussianPulse, Simulation
    from rfx.materials.lorentz import drude_pole
    sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=1e-3, boundary="pec")
    dt = float(sim._build_realized_grid().dt)
    sim.add_material("pole", eps_r=1., lorentz_poles=[drude_pole(.25 / dt, 0.)])
    sim.add_material("kerr", eps_r=2., chi3=1e-18)
    sim.add(Box((0, 0, 0), (.003, .008, .008)), material="pole")
    sim.add(Box((.005, 0, 0), (.008, .008, .008)), material="kerr")
    sim.add_source((.004, .004, .004), "ez", waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.004, .005, .004), "ez")
    series = np.asarray(sim.run(n_steps=4, compute_s_params=False, skip_preflight=True).time_series)
    assert np.all(np.isfinite(series))


def test_query_reads_record_mask_instead_of_spec_mask():
    records = operands()
    records = tuple(replace(r, array=np.zeros_like(r.array)) if r.pole_spec is not None else r
                    for r in records)
    assert pole_set_extremes(records) == ()


@pytest.mark.parametrize("traced", [False, True])
def test_query_owns_cells_excluding_halos_with_broadcast_mask(traced):
    records = operands(mask=np.ones((1, 2, 1), bool))
    records = tuple(replace(r, array=np.full((4, 2, 2), 2.).astype(float)
                           if r.name == "materials.eps_r" else r.array) for r in records)
    eps = records[0].array.copy()
    eps[[0, 3]] = .01
    records = (replace(records[0], array=eps), *records[1:])
    view = SimpleNamespace(halo_lo=1, x0=3, x1=5, nx=8)
    def reduce(mask):
        selected = tuple(replace(r, array=mask) if r.pole_spec is not None else r for r in records)
        (result,) = pole_set_extremes(selected, view=view)
        return result.eps_low, result.count, result.occupied[0]
    low, count, occupied = (jax.jit(reduce) if traced else reduce)(jnp.ones((1, 2, 1), bool))
    assert low == 2.
    assert count == 8
    np.testing.assert_array_equal(occupied, [False, False, False, True, True, False, False, False])


def test_prebuilt_pole_record_is_not_mistaken_for_a_spec():
    records = tuple(replace(r, pole_spec=None) for r in operands())
    assert pole_set_extremes(records) == ()


@pytest.mark.parametrize("masks", [None, [None] * 9])
def test_absent_masks_stay_static_under_jit_above_pattern_cap(masks):
    @jax.jit
    def checked(eps):
        cells = MaterialArrays(jnp.full((2, 2, 2), eps), jnp.zeros((2, 2, 2)), None)
        records = realized_operands(None, cells, {
            "lorentz_spec": ([(0., 0., 1.)] * 9, masks)}, ())
        (result,) = pole_set_extremes(records)
        return result.eps_low, result.count
    low, count = checked(2.)
    assert low == 2.
    assert count == 8
