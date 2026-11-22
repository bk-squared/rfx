"""A concrete f0 warning must preserve the sheet under caller jit."""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.errors import AccuracyWarning
from tests.unit.autodiff.test_pinned_sheet_on_traced_mesh import (
    D_MIN, DT_PIN, DX, K_MID, N_X, N_Y, N_Z, Z_PATCH, _prof_x,
)


DELTA0 = 0.5 * DX


def _series(widths, sheet=True):
    sim = Simulation(
        freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
        dx=DX, dx_profile=widths, cpml_layers=0, boundary="pec",
        nominal_profiles=({"x": _prof_x(DELTA0)}
                          if isinstance(widths, jax.core.Tracer) else {}),
        dt=float(DT_PIN), dt_min_cell=float(D_MIN), snap="declared",
    )
    sim.add_material("diel", eps_r=4.0)
    sim.add_source((3 * DX, 8 * DX, K_MID * DX), "ez", amplitude_kind="current")
    sim.add_probe((20 * DX, 8 * DX, K_MID * DX), "ez")
    if sheet:
        sim.add_thin_conductor(
            Box((8 * DX, 5 * DX, Z_PATCH), (17.1 * DX, 11 * DX, Z_PATCH)),
            surface_impedance_f0=10e9,
        )
    sim.add(Cylinder((18 * DX, 8 * DX, 16 * DX), 1.6 * DX, 2 * DX), material="diel")
    sim.add(Sphere((5 * DX, 8 * DX, 16 * DX), 1.6 * DX), material="pec")
    return sim.forward(n_steps=40).time_series


def _peak_bar(series):
    return 9 * float(np.spacing(np.float32(np.max(np.abs(series)))))


def _rs_text(caught):
    return [re.search(r"Rs0 = .*? ohm/sq", str(w.message)).group(0)
            for w in caught if isinstance(w.message, AccuracyWarning)
            and "Leontovich surface-" in str(w.message)]


def test_concrete_caller_jit_series_and_warning():
    """Row (i): nine float32 ULP at the eager series peak, identical Rs0 text."""
    def g(amplitude):
        return amplitude * _series(_prof_x(DELTA0))

    with warnings.catch_warnings(record=True) as eager_warnings:
        warnings.simplefilter("always")
        eager = np.asarray(g(1.0))
    with warnings.catch_warnings(record=True) as jit_warnings:
        warnings.simplefilter("always")
        compiled = np.asarray(jax.jit(g)(1.0))
    error, bar = float(np.max(np.abs(compiled - eager))), _peak_bar(eager)
    print(f"ROW i: max_abs_error={error!r}, bar={bar!r}, peak={np.max(np.abs(eager))!r}")
    assert np.isfinite(eager).all() and np.isfinite(compiled).all()
    assert error <= bar
    eager_rs, jit_rs = _rs_text(eager_warnings), _rs_text(jit_warnings)
    print(f"ROW i warning: eager={eager_rs!r}, jit={jit_rs!r}")
    assert len(eager_rs) == 1
    assert jit_rs == eager_rs


def test_traced_widths_caller_jit_loss_and_gradient():
    """Row (ii): summed loss and its width gradient have relative bar 1e-4."""
    def loss(widths):
        return jnp.sum(_series(widths) ** 2)

    widths = jnp.asarray(_prof_x(DELTA0), dtype=jnp.float32)
    eager_loss, eager_grad = jax.value_and_grad(loss)(widths)
    jit_loss, jit_grad = jax.jit(jax.value_and_grad(loss))(widths)
    for name, actual, expected in (
        ("loss", jit_loss, eager_loss), ("gradient", jit_grad, eager_grad),
    ):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        bar = 1e-4 * np.abs(expected.astype(np.float64))
        print(f"ROW ii {name}: eager={expected.tolist()!r}, jit={actual.tolist()!r}, "
              f"abs_error={error.tolist()!r}, bar={bar.tolist()!r}")
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        assert np.all(error <= bar)


def test_eager_sheet_anchor():
    """A missing sheet changes the eager series by more than 100 row (i) bars."""
    eager = np.asarray(_series(_prof_x(DELTA0)))
    absent = np.asarray(_series(_prof_x(DELTA0), sheet=False))
    difference = float(np.max(np.abs(eager - absent)))
    bar = 100 * _peak_bar(eager)
    print(f"ANCHOR: max_abs_difference={difference!r}, bar={bar!r}")
    assert np.isfinite(eager).all() and np.isfinite(absent).all()
    assert difference > bar
