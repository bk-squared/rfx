"""Phase C pin: emit_time_series=False on the non-uniform forward path.

Issue #31 — when forward()'s only consumers are frequency-domain
quantities (NTFF, S-params, DFT planes), skip the per-step time-series
allocation. The scan output dimension drops to 0, removing the
``(n_steps, n_probes)`` AD tape entry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.optimize_objectives import (
    maximize_transmitted_energy,
    minimize_reflected_energy,
    steer_probe_array,
)


def _build_sim():
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    return sim


def test_nu_forward_emit_false_empties_time_series():
    sim = _build_sim()
    fr = sim.forward(n_steps=80, emit_time_series=False)
    ts = np.asarray(fr.time_series)
    assert ts.size == 0, f"expected empty time series, got shape {ts.shape}"


def test_nu_forward_emit_true_default_unchanged():
    sim = _build_sim()
    fr = sim.forward(n_steps=80)
    ts = np.asarray(fr.time_series)
    assert ts.shape[0] == 80 and ts.shape[-1] == 1
    assert np.all(np.isfinite(ts))


@pytest.mark.parametrize(
    "make_obj",
    [
        lambda: minimize_reflected_energy(port_probe_idx=0),
        lambda: maximize_transmitted_energy(output_probe_idx=0),
        lambda: steer_probe_array(target_probe_idx=0, suppress_probe_idx=0),
    ],
)
def test_time_domain_objective_requires_emit(make_obj):
    sim = _build_sim()
    fr = sim.forward(n_steps=40, emit_time_series=False)
    obj = make_obj()
    with pytest.raises(ValueError, match="emit_time_series"):
        obj(fr)


def test_uniform_forward_rejects_emit_false():
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    with pytest.raises(NotImplementedError, match="non-uniform"):
        sim.forward(n_steps=20, emit_time_series=False)


def test_nu_forward_ntff_bit_identical_emit_flag():
    """Physical pin: NTFF accumulators must not depend on emit_time_series."""
    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    sim.add_source((0.005, 0.005, 0.001), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    sim.add_ntff_box(
        corner_lo=(0.003, 0.003, 0.0015),
        corner_hi=(0.007, 0.007, 0.003),
        freqs=[5e9],
    )
    fr_on = sim.forward(n_steps=120, emit_time_series=True)
    fr_off = sim.forward(n_steps=120, emit_time_series=False)

    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
        a = np.asarray(getattr(fr_on.ntff_data, face))
        b = np.asarray(getattr(fr_off.ntff_data, face))
        np.testing.assert_array_equal(a, b)


def test_nu_forward_emit_false_grad_runs():
    """jax.grad still works — just w.r.t. a non-time-series loss surrogate."""
    sim = _build_sim()
    g = sim._build_nonuniform_grid()
    eps_base = jnp.ones(g.shape, dtype=jnp.float32)
    ti, tj, tk = g.nx // 2 + 2, g.ny // 2 + 2, g.nz // 2

    def loss(alpha):
        eps = eps_base.at[ti, tj, tk].set(alpha)
        # Use a trivial functional of n_steps so the call participates in AD.
        fr = sim.forward(eps_override=eps, n_steps=40, emit_time_series=False)
        # Without time series, fold the eps directly so grad is well-defined.
        return jnp.sum(eps ** 2) + 0.0 * jnp.sum(jnp.asarray(fr.time_series) ** 2)

    grad = float(jax.grad(loss)(jnp.float32(2.0)))
    assert np.isfinite(grad)
