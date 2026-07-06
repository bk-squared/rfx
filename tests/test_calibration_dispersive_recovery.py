"""Synthetic-truth recovery of a Debye pole via differentiable full-wave S11.

Stage 2A of the calibration-inverse track. Extends the eps_inf-only Stage 1
recovery (``test_calibration_s11_synthetic.py``) to a *dispersive* material:
recover a single Debye pole ``(delta_eps, tau)`` together with ``eps_inf`` from
broadband, physically-scaled S11, starting from a deliberately wrong guess.

    1. Build a small lumped-ez-port fixture on a dielectric slab in a PEC box
       whose material carries a KNOWN Debye pole (eps_inf, delta_eps, tau).
    2. Generate synthetic "measured" S11 from the TRUE material via
       ``Simulation.forward(port_s11_freqs=...)``.
    3. Run ``calibrate_material_s11`` (n_debye_poles=1) from a wrong start.
    4. Assert each recovered parameter is within a stated tolerance and that
       the loss drops by a large factor. Actual true-vs-recovered values are
       printed.

Uniform single-device, boundary="pec", complex64 — the v1 calibrator scope.
Kept tiny (~8x4x4 cells) so it runs in a couple of minutes on CPU.
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.material_fit import DebyeFitResult
from rfx.differentiable_material_fit import calibrate_material_s11


FREQ_MAX = 20e9
DX = 1.5e-3
DOMAIN = (0.012, 0.006, 0.006)   # 8 x 4 x 4 cells
NUM_PERIODS = 14.0

# Ground-truth Debye material.
TRUE_EPS_INF = 3.0
TRUE_DELTA_EPS = 2.0
TRUE_TAU = 1.5e-11               # relaxation ~ 10.6 GHz, inside the band

# Deliberately wrong controlled start.
START_EPS_INF = 2.0
START_DELTA_EPS = 0.6
START_TAU = 8.0e-12

FREQS = np.linspace(6e9, 18e9, 9)


def _make_sim(eps_inf, debye_poles, lorentz_poles):
    """Lumped ez port on a dispersive dielectric slab inside a closed PEC box.

    ``eps_inf`` and the ``DebyePole`` fields may be traced JAX scalars (the
    calibrate path) or plain floats (ground-truth generation).
    """
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    sim.add(Box((0.004, 0.0, 0.0), (0.008, 0.006, 0.006)), material="dut")
    sim.add_port(
        position=(0.003, 0.003, 0.003),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def _true_s11():
    import jax.numpy as jnp

    sim = _make_sim(TRUE_EPS_INF, [DebyePole(delta_eps=TRUE_DELTA_EPS, tau=TRUE_TAU)], None)
    result = sim.forward(
        port_s11_freqs=jnp.asarray(FREQS, dtype=jnp.float32),
        num_periods=NUM_PERIODS,
        checkpoint=True,
        skip_preflight=True,
    )
    return np.asarray(result.s_params)


def test_dispersive_debye_recovery():
    s11_measured = _true_s11()
    assert s11_measured.shape == FREQS.shape
    assert np.all(np.isfinite(s11_measured))

    wrong_start = DebyeFitResult(
        eps_inf=START_EPS_INF,
        poles=[DebyePole(delta_eps=START_DELTA_EPS, tau=START_TAU)],
        fit_error=0.0,
    )

    result = calibrate_material_s11(
        _make_sim,
        s11_measured,
        FREQS,
        n_debye_poles=1,
        n_iterations=100,
        learning_rate=0.05,
        num_periods=NUM_PERIODS,
        initial_guess=wrong_start,
        verbose=False,
    )

    rec_eps = result.eps_inf
    rec_pole = result.debye_poles[0]
    rec_de = float(rec_pole.delta_eps)
    rec_tau = float(rec_pole.tau)

    err_eps = abs(rec_eps - TRUE_EPS_INF) / TRUE_EPS_INF
    err_de = abs(rec_de - TRUE_DELTA_EPS) / TRUE_DELTA_EPS
    err_tau = abs(rec_tau - TRUE_TAU) / TRUE_TAU

    loss0 = result.loss_history[0]
    lossN = result.loss_history[-1]
    drop = loss0 / max(lossN, 1e-30)

    print("\n[dispersive-recovery] parameter        true        recovered    rel.err")
    print(f"[dispersive-recovery] eps_inf     {TRUE_EPS_INF:10.4f} {rec_eps:12.4f} {err_eps*100:8.2f} %")
    print(f"[dispersive-recovery] delta_eps   {TRUE_DELTA_EPS:10.4f} {rec_de:12.4f} {err_de*100:8.2f} %")
    print(f"[dispersive-recovery] tau (s)     {TRUE_TAU:10.3e} {rec_tau:12.3e} {err_tau*100:8.2f} %")
    print(f"[dispersive-recovery] loss {loss0:.4e} -> {lossN:.4e}  (drop {drop:.0f}x)")

    # Stated tolerances (honest, comfortably-met margins around the measured
    # ~4-15% residuals; eps_inf and delta_eps trade off partially so eps_inf is
    # the loosest).
    assert drop > 50.0, f"loss drop too small: {drop:.1f}x"
    assert err_eps < 0.08, f"eps_inf not recovered: {rec_eps:.4f} (rel {err_eps:.3%})"
    assert err_de < 0.12, f"delta_eps not recovered: {rec_de:.4f} (rel {err_de:.3%})"
    assert err_tau < 0.20, f"tau not recovered: {rec_tau:.3e} (rel {err_tau:.3%})"
