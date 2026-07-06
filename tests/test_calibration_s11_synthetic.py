"""Synthetic-truth recovery test for calibrate_material_s11 (Stage 1).

Ground-truth self-consistency check for the physically-scaled, AD-traceable
S11 material calibrator in ``rfx.differentiable_material_fit``:

    1. Build a small lumped-port fixture on a dielectric region inside a PEC
       box with a KNOWN substrate permittivity (eps_inf = 4.3).
    2. Generate the synthetic "measured" S11 by running
       ``Simulation.forward(port_s11_freqs=...)`` on the TRUE eps_inf.
    3. Run ``calibrate_material_s11`` from a deliberately WRONG initial guess
       (eps_inf = 3.0).
    4. Assert the fit recovers the true eps_inf within tolerance and that the
       loss drops by a large factor.

Uniform single-device, boundary="pec", complex64 — the v1 scope of the
calibrator. Kept tiny so it runs in well under a couple of minutes on CPU.
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.material_fit import DebyeFitResult
from rfx.differentiable_material_fit import calibrate_material_s11


# Small, fast fixture ------------------------------------------------------

FREQ_MAX = 20e9
DX = 1.5e-3
DOMAIN = (0.012, 0.006, 0.006)   # 8 x 4 x 4 cells
NUM_PERIODS = 14.0

TRUE_EPS_INF = 4.3
WRONG_EPS_INF = 3.0

# Frequencies (Hz) at which S11 is measured / fit.
FREQS = np.linspace(6e9, 18e9, 9)


def _make_sim(eps_inf, debye_poles, lorentz_poles):
    """Lumped port on a dielectric slab inside a closed PEC box.

    ``eps_inf`` may be a traced JAX scalar (calibrate path) or a plain float
    (ground-truth generation) — both flow through ``add_material(eps_r=...)``.
    """
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=DOMAIN,
        dx=DX,
        boundary="pec",
    )
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    # Dielectric slab fills the middle of the box along x.
    sim.add(Box((0.004, 0.0, 0.0), (0.008, 0.006, 0.006)), material="dut")
    sim.add_port(
        position=(0.003, 0.003, 0.003),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def _true_s11():
    """Generate the synthetic 'measured' S11 at the TRUE eps_inf."""
    import jax.numpy as jnp

    sim = _make_sim(TRUE_EPS_INF, None, None)
    result = sim.forward(
        port_s11_freqs=jnp.asarray(FREQS, dtype=jnp.float32),
        num_periods=NUM_PERIODS,
        checkpoint=True,
        skip_preflight=True,
    )
    return np.asarray(result.s_params)


def test_calibrate_material_s11_recovers_eps_inf():
    s11_measured = _true_s11()
    assert s11_measured.shape == FREQS.shape, s11_measured.shape
    assert np.all(np.isfinite(s11_measured))

    # Deliberately WRONG controlled start: eps_inf = 3.0, no poles.
    wrong_start = DebyeFitResult(eps_inf=WRONG_EPS_INF, poles=[], fit_error=0.0)

    result = calibrate_material_s11(
        _make_sim,
        s11_measured,
        FREQS,
        n_debye_poles=0,
        n_lorentz_poles=0,
        n_iterations=45,
        learning_rate=0.03,
        num_periods=NUM_PERIODS,
        initial_guess=wrong_start,
        verbose=True,
    )

    recovered = result.eps_inf
    rel_err = abs(recovered - TRUE_EPS_INF) / TRUE_EPS_INF

    loss0 = result.loss_history[0]
    lossN = result.loss_history[-1]
    drop_factor = loss0 / max(lossN, 1e-30)

    print(f"\n[synthetic-truth] true eps_inf   = {TRUE_EPS_INF}")
    print(f"[synthetic-truth] recovered       = {recovered:.4f}")
    print(f"[synthetic-truth] rel error       = {rel_err * 100:.2f} %")
    print(f"[synthetic-truth] loss initial    = {loss0:.6e}")
    print(f"[synthetic-truth] loss final      = {lossN:.6e}")
    print(f"[synthetic-truth] loss drop factor= {drop_factor:.1f}x")

    assert result.final_s_params is not None, "final_s_params not populated"
    assert result.final_s_params.shape == FREQS.shape
    assert drop_factor > 5.0, f"loss did not drop enough: {drop_factor:.1f}x"
    assert rel_err < 0.05, f"eps_inf not recovered: {recovered:.4f} (rel {rel_err:.3%})"
