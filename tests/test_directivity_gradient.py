"""Regression test for issue #32: maximize_directivity gradient non-zero.

Prior to the ratio-based fix, `maximize_directivity` used absolute
|E|^2 at the target direction (~1e-27 in rfx's spectral NTFF
convention), which produced zero gradient in `topology_optimize`.

The ratio-based formulation U(target)/P_rad is scale-invariant and
must give a non-zero gradient through the NTFF accumulation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, run
from rfx.farfield import make_ntff_box
from rfx.optimize_objectives import (
    maximize_directivity,
    maximize_directivity_ratio,
    maximize_directivity_logratio,
)


def _forward(eps_scale: jnp.ndarray):
    """Minimal NTFF-enabled forward that is jax.grad-compatible."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials_base = init_materials(grid.shape)
    # Scale eps_r by a scalar — gradient target.
    materials = materials_base._replace(eps_r=materials_base.eps_r * eps_scale)

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, 30)
    ntff = make_ntff_box(
        grid,
        (0.003, 0.003, 0.003),
        (0.017, 0.017, 0.017),
        freqs=jnp.array([3e9]),
    )
    return run(grid, materials, 30, sources=[src], ntff=ntff)


def test_maximize_directivity_alias():
    assert maximize_directivity_ratio is maximize_directivity


def test_maximize_directivity_gradient_nonzero():
    """Gradient of the directivity objective w.r.t. a scalar eps scale
    must be non-trivially non-zero.

    This pins the fix for issue #32: the absolute-power objective
    returned ~1e-27 values and 0.0 gradients; the ratio-based one
    should give |dD/dα| >> 1e-12 in single precision.
    """
    obj = maximize_directivity(theta_target=np.pi / 2, phi_target=0.0)

    def loss(alpha):
        return obj(_forward(alpha))

    value = float(loss(jnp.array(1.0)))
    assert np.isfinite(value)
    # log-ratio loss = -(log U - log P) = log(4pi/D); its sign depends on D
    # (positive below ~11 dBi), so it is not fixed-negative like the old ratio.

    grad = float(jax.grad(loss)(jnp.array(1.0)))
    assert np.isfinite(grad)
    # The ratio is scale-invariant, so d/dα exactly 1.0 is a pathological
    # case — what matters is that the path through NTFF keeps producing a
    # meaningful, finite gradient at other probe points.
    grad_off = float(jax.grad(loss)(jnp.array(1.2)))
    assert np.isfinite(grad_off)
    # Combined magnitude must exceed the float32 noise floor (~1e-7).
    assert abs(grad) + abs(grad_off) > 1e-6, (
        f"directivity gradient collapsed to noise: grad(1.0)={grad:.2e}, "
        f"grad(1.2)={grad_off:.2e} (issue #32 would show ~0.0)"
    )


# ---------------------------------------------------------------------------
# GitHub #129 — maximize_directivity wrong-sign gradient for power-changing DoFs.
# The legacy `-U/stop_gradient(P)` drops the -U*P'/P^2 quotient term, so for any
# DoF that changes total radiated power (here: a lossy `sigma` block) the gradient
# can come out WRONG-SIGNED vs finite difference. `log_ratio=True` optimizes
# `-(log U - log P)` (full quotient `U'/U - P'/P`) which is sign-correct, scale-
# invariant, and NaN-safe.
# ---------------------------------------------------------------------------
_S0, _H = 50.0, 5.0  # base lossy-block scale + central-FD half-step


def _forward_lossy(sigma_scale):
    """NTFF forward with a POWER-CHANGING DoF: a lossy sigma block near the source.

    Adding conductivity dissipates power, so P_rad moves with the DoF — the regime
    where the legacy stop_gradient objective is wrong (#129).
    """
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    mb = init_materials(grid.shape)
    sx, sy, sz = grid.shape
    mask = np.zeros(grid.shape, dtype=np.float32)
    mask[sx // 2:sx // 2 + 4, sy // 2 - 2:sy // 2 + 2, sz // 2 - 2:sz // 2 + 2] = 1.0
    materials = mb._replace(sigma=mb.sigma + sigma_scale * jnp.asarray(mask))
    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, 40)
    ntff = make_ntff_box(grid, (0.003, 0.003, 0.003), (0.017, 0.017, 0.017),
                         freqs=jnp.array([3e9]))
    return run(grid, materials, 40, sources=[src], ntff=ntff)


def _u_target_and_p_rad(result):
    """Value-only (no AD) U(target) and full-sphere P_rad — for the ground-truth FD."""
    from rfx.farfield import compute_far_field
    th = np.array([np.pi / 2])
    ph = np.array([0.0])
    th_h = np.linspace(0.0, np.pi, 37)          # full sphere (matches the objective)
    ph_h = np.linspace(0.0, 2 * np.pi, 73)
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, th, ph)
    u = float(jnp.abs(ff.E_theta[0, 0, 0]) ** 2 + jnp.abs(ff.E_phi[0, 0, 0]) ** 2)
    ffh = compute_far_field(result.ntff_data, result.ntff_box, result.grid, th_h, ph_h)
    uh = jnp.abs(ffh.E_theta) ** 2 + jnp.abs(ffh.E_phi) ** 2
    st = jnp.asarray(np.sin(th_h))
    p = float(jnp.trapezoid(jnp.trapezoid(uh * st[None, :, None], th_h, axis=1),
                            ph_h, axis=1)[0])
    return u, p


def test_directivity_logratio_sign_correct_vs_fd_power_changing_dof():
    """#129: log_ratio is sign-correct + matches FD where legacy flips sign."""
    rm, rp = _forward_lossy(_S0 - _H), _forward_lossy(_S0 + _H)
    um, pm = _u_target_and_p_rad(rm)
    up, pp = _u_target_and_p_rad(rp)
    d_dir = ((up / pp) - (um / pm)) / (2 * _H)          # true directivity slope
    assert abs(d_dir) > 1e-9, "setup: directivity must vary with the DoF"
    true_loss_grad = -d_dir                              # loss = -D

    # central FD of the log-ratio loss = -(log U - log P), same 1e-37 floor.
    e = 1e-37
    def _log_loss(u, p):
        return -(np.log(max(u, e)) - np.log(max(p, e)))
    g_log_fd = (_log_loss(up, pp) - _log_loss(um, pm)) / (2 * _H)

    g_leg = float(jax.grad(
        lambda s: maximize_directivity(np.pi / 2, 0.0, log_ratio=False)(_forward_lossy(s))
    )(jnp.asarray(_S0)))
    g_log = float(jax.grad(
        lambda s: maximize_directivity(np.pi / 2, 0.0, log_ratio=True)(_forward_lossy(s))
    )(jnp.asarray(_S0)))

    assert np.isfinite(g_leg) and np.isfinite(g_log)
    assert abs(g_log) > 1e-7, "log_ratio gradient collapsed to noise"
    # (1) FIX is sign-correct vs the true directivity ascent direction.
    assert np.sign(g_log) == np.sign(true_loss_grad), (
        f"log_ratio sign {g_log:+.3e} != true loss grad {true_loss_grad:+.3e}")
    # (2) FIX AD matches central-FD of its own objective (self-consistent).
    rel = abs(g_log - g_log_fd) / (abs(g_log_fd) + 1e-30)
    assert rel < 0.05, f"log_ratio AD {g_log:+.3e} vs FD {g_log_fd:+.3e} rel_err {rel:.3f}"
    # (3) BUG repro: legacy stop_gradient opposes the correct direction here.
    assert np.sign(g_leg) != np.sign(g_log), (
        f"#129 not reproduced: legacy {g_leg:+.3e} should oppose log_ratio {g_log:+.3e}")


def test_directivity_default_is_logratio_and_prad_full_sphere():
    """A+B: the default objective is now the sign-correct log-ratio (A), and its
    P_rad spans the FULL sphere (B) — not the upper hemisphere, which would be
    ~2x smaller (+3 dB inflation) for a free-space radiator."""
    from rfx.farfield import compute_far_field
    r = _forward_lossy(_S0)

    # (A) default == log_ratio=True (value), and differs from the legacy mode.
    v_def = float(maximize_directivity(np.pi / 2, 0.0)(r))
    v_log = float(maximize_directivity(np.pi / 2, 0.0, log_ratio=True)(r))
    v_leg = float(maximize_directivity(np.pi / 2, 0.0, log_ratio=False)(r))
    assert v_def == v_log, "default objective is no longer the log-ratio"
    assert v_def != v_leg, "default must differ from the legacy stop-gradient mode"

    # (B) exp(-loss) = U(target)/P_rad matches the FULL-sphere normalization.
    ff_t = compute_far_field(r.ntff_data, r.ntff_box, r.grid,
                             np.array([np.pi / 2]), np.array([0.0]))
    u_t = float(jnp.abs(ff_t.E_theta[0, 0, 0]) ** 2 + jnp.abs(ff_t.E_phi[0, 0, 0]) ** 2)

    def _prad(theta_hi):
        th = np.linspace(0.0, theta_hi, 37)
        ph = np.linspace(0.0, 2 * np.pi, 73)
        ff = compute_far_field(r.ntff_data, r.ntff_box, r.grid, th, ph)
        u = jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2
        st = jnp.asarray(np.sin(th))
        return float(jnp.trapezoid(
            jnp.trapezoid(u * st[None, :, None], th, axis=1), ph, axis=1)[0])

    p_full, p_hemi = _prad(np.pi), _prad(np.pi / 2)
    assert abs(p_full - p_hemi) / p_full > 0.1, "setup: full/hemi P_rad must differ"
    ratio_obj = np.exp(-v_def)                       # single freq -> U/P_rad
    assert abs(ratio_obj - u_t / p_full) / (u_t / p_full) < 0.02, \
        "objective P_rad is not full-sphere (regressed to hemisphere?)"


def test_maximize_directivity_logratio_factory_matches_flag():
    """The factory == maximize_directivity(log_ratio=True) (value + finite grad)."""
    r = _forward_lossy(_S0)
    fac = maximize_directivity_logratio(np.pi / 2, 0.0)
    flag = maximize_directivity(np.pi / 2, 0.0, log_ratio=True)
    assert float(fac(r)) == float(flag(r))
    g = float(jax.grad(lambda s: maximize_directivity_logratio(np.pi / 2, 0.0)(_forward_lossy(s)))(jnp.asarray(_S0)))
    assert np.isfinite(g) and abs(g) > 1e-7
