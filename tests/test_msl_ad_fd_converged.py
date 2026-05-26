"""MSL-FD-TIGHT: Converged tight AD-vs-FD cross-check for end-to-end gradient.

MSL-FD-TIGHT (2026-05-25) adds a slow-marked test that runs
compute_msl_s_matrix(eps_override=...) at num_periods=20 (converged DFT)
and asserts jax.grad agrees with a central finite-difference to a tight
tolerance (rel_err <= 0.10, tightened to 0.05 if the converged value lands
there).

This converts the "AD tape flows + roughly right" evidence from
test_sparam_ad_end_to_end.py (num_periods=3, rel_err=16%) into
"AD gradient is accurate" evidence.

Geometry mirrors _build_msl_sim() in test_sparam_ad_end_to_end.py exactly.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

# ---------------------------------------------------------------------------
# Geometry — identical to _build_msl_sim() in test_sparam_ad_end_to_end.py
# ---------------------------------------------------------------------------

_MSL_EPS_R = 3.66
_MSL_H_SUB = 254e-6
_MSL_W_TRACE = 600e-6
_MSL_DX = 80e-6
_MSL_L_LINE = 6e-3
_MSL_PORT_MARGIN = 2e-3
_MSL_F_MAX = 5e9


def _build_msl_sim() -> Simulation:
    """Tiny MSL thru-line sim (2 ports, minimal domain)."""
    lx = _MSL_L_LINE + 2 * _MSL_PORT_MARGIN
    ly = _MSL_W_TRACE + 2 * (2 * _MSL_H_SUB + 8 * _MSL_DX)
    lz = _MSL_H_SUB + 0.5e-3

    sim = Simulation(
        freq_max=_MSL_F_MAX,
        domain=(lx, ly, lz),
        dx=_MSL_DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )

    sim.add_material("ro4350b", eps_r=_MSL_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MSL_H_SUB)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - _MSL_W_TRACE / 2.0
    trace_y_hi = y_centre + _MSL_W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, _MSL_H_SUB), (lx, trace_y_hi, _MSL_H_SUB + _MSL_DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN + _MSL_L_LINE, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


# ---------------------------------------------------------------------------
# Converged tight AD-vs-FD test
# ---------------------------------------------------------------------------

# Number of periods for converged DFT — must be >= 20 per MSL-FD-TIGHT spec.
_NUM_PERIODS = 20
_N_FREQS = 8
_FD_H = 1e-3
# Tolerance: start at 0.10; tighten to 0.05 if converged value lands there.
_REL_ERR_THRESHOLD = 0.10


def _closest_divisor(n: int, target: int) -> int:
    """Divisor of ``n`` nearest ``target`` (for checkpoint_segments, which must
    divide n_steps exactly — see forward(checkpoint_segments=) issue #73)."""
    best = 1
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            for cand in (d, n // d):
                if abs(cand - target) < abs(best - target):
                    best = cand
    return best


# G-AD-CHECKPOINT (2026-05-26): un-skipped. compute_msl_s_matrix now forwards
# checkpoint_every into forward(), so the reverse-mode AD tape is segmented
# (scan-of-scan remat) instead of storing the entire num_periods=20 trajectory.
# Memory scales ~sqrt(n_steps); the OOM (EXIT 137) that forced the prior skip is
# removed. Marked gpu+slow: still a heavy converged run, owned by the VESSL
# physics harness, excluded from the default CPU suite.
@pytest.mark.gpu
@pytest.mark.slow
def test_msl_ad_fd_converged_tight():
    """MSL-FD-TIGHT: converged (num_periods=20) AD gradient matches FD to <=10%.

    SKIPPED: num_periods=20 reverse-AD OOMs without gradient checkpointing (see
    skip reason). Kept as the spec + a ready harness for when checkpointed AD lands.


    R5 instrumentation: prints g_ad, g_fd, rel_err, and forward |S| range.
    If forward |S| is outside [0, 1.2], the test fails explicitly rather than
    silently reporting a gradient on an exploded impedance.

    If rel_err stays >10% this test will fail (deliberately — do NOT loosen
    the gate to force a pass; report as a gradient accuracy finding instead).
    """
    t_start = time.perf_counter()

    sim = _build_msl_sim()
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    # G-AD-CHECKPOINT: the uniform forward path uses checkpoint_segments
    # (issue #73; checkpoint_every is NU-only). The segment count must DIVIDE
    # n_steps exactly — padding is rejected because it would shift the DFT
    # accumulator windows. Pick the divisor nearest sqrt(n_steps) so backward
    # memory scales ~sqrt(n_steps)*carry instead of n_steps*carry (the OOM cause).
    n_steps = int(grid.num_timesteps(num_periods=_NUM_PERIODS))
    checkpoint_segments = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    print(f"\n[MSL-FD-TIGHT] n_steps={n_steps}, "
          f"checkpoint_segments={checkpoint_segments} (~sqrt={np.sqrt(n_steps):.1f})")

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_msl_s_matrix(
                n_freqs=_N_FREQS,
                num_periods=_NUM_PERIODS,
                eps_override=eps_base * alpha,
                checkpoint_segments=checkpoint_segments,
            )
        S = result.S
        # Sum of |S|^2 over all matrix entries and all frequency bins —
        # a smooth scalar that depends on eps at every grid cell.
        return jnp.real(jnp.sum(jnp.abs(S) ** 2))

    alpha0 = jnp.float32(1.0)

    # --- Forward sanity gate (R5) -------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_msl_s_matrix(
            n_freqs=_N_FREQS,
            num_periods=_NUM_PERIODS,
            eps_override=eps_base * alpha0,
            checkpoint_segments=checkpoint_segments,
        )
    S_fwd = np.asarray(fwd_result.S)
    s_vals = np.abs(S_fwd)
    s_min = float(np.min(s_vals))
    s_max = float(np.max(s_vals))
    print(f"\n[MSL-FD-TIGHT] forward |S| range: [{s_min:.4f}, {s_max:.4f}]")

    assert s_max <= 1.2, (
        f"[MSL-FD-TIGHT] Forward |S|_max = {s_max:.4f} exceeds 1.2 — "
        "physically implausible. Gradient on an exploded impedance is meaningless. "
        "Check MSL forward path or geometry."
    )
    assert s_max > 0.0, (
        "[MSL-FD-TIGHT] Forward |S| = 0 everywhere — likely a broken forward pass."
    )

    # --- AD gradient ---------------------------------------------------------
    t_ad_start = time.perf_counter()
    loss_val, g = jax.value_and_grad(objective)(alpha0)
    t_ad = time.perf_counter() - t_ad_start

    g_ad = float(g)
    print(f"[MSL-FD-TIGHT] loss = {float(loss_val):.6e}")
    print(f"[MSL-FD-TIGHT] g_ad = {g_ad:.6e}  (AD wall-time: {t_ad:.1f}s)")

    assert jnp.isfinite(g), f"[MSL-FD-TIGHT] AD gradient is not finite: {g}"
    assert abs(g_ad) > 1e-10, (
        f"[MSL-FD-TIGHT] AD gradient is effectively zero ({g_ad:.3e}): "
        "tape may still be broken."
    )

    # --- Central finite-difference -------------------------------------------
    t_fd_start = time.perf_counter()
    f_plus = float(objective(jnp.float32(float(alpha0) + _FD_H)))
    f_minus = float(objective(jnp.float32(float(alpha0) - _FD_H)))
    t_fd = time.perf_counter() - t_fd_start
    g_fd = (f_plus - f_minus) / (2.0 * _FD_H)
    print(f"[MSL-FD-TIGHT] g_fd = {g_fd:.6e}  (FD wall-time: {t_fd:.1f}s, h={_FD_H})")

    assert abs(g_fd) > 1e-10, (
        f"[MSL-FD-TIGHT] FD gradient is effectively zero ({g_fd:.3e}): "
        "objective may be constant w.r.t. alpha at num_periods={_NUM_PERIODS}."
    )

    # --- Accuracy gate -------------------------------------------------------
    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    # Tighten threshold if the converged value lands well below 0.10
    threshold = _REL_ERR_THRESHOLD
    if rel_err < 0.05:
        threshold = 0.05

    t_total = time.perf_counter() - t_start
    print(f"[MSL-FD-TIGHT] rel_err = {rel_err:.4f}  (threshold: {threshold:.2f})")
    print(f"[MSL-FD-TIGHT] sign agreement: g_ad={g_ad:.4e} g_fd={g_fd:.4e}")
    print(f"[MSL-FD-TIGHT] total wall-time: {t_total:.1f}s")
    print(f"[MSL-FD-TIGHT] num_periods={_NUM_PERIODS}, n_freqs={_N_FREQS}")

    assert g_ad * g_fd > 0, (
        f"[MSL-FD-TIGHT] AD and FD gradients have OPPOSITE SIGNS: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}. "
        "This is a gradient accuracy failure, not a tolerance issue."
    )

    assert rel_err <= _REL_ERR_THRESHOLD, (
        f"[MSL-FD-TIGHT] AD gradient inaccurate at num_periods={_NUM_PERIODS}: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}, rel_err={rel_err:.4f} > {_REL_ERR_THRESHOLD}. "
        "This is a genuine gradient accuracy finding — do not loosen the gate. "
        "Investigate: (1) DFT window vs transient drain, (2) JAX float32 precision, "
        "(3) port extractor AD path for residual non-differentiable ops."
    )

    print("[MSL-FD-TIGHT] PASS")
