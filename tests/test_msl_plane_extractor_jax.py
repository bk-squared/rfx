"""Plane-integrated JAX MSL S-extractor must match compute_msl_s_matrix.

Phase 2 + 3 of gap #2/#4 closure (see
``docs/agent-memory/rfx-known-issues.md``, 2026-05-05).

Phase 2 (commit 0e0183c) added the plane-integrated JAX extractor
:func:`rfx.probes.msl_wave_decomp.extract_msl_s_params_jax_plane`,
mirroring ``compute_msl_s_matrix``'s plane V/I integration formulas.
That alone left a ~0.10–0.15 absolute residual against the imperative
path because the two orchestrators
(``runners/uniform.py::run_uniform_path`` for the imperative path,
``_forward_from_materials`` for ``forward()`` / the plane lane)
launched MSL ports with *different source mode profiles* — the
imperative path honoured the port's ``mode='laplace'`` field, the
forward path silently fell through to a uniform Ez source.

Phase 3 (commit 1c50dff) brought the forward path's MSL-port
construction into parity with the imperative path.  After that fix
the plane lane is bit-identical to ``compute_msl_s_matrix`` at FP32
numerical noise on the cv06b-class thru-line:
  * |S11| max diff 0.0000, RMS 0.0000
  * |S21| max diff 0.0009, RMS 0.0002

The scalar lane (``extract_msl_s_params_jax``) does *not* use
``add_dft_plane_probe`` and so does not exercise Phase 1's
``ForwardResult.dft_planes``; it remains the documented-biased
fallback for users who want a single-time-series S-extractor without
plane probes.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.msl_wave_decomp import (
    register_msl_wave_probes, extract_msl_s_params_jax,
    register_msl_plane_probes, extract_msl_s_params_jax_plane,
)


EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 127e-6
L_LINE = 30e-3
PORT_MARGIN = 1.6e-3
F_MAX = 9e9


def _build_thru_line(*, l_line: float = L_LINE):
    """Two-port through-line MSL geometry, no stub.  cv06b-class."""
    LX = l_line + 2 * PORT_MARGIN
    L_STUB_MAX = 14e-3
    LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX) + L_STUB_MAX + 2 * (2 * H_SUB + 8 * DX)
    LZ = H_SUB + 1.0e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * DX) + W_TRACE / 2
    sim.add(
        Box((0, y_trace - W_TRACE / 2, H_SUB),
            (LX, y_trace + W_TRACE / 2, H_SUB + DX)),
        material="pec",
    )
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + l_line, y_trace, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)
    return sim, y_trace


# ---------------------------------------------------------------------------
# Imperative reference — single shared run across this file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def imperative_reference():
    """Run compute_msl_s_matrix once, return (S, freqs).  Slow."""
    sim, _ = _build_thru_line()
    res = sim.compute_msl_s_matrix(n_freqs=20, num_periods=15.0)
    return np.asarray(res.S), np.asarray(res.freqs)


@pytest.fixture(scope="module")
def plane_lane_result(imperative_reference):
    """Run plane lane once at the imperative reference's frequencies."""
    _, freqs = imperative_reference
    s11, s21 = _run_plane_lane(jnp.asarray(freqs, dtype=jnp.float32))
    return s11, s21


@pytest.fixture(scope="module")
def scalar_lane_result(imperative_reference):
    """Run scalar lane once at the imperative reference's frequencies."""
    _, freqs = imperative_reference
    s11, s21 = _run_scalar_lane(jnp.asarray(freqs, dtype=jnp.float32))
    return s11, s21


def _run_plane_lane(freqs):
    """Single forward() with plane probes registered, S11/S21 via plane lane."""
    sim, y_trace = _build_thru_line()
    # Drive port 0, leave port 1 passive — same convention as
    # compute_msl_s_matrix when computing S[*, 0, :].
    object.__setattr__(sim._msl_ports[1], "excite", False)
    d_set = register_msl_plane_probes(sim, port_index=0, freqs=freqs,
                                       name_prefix="d")
    p_set = register_msl_plane_probes(sim, port_index=1, freqs=freqs,
                                       name_prefix="p")
    fr = sim.forward(num_periods=15.0, skip_preflight=True)
    s11, s21 = extract_msl_s_params_jax_plane(fr, d_set, p_set)
    return np.asarray(s11), np.asarray(s21)


def _run_scalar_lane(freqs):
    """Single forward() with point probes only; S11/S21 via scalar lane."""
    sim, y_trace = _build_thru_line()
    object.__setattr__(sim._msl_ports[1], "excite", False)
    d_set = register_msl_wave_probes(
        sim, feed_x=PORT_MARGIN, direction="+x",
        y_centre=y_trace, z_ez=H_SUB / 2.0, z_hy=H_SUB + 0.5 * DX,
    )
    p_set = register_msl_wave_probes(
        sim, feed_x=PORT_MARGIN + L_LINE, direction="-x",
        y_centre=y_trace, z_ez=H_SUB / 2.0, z_hy=H_SUB + 0.5 * DX,
    )
    fr = sim.forward(num_periods=15.0, skip_preflight=True)
    grid = sim._build_grid()
    s11, s21 = extract_msl_s_params_jax(
        fr.time_series, d_set, p_set,
        dt=float(grid.dt), freqs=jnp.asarray(freqs, dtype=jnp.float32),
    )
    return np.asarray(s11), np.asarray(s21)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Tight absolute-magnitude gates.  After Phase 3 (commit 1c50dff,
# 2026-05-07) closed the MSL-port `mode` parity gap between
# `runners/uniform.py::run_uniform_path` and `_forward_from_materials`,
# the plane lane and the imperative `compute_msl_s_matrix` are
# bit-identical at FP32 numerical noise — measured on the cv06b-class
# 30 mm thru-line at 2 substrate cells:
#   plane vs imperative |S11|: max diff 0.0000, RMS 0.0000
#   plane vs imperative |S21|: max diff 0.0009, RMS 0.0002
# Gates are set just above the measured FP32 noise floor.  The earlier
# loose gates (0.15 / 0.20) reflected the Phase 2 mode-mismatch and
# are no longer the right target.
_GATE_ABS_S21 = 0.005
_GATE_ABS_S11 = 0.001


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Issue #80 stage S1 (2026-05-19) moved compute_msl_s_matrix — the "
        "imperative lane — onto the OpenEMS V·I wave split + closed "
        "Ampère-loop current. The plane lane extract_msl_s_params_jax_plane "
        "still runs the legacy 3-probe quadratic, so the two lanes "
        "legitimately diverge until the S2 consolidation brings the plane "
        "lane to the same V·I split. strict=True so this self-removes when "
        "S2 re-unifies the lanes. See "
        "docs/agent-memory/port_sparam_review_2026-05-19.md."
    ),
)
def test_plane_lane_s21_matches_imperative(
    imperative_reference, plane_lane_result,
):
    """|S21| must agree with the imperative reference to within ≤ 0.005
    absolute on the 2-substrate-cell thru-line mesh — i.e. bit-identical
    at FP32 noise after the Phase 3 mode-parity fix."""
    S_imp, _ = imperative_reference
    _, s21_plane = plane_lane_result
    s21_imp = S_imp[1, 0, :]
    diff = np.abs(np.abs(s21_plane) - np.abs(s21_imp))
    assert diff.max() <= _GATE_ABS_S21, (
        f"plane-lane |S21| disagrees with imperative by {diff.max():.3f} "
        f"> {_GATE_ABS_S21}; plane={np.abs(s21_plane)!r}\nimp={np.abs(s21_imp)!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Issue #80 stage S1 (2026-05-19) moved compute_msl_s_matrix — the "
        "imperative lane — onto the OpenEMS V·I wave split + closed "
        "Ampère-loop current. The plane lane extract_msl_s_params_jax_plane "
        "still runs the legacy 3-probe quadratic, so the two lanes "
        "legitimately diverge until the S2 consolidation brings the plane "
        "lane to the same V·I split. strict=True so this self-removes when "
        "S2 re-unifies the lanes. See "
        "docs/agent-memory/port_sparam_review_2026-05-19.md."
    ),
)
def test_plane_lane_s11_matches_imperative(
    imperative_reference, plane_lane_result,
):
    """|S11| must agree with the imperative reference to within ≤ 0.001
    absolute on the 2-substrate-cell thru-line mesh — i.e. bit-identical
    at FP32 noise after the Phase 3 mode-parity fix."""
    S_imp, _ = imperative_reference
    s11_plane, _ = plane_lane_result
    s11_imp = S_imp[0, 0, :]
    diff = np.abs(np.abs(s11_plane) - np.abs(s11_imp))
    assert diff.max() <= _GATE_ABS_S11, (
        f"plane-lane |S11| disagrees with imperative by {diff.max():.3f} "
        f"> {_GATE_ABS_S11}; plane={np.abs(s11_plane)!r}\nimp={np.abs(s11_imp)!r}"
    )


def test_plane_lane_strictly_tighter_than_scalar_lane(
    imperative_reference, plane_lane_result, scalar_lane_result,
):
    """**Headline gate** for Phase 2 + 3.  The plane lane must agree with
    the imperative path more tightly than the scalar lane on the same
    mesh.  After Phase 3 closure (commit 1c50dff) the plane RMS is
    bit-identical (~2e-4) while the scalar RMS sits at ~1.5 — the
    plane > scalar margin is now several orders of magnitude (gap #4
    in `docs/agent-memory/rfx-known-issues.md`)."""
    S_imp, _ = imperative_reference
    s21_imp = S_imp[1, 0, :]
    _, s21_plane = plane_lane_result
    _, s21_scalar = scalar_lane_result
    rms_plane = float(np.sqrt(np.mean(
        (np.abs(s21_plane) - np.abs(s21_imp)) ** 2
    )))
    rms_scalar = float(np.sqrt(np.mean(
        (np.abs(s21_scalar) - np.abs(s21_imp)) ** 2
    )))
    assert rms_plane <= rms_scalar, (
        f"plane lane (RMS={rms_plane:.4f}) must be at least as accurate as "
        f"scalar lane (RMS={rms_scalar:.4f}) vs imperative reference."
    )
