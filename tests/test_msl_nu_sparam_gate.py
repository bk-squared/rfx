"""Gate-1 — NU MSL S-param settled-S11 witness (edge-fed patch).

The full fenced NU MSL build (PR-after-#238) routes compute_msl_s_matrix through
run_nonuniform_path when a *_profile is set: _setup_msl_ports_nu injects the Ez
static-Laplace feed, the per-probe DFT planes ride the existing NU dft_plane_probe
accumulation, and the grid-agnostic V·I + N-probe-fit + S-assembly is reused
verbatim. This gate validates that the NU lane produces the SAME physically-correct
edge-fed-patch S11 the uniform lane is validated for (the committed issue-80/#118
gate), against the NU path's own grid.

Validation rationale (R5 / canonical anchor):
  * We assert the COMMITTED edge-fed physics gate (passivity + edge-fed signature +
    soft dip-above-band) — NEVER a re-spec to a dip-at-9.3 gate (that is the issue
    #118 category error). These are the exact assertions from
    test_issue80_patch_s11_regression.py.
  * We do NOT gate on NU == uniform bit-for-bit: rfx's uniform Grid and
    NonUniformGrid constructors differ by ~1 cell per axis for the same domain/dx
    (verified), so the two rasterise the geometry slightly differently. The NU path
    is validated against PHYSICS on its own grid, exactly as the waveguide NU lane is
    (vs analytic/Meep, not vs a uniform-rfx run). A uniform cross-run is printed as an
    informational witness only.
  * The mesh here is dz_profile = full(nz, dx) — the NU CODE PATH with uniform cell
    sizes — so this isolates 'the NU runner reproduces correct MSL physics' from the
    separate question 'does coarsening the far-air dz preserve the patch resonance'
    (a graded-z efficiency follow-up; the runner is identical).

Marked gpu + slow: dx=0.197 mm over ~30x18x13 mm with num_periods=200 — GPU-scale,
run by the VESSL validation harness, excluded from the default CPU suite (mirrors the
uniform issue-80 gate's resourcing).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

# --- identical geometry to tests/test_issue80_patch_s11_regression.py ---
EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
L_MSL = 8.0e-3
PORT_MARGIN = 5.0e-3
DX = 0.197e-3
DOM_X = 29.747e-3
DOM_Y = 18.130e-3
DOM_Z = 12.787e-3
Y_C = DOM_Y / 2.0

PASSIVE_TOL = 1.05
RES_BAND_GHZ = (9.0, 9.42)
RES_BAND_S11_MIN = 0.70


def _build_patch_sim_nu() -> Simulation:
    """The committed edge-fed patch, built on the NON-UNIFORM lane.

    dz_profile = full(nz, DX) drives the NU constructor + NU runner with uniform
    cell sizes, so any failure is the NU MSL path, not a grading effect.
    """
    nz = int(round(DOM_Z / DX))
    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
        dz_profile=np.full(nz, DX),
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, 4e-3), (DOM_X, DOM_Y, 4e-3 + DX)), material="pec")
    sim.add(Box((0, 0, 4e-3 + DX), (DOM_X, DOM_Y, 4e-3 + DX + H_SUB)),
            material="ro4003c")
    sim.add(Box((0, Y_C - W_MSL / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    sim.add(Box((PORT_MARGIN + L_MSL, Y_C - W / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL + L, Y_C + W / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, 4e-3 + DX),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    return sim


@pytest.mark.gpu
@pytest.mark.slow
def test_nu_msl_patch_s11_passive_and_edge_fed_match():
    """NU-lane edge-fed patch |S11| is passive AND shows the edge-fed signature,
    exactly like the validated uniform lane. Gates the full fenced NU MSL build."""
    sim = _build_patch_sim_nu()

    # R: never ignore preflight — surface any warning before trusting |S| numbers.
    sim.preflight()

    freqs = np.linspace(6e9, 14e9, 81)
    res = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs), num_periods=200.0)

    fr = np.asarray(res.freqs, dtype=float) / 1e9
    s = np.asarray(res.S)[0, 0, :]
    z0 = np.asarray(res.Z0)[0, :]
    zin = z0 * (1.0 + s) / (1.0 - s)
    s11 = np.abs(s)

    i_dip = int(np.argmin(s11))
    f_dip = fr[i_dip]
    s11_max = float(np.max(s11))
    band = (fr >= RES_BAND_GHZ[0]) & (fr <= RES_BAND_GHZ[1])
    s11_res_band_min = float(np.min(s11[band]))

    # --- R5 witnesses: full per-frequency trace, never a bare headline ---
    print(f"\n[NU-MSL-GATE] max|S11| = {s11_max:.4f}  dip @ {f_dip:.3f} GHz "
          f"(|S11|={s11[i_dip]:.4f}, the off-resonance MATCH point)")
    print(f"[NU-MSL-GATE] min|S11| over resonance band {RES_BAND_GHZ} GHz = "
          f"{s11_res_band_min:.4f} (HIGH => edge-fed signature)")
    print(f"[NU-MSL-GATE] Z0[0] median Re = {np.median(z0.real):.2f} ohm "
          f"(analytic Hammerstad-Jensen ~50.6 ohm)")
    for f, a, zr, zi in zip(fr, s11, zin.real, zin.imag):
        print(f"[NU-MSL-TRACE] {f:7.3f} GHz  |S11|={a:.5f}  "
              f"Re(Zin)={zr:9.2f}  Im(Zin)={zi:9.2f}")

    # --- (1) passivity (the #80 fix must hold on the NU lane too) ---
    assert s11_max <= PASSIVE_TOL, (
        f"non-passive NU patch: max|S11| = {s11_max:.4f} > {PASSIVE_TOL}. "
        "The NU MSL runner does not reproduce the validated passive patch S11."
    )

    # --- (2) edge-fed signature: poorly matched at resonance (dip is NOT there) ---
    assert s11_res_band_min > RES_BAND_S11_MIN, (
        f"min|S11| = {s11_res_band_min:.4f} over the resonance band "
        f"{RES_BAND_GHZ} GHz is unexpectedly LOW (<= {RES_BAND_S11_MIN}). "
        "A directly edge-fed patch must be poorly matched at its TM010 resonance; "
        "a deep dip there means the NU lane changed the physics."
    )

    # --- (3) soft: the |S11| minimum (match point) lies ABOVE the resonance band ---
    assert f_dip > RES_BAND_GHZ[1], (
        f"|S11| dip at {f_dip:.3f} GHz is inside/below the resonance band — "
        "expected the off-resonance match point ABOVE it (mesh-limited argmin; "
        "only the lower bound is asserted)."
    )
