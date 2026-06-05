"""Issue #118 — falsifiable witness: is the |S11| dip the resonance or the match point?

For a 50-ohm microstrip line BUTT-COUPLED directly to a patch radiating edge, standard
patch theory (Balanis/Pozar) says the edge input resistance at the TM010 resonance is high
(~230-470 ohm for this geometry), so the feed is *badly matched at resonance* and the
|S11| MAGNITUDE minimum is the off-resonance MATCH point (Re(Zin) -> 50), which sits ABOVE
the resonance. The resonance itself is where Im(Zin) crosses zero.

This runs the *production* compute_msl_s_matrix on the issue-80 patch (same geometry as
tests/test_issue80_patch_s11_regression.py), derives Zin(f) = Z0 (1+S11)/(1-S11), and reports:
  - f_dip          : argmin |S11|                     (the DIP)
  - f_match        : argmin |Zin - 50|                (the MATCH point; expect == f_dip)
  - f_res_ImZ0     : zero-crossing of Im(Zin)         (the RESONANCE; expect ~9.2-9.3 GHz)

FALSIFIER (this is the gate on the whole #118 re-spec):
  PASS  (re-spec is a legitimate physics correction): Im(Zin)=0 near ~9.2-9.3 GHz, clearly
        BELOW f_dip, and f_match == f_dip. => the dip is the match point, resonance is correct.
  FAIL  (real MSL-port coupling bug, STOP and reject the re-spec): Im(Zin)=0 coincides with
        f_dip => the fields actually resonate at the dip, so the gate was right to flag it.

R5: dumps the full per-frequency trace (|S11|, Re/Im Zin), never a bare headline.
Run: JAX_PLATFORMS=cpu python scripts/diagnostics/issue118_match_vs_resonance_witness.py
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rfx import Box, Simulation  # noqa: E402
from rfx.sources import GaussianPulse  # noqa: E402

# --- issue #80 reproduction geometry (identical to the regression test) ---
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

NUM_PERIODS = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
FREQS = np.linspace(6e9, 14e9, 81)


def build() -> Simulation:
    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
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


def zero_crossing_ghz(fr_ghz, y):
    """First sign change of y, linearly interpolated; None if no crossing."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        f0, f1 = fr_ghz[i], fr_ghz[i + 1]
        y0, y1 = y[i], y[i + 1]
        out.append(f0 - y0 * (f1 - f0) / (y1 - y0))
    return out


def main() -> int:
    sim = build()
    print(f"[ISSUE118-WITNESS] num_periods={NUM_PERIODS}  dx={DX*1e6:.1f}um "
          f"grid={sim._build_grid().shape}", flush=True)

    # R: never ignore preflight — surface warnings before trusting numbers.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.preflight()
    for wm in caught:
        print(f"[PREFLIGHT] {wm.category.__name__}: {wm.message}", flush=True)

    res = sim.compute_msl_s_matrix(freqs=jnp.asarray(FREQS), num_periods=NUM_PERIODS)

    fr = np.asarray(res.freqs, dtype=float) / 1e9
    s11 = np.asarray(res.S)[0, 0, :]
    z0 = np.asarray(res.Z0)[0, :]
    zin = z0 * (1.0 + s11) / (1.0 - s11)
    s11_abs = np.abs(s11)

    i_dip = int(np.argmin(s11_abs))
    i_match = int(np.argmin(np.abs(zin - 50.0)))
    f_dip = fr[i_dip]
    f_match = fr[i_match]
    res_cross = zero_crossing_ghz(fr, zin.imag)

    print("\n[ISSUE118-TRACE]   f(GHz)   |S11|    Re(Zin)    Im(Zin)   |Zin-50|", flush=True)
    for k in range(len(fr)):
        print(f"  {fr[k]:7.3f}  {s11_abs[k]:7.4f}  {zin.real[k]:9.2f}  {zin.imag[k]:9.2f}  "
              f"{abs(zin[k]-50.0):9.2f}", flush=True)

    print("\n========== ISSUE118 WITNESS SUMMARY ==========", flush=True)
    print(f"max|S11|              = {s11_abs.max():.4f}  (passive if <= 1.05)", flush=True)
    print(f"f_dip   (argmin|S11|) = {f_dip:.3f} GHz   |S11|={s11_abs[i_dip]:.4f}", flush=True)
    print(f"f_match (argmin|Zin-50|)= {f_match:.3f} GHz  Zin={zin[i_match]:.1f}", flush=True)
    print(f"Im(Zin)=0 crossings   = {[round(x,3) for x in res_cross]} GHz "
          f"(RESONANCE; expect ~9.2-9.3)", flush=True)

    # Verdict logic
    res_near_93 = any(8.9 <= x <= 9.6 for x in res_cross)
    dip_above_res = bool(res_cross) and f_dip > (min(res_cross, key=lambda x: abs(x - 9.3)) + 0.3)
    dip_is_match = abs(f_dip - f_match) <= 0.3
    print("\n--- FALSIFIER ---", flush=True)
    print(f"  Im(Zin)=0 near 9.2-9.3 GHz?         {res_near_93}", flush=True)
    print(f"  dip clearly ABOVE the resonance?    {dip_above_res}", flush=True)
    print(f"  dip == match point (|Zin-50| min)?  {dip_is_match}", flush=True)
    if res_near_93 and dip_above_res and dip_is_match:
        print("  => PASS: dip is the off-resonance MATCH point; resonance ~9.3 is correct. "
              "Re-spec is a legitimate physics correction.", flush=True)
        return 0
    print("  => INSPECT: the clean PASS pattern did not hold — read the full trace above "
          "before trusting the re-spec (do NOT auto-promote).", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
