# ruff: noqa: E741  (V, I are standard EM notation for voltage/current here)
"""Issue #80 Stage-0 R5 falsification: is the MSL |S11|>1 a Yee half-step phase error?

Round-2 architect+codex consensus hypothesis: the MSL loop current I (from Hy/Hz
DFT plane accumulators) carries an uncorrected exp(-jω·dt/2) phase relative to the
Ez voltage V, because the DFT plane probe stamps E and H at the SAME t=step·dt
(rfx/simulation.py) while H physically lives at (step-1/2)·dt. The flux monitor and
the waveguide port both apply the exp(+jω·dt/2) correction; the MSL path does not.

This script does NOT change production code. It runs compute_msl_s_matrix on a
small REFLECTING single-port stub (CPU), reads the REAL extractor's raw V and I
(result.raw_v / result.raw_i1), and tests the hypothesis with a witness dump:

  raw   : phase(I) - phase(V) vs f  → if the bug is present, slope ≈ -π·dt
          (= d/df of -ω·dt/2), with |S11|>1 in the sign-fragile bins.
  corr  : apply I_corr = I·exp(+jω·dt/2) → Zin_corr = Zin·exp(-jω·dt/2);
          recompute |S11| with the SAME effective Z0. If the half-step is the
          DOMINANT cause, Re(Zin_corr)≥0 band-wide and |S11_corr|≤1.

Discriminator (architect+codex): linear slope ≈ -π·dt ⇒ half-step. Flat ±π ⇒ sign
error. Different slope ⇒ spatial V/I plane mismatch (H2). If raw |S11|≫1 but the
half-step correction does NOT pull Re(Zin)≥0, Hypothesis 1 is FALSIFIED (the tiny
~0.6° rotation cannot explain large |S11|) ⇒ escalate to H2, do NOT implement H1.
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

# --- small reflecting single-port MSL stub (CPU-runnable) ---
_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6
_F_MAX = 12e9
_PORT_MARGIN = 2e-3
_STUB_LEN = 6e-3          # trace runs PORT_MARGIN .. PORT_MARGIN+STUB_LEN, then SHORTED


def _build_reflecting_stub() -> Simulation:
    """Single MSL port feeding a short-circuited stub → strong standing-wave reflector."""
    lx = _PORT_MARGIN + _STUB_LEN + _PORT_MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * _DX)
    lz = _H_SUB + 0.6e-3
    sim = Simulation(
        freq_max=_F_MAX, domain=(lx, ly, lz), dx=_DX,
        cpml_layers=8, boundary="cpml",
    )
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="ro4350b")

    y_c = ly / 2.0
    trace_lo, trace_hi = y_c - _W_TRACE / 2.0, y_c + _W_TRACE / 2.0
    x_end = _PORT_MARGIN + _STUB_LEN
    # microstrip trace (one cell thick, on top of substrate)
    sim.add(Box((0.0, trace_lo, _H_SUB), (x_end, trace_hi, _H_SUB + _DX)), material="pec")
    # SHORT: vertical PEC wall from ground plane up to the trace at the stub end
    sim.add(Box((x_end, trace_lo, 0.0), (x_end + _DX, trace_hi, _H_SUB + _DX)),
            material="pec")

    sim.add_msl_port(
        position=(_PORT_MARGIN, y_c, 0.0),
        width=_W_TRACE, height=_H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=6e9, bandwidth=1.6),
    )
    return sim


def main() -> None:
    sim = _build_reflecting_stub()
    sim.preflight()
    grid = sim._build_grid()
    dt = float(grid.dt)

    dump_path = "/tmp/issue80_stage0_dump.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(
            n_freqs=121, num_periods=60.0, raw_3probe_dump_path=dump_path,
        )

    # raw V/I come from the extractor's own dump (raw_3probe_dump_path) so we
    # read the REAL extractor output, not a reimplementation.
    d = np.load(dump_path, allow_pickle=True)
    f = np.asarray(res.freqs, dtype=float)
    w = 2.0 * np.pi * f
    V = np.asarray(d["raw_v"])[0, 0, 0, :]      # driven=0, port=0, probe=0
    I = np.asarray(d["raw_i1"])[0, 0, :]        # driven=0, port=0
    S11 = np.asarray(res.S)[0, 0, :]

    Zin = V / I
    # Back out the effective Z0 used by the V·I split from the recorded S11:
    #   S11 = (Zin - Z0)/(Zin + Z0)  ⇒  Z0 = Zin·(1 - S11)/(1 + S11)
    Z0 = Zin * (1.0 - S11) / (1.0 + S11 + 1e-30)
    Z0_med = np.median(Z0.real[np.isfinite(Z0.real)])

    # Candidate half-step correction: I_true = I_raw · exp(+jω·dt/2)
    half = np.exp(+1j * w * dt / 2.0)
    I_corr = I * half
    Zin_corr = V / I_corr
    S11_corr = (Zin_corr - Z0_med) / (Zin_corr + Z0_med + 1e-30)

    dphi = np.unwrap(np.angle(I) - np.angle(V))   # phase(I) - phase(V)
    # Fit slope of dphi vs f over the interior (avoid band-edge wrap noise)
    sl = slice(len(f) // 8, -len(f) // 8)
    slope, intercept = np.polyfit(f[sl], dphi[sl], 1)
    expected_slope = -np.pi * dt   # d/df of -ω·dt/2 = -π·dt

    print("\n=== Stage-0 half-step witness (issue #80) ===")
    print(f"dt = {dt:.4e} s   ω·dt/2 @ {f[-1]/1e9:.1f}GHz = "
          f"{np.degrees(w[-1]*dt/2):.3f}°   (tiny rotation — see caveat)")
    print(f"effective Z0 (median Re) = {Z0_med:.2f} Ω")
    print(f"raw   max|S11| = {np.max(np.abs(S11)):.3f}   "
          f"# bins |S11|>1.001: {int(np.sum(np.abs(S11) > 1.001))}/{len(f)}")
    print(f"corr  max|S11| = {np.max(np.abs(S11_corr)):.3f}   "
          f"# bins |S11|>1.001: {int(np.sum(np.abs(S11_corr) > 1.001))}/{len(f)}")
    print(f"# bins Re(Zin)<0  raw={int(np.sum(Zin.real < 0))}  "
          f"corr={int(np.sum(Zin_corr.real < 0))}")
    print(f"slope d[phase(I)-phase(V)]/df = {slope:.4e} rad/Hz")
    print(f"expected half-step slope -π·dt = {expected_slope:.4e} rad/Hz   "
          f"ratio = {slope/expected_slope:.3f}")
    print(f"slope/(±π flat? no) intercept = {np.degrees(intercept):.1f}°")

    print("\n--- per-freq trace (sign-fragile region) ---")
    print(f"{'f[GHz]':>8} {'|S11|raw':>9} {'|S11|cor':>9} "
          f"{'Re(Zin)':>10} {'Re(Zcor)':>10} {'dphi[deg]':>10}")
    for i in range(len(f)):
        if np.abs(S11[i]) > 1.001 or abs(Zin[i].real) < 0.2 * abs(Zin[i].imag):
            print(f"{f[i]/1e9:8.3f} {np.abs(S11[i]):9.4f} {np.abs(S11_corr[i]):9.4f} "
                  f"{Zin[i].real:10.2f} {Zin_corr[i].real:10.2f} "
                  f"{np.degrees(dphi[i]):10.2f}")

    # --- verdict ---
    raw_bad = int(np.sum(np.abs(S11) > 1.001))
    corr_bad = int(np.sum(np.abs(S11_corr) > 1.001))
    slope_ratio = slope / expected_slope if expected_slope != 0 else float("nan")
    print("\n=== VERDICT ===")
    if raw_bad == 0:
        print("INCONCLUSIVE: stub did not produce |S11|>1 — not reflective enough; "
              "lengthen stub or widen freq band.")
    elif 0.5 < slope_ratio < 1.5 and corr_bad < raw_bad:
        print(f"CONFIRMED (H1): phase slope ≈ -π·dt (ratio {slope_ratio:.2f}) AND "
              f"correction reduces |S11|>1 bins {raw_bad}→{corr_bad}.")
    elif corr_bad >= raw_bad:
        print(f"FALSIFIED (H1 not dominant): half-step correction did NOT reduce "
              f"|S11|>1 bins ({raw_bad}→{corr_bad}). Escalate to H2 (spatial V/I plane).")
    else:
        print(f"PARTIAL: slope ratio {slope_ratio:.2f} (not ≈1) — phase error present "
              f"but not pure half-step; correction {raw_bad}→{corr_bad}. Inspect.")


if __name__ == "__main__":
    main()
