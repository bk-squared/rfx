"""Issue #80 Fix-B premise discriminator: source near-field vs alpha->0.

Question
--------
PR #99 fixed the MSL extractor (S = gamma/alpha spatial fit, current-free) but
the resonant-patch test still reads |S11| = 1.166 > 1. The runtime honesty guard
offers TWO candidate causes:

  H1 (near-field): the default probe offset is 0.5 * lambda/(2pi), i.e. the
       probes sit INSIDE the source reactive near-field, so a spurious
       (evanescent) component biases the measured V and inflates |S11|.
  H2 (alpha->0): at a sharp resonance the forward wave amplitude alpha -> 0 at
       the notch, so S11 = gamma/alpha is ill-conditioned and overshoots 1.
       This is NOT cured by a larger offset (could be made worse).

This script settles the PREMISE of H1 on a clean, CPU-scale MATCHED thru-line
(no resonance -> H2 cannot apply, alpha is large everywhere). It runs ONE FDTD
with a DENSE 30-probe array, then slides a 3-probe window outward and re-fits
|S11| with the REAL production extractor (extract_msl_nprobe). On a matched line
the true |S11| is ~0; any residual is bias.

  - If H1 is real: |S11|(offset) starts elevated near the feed and DECAYS as the
    window moves out, flattening to a floor once past the evanescent zone.
  - If H1 is wrong: |S11|(offset) is ~flat -> the 0.118 residual is far-port
    reflection / Yee-staircase / mode mismatch, none of which Fix B addresses.

The decay length L (exp fit) is compared against lambda/(2pi) [the guard's
yardstick] and a higher-order-mode evanescent estimate [the physically correct
one for a guided mode].

Run:  python scripts/diagnostics/issue80_nearfield_offset_sweep.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

# NB: run in default c64 — the FDTD time-stepping scan is NOT x64-clean (the
# DFT-plane carry is initialized c64 and the body would output c128 under x64,
# which jax.lax.scan rejects). This matches how production runs a LIVE solve
# (the f64 extractor path is only exercised in dump-replay, never a live FDTD),
# and it is the same regime the GPU patch acceptance run used. c64 precision
# (~1e-6 relative) is far below the |S11| values being compared here.
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

# --- geometry (RO4350B thru, mirrors tests/test_msl_port_integration.py) ---
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 25e-3          # long line: dense probe array stays mid-line
PORT_MARGIN = 2e-3
DX = 80e-6
F_MAX = 5e9

LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3

# Dense probe array on the driven port.
N_PROBES = 30
N_OFFSET = 4            # first probe 4 cells from feed (deep into reactive zone)
N_SPACING = 4          # 0.32 mm adjacent spacing
WIN = 3                # production default window width


def eps_eff_hj(eps_r: float, w: float, h: float) -> float:
    """Hammerstad-Jensen quasi-static eps_eff (W/h > 1 branch)."""
    u = w / h
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** -0.5


def build_sim() -> Simulation:
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2.0
    sim.add(
        Box((0.0, y_c - W_TRACE / 2, H_SUB), (LX, y_c + W_TRACE / 2, H_SUB + DX)),
        material="pec",
    )
    # Driven port 0 (+x) with the DENSE probe array.
    sim.add_msl_port(
        position=(PORT_MARGIN, y_c, 0.0), width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
        n_probes=N_PROBES, n_probe_offset=N_OFFSET, n_probe_spacing=N_SPACING,
    )
    # Passive matched port 1 (-x) at the far end -> clean termination.
    sim.add_msl_port(
        position=(PORT_MARGIN + L_LINE, y_c, 0.0), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0,
    )
    return sim


def main() -> None:
    sim = build_sim()
    grid = sim._build_grid()

    dump_path = os.path.join(tempfile.gettempdir(), "issue80_nf_sweep.npz")
    print(f"[run] FDTD matched thru-line, dense {N_PROBES}-probe array ...")
    res = sim.compute_msl_s_matrix(
        n_freqs=20, num_periods=14.0, raw_3probe_dump_path=dump_path,
    )
    freqs = np.asarray(res.freqs, dtype=float)
    print(f"[run] done. freqs {freqs[0]/1e9:.2f}-{freqs[-1]/1e9:.2f} GHz")

    d = np.load(dump_path, allow_pickle=True)
    raw_v = np.asarray(d["raw_v"])          # (n_driven, n_ports, n_probes, n_freqs)
    prod_beta = np.asarray(d["production_beta"])  # fitted beta anchor
    meta = json.loads(str(d["metadata_json"]))
    print(f"[dump] raw_v shape {raw_v.shape}")

    # Driven port 0, its own probes: (n_probes, n_freqs)
    v_probes = raw_v[0, 0, :N_PROBES, :]    # complex
    # Production S11 from the full array (sanity anchor).
    prod_s11 = np.abs(np.asarray(res.S)[0, 0, :])

    # Matching x-coords for the dense array (signed physical x).
    mp0 = MSLPort(
        feed_x=PORT_MARGIN, y_lo=LY / 2 - W_TRACE / 2, y_hi=LY / 2 + W_TRACE / 2,
        z_lo=0.0, z_hi=H_SUB, direction="+x", impedance=50.0, excitation=None,
    )
    x_all = np.asarray(msl_probe_x_coords_n(
        grid, mp0, n_probes=N_PROBES, n_offset_cells=N_OFFSET,
        n_spacing_cells=N_SPACING,
    ), dtype=float)
    print(f"[probes] x range {x_all.min()*1e3:.2f}-{x_all.max()*1e3:.2f} mm "
          f"(feed at {PORT_MARGIN*1e3:.2f} mm)")

    # beta0 anchor per freq for the windowed re-fit (real part of fitted beta).
    beta0 = np.real(prod_beta).astype(float)
    if beta0.ndim == 0:
        beta0 = np.full(freqs.shape[0], float(beta0))

    # Slide a WIN-probe window; |S11| per window per freq.
    n_win = N_PROBES - WIN + 1
    win_off_cells = np.zeros(n_win)         # window-center offset from feed [cells]
    s11_win = np.zeros((n_win, freqs.shape[0]))
    i1_dummy = np.ones(freqs.shape[0], dtype=complex)
    for k in range(n_win):
        sl = slice(k, k + WIN)
        vk = v_probes[sl, :].T              # (n_freqs, WIN)
        xk = x_all[sl]
        out = extract_msl_nprobe(vk, xk, i1_dummy, beta0)
        s11_win[k, :] = np.abs(np.asarray(out["s11"]))
        # center probe offset from feed in cells
        win_off_cells[k] = N_OFFSET + (k + (WIN - 1) / 2) * N_SPACING

    win_off_mm = win_off_cells * DX * 1e3

    # Evaluate at a few representative freqs in the gate band.
    eps_eff = eps_eff_hj(EPS_R, W_TRACE, H_SUB)
    print(f"\n[analytic] eps_eff(HJ) = {eps_eff:.3f}")
    C0 = 2.99792458e8

    eval_ghz = [3.0, 4.0, 5.0]
    f_idx = [int(np.argmin(np.abs(freqs - g * 1e9))) for g in eval_ghz]

    print("\n=== |S11|(offset) on a MATCHED line (true |S11| ~ 0) ===")
    header = "off[cell] off[mm] " + " ".join(
        f"|S11|@{freqs[i]/1e9:4.2f}G" for i in f_idx)
    print(header)
    for k in range(n_win):
        row = f"{win_off_cells[k]:7.0f} {win_off_mm[k]:6.2f} " + " ".join(
            f"{s11_win[k, i]:12.4f}" for i in f_idx)
        print(row)

    # lambda/(2pi) yardstick (guard) + decay-length fit per eval freq.
    print("\n=== decay analysis per frequency ===")
    print("freq   lam/(2pi)[cell]  |S11|@first  |S11|@last  fit_L[cell]  fit_L[mm]")
    for i in f_idx:
        lam = C0 / freqs[i] / math.sqrt(eps_eff)
        lam_2pi_cells = lam / (2 * math.pi) / DX
        y = s11_win[:, i]
        # exp-decay fit: y = A exp(-off/L) + C, via simple log-linear on (y - floor)
        floor = float(np.min(y[-5:]))       # tail as floor estimate
        yy = y - floor
        mask = yy > 1e-6
        if mask.sum() >= 3:
            L = float(-1.0 / np.polyfit(win_off_cells[mask], np.log(yy[mask]), 1)[0])
        else:
            L = float("nan")
        print(f"{freqs[i]/1e9:4.2f}G {lam_2pi_cells:13.1f}  {y[0]:10.4f}  "
              f"{y[-1]:9.4f}  {L:10.1f}  {L*DX*1e3:8.2f}")

    print("\n[interpretation]")
    print(" - DECAY with offset (|S11|@first >> |S11|@last)  => H1 near-field REAL")
    print("   -> Fix B has merit; decay length L sets the required offset.")
    print(" - FLAT vs offset                                  => H1 WRONG")
    print("   -> residual is far-port/Yee/mode, Fix B won't help; investigate H2.")
    print(f"\n[sanity] production full-array |S11| band-mean = "
          f"{float(np.mean(prod_s11)):.4f} "
          f"(default offset {meta['port_definitions'][0]['n_probe_offset']} cells)")


if __name__ == "__main__":
    main()
