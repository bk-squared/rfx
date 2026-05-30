"""Issue #80 — definitive patch diagnosis via raw-probe dump (GPU).

The CPU experiments (docs/research_notes/20260530_issue80_nearfield_falsified.md)
falsified near-field and conditioning-alone for the patch |S11|=1.166>1, and
pointed to single-mode MODEL MISMATCH at the short patch feed. CPU proxies are
clean single-mode lines and cannot reproduce it — only the actual patch can.

This runs the patch geometry (mirrors test_issue80_patch_s11_regression) with a
DENSE 18-probe array spanning the whole feed (near-port -> near-patch), dumps
raw probe voltages, and analyzes OFFLINE with the production extractor:

  (1) per-3-probe-window single-mode fit RESIDUAL vs window position. If the
      relative residual GROWS toward the patch junction -> non-TEM/higher-order
      field there -> model mismatch CONFIRMED (and it is local to the junction).
  (2) production full 18-probe (well-conditioned) |S11|. If still >1 -> NOT
      conditioning; the field genuinely is not a single-mode 2-wave standing
      wave at the reference plane.
  (3) |S11| vs SPAN (fixed start) and vs OFFSET (fixed span) — re-confirm
      conditioning and near-field are not the levers on the real patch.
  (4) num_periods 200 vs 400 — settling check (does max|S11| / dip move?).

Decision table:
  residual small everywhere AND full-array |S11|<=1  -> conditioning was it after
      all (revisit); residual small but |S11|>1 -> a DIFFERENT cause (de-embed /
      assembly at near-total reflection) — investigate that.
  residual large near patch AND |S11|>1 at all spans -> model mismatch CONFIRMED;
      honest resolution = flag (keep xfail) or multimode-aware reference plane.

Run on GPU (VESSL). Dumps land in the NFS workspace for local re-analysis.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

# --- patch geometry (mirrors tests/test_issue80_patch_s11_regression.py) ---
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
Z_TRACE = 4e-3 + DX + H_SUB + DX        # trace z (= port z_lo + ... matches test)
TARGET_GHZ = 9.21

# Dense probe array across the WHOLE feed (port x=5mm -> patch edge x=13mm).
# offset 3 + (N-1)*spacing*... must stay below 40 cells (8mm feed).
N_PROBES = 18
N_OFFSET = 3
N_SPACING = 2                            # probes 3,5,...,37 cells => x 5.59..12.29mm
WIN = 3
ART_DIR = os.path.join(os.path.dirname(__file__), "_artifacts")


def build_patch(n_probes, n_offset, n_spacing):
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=DX, cpml_layers=8, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, 4e-3), (DOM_X, DOM_Y, 4e-3 + DX)), material="pec")
    sim.add(Box((0, 0, 4e-3 + DX), (DOM_X, DOM_Y, 4e-3 + DX + H_SUB)),
            material="ro4003c")
    sim.add(Box((0, Y_C - W_MSL / 2, Z_TRACE),
                (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2, Z_TRACE + DX)),
            material="pec")
    sim.add(Box((PORT_MARGIN + L_MSL, Y_C - W / 2, Z_TRACE),
                (PORT_MARGIN + L_MSL + L, Y_C + W / 2, Z_TRACE + DX)),
            material="pec")
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, 4e-3 + DX), width=W_MSL, height=H_SUB,
        direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
        n_probes=n_probes, n_probe_offset=n_offset, n_probe_spacing=n_spacing,
    )
    return sim


def fit_window(v_probes, x_all, idx, beta0, i1):
    xk = x_all[idx]
    vk = v_probes[idx, :].T               # (n_freqs, len(idx))
    out = extract_msl_nprobe(vk, xk, i1, beta0)
    s11 = np.abs(np.asarray(out["s11"]))
    # relative residual: ||V - fit|| / ||V|| per freq, recomputed here.
    a = np.asarray(out["alpha"]); g = np.asarray(out["gamma"]); b = np.asarray(out["beta"])
    xc = (xk - xk[0])[None, :]
    pred = a[:, None] * np.exp(-1j * b[:, None] * xc) + g[:, None] * np.exp(1j * b[:, None] * xc)
    num = np.linalg.norm(vk - pred, axis=1)
    den = np.linalg.norm(vk, axis=1) + 1e-30
    return s11, num / den


def cond_2wave(x, beta):
    xc = np.asarray(x) - x[0]
    a = np.stack([np.exp(-1j * beta * xc), np.exp(1j * beta * xc)], axis=-1)
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-30))


def analyze(dump, tag):
    d = np.load(dump, allow_pickle=True)
    raw_v = np.asarray(d["raw_v"])
    prod_beta = np.real(np.asarray(d["production_beta"])).astype(float)
    freqs = np.asarray(d["freqs_hz"], dtype=float)
    prod_s11 = np.abs(np.asarray(d["production_smatrix"])[0, 0, :])
    v_probes = raw_v[0, 0, :N_PROBES, :]

    mp0 = MSLPort(feed_x=PORT_MARGIN, y_lo=Y_C - W_MSL / 2, y_hi=Y_C + W_MSL / 2,
                  z_lo=4e-3 + DX, z_hi=4e-3 + DX + H_SUB, direction="+x",
                  impedance=50.0, excitation=None)
    grid = build_patch(N_PROBES, N_OFFSET, N_SPACING)._build_grid()
    x_all = np.asarray(msl_probe_x_coords_n(
        grid, mp0, n_probes=N_PROBES, n_offset_cells=N_OFFSET,
        n_spacing_cells=N_SPACING), dtype=float)
    beta0 = prod_beta if prod_beta.ndim else np.full(freqs.shape[0], float(prod_beta))
    i1 = np.ones(freqs.shape[0], dtype=complex)

    ir = int(np.argmin(np.abs(freqs - TARGET_GHZ * 1e9)))   # near resonance
    imax = int(np.argmax(prod_s11))

    print(f"\n########## {tag} ##########")
    print(f"production full 18-probe |S11|: max {prod_s11.max():.4f} @ "
          f"{freqs[imax]/1e9:.3f} GHz ; @9.21G = {prod_s11[ir]:.4f} ; "
          f"min {prod_s11.min():.4f} @ {freqs[int(np.argmin(prod_s11))]/1e9:.3f} GHz")

    # (1) residual + |S11| vs WINDOW POSITION (near-port -> near-patch)
    print("\n-- 3-probe window: position (near-port->near-patch) --")
    print(" win_x[mm]  dist_to_patch[mm]  |S11|@max  relres@max  |S11|@9.21  relres@9.21")
    n_win = N_PROBES - WIN + 1
    for k in range(n_win):
        idx = [k, k + 1, k + 2]
        s11, relres = fit_window(v_probes, x_all, idx, beta0, i1)
        xc_mm = x_all[idx[1]] * 1e3
        d2patch = (PORT_MARGIN + L_MSL - x_all[idx[1]]) * 1e3
        print(f" {xc_mm:8.2f}  {d2patch:16.2f}  {s11[imax]:9.4f}  {relres[imax]:10.4f}  "
              f"{s11[ir]:10.4f}  {relres[ir]:11.4f}")

    # (2)+(3) |S11| vs SPAN at fixed near-port start
    print("\n-- |S11| vs SPAN (fixed near-port start, stride sub-sample) --")
    print(" stride span[mm] cond@max  |S11|@max  |S11|@9.21")
    for s in range(1, (N_PROBES - 1) // 2 + 1):
        idx = [0, s, 2 * s]
        if idx[-1] >= N_PROBES:
            break
        s11, _ = fit_window(v_probes, x_all, idx, beta0, i1)
        span = (x_all[idx[-1]] - x_all[idx[0]])
        c = cond_2wave(x_all[idx], beta0[imax])
        print(f" {s:6d} {span*1e3:7.2f} {c:8.1f}  {s11[imax]:9.4f}  {s11[ir]:10.4f}")

    return freqs, prod_s11, imax, ir


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    summary = {}
    for NP in (200, 400):
        dump = os.path.join(ART_DIR, f"patch_rawdump_np{NP}.npz")
        print(f"\n[run] patch FDTD num_periods={NP}, dense {N_PROBES}-probe array ...")
        sim = build_patch(N_PROBES, N_OFFSET, N_SPACING)
        sim.preflight()
        res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=float(NP),
                                       raw_3probe_dump_path=dump)
        freqs, prod_s11, imax, ir = analyze(dump, f"num_periods={NP}")
        summary[NP] = dict(maxs11=float(prod_s11.max()),
                           fmax=float(freqs[imax] / 1e9),
                           s11_res=float(prod_s11[ir]),
                           fdip=float(freqs[int(np.argmin(prod_s11))] / 1e9))

    print("\n########## SETTLING (num_periods 200 vs 400) ##########")
    for NP in (200, 400):
        s = summary[NP]
        print(f" np={NP}: max|S11|={s['maxs11']:.4f}@{s['fmax']:.3f}G "
              f"|S11|@9.21={s['s11_res']:.4f}  dip@{s['fdip']:.3f}G")
    d200, d400 = summary[200], summary[400]
    print(f" delta max|S11| = {abs(d400['maxs11']-d200['maxs11']):.4f} "
          f"(small => settled => settling NOT the cause)")
    print("\n[verdict guide] full-array |S11|>1 with SMALL residual that does NOT "
          "grow toward the patch => not model-mismatch-at-probes; |S11|>1 with "
          "residual GROWING toward the patch => model mismatch confirmed.")


if __name__ == "__main__":
    main()
