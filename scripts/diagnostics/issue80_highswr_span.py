"""Issue #80: confirm the conditioning fix on a HIGH-SWR (reflecting) load.

The matched-line experiments showed (a) offset is irrelevant (near-field
falsified) and (b) probe SPAN controls the 2-wave fit conditioning (cond 34->4
as span grows). On the low-SWR matched line the ill-conditioning only scatters
|S11| within a bounded band (<1). The resonant patch reads |S11|>1 — the claim
is that the SAME ill-conditioning, acting on a high-SWR load where true
|gamma/alpha| ~ 1, pushes |S11| over 1, and that a wider span cures it.

This builds a CPU-scale OPEN-ENDED microstrip stub (|Gamma| ~ 1 across the band,
a resonant high-SWR load like the patch's input), dumps a dense 30-probe array,
then for a fixed start probe sweeps the 3-probe SPAN (stride 1..9) and reports
|S11| + lstsq condition number per frequency.

Decisive outcomes:
  - short span gives |S11| > 1 AND wider span brings it <= 1  => conditioning is
    the cause and a wider probe span (n_probe_spacing/n_probes) is the real fix
    (NOT n_probe_offset / Fix B).
  - |S11| stays > 1 even at wide span                          => the >1 is not
    pure conditioning; the 2-wave model itself breaks down at near-total
    reflection (then the honest gate is to flag, not to "fix" the number).

Run:  python scripts/diagnostics/issue80_highswr_span.py
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
PORT_MARGIN = 2e-3
OPEN_X = 14e-3          # trace open end (feed at PORT_MARGIN -> 12 mm stub)
DX = 80e-6
F_MAX = 6e9
LX = 18e-3
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3

N_PROBES = 30
N_OFFSET = 4
N_SPACING = 4


def eps_eff_hj(eps_r, w, h):
    u = w / h
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** -0.5


def cond_2wave(x, beta):
    xc = np.asarray(x) - x[0]
    a = np.stack([np.exp(-1j * beta * xc), np.exp(+1j * beta * xc)], axis=-1)
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-30))


def build_sim():
    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2.0
    # Open-ended trace: from x=0 to OPEN_X (open circuit at OPEN_X).
    sim.add(Box((0.0, y_c - W_TRACE / 2, H_SUB),
                (OPEN_X, y_c + W_TRACE / 2, H_SUB + DX)), material="pec")
    # Single driven port (+x); only port -> ONE FDTD run.
    sim.add_msl_port(
        position=(PORT_MARGIN, y_c, 0.0), width=W_TRACE, height=H_SUB,
        direction="+x", impedance=50.0,
        n_probes=N_PROBES, n_probe_offset=N_OFFSET, n_probe_spacing=N_SPACING,
    )
    return sim


def main():
    sim = build_sim()
    grid = sim._build_grid()
    dump = os.path.join(tempfile.gettempdir(), "issue80_highswr.npz")
    print(f"[run] FDTD open-ended stub (feed->open = {(OPEN_X-PORT_MARGIN)*1e3:.0f} mm) ...")
    res = sim.compute_msl_s_matrix(n_freqs=24, num_periods=16.0,
                                   raw_3probe_dump_path=dump)
    freqs = np.asarray(res.freqs, dtype=float)
    prod_s11 = np.abs(np.asarray(res.S)[0, 0, :])
    print(f"[run] done. freqs {freqs[0]/1e9:.2f}-{freqs[-1]/1e9:.2f} GHz")
    print(f"[run] production full-array |S11|: min {prod_s11.min():.3f} "
          f"max {prod_s11.max():.3f} (open stub -> expect ~1, >1 = non-physical)")

    d = np.load(dump, allow_pickle=True)
    raw_v = np.asarray(d["raw_v"])
    prod_beta = np.real(np.asarray(d["production_beta"])).astype(float)
    v_probes = raw_v[0, 0, :N_PROBES, :]

    mp0 = MSLPort(feed_x=PORT_MARGIN, y_lo=LY / 2 - W_TRACE / 2,
                  y_hi=LY / 2 + W_TRACE / 2, z_lo=0.0, z_hi=H_SUB,
                  direction="+x", impedance=50.0, excitation=None)
    x_all = np.asarray(msl_probe_x_coords_n(
        grid, mp0, n_probes=N_PROBES, n_offset_cells=N_OFFSET,
        n_spacing_cells=N_SPACING), dtype=float)

    beta0 = prod_beta if prod_beta.ndim else np.full(freqs.shape[0], float(prod_beta))
    eps_eff = eps_eff_hj(EPS_R, W_TRACE, H_SUB)
    C0 = 2.99792458e8
    i1 = np.ones(freqs.shape[0], dtype=complex)

    # Report max over frequency of |S11| for each span (passivity is a max test).
    print("\n=== HIGH-SWR open stub: |S11| vs probe SPAN (fixed start probe) ===")
    print("True |S11| ~ 1 (open). PASS gate = max|S11| <= 1.05.")
    print("stride span[mm] 2*b*span[rad]  max|S11|  mean|S11|  cond(median)")
    for s in range(1, 10):
        idx = [0, s, 2 * s]
        if idx[-1] >= N_PROBES:
            break
        xk = x_all[idx]
        span = xk[-1] - xk[0]
        vk = v_probes[idx, :].T
        out = extract_msl_nprobe(vk, xk, i1, beta0)
        s11 = np.abs(np.asarray(out["s11"]))
        conds = [cond_2wave(xk, beta0[i]) for i in range(freqs.shape[0])]
        ang = 2 * float(np.median(beta0)) * span
        print(f"{s:6d} {span*1e3:7.2f} {ang:12.3f}  {s11.max():8.4f}  "
              f"{s11.mean():9.4f}  {np.median(conds):11.1f}")

    # Also confirm offset-independence on this high-SWR load (sliding 3-probe
    # window at the production default span, stride 1).
    print("\n=== HIGH-SWR: |S11| vs OFFSET at fixed (short) span — should NOT "
          "improve with offset if near-field is irrelevant ===")
    print("off[cell]  max|S11|  mean|S11|")
    for k in range(0, N_PROBES - 2, 3):
        idx = [k, k + 1, k + 2]
        xk = x_all[idx]
        vk = v_probes[idx, :].T
        out = extract_msl_nprobe(vk, xk, i1, beta0)
        s11 = np.abs(np.asarray(out["s11"]))
        off = N_OFFSET + k * N_SPACING
        print(f"{off:8d}  {s11.max():8.4f}  {s11.mean():9.4f}")


if __name__ == "__main__":
    main()
