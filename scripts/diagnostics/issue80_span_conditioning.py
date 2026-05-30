"""Issue #80: is the residual |S11| driven by probe SPAN (fit conditioning),
not offset? Re-fits the EXISTING matched-line dump at increasing spans.

Background
----------
issue80_nearfield_offset_sweep.py falsified the near-field hypothesis: on a
matched line |S11| is FLAT vs offset. The remaining mechanism is the 2-wave
least-squares conditioning: fitting V = a e^{-jbx} + g e^{+jbx} over a probe
span that is short in wavelengths makes the two basis columns near-collinear
(angle ~ 2*beta*span), so a/g have huge variance and |S11|=|g/a| is biased
upward (and can exceed 1 in the high-SWR resonant regime).

This script reuses /tmp/issue80_nf_sweep.npz (dense 30-probe array, spacing 4
cells). It forms 3-probe windows at a FIXED start probe but increasing stride
s=1..8 (span = 8s cells), so OFFSET is held ~fixed and only SPAN grows. For each
span it reports |S11| and the lstsq matrix condition number. If |S11| drops and
cond() falls as span grows -> conditioning is the lever (span/spacing), not
offset (near-field).

Run:  python scripts/diagnostics/issue80_span_conditioning.py
"""
from __future__ import annotations

import json
import math

import numpy as np

from rfx.api import Simulation  # noqa: F401  (ensures rfx import side-effects)
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

import os, tempfile
DUMP = os.path.join(tempfile.gettempdir(), "issue80_nf_sweep.npz")
DX = 80e-6
EPS_R = 3.66
W_TRACE = 600e-6
H_SUB = 254e-6
PORT_MARGIN = 2e-3
L_LINE = 25e-3
N_PROBES = 30
N_OFFSET = 4
N_SPACING = 4


def eps_eff_hj(eps_r, w, h):
    u = w / h
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** -0.5


def cond_2wave(x, beta):
    """Condition number of the [e^{-jbx}, e^{+jbx}] lstsq design matrix."""
    xc = np.asarray(x) - x[0]
    a = np.stack([np.exp(-1j * beta * xc), np.exp(+1j * beta * xc)], axis=-1)
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-30))


def main():
    d = np.load(DUMP, allow_pickle=True)
    raw_v = np.asarray(d["raw_v"])           # (n_driven, n_ports, n_probes, n_freqs)
    prod_beta = np.real(np.asarray(d["production_beta"])).astype(float)
    freqs = np.asarray(d["freqs_hz"], dtype=float)
    meta = json.loads(str(d["metadata_json"]))

    v_probes = raw_v[0, 0, :N_PROBES, :]     # (n_probes, n_freqs)

    LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
    mp0 = MSLPort(feed_x=PORT_MARGIN, y_lo=LY / 2 - W_TRACE / 2,
                  y_hi=LY / 2 + W_TRACE / 2, z_lo=0.0, z_hi=H_SUB,
                  direction="+x", impedance=50.0, excitation=None)
    x_all = np.asarray(msl_probe_x_coords_n(
        Simulation(freq_max=5e9, domain=(L_LINE + 2 * PORT_MARGIN, LY, H_SUB + 1.5e-3),
                   dx=DX)._build_grid(),
        mp0, n_probes=N_PROBES, n_offset_cells=N_OFFSET, n_spacing_cells=N_SPACING,
    ), dtype=float)

    beta0 = prod_beta
    if beta0.ndim == 0:
        beta0 = np.full(freqs.shape[0], float(beta0))

    eps_eff = eps_eff_hj(EPS_R, W_TRACE, H_SUB)
    C0 = 2.99792458e8

    eval_ghz = [3.0, 4.0, 5.0]
    f_idx = [int(np.argmin(np.abs(freqs - g * 1e9))) for g in eval_ghz]
    i1 = np.ones(freqs.shape[0], dtype=complex)

    print("Matched line — |S11| vs probe SPAN at fixed start probe (offset ~4 cells).")
    print("True |S11| ~ 0; lower is better. span[cell]=8*stride.\n")
    print("stride span[cell] span[mm] " + " ".join(
        f" |S11|@{freqs[i]/1e9:.1f}G cond@{freqs[i]/1e9:.1f}G" for i in f_idx))

    for s in range(1, 9):
        idx = [0, s, 2 * s]
        if idx[-1] >= N_PROBES:
            break
        xk = x_all[idx]
        span_cells = (xk[-1] - xk[0]) / DX
        vk = v_probes[idx, :].T              # (n_freqs, 3)
        out = extract_msl_nprobe(vk, xk, i1, beta0)
        s11 = np.abs(np.asarray(out["s11"]))
        cells_per_lam = []
        row = f"{s:6d} {span_cells:9.0f} {span_cells*DX*1e3:7.2f}  "
        for i in f_idx:
            lam = C0 / freqs[i] / math.sqrt(eps_eff)
            c = cond_2wave(xk, beta0[i])
            row += f" {s11[i]:9.4f} {c:9.1f}"
        print(row)

    # Reference: span/lambda and the rad-angle the basis columns subtend.
    print("\nbeta*span (rad) — basis columns separable when this is O(1), "
          "collinear when << 1:")
    for s in [1, 4, 8]:
        idx = [0, s, 2 * s]
        if idx[-1] >= N_PROBES:
            continue
        span = x_all[idx][-1] - x_all[idx][0]
        for i in f_idx:
            ang = 2 * beta0[i] * span
            print(f"  stride {s}, {freqs[i]/1e9:.1f} GHz: span={span*1e3:.2f}mm "
                  f"2*beta*span={ang:.3f} rad  span/lam={span/(C0/freqs[i]/math.sqrt(eps_eff)):.3f}")

    print(f"\nproduction full 30-probe |S11| band-mean = {float(np.mean(np.abs(np.asarray(d['production_smatrix'])[0,0,:]))):.4f}")
    print(f"default offset used in dump = {meta['port_definitions'][0]['n_probe_offset']} cells, "
          f"spacing = {meta['port_definitions'][0]['n_probe_spacing']} cells, "
          f"n_probes = {meta['port_definitions'][0]['n_probes']}")


if __name__ == "__main__":
    main()
