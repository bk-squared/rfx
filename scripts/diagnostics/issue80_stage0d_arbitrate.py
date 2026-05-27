# ruff: noqa: E741  (V, I are standard EM notation for voltage/current here)
"""Issue #80 Stage-0d: arbitrate architect (lumped feed) vs codex (single-cell curl).

Both panel agents converge that the multi-cell ∮H·dl loop is the problem (catastrophic
cancellation / enclosed reactive flux, worsens with dx). They diverge on the fix:
  - Codex: KEEP the MSL feed, replace I with a SINGLE-CELL discrete curl at the feed
    cell (∂Hz/∂y − ∂Hy/∂z), same passive formula as the lumped port.
  - Architect: switch to a lumped-port feed; warns codex's single-cell I is not a
    conjugate pair with the multi-cell V → passivity NOT guaranteed.

This arbitrates EMPIRICALLY and cheaply (no production change). Passivity criterion
|S11|≤1 ⟺ Re(Zin)≥0 is Z0-independent, so we just compare the SIGN of Re(V/I):
  - production loop I (raw_i1)  vs
  - single-cell curl I at the trace-center column, a few candidate feed cells,
both at dx=80µm and 50.8µm. If the single-cell curl flips Re(Zin)≥0 band-wide AND
holds/improves under refinement ⇒ codex Option C works (minimal fix). If it stays
negative or is cell-choice-sensitive ⇒ architect's concern validated ⇒ lumped feed.
"""
from __future__ import annotations

import warnings

import numpy as np

import rfx.sources.msl_port as _mslmod
from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.sources.msl_port import msl_loop_current as _real_loop

_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_F_MAX, _PORT_MARGIN, _STUB_LEN = 12e9, 2e-3, 6e-3
_CAP: dict = {}


def _cap_loop(hy_plane, hz_plane, *, j_lo, j_hi, k_trace_lo, k_trace_hi,
              dy_arr, dz_arr, direction):
    _CAP.update(hy=np.asarray(hy_plane), hz=np.asarray(hz_plane),
                j_lo=j_lo, j_hi=j_hi, k_lo=k_trace_lo, k_hi=k_trace_hi,
                dy=np.asarray(dy_arr, float), dz=np.asarray(dz_arr, float),
                direction=direction)
    return _real_loop(hy_plane, hz_plane, j_lo=j_lo, j_hi=j_hi,
                      k_trace_lo=k_trace_lo, k_trace_hi=k_trace_hi,
                      dy_arr=dy_arr, dz_arr=dz_arr, direction=direction)


def _single_cell_curl(jc, k, direction):
    """Codex Option C: discrete Ampere curl ∂Hz/∂y − ∂Hy/∂z at one feed cell."""
    hy, hz, dy, dz = _CAP["hy"], _CAP["hz"], _CAP["dy"], _CAP["dz"]
    i_f = ((hz[:, jc, k] - hz[:, jc - 1, k]) * float(dz[k])
           - (hy[:, jc, k] - hy[:, jc, k - 1]) * float(dy[jc]))
    if direction == "+x":
        i_f = -i_f
    return i_f


def _build_stub(dx: float) -> Simulation:
    lx = _PORT_MARGIN + _STUB_LEN + _PORT_MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = _H_SUB + 0.6e-3
    sim = Simulation(freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, _H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    tl, th = y_c - _W_TRACE / 2, y_c + _W_TRACE / 2
    x_end = _PORT_MARGIN + _STUB_LEN
    sim.add(Box((0, tl, _H_SUB), (x_end, th, _H_SUB + dx)), material="pec")
    sim.add(Box((x_end, tl, 0), (x_end + dx, th, _H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(_PORT_MARGIN, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1.6))
    return sim


def _neg_bins(V, I, sl):
    Z = V / I
    return int(np.sum(Z.real[sl] < 0)), float(np.median((np.abs(np.degrees(np.angle(Z))) - 90)[sl]))


def main() -> None:
    _mslmod.msl_loop_current = _cap_loop
    print("\n=== Stage-0d: loop vs single-cell-curl current (passivity sign test) ===")
    for dx in (80e-6, 50.8e-6):
        sim = _build_stub(dx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.preflight()
            dump = f"/tmp/issue80_s0d_{int(dx*1e6)}.npz"
            res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=60.0,
                                           raw_3probe_dump_path=dump)
        d = np.load(dump, allow_pickle=True)
        f = np.asarray(res.freqs, float)
        nf = len(f)
        sl = slice(nf // 8, -nf // 8)
        V = np.asarray(d["raw_v"])[0, 0, 0, :]
        I_loop = np.asarray(d["raw_i1"])[0, 0, :]
        jc = (_CAP["j_lo"] + _CAP["j_hi"]) // 2
        kt = _CAP["k_lo"]   # trace lower cell
        print(f"\n--- dx={dx*1e6:.1f}µm  (j_lo={_CAP['j_lo']},j_hi={_CAP['j_hi']},"
              f"k_trace_lo={kt}) ---")
        nb, em = _neg_bins(V, I_loop, sl)
        print(f"  production ∮H·dl loop : Re(Zin)<0 bins {nb}/{nf-2*(nf//8)}  ε_med={em:+.2f}°")
        # single-cell curl at candidate feed cells around the trace
        for k in (kt - 1, kt, kt + 1):
            I_sc = _single_cell_curl(jc, k, _CAP["direction"])
            nb_s, em_s = _neg_bins(V, I_sc, sl)
            print(f"  single-cell curl @ (jc={jc},k={k}): Re(Zin)<0 bins {nb_s}"
                  f"/{nf-2*(nf//8)}  ε_med={em_s:+.2f}°")

    print("\n=== READING ===")
    print("If single-cell curl → ~0 negative-Re bins at BOTH dx and is stable across "
          "k ⇒ codex Option C works (minimal MSL-preserving fix).")
    print("If it stays negative or flips with k/dx ⇒ architect is right "
          "(single-cell I ≠ conjugate pair with multi-cell V) ⇒ lumped-port feed.")


if __name__ == "__main__":
    main()
