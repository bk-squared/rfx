# ruff: noqa: E741  (V, I are standard EM notation for voltage/current here)
"""Issue #80 Stage-0c: is the ~7° I-vs-V phase excess an EXTRACTOR flaw or a
coarse-mesh artifact?

Stage-0/0b ruled out the Yee half-step and the enclosed displacement current, and
quantified a contour-invariant ~+7° median (up to +19°) phase excess in I_loop vs V
on a shorted stub ⇒ Re(Zin)<0 ⇒ |S11|>1. The stub was coarsely resolved
(h_sub≈3 cells, trace=1 cell), and the bottom/top Hy legs came out near-equal
(quasi-TEM possibly under-resolved). DECISIVE TEST: refine dx. If ε → 0 with finer
mesh, the |S11|>1 is a numerical/resolution artifact (extractor is fine in the
converged limit). If ε persists, the loop-current V·I split is a genuine structural
flaw on reflectors (⇒ architectural fix / lumped-port feed).
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_F_MAX, _PORT_MARGIN, _STUB_LEN = 12e9, 2e-3, 6e-3


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


def main() -> None:
    print("\n=== Stage-0c: phase-excess vs mesh resolution ===")
    print(f"{'dx[µm]':>7} {'sub_cells':>9} {'max|S11|':>9} {'bins>1':>7} "
          f"{'ε_med[°]':>9} {'ε_max[°]':>9}")
    for dx in (80e-6, 50.8e-6):   # h_sub/3.18, /5  (drop /7 — 24x CPU cost)
        sim = _build_stub(dx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.preflight()
            dump = f"/tmp/issue80_s0c_{int(dx*1e6)}.npz"
            res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=60.0,
                                           raw_3probe_dump_path=dump)
        d = np.load(dump, allow_pickle=True)
        f = np.asarray(res.freqs, float)
        V = np.asarray(d["raw_v"])[0, 0, 0, :]
        I = np.asarray(d["raw_i1"])[0, 0, :]
        S11 = np.asarray(res.S)[0, 0, :]
        eps = np.abs(np.degrees(np.angle(V / I))) - 90.0
        sl = slice(len(f) // 8, -len(f) // 8)
        print(f"{dx*1e6:7.1f} {_H_SUB/dx:9.2f} {np.max(np.abs(S11)):9.3f} "
              f"{int(np.sum(np.abs(S11) > 1.001)):7d} "
              f"{np.median(eps[sl]):9.2f} {np.max(eps[sl]):9.2f}")

    print("\n=== READING ===")
    print("ε shrinks ≥~2x toward finer dx ⇒ |S11|>1 is a RESOLUTION/numerical "
          "artifact (extractor OK in converged limit; patch needs finer mesh).")
    print("ε ~flat across dx ⇒ STRUCTURAL extractor flaw on reflectors "
          "(architectural fix: lumped-port feed / different I extraction).")


if __name__ == "__main__":
    main()
