# ruff: noqa: E741  (V, I are standard EM notation for voltage/current here)
"""Issue #80 Stage-0b: localize the ~10-18° current phase excess that makes Re(Zin)<0.

Stage-0 falsified the Yee half-step (0.33°, negligible). The real error is a
systematic phase excess in the extracted loop current I vs the voltage V on a
REFLECTING load (dphi≈100° vs the ideal ±90° for a lossless short ⇒ Re(Zin)<0 ⇒
|S11|>1). This script localizes it WITHOUT changing production code.

Mechanism under test (architect/codex H2 refined): the closed Ampere loop
∮H·dl runs ONE cell outside the trace, so it encloses the conduction current PLUS
the displacement current jω∫D·dA in the surrounding ring. On a matched traveling
wave that term is benign; on a standing-wave reflector it is significant and ~90°
out of phase with conduction → it ROTATES I_loop. Prediction: the phase error
grows as the loop contour is enlarged (more enclosed displacement area). If instead
ε is invariant to contour size, displacement current is NOT the cause.

Method: monkeypatch msl_loop_current to capture (hy_plane, hz_plane, trace
indices) on the EXACT production geometry, then recompute I_loop at contour
margins m=1 (production), 2, 3 and report ε(f)=|angle(Zin)|-90° per margin, plus
a matched thru-line control (expect ε≈0).
"""
from __future__ import annotations

import warnings

import numpy as np

import rfx.sources.msl_port as _mslmod
from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.sources.msl_port import msl_loop_current as _real_loop

_EPS_R, _H_SUB, _W_TRACE, _DX = 3.66, 254e-6, 600e-6, 80e-6
_F_MAX, _PORT_MARGIN, _STUB_LEN = 12e9, 2e-3, 6e-3

_CAP: dict = {}


def _loop_at_margin(hy, hz, j_lo, j_hi, k_lo, k_hi, dy, dz, direction, m):
    """∮H·dl on a contour m cells outside the trace block (m=1 = production)."""
    dy = np.asarray(dy, float)
    dz = np.asarray(dz, float)
    js = slice(j_lo, j_hi + 1)
    ks = slice(k_lo, k_hi + 1)
    # expand contour by (m-1) extra cells on each side
    kb, kt = k_lo - m, k_hi + (m - 1)
    jl, jr = j_lo - m, j_hi + (m - 1)
    js_w = slice(jl + 1, jr + 1) if False else js  # keep horizontal span = trace width
    bottom = (hy[:, js_w, kb] * dy[js_w]).sum(axis=1)
    top = (hy[:, js_w, kt] * dy[js_w]).sum(axis=1)
    right = (hz[:, jr, ks] * dz[ks]).sum(axis=1)
    left = (hz[:, jl, ks] * dz[ks]).sum(axis=1)
    i_loop = bottom - top + right - left
    if direction == "+x":
        i_loop = -i_loop
    return i_loop, bottom, top, right, left


def _capturing_loop(hy_plane, hz_plane, *, j_lo, j_hi, k_trace_lo, k_trace_hi,
                    dy_arr, dz_arr, direction):
    ny, nz = int(hy_plane.shape[1]), int(hy_plane.shape[2])
    _CAP.update(hy=np.asarray(hy_plane), hz=np.asarray(hz_plane),
                j_lo=j_lo, j_hi=j_hi, k_lo=k_trace_lo, k_hi=k_trace_hi,
                dy=np.asarray(dy_arr, float), dz=np.asarray(dz_arr, float),
                direction=direction, ny=ny, nz=nz)
    return _real_loop(hy_plane, hz_plane, j_lo=j_lo, j_hi=j_hi,
                      k_trace_lo=k_trace_lo, k_trace_hi=k_trace_hi,
                      dy_arr=dy_arr, dz_arr=dz_arr, direction=direction)


def _build_stub() -> Simulation:
    lx = _PORT_MARGIN + _STUB_LEN + _PORT_MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * _DX)
    lz = _H_SUB + 0.6e-3
    sim = Simulation(freq_max=_F_MAX, domain=(lx, ly, lz), dx=_DX,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, _H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    tl, th = y_c - _W_TRACE / 2, y_c + _W_TRACE / 2
    x_end = _PORT_MARGIN + _STUB_LEN
    sim.add(Box((0, tl, _H_SUB), (x_end, th, _H_SUB + _DX)), material="pec")
    sim.add(Box((x_end, tl, 0), (x_end + _DX, th, _H_SUB + _DX)), material="pec")
    sim.add_msl_port(position=(_PORT_MARGIN, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1.6))
    return sim


def _eps_excess(V, I):
    """Phase error magnitude: |angle(Zin)| - 90° (deg). >0 ⇒ Re(Zin)<0 ⇒ |S11|>1."""
    ang = np.degrees(np.angle(V / I))
    return np.abs(ang) - 90.0


def main() -> None:
    _mslmod.msl_loop_current = _capturing_loop  # patched at source (imported locally in _sparams)
    sim = _build_stub()
    sim.preflight()
    dump = "/tmp/issue80_stage0b.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_freqs=121, num_periods=60.0,
                                       raw_3probe_dump_path=dump)
    d = np.load(dump, allow_pickle=True)
    f = np.asarray(res.freqs, float)
    V = np.asarray(d["raw_v"])[0, 0, 0, :]
    S11 = np.asarray(res.S)[0, 0, :]

    print("\n=== Stage-0b: loop-contour / displacement-current witness ===")
    print(f"raw production max|S11| = {np.max(np.abs(S11)):.3f}  "
          f"({int(np.sum(np.abs(S11)>1.001))}/{len(f)} bins >1)")

    legs = {}
    for m in (1, 2, 3):
        I_m, b, t, r, l = _loop_at_margin(
            _CAP["hy"], _CAP["hz"], _CAP["j_lo"], _CAP["j_hi"],
            _CAP["k_lo"], _CAP["k_hi"], _CAP["dy"], _CAP["dz"],
            _CAP["direction"], m)
        legs[m] = (I_m, b, t, r, l)
        eps = _eps_excess(V, I_m)
        # band-interior median to avoid edge noise
        sl = slice(len(f)//8, -len(f)//8)
        print(f"margin m={m}: median ε(|∠Zin|-90°) = {np.median(eps[sl]):+6.2f}°  "
              f"max = {np.max(eps[sl]):+6.2f}°")

    # leg magnitudes/phases at a mid-band bin
    i_mid = len(f)//2
    I1, b, t, r, l = legs[1]
    print(f"\n--- loop legs @ f={f[i_mid]/1e9:.2f} GHz (m=1) ---")
    for nm, leg in (("bottom Hy", b), ("top Hy", t), ("right Hz", r), ("left Hz", l)):
        print(f"  {nm:9}: |{abs(leg[i_mid]):.3e}|  ∠{np.degrees(np.angle(leg[i_mid])):7.2f}°")
    print(f"  I_loop   : |{abs(I1[i_mid]):.3e}|  ∠{np.degrees(np.angle(I1[i_mid])):7.2f}°")
    print(f"  V        : |{abs(V[i_mid]):.3e}|  ∠{np.degrees(np.angle(V[i_mid])):7.2f}°")

    # interpretation hint
    e1 = np.median(_eps_excess(V, legs[1][0])[len(f)//8:-len(f)//8])
    e3 = np.median(_eps_excess(V, legs[3][0])[len(f)//8:-len(f)//8])
    print("\n=== READING ===")
    print(f"ε(m=1)={e1:+.2f}°  ε(m=3)={e3:+.2f}°  Δ={e3-e1:+.2f}°")
    if abs(e3 - e1) > 2.0:
        print("ε GROWS with contour size ⇒ DISPLACEMENT-CURRENT enclosed by the "
              "loop is the dominant phase error (architectural: shrink contour / "
              "subtract jω∫D·dA / use conduction current directly).")
    else:
        print("ε ~INVARIANT to contour size ⇒ NOT displacement current. Inspect "
              "leg balance/sign (H3) or V-integration convention next.")


if __name__ == "__main__":
    main()
