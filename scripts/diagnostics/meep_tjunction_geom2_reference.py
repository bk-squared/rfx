"""Far-port MEEP reference for T-junction GEOMETRY 2 (W=0.036 m).

Mirrors scripts/diagnostics/meep_tjunction_farport_reference.py (geometry 1,
W=0.04) for the second junction geometry demanded by the broad-claim breadth
bars: same domain 0.30x0.24, same far-port discipline, guide width W=0.036,
band 5.2-7.3 GHz. FLUX-based (MPB-free; MEEP MPB rejects PEC walls); per drive
it runs a hand-built straight-guide reference (P_inc) then the junction.

Geometry (air channels): horizontal arm y in [0.102, 0.138] all x; vertical
stub x in [0.132, 0.168], y in [0.138, 0.24] (open top).

Run (numpy<2 venv + system meep):
  PYTHONPATH=/usr/lib/python3/dist-packages /tmp/meepenv/bin/python \
      scripts/diagnostics/meep_tjunction_geom2_reference.py --drive D --res R
"""
import argparse
import os

import numpy as np
import meep as mp

C0 = 299792458.0
LX, LY, W, DPML = 0.30, 0.24, 0.036, 0.02
Y_LO, Y_HI = 0.102, 0.138       # horizontal channel
X_LO, X_HI = 0.132, 0.168       # vertical stub
Y_MID = 0.5 * (Y_LO + Y_HI)     # 0.120
X_MID = 0.5 * (X_LO + X_HI)     # 0.150

PORTS = [
    dict(name="left",  center=(0.05, Y_MID), axis="x", outsign=-1),
    dict(name="right", center=(0.25, Y_MID), axis="x", outsign=+1),
    dict(name="top",   center=(X_MID, 0.19), axis="y", outsign=+1),
]


def _v(x, y):
    return mp.Vector3(x - LX / 2, y - LY / 2)


def _blk(cx, cy, sx, sy):
    return mp.Block(center=_v(cx, cy), size=mp.Vector3(sx, sy, mp.inf), material=mp.metal)


def _tjunction_blocks():
    return [
        _blk(LX / 2, Y_LO / 2, LX, Y_LO),                                  # bottom
        _blk(X_LO / 2, (Y_HI + LY) / 2, X_LO, LY - Y_HI),                  # top-left
        _blk((X_HI + LX) / 2, (Y_HI + LY) / 2, LX - X_HI, LY - Y_HI),      # top-right
    ]


def _straight_blocks(axis):
    if axis == "x":  # straight horizontal channel y in [Y_LO, Y_HI]
        return [_blk(LX / 2, Y_LO / 2, LX, Y_LO),
                _blk(LX / 2, (Y_HI + LY) / 2, LX, LY - Y_HI)]
    # straight vertical channel x in [X_LO, X_HI]
    return [_blk(X_LO / 2, LY / 2, X_LO, LY),
            _blk((X_HI + LX) / 2, LY / 2, LX - X_HI, LY)]


def _src(p, fcen, df):
    cx, cy = p["center"]
    setback = 0.016
    if p["axis"] == "x":
        sx, sy = cx + p["outsign"] * setback, cy
        size = mp.Vector3(0, W, mp.inf)
        amp = lambda v: np.sin(np.pi * (v.y + W / 2) / W)
    else:
        sx, sy = cx, cy + p["outsign"] * setback
        size = mp.Vector3(W, 0, mp.inf)
        amp = lambda v: np.sin(np.pi * (v.x + W / 2) / W)
    return mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                     center=_v(sx, sy), size=size, amp_func=amp)


def _flux_inward(sim, p, fcen, df, nf):
    cx, cy = p["center"]
    if p["axis"] == "x":
        size, direction = mp.Vector3(0, W, mp.inf), mp.X
    else:
        size, direction = mp.Vector3(W, 0, mp.inf), mp.Y
    fr = mp.FluxRegion(center=_v(cx, cy), size=size, direction=direction,
                       weight=float(-p["outsign"]))
    return sim.add_flux(fcen, df, nf, fr)


def run_case(geometry, drive, fcen, df, nf, res):
    p = PORTS[drive]
    sim = mp.Simulation(cell_size=mp.Vector3(LX, LY, 0), resolution=res,
                        geometry=geometry, sources=[_src(p, fcen, df)],
                        boundary_layers=[mp.PML(DPML)], dimensions=2,
                        force_complex_fields=True)
    fluxes = [_flux_inward(sim, q, fcen, df, nf) for q in PORTS]
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, _v(*p["center"]), 1e-6))
    return (np.array(mp.get_flux_freqs(fluxes[0])),
            np.array([np.array(mp.get_fluxes(f)) for f in fluxes]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=int, required=True)
    ap.add_argument("--res", type=int, default=500)
    ap.add_argument("--nfreq", type=int, default=11)
    ap.add_argument("--fmin", type=float, default=5.2e9)
    ap.add_argument("--fmax", type=float, default=7.3e9)
    args = ap.parse_args()
    fmin, fmax = args.fmin / C0, args.fmax / C0
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin)
    p = PORTS[args.drive]
    fr, Pn = run_case(_straight_blocks(p["axis"]), args.drive, fcen, df, args.nfreq, args.res)
    P_inc = Pn[args.drive]
    fr2, Pd = run_case(_tjunction_blocks(), args.drive, fcen, df, args.nfreq, args.res)
    nf = len(fr)
    col2 = np.zeros((3, nf))
    for j in range(3):
        col2[j] = np.clip((P_inc - Pd[j]) / np.abs(P_inc), 0, None) if j == args.drive \
                  else np.clip(-Pd[j] / np.abs(P_inc), 0, None)
    col = np.sqrt(col2)
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_artifacts")
    os.makedirs(art, exist_ok=True)
    out = os.path.join(art, f"meep_tjunction_geom2_r{args.res}_drive{args.drive}.npz")
    np.savez(out, freqs_hz=fr2 * C0, col=col, P_inc=P_inc, drive=args.drive, W=W)
    print(f"[meep-geom2] drive={args.drive} res={args.res} -> {out}")
    print(f"[meep-geom2] |S[:,{args.drive}]| band-mean = {np.mean(col, axis=1)} "
          f"colsum={np.mean(np.sum(col2, axis=0)):.4f}")


if __name__ == "__main__":
    main()
