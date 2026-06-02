"""External MEEP reference for the H-plane T-junction |S|-matrix (gates the
broad-E5 promotion of rectangular_waveguide_port — no closed-form truth exists).

FLUX-based, MPB-free: MEEP's MPB eigenmode solver rejects PEC (mp.metal) walls
("invalid dielectric function for MPB"), so instead of get_eigenmode_coefficients
we use power flux (sim.add_flux) through each port plane. In the single-mode band
(5-7 GHz; TE10 cutoff C0/2W=3.75 GHz, TE20 onset C0/W=7.5 GHz) the port flux IS
the TE10 modal power, so |S_ji|^2 = P_j / P_inc (magnitudes only — exactly what
the envelope gate compares). A straight-guide normalization run gives P_inc.

Geometry matches tests/test_simulation.py::test_extract_waveguide_s_matrix_mixed_normal_branch_reciprocity:
2D H-plane T (Ez), guide width W=0.04 m. Air channels:
  horizontal arm y in [0.04,0.08], x in [0,0.12]  (left, right ports)
  vertical stub  x in [0.04,0.08], y in [0.08,0.12] (top port)

Per drive port i:
  P_inc  = flux at port i in a STRAIGHT guide of width W (no junction).
  device = flux at all 3 ports, monitors oriented INWARD (toward junction):
    driven i:  flux_inward = P_inc - P_refl  -> |S_ii|^2 = (P_inc-flux_i)/P_inc
    other  j:  transmission leaves OUTWARD -> P_trans_j = -flux_j (=|flux_j|)
                                              |S_ji|^2 = P_trans_j / P_inc
Validation gates (no external truth needed for these): reciprocity |S_ij|~|S_ji|,
passivity sum_i|S_ij|^2 <= 1.

Run (numpy<2 venv + system meep), per drive:
  PYTHONPATH=/usr/lib/python3/dist-packages /tmp/meepenv/bin/python \
      scripts/diagnostics/meep_tjunction_reference.py --drive 0
"""
import argparse, os
import numpy as np
import meep as mp

C0 = 299792458.0
LX, LY = 0.12, 0.12
W = 0.04
DPML = 0.02

# air-channel bounds
H_ARM = (0.04, 0.08)   # horizontal arm y-range
V_STUB = (0.04, 0.08)  # vertical stub x-range

# Port planes: (center, size, axis, outsign). Monitors oriented INWARD below.
PORTS = [
    dict(name="left",  center=(0.030, 0.06), axis="x", outsign=-1),
    dict(name="right", center=(0.090, 0.06), axis="x", outsign=+1),
    dict(name="top",   center=(0.06, 0.095), axis="y", outsign=+1),
]


def _v(x, y):
    return mp.Vector3(x - LX / 2, y - LY / 2)


def _tjunction_blocks():
    return [
        mp.Block(center=_v(0.06, 0.02), size=mp.Vector3(0.12, 0.04, mp.inf), material=mp.metal),
        mp.Block(center=_v(0.02, 0.10), size=mp.Vector3(0.04, 0.04, mp.inf), material=mp.metal),
        mp.Block(center=_v(0.10, 0.10), size=mp.Vector3(0.04, 0.04, mp.inf), material=mp.metal),
    ]


def _straight_blocks(axis):
    # Two PEC half-planes bounding a width-W channel along `axis` (no junction).
    if axis == "x":  # horizontal channel y in [0.04,0.08]
        return [mp.Block(center=_v(0.06, 0.02), size=mp.Vector3(0.12, 0.04, mp.inf), material=mp.metal),
                mp.Block(center=_v(0.06, 0.10), size=mp.Vector3(0.12, 0.04, mp.inf), material=mp.metal)]
    else:            # vertical channel x in [0.04,0.08]
        return [mp.Block(center=_v(0.02, 0.06), size=mp.Vector3(0.04, 0.12, mp.inf), material=mp.metal),
                mp.Block(center=_v(0.10, 0.06), size=mp.Vector3(0.04, 0.12, mp.inf), material=mp.metal)]


def _src(p, fcen, df):
    cx, cy = p["center"]
    # bidirectional Ez line source with the TE10 sin profile, set just OUTWARD
    # of the monitor; the inward half is the incident TE10, outward half -> PML.
    setback = 0.016
    if p["axis"] == "x":
        sx, sy = cx + p["outsign"] * setback, cy
        size = mp.Vector3(0, W, mp.inf)
        def amp(v):  # v is relative to source center; transverse coord = y
            return np.sin(np.pi * (v.y + W / 2) / W)
    else:
        sx, sy = cx, cy + p["outsign"] * setback
        size = mp.Vector3(W, 0, mp.inf)
        def amp(v):
            return np.sin(np.pi * (v.x + W / 2) / W)
    return mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                     center=_v(sx, sy), size=size, amp_func=amp)


def _flux_inward(sim, p, fcen, df, nf):
    # FluxRegion direction = INWARD (toward junction) = -outsign along axis.
    cx, cy = p["center"]
    if p["axis"] == "x":
        size = mp.Vector3(0, W, mp.inf)
        direction = mp.X
    else:
        size = mp.Vector3(W, 0, mp.inf)
        direction = mp.Y
    fr = mp.FluxRegion(center=_v(cx, cy), size=size, direction=direction,
                       weight=float(-p["outsign"]))  # +weight => inward
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
    freqs = np.array(mp.get_flux_freqs(fluxes[0]))
    P = np.array([np.array(mp.get_fluxes(f)) for f in fluxes])  # (3, nf), inward
    return freqs, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=int, required=True)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--nfreq", type=int, default=11)
    ap.add_argument("--fmin", type=float, default=5.0e9)
    ap.add_argument("--fmax", type=float, default=7.0e9)
    args = ap.parse_args()

    fmin, fmax = args.fmin / C0, args.fmax / C0
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin)
    p = PORTS[args.drive]

    # 1) normalization: straight guide along the driven axis -> P_inc (inward).
    fr, Pn = run_case(_straight_blocks(p["axis"]), args.drive, fcen, df, args.nfreq, args.res)
    P_inc = Pn[args.drive]  # inward flux at the driven port, no junction => incident
    # 2) device run.
    fr2, Pd = run_case(_tjunction_blocks(), args.drive, fcen, df, args.nfreq, args.res)

    nf = len(fr)
    col2 = np.zeros((3, nf))  # |S_ji|^2
    for j in range(3):
        if j == args.drive:
            col2[j] = np.clip((P_inc - Pd[j]) / np.abs(P_inc), 0, None)   # reflected
        else:
            col2[j] = np.clip(-Pd[j] / np.abs(P_inc), 0, None)            # transmitted (outward)
    col = np.sqrt(col2)

    art = os.path.join(os.path.dirname(__file__), "_artifacts")
    os.makedirs(art, exist_ok=True)
    out = os.path.join(art, f"meep_tjunction_drive{args.drive}.npz")
    np.savez(out, freqs_hz=fr2 * C0, col=col, P_inc=P_inc, drive=args.drive)
    print(f"[meep] drive={args.drive} -> {out}")
    print(f"[meep] P_inc band-mean = {np.mean(P_inc):.4e}")
    print(f"[meep] |S[:, {args.drive}]| band-mean = {np.mean(col, axis=1)}")
    print(f"[meep] column power sum band-mean = {np.mean(np.sum(col2, axis=0)):.4f} (<=1 lossless)")


if __name__ == "__main__":
    main()
