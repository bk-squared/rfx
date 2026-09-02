"""Meep leg for crossval 26 (oblique slab Fresnel): one arm -> one JSON.

cv04's slab (eps = 4, 1 cm) in Meep 2-D with a Bloch-periodic transverse
axis: ``k_point = (0, k_y a / 2 pi, 0)`` with the SAME fixed
``k_y = k0(f0) sin theta0`` as the rfx arm (``oblique_fresnel.meep_k_point``;
Meep's k_point is in units of 2 pi / distance and its Bloch phase is
exp(i k . r)), a line source whose amplitude carries exp(i 2 pi k_meep y)
(Meep's documented oblique-planewave construction), and a GaussianSource
whose spectrum equals the rfx arm's (``meep_fwidth_for``). TE arms drive
``Ez`` (E perpendicular to the plane of incidence); TM arms drive ``Hz`` --
the REAL p-polarization on the eps-slab, which rfx reaches only through the
eps <-> mu duality. R, T come from cv04's two-run flux subtraction; every bin
is at the realized angle theta(f) of that k_y, as in the rfx arm.

Before any FDTD the k_point is mapped back to k_y and the realized angle at
f0 must equal the declared theta0 to 1e-9 (pre-check); with ``--falsifier
k_2pi`` (F4: k_point in rad/a) the check is RECORDED as failed and the run
proceeds so the E4 gates see real Meep output of the wrong convention.

The block is drawn with its centre shifted by +0.5 pixel (cv23 note section
14.3: a Block of nominal width N a/res centred on a node holds N + 1
integer-position E nodes without eps_averaging; off-centre by half a pixel
it holds N).

This script never imports rfx; it needs numpy + meep + the comparator.

Exit codes: 0 JSON written; 1 pre-check failed (non-falsifier) or non-finite
output; 2 Meep not importable.

Run (conda-forge pymeep env):
  python scripts/crossval/meep_cv26_oblique_slab.py --arm te_45 --out-dir <dir>
  python scripts/crossval/meep_cv26_oblique_slab.py --arm te_45 --falsifier k_2pi --out-dir <dir>
"""

from __future__ import annotations

import argparse
import cmath
import datetime as _dt
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "validation", "crossval", "comparators"))
import oblique_fresnel as O  # noqa: E402

C0 = O.C0
PRECHECK_TOL = 1e-9


def precheck(k_point, a_m: float, theta0_deg: float, f0_hz: float) -> dict:
    """The k_point mapped back: k_y = 2 pi k_meep / a; theta(f0) must be theta0."""
    ky = O.ky_from_meep_k_point(k_point, a_m)
    th = O.realized_theta_rad(np.array([f0_hz]), ky)[0]
    ok = bool(np.isfinite(th)) and abs(math.degrees(th) - theta0_deg) <= PRECHECK_TOL * max(1.0, theta0_deg)
    ky_decl = O.ky_from(f0_hz, theta0_deg)
    return {"ky_from_k_point_rad_m": ky, "ky_declared_rad_m": ky_decl,
            "theta_at_f0_deg": (float(math.degrees(th)) if np.isfinite(th) else None),
            "theta0_declared_deg": theta0_deg, "rel_err_ky": abs(ky - ky_decl) / max(ky_decl, 1e-30),
            "tol": PRECHECK_TOL, "passed": ok}


def run_two_pass(mp, *, arm: str, k_point, a_m: float, resolution: int, courant: float, nfreq: int,
                 fcen: float, fwidth: float, decay: float, center_offset_cells: float, pol: str) -> dict:
    cells = O.rig_cells(O.NX_INTERIOR)
    sx = O.NX_INTERIOR * O.DX_M / a_m
    sy = O.NY_CELLS * O.DX_M / a_m
    dpml = O.N_CPML * O.DX_M / a_m
    d_slab = O.D_SLAB_M / a_m
    off = (cells["probe_refl"] - cells["slab_lo"]) * O.DX_M / a_m     # -30 cells
    refl_x = -d_slab / 2 + off
    trans_x = d_slab / 2 - off
    src_x = -d_slab / 2 + (cells["x_lo"] - cells["slab_lo"]) * O.DX_M / a_m
    comp = mp.Ez if pol == "te" else mp.Hz
    kvec = mp.Vector3(*k_point)

    def amp(p):
        return cmath.exp(1j * 2 * math.pi * kvec.dot(p))

    src = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=fwidth), component=comp,
                     center=mp.Vector3(src_x, 0), size=mp.Vector3(0, sy), amp_func=amp)]
    pml = [mp.PML(dpml, direction=mp.X)]
    cell = mp.Vector3(sx, sy, 0)
    refl_fr = mp.FluxRegion(center=mp.Vector3(refl_x, 0), size=mp.Vector3(0, sy))
    trans_fr = mp.FluxRegion(center=mp.Vector3(trans_x, 0), size=mp.Vector3(0, sy))
    stop = mp.stop_when_fields_decayed(50, comp, mp.Vector3(trans_x, 0), decay)

    t0 = time.time()
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=src, resolution=resolution,
                        k_point=kvec, Courant=courant, eps_averaging=False)
    refl = sim.add_flux(fcen, fwidth, nfreq, refl_fr)
    trans = sim.add_flux(fcen, fwidth, nfreq, trans_fr)
    sim.run(until_after_sources=stop)
    refl_data = sim.get_flux_data(refl)
    inc_flux = np.array(mp.get_fluxes(trans))
    freqs = np.array(mp.get_flux_freqs(refl))
    t_ref = time.time() - t0
    dt_meep_s = courant * (a_m / resolution) / C0
    sim.reset_meep()

    t0 = time.time()
    geometry = [mp.Block(center=mp.Vector3(center_offset_cells * a_m / resolution / a_m, 0),
                         size=mp.Vector3(d_slab, mp.inf, mp.inf), material=mp.Medium(epsilon=O.EPS_R_SLAB))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, geometry=geometry, sources=src,
                        resolution=resolution, k_point=kvec, Courant=courant, eps_averaging=False)
    refl = sim.add_flux(fcen, fwidth, nfreq, refl_fr)
    trans = sim.add_flux(fcen, fwidth, nfreq, trans_fr)
    sim.load_minus_flux_data(refl, refl_data)
    sim.run(until_after_sources=stop)
    R = -np.array(mp.get_fluxes(refl)) / inc_flux
    T = np.array(mp.get_fluxes(trans)) / inc_flux
    t_slab = time.time() - t0
    return {"R": R, "T": T, "freqs_hz": freqs * C0 / a_m, "t_ref_s": t_ref, "t_slab_s": t_slab,
            "dt_meep_s": dt_meep_s, "fcen": fcen, "fwidth": fwidth, "sx": sx, "sy": sy, "dpml": dpml,
            "refl_x": refl_x, "trans_x": trans_x, "src_x": src_x, "d_slab": d_slab}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(O.MEEP_ARMS))
    ap.add_argument("--falsifier", choices=sorted(O.MEEP_FALSIFIERS), default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=O.MEEP_PRIMARY_RESOLUTION, help="px/cm")
    ap.add_argument("--courant", type=float, default=O.MEEP_COURANT)
    ap.add_argument("--nfreq", type=int, default=400)
    ap.add_argument("--decay", type=float, default=1e-3)
    ap.add_argument("--center-offset-cells", type=float, default=O.MEEP_CENTER_OFFSET_CELLS)
    ap.add_argument("--tag", default=None, help="write meep_<arm>__<tag>.json (ladder rungs / diagnostics)")
    a = ap.parse_args(argv)
    try:
        import meep as mp
    except Exception as exc:
        print(f"[SKIP] meep not importable: {type(exc).__name__}: {exc} (exit 2)")
        return 2
    if a.falsifier and a.arm != O.MEEP_FALSIFIER_ARM:
        print(f"falsifiers are declared on the {O.MEEP_FALSIFIER_ARM} arm only")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    spec = O.arm_spec(a.arm)
    a_m = O.MEEP_A_M
    f0 = spec["f0_hz"]
    k_point = O.meep_k_point(f0, spec["theta0_deg"], a_m)
    if a.falsifier == "k_2pi":
        k_point = O.meep_k_point_wrong_2pi(f0, spec["theta0_deg"], a_m)
        print(f"FALSIFIER k_2pi: {O.MEEP_FALSIFIERS['k_2pi']}")
    pre = precheck(k_point, a_m, spec["theta0_deg"], f0)
    print(f"arm={a.arm} pol={spec['pol']} theta0={spec['theta0_deg']} bw={spec['bw']} k_point={k_point} "
          f"precheck: theta(f0)={pre['theta_at_f0_deg']} rel_err_ky={pre['rel_err_ky']:.3e} -> "
          f"{'PASS' if pre['passed'] else 'FAIL'}")
    if not pre["passed"] and not a.falsifier:
        print("ABORT: k_point does not map back to the declared angle (exit 1)")
        return 1
    fcen = f0 * a_m / C0
    fwidth = O.meep_fwidth_for(spec["bw"], f0) * a_m / C0
    res = run_two_pass(mp, arm=a.arm, k_point=k_point, a_m=a_m, resolution=a.resolution, courant=a.courant,
                       nfreq=a.nfreq, fcen=fcen, fwidth=fwidth, decay=a.decay,
                       center_offset_cells=a.center_offset_cells, pol=spec["pol"])
    R, T, freqs_hz = res["R"], res["T"], res["freqs_hz"]
    finite = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(T)))
    g = O.gated_mask(freqs_hz, spec)
    R_an, T_an = O.oracle_RT(freqs_hz, spec["ky"], spec["pol"])
    if finite and g.any():
        print(f"  band {freqs_hz.min()/1e9:.2f}-{freqs_hz.max()/1e9:.2f} GHz, gated bins {int(g.sum())}; "
              f"Meep vs Fresnel(theta(f)) mean|dR|={float(np.nanmean(np.abs(R-R_an)[g])):.4f} "
              f"mean|dT|={float(np.nanmean(np.abs(T-T_an)[g])):.4f} max R+T {float((R+T)[g].max()):.4f} "
              f"(diagnostic on Meep's own bins; the verdict is the case script's)")
    doc = {
        "schema": "cv26-meep-leg/v1", "case_id": O.CASE_ID, "arm": a.arm, "pol": spec["pol"],
        "falsifier": a.falsifier, "tag": a.tag, "meep_version": getattr(mp, "__version__", "unknown"),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "a_m": a_m, "resolution": a.resolution, "courant": a.courant, "dt_meep_s": res["dt_meep_s"],
        "nfreq": a.nfreq, "fcen_meep": res["fcen"], "fwidth_meep": res["fwidth"], "decay": a.decay,
        "eps_averaging": False, "center_offset_cells": a.center_offset_cells,
        "theta0_deg": spec["theta0_deg"], "bw": spec["bw"], "ky_declared_rad_m": spec["ky"],
        "k_point": list(k_point), "source_component": "Ez" if spec["pol"] == "te" else "Hz",
        "geometry": {k: res[k] for k in ("sx", "sy", "dpml", "refl_x", "trans_x", "src_x", "d_slab")},
        "precheck": pre,
        "freqs_hz": freqs_hz.tolist(), "R": R.tolist(), "T": T.tolist(),
        "run": {"t_ref_s": res["t_ref_s"], "t_slab_s": res["t_slab_s"], "finite": finite},
    }
    out = os.path.join(a.out_dir, O.meep_json_name(a.arm, a.falsifier, a.tag))
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {out}")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
