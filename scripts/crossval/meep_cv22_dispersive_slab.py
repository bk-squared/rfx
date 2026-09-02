"""Meep leg for crossval 22 (dispersive slab): one arm -> one JSON.

Builds the SAME slab as ``validation/crossval/22_dispersive_slab_fresnel.py``
in Meep (cv04's Meep geometry: a = 1 cm, 60 x 0.4 cm cell, 2 cm PML in x,
periodic in y, resolution 10 px/cm = the rfx dx, slab 1 cm at the centre,
flux monitors 3 cm either side of the slab, two-run reference subtraction)
with the material mapped by ``dispersive_eps.to_meep(...)``.

Before any FDTD, the mapped ε is evaluated at three band frequencies --
through ``meep.Medium.epsilon(f)`` when the installed Meep exposes it (Meep's
OWN evaluation of its OWN convention), and always through the comparator's
reconstruction -- and compared with ``eps_analytic`` (Debye: with the mapped
overdamped-Lorentz target; the residual vs true Debye is recorded). Any
relative error above 1e-9 ABORTS (exit 1) unless ``--falsifier`` is given, in
which case the failure is RECORDED and the run proceeds so that the E4 gates
are exercised on real Meep output of a wrongly mapped material (F3).

This script never imports rfx; it needs numpy + meep + the comparator.

Exit codes: 0 JSON written; 1 pre-check failed (non-falsifier) or Meep blew
up; 2 Meep not importable.

Run (conda-forge pymeep env):
  python scripts/crossval/meep_cv22_dispersive_slab.py --arm debye --out-dir <dir>
  python scripts/crossval/meep_cv22_dispersive_slab.py --arm lorentz --falsifier no_2pi --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "validation", "crossval", "comparators"))
import cv22_dispersive_gates as G  # noqa: E402
import dispersive_eps as DE  # noqa: E402

PRECHECK_FREQS_HZ = (4.5e9, 7.0e9, 9.5e9)
PRECHECK_REL_TOL = 1e-9


def _precheck(mp, medium, model, params, meep_params, fn_map_hz):
    """Return the precheck record; ``passed`` is the verdict."""
    f = np.asarray(PRECHECK_FREQS_HZ)
    if model == "debye":
        target = DE.eps_debye_mapped_target(f, params, fn_debye_map_hz=fn_map_hz)
        resid = DE.debye_mapping_residual(f, params, fn_debye_map_hz=fn_map_hz)
    else:
        target = DE.eps_analytic(f, model, params)
        resid = np.zeros_like(f)
    recon = np.conj(DE.eps_meep_convention(f, meep_params))
    err_recon = np.abs(recon - target) / np.abs(target)
    rec = {
        "freqs_hz": f.tolist(), "rel_tol": PRECHECK_REL_TOL,
        "target_rfx_convention": [[z.real, z.imag] for z in target],
        "reconstruction_rfx_convention": [[z.real, z.imag] for z in recon],
        "max_rel_err_reconstruction": float(err_recon.max()),
        "debye_map_residual_rel": resid.tolist(),
        "debye_map_residual_bound": G.DEBYE_MAP_RESIDUAL_REL_BOUND,
    }
    # Meep's own evaluation, when available (Medium.epsilon(freq) -> 3x3).
    err_meep = None
    try:
        vals = []
        for fi in f:
            e = np.asarray(medium.epsilon(float(fi * meep_params["a_m"] / DE.C0)))
            vals.append(complex(e[2, 2]))  # Ez polarization
        meep_eval = np.conj(np.asarray(vals))
        err_meep = np.abs(meep_eval - target) / np.abs(target)
        rec["meep_medium_epsilon_rfx_convention"] = [[z.real, z.imag] for z in meep_eval]
        rec["max_rel_err_meep_medium_epsilon"] = float(err_meep.max())
    except Exception as exc:  # older Meep without Medium.epsilon
        rec["meep_medium_epsilon_rfx_convention"] = None
        rec["max_rel_err_meep_medium_epsilon"] = None
        rec["meep_medium_epsilon_note"] = f"unavailable: {type(exc).__name__}: {exc}"
    worst = float(err_recon.max()) if err_meep is None else float(max(err_recon.max(), err_meep.max()))
    rec["max_rel_err"] = worst
    rec["passed"] = bool(worst < PRECHECK_REL_TOL
                         and (model != "debye" or resid.max() < G.DEBYE_MAP_RESIDUAL_REL_BOUND))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(G.ARM_ORDER))
    ap.add_argument("--falsifier", choices=sorted(G.MEEP_FALSIFIERS), default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=int(round(G.MEEP_A_M / G.DX_M)))
    ap.add_argument("--courant", type=float, default=G.MEEP_COURANT)
    ap.add_argument("--nfreq", type=int, default=200)
    ap.add_argument("--fn-debye-map-hz", type=float, default=G.DEBYE_MEEP_MAP_FN_HZ)
    ap.add_argument("--decay", type=float, default=1e-3)
    ap.add_argument("--fcen-ghz", type=float, default=G.TFSF_F0_HZ / 1e9,
                    help="Meep GaussianSource centre (default cv04's 10 GHz)")
    ap.add_argument("--fwidth-ghz", type=float, default=1.5 * G.TFSF_F0_HZ / 1e9,
                    help="Meep GaussianSource fwidth (default cv04's 15 GHz)")
    ap.add_argument("--tag", default=None,
                    help="write meep_<arm>__<tag>.json (diagnostic legs, note 11.2(b)/(b')/(c)); "
                         "without it the leg is the one the case script reads")
    a = ap.parse_args(argv)

    try:
        import meep as mp
    except Exception as exc:
        print(f"[SKIP] meep not importable: {type(exc).__name__}: {exc} (exit 2)")
        return 2

    if a.falsifier and a.arm != G.MEEP_FALSIFIER_ARM:
        print(f"falsifiers are declared on the {G.MEEP_FALSIFIER_ARM} arm only")
        return 1

    os.makedirs(a.out_dir, exist_ok=True)
    arm = G.ARMS[a.arm]
    model, params = arm["model"], dict(arm["params"])
    a_m = G.MEEP_A_M
    meep_params = DE.to_meep(model, params, a_m=a_m, fn_debye_map_hz=a.fn_debye_map_hz)
    if a.falsifier:
        meep_params = G.apply_meep_falsifier(meep_params, a.falsifier)
        print(f"FALSIFIER {a.falsifier}: {G.MEEP_FALSIFIERS[a.falsifier]}")

    if meep_params["kind"] == "LorentzianSusceptibility":
        sus = mp.LorentzianSusceptibility(frequency=meep_params["frequency"],
                                          gamma=meep_params["gamma"], sigma=meep_params["sigma"])
    else:
        sus = mp.DrudeSusceptibility(frequency=meep_params["frequency"],
                                     gamma=meep_params["gamma"], sigma=meep_params["sigma"])
    medium = mp.Medium(epsilon=meep_params["eps_inf"], E_susceptibilities=[sus])
    print(f"arm={a.arm} model={model} meep_params={meep_params}")

    # ---- pre-run material check (1e-9) ----
    pre = _precheck(mp, medium, model, params, meep_params, a.fn_debye_map_hz)
    print(f"precheck: max_rel_err={pre['max_rel_err']:.3e} (tol {PRECHECK_REL_TOL:g}) "
          f"meep_medium_epsilon={pre['max_rel_err_meep_medium_epsilon']} -> "
          f"{'PASS' if pre['passed'] else 'FAIL'}")
    if not pre["passed"] and not a.falsifier:
        print("ABORT: Meep material mapping does not reproduce eps_analytic to 1e-9 (exit 1)")
        return 1

    # ---- cv04 Meep geometry ----
    fcen = a.fcen_ghz * 1e9 * a_m / DE.C0
    fwidth = a.fwidth_ghz * 1e9 * a_m / DE.C0
    sx = G.NX_INTERIOR * G.DX_M / a_m
    sy = 0.4
    dpml = G.N_CPML * G.DX_M / a_m
    d_slab = G.D_SLAB_M / a_m
    refl_x = -d_slab / 2 - 30 * G.DX_M / a_m
    trans_x = d_slab / 2 + 30 * G.DX_M / a_m
    src_x = -d_slab / 2 - 50 * G.DX_M / a_m
    cell = mp.Vector3(sx, sy, 0)
    pml = [mp.PML(dpml, direction=mp.X)]
    src = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=fwidth), component=mp.Ez,
                     center=mp.Vector3(src_x, 0), size=mp.Vector3(0, sy))]
    refl_fr = mp.FluxRegion(center=mp.Vector3(refl_x, 0), size=mp.Vector3(0, sy))
    trans_fr = mp.FluxRegion(center=mp.Vector3(trans_x, 0), size=mp.Vector3(0, sy))
    stop = lambda: mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(trans_x, 0), a.decay)

    print(f"Meep: cell {sx}x{sy} (a={a_m} m), pml {dpml}, res {a.resolution}, Courant {a.courant}, "
          f"fcen {fcen:.4f} fwidth {fwidth:.4f}, nfreq {a.nfreq}")
    t0 = time.time()
    sim_ref = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=src,
                            resolution=a.resolution, Courant=a.courant, k_point=mp.Vector3())
    refl_ref = sim_ref.add_flux(fcen, fwidth, a.nfreq, refl_fr)
    trans_ref = sim_ref.add_flux(fcen, fwidth, a.nfreq, trans_fr)
    sim_ref.run(until_after_sources=stop())
    straight_refl_data = sim_ref.get_flux_data(refl_ref)
    straight_tran_flux = np.array(mp.get_fluxes(trans_ref))
    flux_freqs = np.array(mp.get_flux_freqs(refl_ref))
    t_ref = time.time() - t0
    print(f"  reference run {t_ref:.1f}s")

    t0 = time.time()
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=src,
                        resolution=a.resolution, Courant=a.courant, k_point=mp.Vector3(),
                        geometry=[mp.Block(center=mp.Vector3(0, 0),
                                           size=mp.Vector3(d_slab, mp.inf, mp.inf),
                                           material=medium)])
    refl = sim.add_flux(fcen, fwidth, a.nfreq, refl_fr)
    trans = sim.add_flux(fcen, fwidth, a.nfreq, trans_fr)
    sim.load_minus_flux_data(refl, straight_refl_data)
    sim.run(until_after_sources=stop())
    slab_refl_flux = np.array(mp.get_fluxes(refl))
    slab_tran_flux = np.array(mp.get_fluxes(trans))
    t_slab = time.time() - t0
    print(f"  slab run {t_slab:.1f}s")

    T = slab_tran_flux / straight_tran_flux
    R = -slab_refl_flux / straight_tran_flux
    freqs_hz = flux_freqs * DE.C0 / a_m
    dt_meep_s = a.courant / a.resolution * a_m / DE.C0
    finite = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(T)))
    g = G.gated_mask(freqs_hz)
    passive = bool(finite and np.all((R + T)[g] <= 1.0 + G.CONS_MAX_LIMIT))
    print(f"  band {freqs_hz.min()/1e9:.2f}-{freqs_hz.max()/1e9:.2f} GHz; finite={finite}; "
          f"passive in gated band={passive}; max R+T (gated) = {float((R+T)[g].max()):.4f}")

    # Diagnostic (not the verdict): Meep vs TMM on Meep's own bins.
    eps_c = DE.eps_analytic(freqs_hz, model, params)
    R_an, T_an = DE.tmm_slab_rt(freqs_hz, eps_c, G.D_SLAB_M)
    print(f"  Meep vs TMM (gated bins): max|dR|={float(np.abs(R-R_an)[g].max()):.4f} "
          f"max|dT|={float(np.abs(T-T_an)[g].max()):.4f} mean {float(np.abs(R-R_an)[g].mean()):.4f}/"
          f"{float(np.abs(T-T_an)[g].mean()):.4f}")

    doc = {
        "schema": "cv22-meep-leg/v1", "case_id": "22_dispersive_slab_fresnel",
        "arm": a.arm, "falsifier": a.falsifier, "tag": a.tag,
        "meep_version": getattr(mp, "__version__", "unknown"),
        "fn_debye_map_hz": a.fn_debye_map_hz if model == "debye" else None,
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "a_m": a_m, "resolution": a.resolution, "courant": a.courant, "dt_meep_s": dt_meep_s,
        "nfreq": a.nfreq, "fcen_meep": fcen, "fwidth_meep": fwidth, "decay": a.decay,
        "material": {"model": model, "params": params},
        "meep_params": meep_params,
        "precheck": pre,
        "freqs_hz": freqs_hz.tolist(), "R": R.tolist(), "T": T.tolist(),
        "run": {"t_ref_s": t_ref, "t_slab_s": t_slab, "finite": finite,
                "passive_gated": passive, "max_RT_gated": float((R + T)[g].max()) if finite else None},
    }
    out = os.path.join(a.out_dir, f"meep_{a.arm}__{a.tag}.json" if a.tag
                       else G.meep_json_name(a.arm, a.falsifier))
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {out}")
    if not finite:
        print("Meep produced non-finite R/T (exit 1)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
