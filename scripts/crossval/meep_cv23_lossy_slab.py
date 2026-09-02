"""Meep leg for crossval 23 (lossy slab): one arm -> one JSON.

The SAME slab as ``validation/crossval/23_lossy_slab_fresnel.py`` in Meep --
cv04/cv22's Meep geometry and two-run reference subtraction
(``meep_cv22_dispersive_slab.run_slab_two_pass``) -- with the material
``Medium(epsilon=eps', D_conductivity=sigma_D)``, ``sigma_D`` mapped by
``dispersive_eps.to_meep("conductive", ...)`` (= sigma a/(c eps0 eps'), note
section 7). ``eps_averaging`` is False on both passes (note section 1).

Before any FDTD the mapped eps is evaluated at three band frequencies through
``meep.Medium.epsilon(f)`` (Meep's OWN evaluation, which includes the
conductivity term) and through the comparator's reconstruction, against
``eps_analytic``; any relative error above 1e-9 ABORTS (exit 1) unless
``--falsifier`` is given, in which case the failure is RECORDED and the run
proceeds so the E4 gates are exercised on real Meep output of a wrongly
scaled sigma_D (F3).

This script never imports rfx; it needs numpy + meep + the comparators.

Exit codes: 0 JSON written; 1 pre-check failed (non-falsifier) or Meep blew
up; 2 Meep not importable.

Run (conda-forge pymeep env):
  python scripts/crossval/meep_cv23_lossy_slab.py --arm tand1 --out-dir <dir>
  python scripts/crossval/meep_cv23_lossy_slab.py --arm tand1 --falsifier sigma_2pi --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "validation", "crossval", "comparators"))
import cv23_lossy_gates as L  # noqa: E402
import dispersive_eps as DE  # noqa: E402


def _load_cv22_leg():
    spec = importlib.util.spec_from_file_location("meep_cv22_dispersive_slab",
                                                  os.path.join(_HERE, "meep_cv22_dispersive_slab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(L.ARM_ORDER))
    ap.add_argument("--falsifier", choices=sorted(L.MEEP_FALSIFIERS), default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=L.MEEP_PRIMARY_RESOLUTION,
                    help="px/cm (default: the 40 px/cm primary reference; the ladder passes 10/20/40)")
    ap.add_argument("--courant", type=float, default=L.MEEP_COURANT)
    ap.add_argument("--nfreq", type=int, default=200)
    ap.add_argument("--decay", type=float, default=1e-3)
    ap.add_argument("--fcen-ghz", type=float, default=10.0)
    ap.add_argument("--fwidth-ghz", type=float, default=15.0)
    ap.add_argument("--thickness-offset-cells", type=float, default=0.0,
                    help="draw the block d + N a/res thick (note section 12: -1 tests the hypothesis that "
                         "Meep's block realizes d + a/res); diagnostic, requires --tag")
    ap.add_argument("--tag", default=None,
                    help="write meep_<arm>__<tag>.json (ladder rungs); without it the leg is the one the case reads")
    a = ap.parse_args(argv)

    try:
        import meep as mp
    except Exception as exc:
        print(f"[SKIP] meep not importable: {type(exc).__name__}: {exc} (exit 2)")
        return 2
    M22 = _load_cv22_leg()

    if a.thickness_offset_cells and not a.tag:
        ap.error("--thickness-offset-cells is a diagnostic and requires --tag")
    if a.falsifier and a.arm != L.MEEP_FALSIFIER_ARM:
        print(f"falsifiers are declared on the {L.MEEP_FALSIFIER_ARM} arm only")
        return 1

    os.makedirs(a.out_dir, exist_ok=True)
    params = dict(L.ARMS[a.arm]["params"])
    a_m = L.MEEP_A_M
    meep_params = DE.to_meep(L.MODEL, params, a_m=a_m)
    if a.falsifier:
        meep_params = L.apply_meep_falsifier(meep_params, a.falsifier)
        print(f"FALSIFIER {a.falsifier}: {L.MEEP_FALSIFIERS[a.falsifier]}")
    medium = mp.Medium(epsilon=meep_params["eps_inf"], D_conductivity=meep_params["D_conductivity"])
    print(f"arm={a.arm} tan delta @ {L.F_CENTRE_HZ/1e9:g} GHz = {L.ARM_TAN_DELTA[a.arm]:g} "
          f"sigma={params['sigma']:.6f} S/m meep_params={meep_params}")

    # ---- pre-run material check (1e-9); Medium.epsilon(f) includes D_conductivity ----
    pre = M22._precheck(mp, medium, L.MODEL, params, meep_params, None)
    print(f"precheck: max_rel_err={pre['max_rel_err']:.3e} (tol {M22.PRECHECK_REL_TOL:g}) "
          f"meep_medium_epsilon={pre['max_rel_err_meep_medium_epsilon']} -> "
          f"{'PASS' if pre['passed'] else 'FAIL'}")
    if not pre["passed"] and not a.falsifier:
        print("ABORT: Meep D_conductivity mapping does not reproduce eps_analytic to 1e-9 (exit 1)")
        return 1

    res = M22.run_slab_two_pass(mp, medium, a_m=a_m, resolution=a.resolution, courant=a.courant,
                                nfreq=a.nfreq, fcen_ghz=a.fcen_ghz, fwidth_ghz=a.fwidth_ghz,
                                decay=a.decay, eps_averaging=L.MEEP_EPS_AVERAGING,
                                d_slab_m=L.D_SLAB_M + a.thickness_offset_cells * a_m / a.resolution)
    R, T, freqs_hz = res["R"], res["T"], res["freqs_hz"]
    A = 1.0 - R - T
    finite = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(T)))
    import cv22_dispersive_gates as G  # noqa: E402
    g = G.gated_mask(freqs_hz)
    passive = bool(finite and np.all((R + T)[g] <= 1.0 + G.CONS_MAX_LIMIT))
    print(f"  band {freqs_hz.min()/1e9:.2f}-{freqs_hz.max()/1e9:.2f} GHz; finite={finite}; "
          f"passive in gated band={passive}; max R+T (gated) = {float((R+T)[g].max()):.4f}")

    # Diagnostic (not the verdict): Meep vs TMM on Meep's own bins, R, T, A.
    R_an, T_an, A_an = L.analytic_rta(freqs_hz, params)
    print(f"  Meep vs TMM (gated bins): mean|dR|={float(np.abs(R-R_an)[g].mean()):.4f} "
          f"mean|dT|={float(np.abs(T-T_an)[g].mean()):.4f} mean|dA|={float(np.abs(A-A_an)[g].mean()):.4f} "
          f"(max {float(np.abs(R-R_an)[g].max()):.4f}/{float(np.abs(T-T_an)[g].max()):.4f}/"
          f"{float(np.abs(A-A_an)[g].max()):.4f})")

    doc = {
        "schema": "cv23-meep-leg/v1", "case_id": L.CASE_ID,
        "arm": a.arm, "falsifier": a.falsifier, "tag": a.tag,
        "meep_version": getattr(mp, "__version__", "unknown"),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "a_m": a_m, "resolution": a.resolution, "courant": a.courant, "dt_meep_s": res["dt_meep_s"],
        "nfreq": a.nfreq, "fcen_meep": res["fcen"], "fwidth_meep": res["fwidth"], "decay": a.decay,
        "eps_averaging": L.MEEP_EPS_AVERAGING,
        "d_slab_m": res["d_slab_m"], "thickness_offset_cells": a.thickness_offset_cells,
        "material": {"model": L.MODEL, "params": params, "tan_delta_at_f_centre": L.ARM_TAN_DELTA[a.arm]},
        "meep_params": meep_params,
        "precheck": pre,
        "freqs_hz": freqs_hz.tolist(), "R": R.tolist(), "T": T.tolist(), "A": A.tolist(),
        "run": {"t_ref_s": res["t_ref_s"], "t_slab_s": res["t_slab_s"], "finite": finite,
                "passive_gated": passive, "max_RT_gated": float((R + T)[g].max()) if finite else None},
    }
    out = os.path.join(a.out_dir, f"meep_{a.arm}__{a.tag}.json" if a.tag
                       else L.meep_json_name(a.arm, a.falsifier))
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {out}")
    if not finite:
        print("Meep produced non-finite R/T (exit 1)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
