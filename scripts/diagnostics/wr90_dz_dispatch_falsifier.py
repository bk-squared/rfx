"""WR-90 dz-only dispatch falsifier (issue #811).

``compute_waveguide_s_matrix`` gated its non-uniform lane on
``dx_profile``/``dy_profile`` only (rfx/api/_sparams.py, waveguide NU
dispatch), so a simulation whose ONLY profile was ``dz_profile`` was
silently solved on the uniform grid built from the scalar ``dx`` while
``preflight()`` described the graded mesh. This script measures that
dispatch honestly, before and after the fix, on one fixed WR-90 two-port
geometry with a dielectric slab between the ports.

Arms (identical physical geometry, only the z mesh differs):

  U       no profiles                    -> uniform lane (also: the answer
                                            every pre-fix dz-only run got)
  A       dz_profile = Z_A (fine->coarse, 19 cells, min 0.4 mm)
  B       dz_profile = Z_B = reversed(Z_A)
  C       dz_profile = Z_C (coarse/fine/mid, 14 cells, min 0.62 mm -> its
                                            dt differs from A/B by ~1.34x; the min-CELL ratio is 1.55x)
  A_shim  dz_profile = Z_A + uniform-valued dy_profile = full(23, 1e-3)
          (matches the NU lane's own synthesized dy; PLUMBING witness only
          -- a uniform-valued profile tests plumbing, never NU metrics)

Falsifiers evaluated (declared in
docs/design_notes/issue811_dz_dispatch_predeclaration.md BEFORE any arm ran):

  F1  dz-only arms across genuinely different z meshes must NOT be
      bit-identical (pre-fix they are: the defect).
  F2  A vs A_shim must agree (expected exactly 0.0; tolerance 1e-6).
  F3  max|S11_A - S11_U| must land in [1e-5, 1e-1].

Exit codes: 0 = all evaluated falsifiers pass (post-fix semantics);
2 = dz-only arms bit-identical (defect signature); 1 = other failure.

The S numbers here are DISPATCH EVIDENCE, not validated S-parameters:
run length (num_periods=20) is shared by every compared arm so truncation
is common-mode, and no external reference is involved. Do not quote them
as physics.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Declared meshes. All sum EXACTLY to b = 10.16 mm (WR-90 narrow wall);
# adjacent-cell ratios stay <= 1.4 (the validated multi-band grading cap).
# ---------------------------------------------------------------------------
Z_A = np.concatenate([
    np.full(10, 0.40e-3),
    np.full(3, 0.52e-3),
    np.full(2, 0.70e-3),
    np.full(4, 0.80e-3),
])                              # 19 cells, sum 10.16 mm, ratios 1.30/1.35/1.14
Z_B = Z_A[::-1].copy()          # same cells, mirrored placement
Z_C = np.concatenate([
    np.full(6, 0.80e-3),
    np.full(4, 0.62e-3),
    np.full(4, 0.72e-3),
])                              # 14 cells, sum 10.16 mm, ratios 1.29/1.16
DY_SHIM = np.full(23, 1.0e-3)   # uniform-valued; equals the synthesized dy

A_WG = 0.02286                  # WR-90 broad wall (y)
B_WG = 0.01016                  # WR-90 narrow wall (z)
DOMAIN_X = 0.10
DX = 1.0e-3
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 9)
NUM_PERIODS = 20.0
EPS_SLAB = 2.2

ARM_PROFILES = {
    "U": (None, None),
    "A": (Z_A, None),
    "B": (Z_B, None),
    "C": (Z_C, None),
    "A_shim": (Z_A, DY_SHIM),
}


def _build_sim(dz_profile, dy_profile):
    import jax.numpy as jnp
    from rfx import Box
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    kwargs = {}
    if dz_profile is not None:
        kwargs["dz_profile"] = dz_profile
    if dy_profile is not None:
        kwargs["dy_profile"] = dy_profile
    sim = Simulation(
        freq_max=12.4e9,
        domain=(DOMAIN_X, A_WG, B_WG),
        dx=DX,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20,
        **kwargs,
    )
    sim.add_material("slab", eps_r=EPS_SLAB)
    sim.add(Box((0.045, 0.0, 0.0), (0.055, A_WG, B_WG)), material="slab")
    for x_position, direction, ref, name in (
            (0.015, "+x", 0.020, "left"),
            (0.085, "-x", 0.080, "right")):
        sim.add_waveguide_port(
            x_position, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(FREQS_HZ), f0=10.3e9, bandwidth=0.5,
            reference_plane=ref, name=name)
    return sim


def _run_arm(name):
    dz_profile, dy_profile = ARM_PROFILES[name]
    sim = _build_sim(dz_profile, dy_profile)

    # Preflight context is part of the result (house rule): capture every
    # issue and warning verbatim.
    preflight_issues: list[str] = []
    preflight_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            report = sim.preflight()
        preflight_issues = [str(i) for i in getattr(report, "issues", [])]
        preflight_warnings = [str(w.message) for w in wrec]
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        preflight_issues = [f"preflight raised {type(exc).__name__}: {exc}"]

    is_nu = dz_profile is not None or dy_profile is not None
    grid = sim._build_nonuniform_grid() if is_nu else sim._build_grid()
    dt = float(grid.dt)
    nz = int(len(dz_profile)) if dz_profile is not None else int(
        round(B_WG / DX))

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        res = sim.compute_waveguide_s_matrix(
            num_periods=NUM_PERIODS, normalize="flux")
    wall = time.perf_counter() - t0
    compute_warnings = [str(w.message) for w in wrec]

    s = np.asarray(res.s_params)
    return {
        "arm": name,
        "dz_profile_m": None if dz_profile is None else [
            float(v) for v in dz_profile],
        "dy_profile_m": None if dy_profile is None else [
            float(v) for v in dy_profile],
        "nz": nz,
        "dt_s": dt,
        "wallclock_s": wall,
        "freqs_hz": [float(f) for f in np.asarray(res.freqs)],
        "s_params_real": np.real(s).tolist(),
        "s_params_imag": np.imag(s).tolist(),
        "preflight_issues": preflight_issues,
        "preflight_warnings": preflight_warnings,
        "compute_warnings": compute_warnings,
    }, s


def _fmt_bins(vals):
    return np.array2string(np.asarray(vals), precision=8,
                           floatmode="maxprec", max_line_width=200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="U,A,B,C,A_shim",
                    help="comma-separated arm names to run")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    results = {}
    s_by_arm = {}
    for name in arm_names:
        print(f"=== arm {name} ===", flush=True)
        rec, s = _run_arm(name)
        results[name] = rec
        s_by_arm[name] = s
        print(f"  nz={rec['nz']} dt={rec['dt_s']:.6e}s "
              f"wall={rec['wallclock_s']:.1f}s")
        print(f"  |S11(left,left)| per bin: "
              f"{_fmt_bins(np.abs(s[0, 0, :]))}")
        for line in rec["preflight_issues"]:
            print(f"  preflight issue: {line}")
        for line in rec["preflight_warnings"] + rec["compute_warnings"]:
            print(f"  warning: {line}")

    verdict = {"pairs": {}}
    dz_only = [n for n in ("A", "B", "C") if n in s_by_arm]

    # F1: pairwise comparison of the dz-only arms.
    f1_pass = None
    if len(dz_only) >= 2:
        f1_pass = True
        for i, na in enumerate(dz_only):
            for nb in dz_only[i + 1:]:
                sa, sb = s_by_arm[na], s_by_arm[nb]
                bit_identical = bool(np.array_equal(sa, sb))
                max_ds = float(np.max(np.abs(sa - sb)))
                per_bin_s11 = np.abs(sa[0, 0, :] - sb[0, 0, :])
                verdict["pairs"][f"{na}-vs-{nb}"] = {
                    "bit_identical": bit_identical,
                    "max_abs_dS": max_ds,
                    "per_bin_abs_dS11": [float(v) for v in per_bin_s11],
                }
                print(f"F1 {na} vs {nb}: bit_identical={bit_identical} "
                      f"max|dS|={max_ds:.6e}")
                print(f"   per-bin |dS11|: {_fmt_bins(per_bin_s11)}")
                if bit_identical:
                    f1_pass = False
        verdict["F1_dz_only_arms_not_bit_identical"] = f1_pass

    # F2: plumbing witness (uniform-valued dy shim).
    f2_pass = None
    if "A" in s_by_arm and "A_shim" in s_by_arm:
        d = float(np.max(np.abs(s_by_arm["A"] - s_by_arm["A_shim"])))
        f2_pass = d <= 1e-6
        verdict["F2_shim_agreement_max_abs_dS"] = d
        verdict["F2_pass"] = f2_pass
        print(f"F2 A vs A_shim: max|dS|={d:.6e} (tolerance 1e-6) "
              f"-> {'PASS' if f2_pass else 'FAIL'}")

    # F3: dz-only answer must move off the uniform answer by a plausible
    # amount (order-of-magnitude window, NOT a tight gate).
    f3_pass = None
    if "U" in s_by_arm and "A" in s_by_arm:
        su, sa = s_by_arm["U"], s_by_arm["A"]
        per_bin_s11 = np.abs(sa[0, 0, :] - su[0, 0, :])
        d11 = float(np.max(per_bin_s11))
        d_all = float(np.max(np.abs(sa - su)))
        f3_pass = 1e-5 <= d11 <= 1e-1
        verdict["F3_A_vs_U_max_abs_dS11"] = d11
        verdict["F3_A_vs_U_max_abs_dS"] = d_all
        verdict["F3_A_vs_U_per_bin_abs_dS11"] = [float(v) for v in per_bin_s11]
        verdict["F3_pass"] = f3_pass
        print(f"F3 A vs U: max|dS11|={d11:.6e} (window [1e-5, 1e-1]) "
              f"max|dS|={d_all:.6e} -> {'PASS' if f3_pass else 'FAIL'}")
        print(f"   per-bin |dS11|: {_fmt_bins(per_bin_s11)}")

    import rfx as _rfx
    out = {
        "schema": "rfx.dz_dispatch_falsifier.v1",
        "issue": "#811",
        # Which rfx actually ran: `python script.py` resolves `import rfx`
        # from sys.path, NOT from the checkout the script lives in — a
        # stale editable install silently shadows the working tree. Pin
        # with PYTHONPATH=<checkout> and check this field.
        "rfx_module_file": str(getattr(_rfx, "__file__", "?")),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_periods": NUM_PERIODS,
        "normalize": "flux",
        "arms": results,
        "verdict": verdict,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"rfx module: {out['rfx_module_file']}")
    print(f"wrote {args.out}")

    evaluated = [p for p in (f1_pass, f2_pass, f3_pass) if p is not None]
    if f1_pass is False:
        print("VERDICT: dz-only arms bit-identical -- dispatch defect "
              "signature (#811) present")
        return 2
    if evaluated and all(evaluated):
        print("VERDICT: all evaluated falsifiers PASS")
        return 0
    if not evaluated:
        print("VERDICT: nothing evaluated (arm subset too small)")
        return 0
    print("VERDICT: FAIL (see falsifier lines above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
