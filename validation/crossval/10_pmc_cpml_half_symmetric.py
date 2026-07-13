"""Cross-validation 10: PMC + CPML composition on the same axis

Regression lock for the v1.7.5 per-face PMC/CPML composition fix.
Pre-fix, a PMC face silently received a full-strength CPML profile,
which `apply_cpml_e` then applied to the first `n` INTERIOR cells
adjacent to the reflector — causing field decay proportional to
`cpml_layers` (free-space peak dropped 10 000× between cpml=2 and
cpml=8). The NU scan body additionally did not call
`apply_pmc_faces` at all, so the "PMC" was effectively a free
boundary on that path. This crossval locks both fixes.

**Setup (both paths):**
  - 3D free-space (no materials)
  - y_lo = PMC (mirror plane), y_hi = CPML
  - x, z axes = CPML on both sides
  - Ez source one cell inside the PMC plane (Taflove FDTD convention;
    source-on-plane is separately flagged by v1.7.5 preflight warn)
  - Probe further inside the interior
  - Sweep cpml_layers ∈ {2, 4, 6, 8}

**Physical expectation:**
  - Peak|probe| is governed by the direct Gaussian pulse from source
    to probe. This path is in the interior, far from y_hi CPML, so
    changing the absorber thickness on the hi face MUST NOT change
    the peak. Pre-fix: peak decayed 10 000× as cpml grew. Post-fix:
    peak stable within ±1 %.
  - Late-time tail should DECAY with more absorber (normal CPML
    behaviour). A larger `tail / peak` ratio at higher cpml would
    indicate a new regression.

**Coverage against v1.7.5 commits:**
  - `e340644` — init_cpml union of pec_faces ∪ pmc_faces → noop
    profile on PMC faces (otherwise the apply_cpml slice hits
    interior cells adjacent to the reflector)
  - `fdc6cc1` — NU path passes `pmc_faces` explicitly to init_cpml
    (NonUniformGrid does not carry a `pmc_faces` attr, so the
    default `getattr` read empty)
  - `3a66c02` / `9072a59` — per-face grid padding so the PMC plane
    aligns at array index 0 (pad_y_lo=0, pad_y_hi=n)
  - `84b11aa` — `apply_pmc_faces` wired into the NU scan body
    (previously never called on NU)
  - `79d2ea2` / `29d6c3d` — preflight warn when user places source
    exactly on a reflector plane (this script intentionally uses
    y=DX, so no warning is expected)

Pass criteria:
  1. (peak_max − peak_min) / peak_max  <  0.02   per path
  2. no NaN, no Inf in either time series at any cpml value
  3. both uniform and NU paths must satisfy (1) and (2)
"""

from __future__ import annotations

import math, os, sys, time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8
DX = 1e-3                       # 1 mm — fast, enough for clean propagation
DOM = (40e-3, 20e-3, 20e-3)     # half-domain y=20mm, interior x=40mm
SRC_Y = DX                      # one cell inside the PMC plane (convention)
CPML_VALUES = (2, 4, 6, 8)
N_STEPS = 800


def _common_spec():
    return dict(
        freq_max=10e9, dx=DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pmc", hi="cpml"),
            z="cpml",
        ),
    )


def run_uniform(n_cpml: int) -> dict:
    sim = Simulation(
        domain=DOM, cpml_layers=n_cpml, **_common_spec(),
    )
    sim.add_source(
        position=(20e-3, SRC_Y, 10e-3), component="ez",
        waveform=GaussianPulse(f0=5e9, bandwidth=1.0),
    )
    sim.add_probe(position=(20e-3, 5e-3, 10e-3), component="ez")
    sim.preflight(strict=False)
    t0 = time.time()
    res = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(res.time_series).ravel()
    return {
        "path": "uniform",
        "cpml": n_cpml,
        "peak": float(np.max(np.abs(ts))),
        "tail_peak": float(np.max(np.abs(ts[-100:]))),
        "nan": bool(np.any(~np.isfinite(ts))),
        "wall": time.time() - t0,
    }


def run_nonuniform(n_cpml: int) -> dict:
    """Same problem, routed through the non-uniform z path."""
    dz_profile = np.full(int(round(DOM[2] / DX)) + 1, DX)
    sim = Simulation(
        domain=DOM, cpml_layers=n_cpml,
        dz_profile=dz_profile,
        **_common_spec(),
    )
    sim.add_source(
        position=(20e-3, SRC_Y, 10e-3), component="ez",
        waveform=GaussianPulse(f0=5e9, bandwidth=1.0),
    )
    sim.add_probe(position=(20e-3, 5e-3, 10e-3), component="ez")
    sim.preflight(strict=False)
    t0 = time.time()
    res = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(res.time_series).ravel()
    return {
        "path": "nonuniform",
        "cpml": n_cpml,
        "peak": float(np.max(np.abs(ts))),
        "tail_peak": float(np.max(np.abs(ts[-100:]))),
        "nan": bool(np.any(~np.isfinite(ts))),
        "wall": time.time() - t0,
    }


def _evaluate(results: list[dict], path_name: str):
    peaks = np.array([r["peak"] for r in results])
    any_nan = any(r["nan"] for r in results)
    peak_max = float(peaks.max())
    peak_min = float(peaks.min())
    peak_range = (peak_max - peak_min) / peak_max if peak_max > 0 else float("nan")
    ok_stable = bool(peak_range < 0.02 and peak_max > 0)
    ok_finite = not any_nan

    print(f"\n  [{path_name}]  (pass gate: peak range < 2 %, no NaN)")
    print(f"    {'cpml':>5} | {'peak':>12} | {'tail':>12} | {'nan':>5} | {'wall':>5}")
    print(f"    {'-'*5} | {'-'*12} | {'-'*12} | {'-'*5} | {'-'*5}")
    for r in results:
        print(f"    {r['cpml']:>5d} | {r['peak']:>12.3e} | {r['tail_peak']:>12.3e} | "
              f"{str(r['nan']):>5s} | {r['wall']:>4.1f}s")
    print(f"    peak range = (max-min)/max = {peak_range*100:.3f} %   "
          f"{'PASS' if ok_stable else 'FAIL'}")
    print(f"    any NaN                     = {any_nan}   "
          f"{'PASS' if ok_finite else 'FAIL'}")
    return ok_stable and ok_finite


def _plot(uniform_results, nu_results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cpmls = [r["cpml"] for r in uniform_results]
    ax.plot(cpmls, [r["peak"] for r in uniform_results], "o-",
            label="uniform path peak", lw=2)
    ax.plot(cpmls, [r["peak"] for r in nu_results], "s--",
            label="non-uniform path peak", lw=2)
    ax.set_xlabel("cpml_layers (y_hi face active count)")
    ax.set_ylabel("peak |Ez(probe)|")
    ax.set_title("Crossval 10 — PMC + CPML composition stability\n"
                 "(peak must be flat across cpml — pre-v1.7.5 decayed 10 000×)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "10_pmc_cpml_composition.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")


def main():
    print("=" * 70)
    print("Crossval 10: PMC + CPML composition regression lock")
    print("=" * 70)
    print(f"  domain = {tuple(f'{d*1e3:.0f}mm' for d in DOM)}, dx = {DX*1e3:.1f} mm")
    print(f"  PMC y_lo, CPML y_hi + all x/z faces; source at y = DX = {SRC_Y*1e3:.1f} mm")
    print(f"  sweep cpml_layers ∈ {CPML_VALUES}")

    print("\n  [uniform path] ...")
    uniform_results = [run_uniform(n) for n in CPML_VALUES]

    print("  [non-uniform path (dz_profile, uniform xy)] ...")
    nu_results = [run_nonuniform(n) for n in CPML_VALUES]

    pass_u = _evaluate(uniform_results, "uniform")
    pass_n = _evaluate(nu_results,     "nonuniform")

    _plot(uniform_results, nu_results)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  uniform path     : {'PASS' if pass_u else 'FAIL'}")
    print(f"  non-uniform path : {'PASS' if pass_n else 'FAIL'}")
    PASS = pass_u and pass_n
    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
