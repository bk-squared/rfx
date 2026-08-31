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
  - The declared PMC wall must actually BE a magnetic wall: H_tan on
    that face is exactly zero, not merely small.
  - The half-domain PMC result must reproduce the AMPLITUDE of the
    equivalent full-domain image-source problem, not just its shape.

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

PMC-plane convention: REALIZE-DECLARED, decided on issue #722's ninth
surface (2026-08-28, see validation/crossval/09_half_symmetric_waveguide.py
for the full argument). apply_pmc_faces zeros H_tan a half-cell 0.5*dx
INSIDE the declared y_lo wall (rfx/boundaries/pmc.py, index 0 on a `_lo`
face; pinned by tests/unit/boundaries/test_boundary_pmc_hi_faces.py), so this script's
y=20 mm half-domain has its H_tan mirror plane at y=DX/2=0.5 mm, not
y=0.

CORRECTION (2026-08-31, issue #812 re-gate; full argument in
docs/design_notes/cv10_pmc_realization_regate.md §5). This paragraph
used to end: "UNLIKE cv09, this offset biases NO comparator here …
so the half-cell shift changes no pass/fail gate … there is no gate-3
class comparison against a mirrored full geometry to protect." That
was true only while EVERY gate here was a within-path relative
spread, and gate 4 below makes it false. Gate 4 IS a comparison
against a mirrored full geometry, so the 0.5*dx plane location is now
load-bearing: it fixes the control domain at y = 39 mm (not 40 mm),
the mirror plane at y = 19.5 mm, and the in-phase Ez image pair at
y = 19 mm / 20 mm. A half-cell error in the convention would move the
control geometry and break gate 4 rather than pass silently.

WHY GATES 3 AND 4 EXIST (issue #812, critical tier). Gates 1 and 2 are
within-path RELATIVE spreads, and `(peak_max − peak_min)/peak_max` is
invariant under `peak_i → c·peak_i` for any constant c > 0. Deleting
`apply_pmc_faces` is exactly such a constant factor here: the array
truncation supplies a PEC (E_tan = 0) wall in place of the absent PMC
(H_tan = 0) wall, which flips the image-source sign and rescales the
whole sweep by one factor the relative spread divides out. Measured on
this host: with `apply_pmc_faces` monkeypatched to a no-op the uniform
spread IMPROVES to 0.0667 % (from 0.2487 %) against a 2 % gate, and
the peaks scale by 0.5077 with the field bit-identical to a PEC wall.
Gate 3 is the direct realization check a no-op cannot pass; gate 4 is
the absolute amplitude reference the relative spread threw away.

Gate 4's control geometry (derived, not fitted — see the design note):
the half domain's H_tan mirror plane at y' = 0.5 mm makes Ez even about
that plane, so interior nodes y' = 1…20 mm image onto y' = 0…−19 mm.
The full symmetric domain is therefore y' ∈ [−19, 20] mm = 39 mm with
the plane 19.5 mm from each end; under y_full = y' + 19 mm the source
at y' = 1 mm becomes an IN-PHASE Ez pair at y_full = 20/19 mm (a PMC
images a tangential electric current in phase; a PEC reverses it — that
is the sign the defect flips), and the probe at y' = 5 mm becomes
y_full = 24 mm. The control leg declares NO PMC face, so it is immune
to any defect in apply_pmc_faces.

Pass criteria:
  1. (peak_max − peak_min) / peak_max  <  0.02   per path
  2. no NaN, no Inf in either time series at any cpml value
  3. PMC realization, bit-exact: max|Hx[:, 0, :]| == 0.0 and
     max|Hz[:, 0, :]| == 0.0 on the declared y_lo face at every cpml
     value on both paths, with max|Hx| over the whole array > 0 so an
     all-zero field cannot satisfy it vacuously. The threshold is
     definitional, not a tolerance: apply_pmc_faces writes literal 0.0.
  4. image-doubling control arm, absolute: with
     R = peak|Ez(probe)|_half-PMC / peak|Ez(probe)|_full-image at
     cpml_layers = 8, |R − 1| < 0.02 on each path. The identity is
     exact in exact arithmetic; the 0.02 is gate 1's own 2 %, re-used
     unchanged on an ABSOLUTE comparator (never widened), and sits
     ≥ 2× above the −40 dB 8-layer CPML reflection floor while being
     more than an order of magnitude inside the O(1) amplitude change
     the image-sign reversal produces.
  5. both uniform and NU paths must satisfy (1)–(4)
"""

from __future__ import annotations

import math
import os
import sys
import time
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
PROBE_Y = 5e-3
CPML_VALUES = (2, 4, 6, 8)
N_STEPS = 800

# --- gate 4 control arm: the mirrored full geometry (design note §4) ---
# H_tan mirror plane sits 0.5*dx inside the declared y_lo wall, so the
# 21 half-domain y nodes {0…20 mm} image onto {0…−19 mm}: the full
# symmetric domain is 39 mm with the plane 19.5 mm from each end.
MIRROR_PLANE_Y = 0.5 * DX                       # 0.5 mm, in half-domain coords
FULL_DOM = (DOM[0], DOM[1] + (DOM[1] - DX), DOM[2])   # (40, 39, 20) mm
Y_SHIFT = DOM[1] - DX                           # 19 mm: y_full = y_half + 19mm
FULL_SRC_Y = SRC_Y + Y_SHIFT                    # 20 mm
FULL_IMAGE_Y = (2.0 * MIRROR_PLANE_Y - SRC_Y) + Y_SHIFT   # 19 mm
FULL_PROBE_Y = PROBE_Y + Y_SHIFT                # 24 mm
CONTROL_CPML = max(CPML_VALUES)                 # 8 — thickest absorber
IMAGE_TOL = 0.02                                # pre-declared, commit a00a53d


def _waveform():
    return GaussianPulse(f0=5e9, bandwidth=1.0)


def _common_spec():
    return dict(
        freq_max=10e9, dx=DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pmc", hi="cpml"),
            z="cpml",
        ),
    )


def _nu_kwargs(dom):
    """dz_profile that reproduces the uniform mesh, to route the NU path."""
    return dict(dz_profile=np.full(int(round(dom[2] / DX)) + 1, DX))


def _face_h_tan(state):
    """max |H_tan| on the declared y_lo face, and over the whole array.

    On a `_lo` face apply_pmc_faces writes literal 0.0 into hx[:, 0, :]
    and hz[:, 0, :] (rfx/boundaries/pmc.py), after the CPML-H stage in
    the scan body, so nothing repopulates them before the state is
    returned. The domain maximum is the non-degeneracy guard.
    """
    hx = np.asarray(state.hx)
    hz = np.asarray(state.hz)
    return {
        "face_hx": float(np.max(np.abs(hx[:, 0, :]))),
        "face_hz": float(np.max(np.abs(hz[:, 0, :]))),
        "dom_h": float(max(np.max(np.abs(hx)), np.max(np.abs(hz)))),
    }


def _run_half(n_cpml: int, path: str) -> dict:
    kwargs = dict(_common_spec())
    if path == "nonuniform":
        kwargs.update(_nu_kwargs(DOM))
    sim = Simulation(domain=DOM, cpml_layers=n_cpml, **kwargs)
    sim.add_source(
        position=(20e-3, SRC_Y, 10e-3), component="ez",
        waveform=_waveform(),
    )
    sim.add_probe(position=(20e-3, PROBE_Y, 10e-3), component="ez")
    sim.preflight(strict=False)
    t0 = time.time()
    res = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(res.time_series).ravel()
    out = {
        "path": path,
        "cpml": n_cpml,
        "peak": float(np.max(np.abs(ts))),
        "tail_peak": float(np.max(np.abs(ts[-100:]))),
        "nan": bool(np.any(~np.isfinite(ts))),
        "wall": time.time() - t0,
    }
    out.update(_face_h_tan(res.state))
    return out


def run_uniform(n_cpml: int) -> dict:
    return _run_half(n_cpml, "uniform")


def run_nonuniform(n_cpml: int) -> dict:
    """Same problem, routed through the non-uniform z path."""
    return _run_half(n_cpml, "nonuniform")


def run_full_image(path: str, n_cpml: int = CONTROL_CPML) -> dict:
    """Gate 4 control leg — the mirrored FULL domain, no PMC face.

    Two in-phase Ez sources straddling the mirror plane replace the
    half-domain's single source plus its PMC image. CPML on both y
    faces. This leg declares no PMC face at all, so a defect in
    apply_pmc_faces cannot reach it.
    """
    kwargs = dict(
        freq_max=10e9, dx=DX,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    if path == "nonuniform":
        kwargs.update(_nu_kwargs(FULL_DOM))
    sim = Simulation(domain=FULL_DOM, cpml_layers=n_cpml, **kwargs)
    for y in (FULL_SRC_Y, FULL_IMAGE_Y):
        sim.add_source(
            position=(20e-3, y, 10e-3), component="ez",
            waveform=_waveform(),
        )
    sim.add_probe(position=(20e-3, FULL_PROBE_Y, 10e-3), component="ez")
    sim.preflight(strict=False)
    t0 = time.time()
    res = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(res.time_series).ravel()
    return {
        "path": path,
        "cpml": n_cpml,
        "peak": float(np.max(np.abs(ts))),
        "nan": bool(np.any(~np.isfinite(ts))),
        "wall": time.time() - t0,
    }


def evaluate_image_control(half_peak: float, full_peak: float) -> tuple[float, bool]:
    """Gate 4 verdict. Returns (R, ok)."""
    if not (full_peak > 0) or not np.isfinite(half_peak):
        return float("nan"), False
    ratio = half_peak / full_peak
    return ratio, bool(abs(ratio - 1.0) < IMAGE_TOL)


def _evaluate(results: list[dict], path_name: str, control: dict | None = None):
    peaks = np.array([r["peak"] for r in results])
    any_nan = any(r["nan"] for r in results)
    peak_max = float(peaks.max())
    peak_min = float(peaks.min())
    peak_range = (peak_max - peak_min) / peak_max if peak_max > 0 else float("nan")
    ok_stable = bool(peak_range < 0.02 and peak_max > 0)
    ok_finite = not any_nan

    # Gate 3 — PMC realization, bit-exact, with non-degeneracy guard.
    face_worst = max(max(r["face_hx"], r["face_hz"]) for r in results)
    dom_min = min(r["dom_h"] for r in results)
    ok_realized = bool(face_worst == 0.0 and dom_min > 0.0)

    print(f"\n  [{path_name}]  (gates: peak range < 2 %, no NaN, H_tan==0 on "
          f"y_lo, |R−1| < {IMAGE_TOL*100:.0f} %)")
    print(f"    {'cpml':>5} | {'peak':>12} | {'tail':>12} | {'nan':>5} | "
          f"{'max|H_tan| y_lo':>15} | {'max|H| domain':>13} | {'wall':>5}")
    print(f"    {'-'*5} | {'-'*12} | {'-'*12} | {'-'*5} | {'-'*15} | "
          f"{'-'*13} | {'-'*5}")
    for r in results:
        print(f"    {r['cpml']:>5d} | {r['peak']:>12.3e} | {r['tail_peak']:>12.3e} | "
              f"{str(r['nan']):>5s} | {max(r['face_hx'], r['face_hz']):>15.6e} | "
              f"{r['dom_h']:>13.3e} | {r['wall']:>4.1f}s")
    print(f"    G1 peak range = (max-min)/max = {peak_range*100:.3f} %   "
          f"{'PASS' if ok_stable else 'FAIL'}")
    print(f"    G2 any NaN                    = {any_nan}   "
          f"{'PASS' if ok_finite else 'FAIL'}")
    print(f"    G3 max|H_tan| on y_lo (bit-exact 0.0) = {face_worst!r}, "
          f"min max|H| in domain = {dom_min:.3e}   "
          f"{'PASS' if ok_realized else 'FAIL'}")

    ok_image = True
    if control is not None:
        ratio, ok_image = evaluate_image_control(control["half_peak"],
                                                 control["full_peak"])
        print(f"    G4 image control @ cpml={control['cpml']}: "
              f"half {control['half_peak']:.6e} / full {control['full_peak']:.6e} "
              f"= {ratio:.6f}, |R−1| = {abs(ratio-1)*100:.4f} %   "
              f"{'PASS' if ok_image else 'FAIL'}")

    return ok_stable and ok_finite and ok_realized and ok_image


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
    print(f"  gate-4 control: full domain "
          f"{tuple(f'{d*1e3:.0f}mm' for d in FULL_DOM)}, mirror plane at "
          f"y = {(MIRROR_PLANE_Y + Y_SHIFT)*1e3:.1f} mm, in-phase Ez pair at "
          f"y = {FULL_SRC_Y*1e3:.0f}/{FULL_IMAGE_Y*1e3:.0f} mm, "
          f"probe y = {FULL_PROBE_Y*1e3:.0f} mm, cpml = {CONTROL_CPML}")

    print("\n  [uniform path] ...")
    uniform_results = [run_uniform(n) for n in CPML_VALUES]

    print("  [non-uniform path (dz_profile, uniform xy)] ...")
    nu_results = [run_nonuniform(n) for n in CPML_VALUES]

    print("  [gate-4 control arm: mirrored full domain, no PMC face] ...")
    controls = {}
    for path, results in (("uniform", uniform_results),
                          ("nonuniform", nu_results)):
        full = run_full_image(path)
        half_peak = next(r["peak"] for r in results if r["cpml"] == CONTROL_CPML)
        controls[path] = {
            "cpml": CONTROL_CPML,
            "half_peak": half_peak,
            "full_peak": full["peak"],
            "wall": full["wall"],
        }
        print(f"    {path:>10}: full-image peak = {full['peak']:.6e} "
              f"({full['wall']:.1f}s)")

    pass_u = _evaluate(uniform_results, "uniform", controls["uniform"])
    pass_n = _evaluate(nu_results,     "nonuniform", controls["nonuniform"])

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
