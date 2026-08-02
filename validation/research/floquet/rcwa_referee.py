#!/usr/bin/env python3
"""External RCWA referee for issue #491 (Floquet calibrated S-params roadmap item).

SCOPE FENCE (read before extending this file): this script is the
COMPARATOR-LEG ONLY. It builds and validates an EXTERNAL reference
instrument (the `grcwa` RCWA package) so that a *future* rfx-side comparison
has something honest to compare against. It does **not** touch rfx's
Floquet port code, runs no rfx simulation, and makes **no** claim about
rfx's own calibrated S-parameters.

rfx-side status (`docs/agent-memory/rfx-known-issues.md`, "RCWA FEASIBILITY
SCOPED 2026-07-05" entry -- grep "RCWA"): the real blocker for a
Floquet-vs-RCWA comparison is DRIVE-ISOLATION on the rfx side (recovering a
physical |S11|<=1 from a real periodic-cell FDTD run), not the availability
of an RCWA reference -- RCWA is MIS-SCOPED as *the* bottleneck. Two-run
reference subtraction was ATTEMPTED and FAILED there (an eps_r=4 slab gave
non-physical |S11|~1.5 at 6 GHz band-center where the analytic Airy value is
0.09) and is an R2-STOP: **do not retry naive two-run subtraction**, and do
not run any rfx Floquet physics from this script or any script derived from
it. When rfx's drive-isolation problem is eventually solved, THIS referee is
the Tier-B external reference the roadmap item (#491 step 1) points at;
until then it stands alone.

Governance note (crossval-registry rule, `.claude/rules/rfx-feature-discovery.md`
+ `feedback_crossval_governance_glob_bypass.md`): this file lives at
`validation/research/floquet/` and is deliberately **OUTSIDE**
`validation/crossval/` and its `manifest.json`. It is a referee harness for
a capability that does not exist on the rfx side yet -- a script outside
`validation/crossval/` is exempt from crossval governance by construction,
so its presence here must NOT be read as a registered crossval pass. When
the rfx-side Floquet comparison lands, this script (or its successor) must
be MOVED into `validation/crossval/` and added to `manifest.json` at that
time -- not before.

What this referee CAN answer (once the rfx side exists): broadside
R(f)/T(f) of periodic unit cells -- uniform slabs today (the
referee-calibration check, gated below); patterned FSS (e.g. a Munk-class
dipole array, per #491 step 2) once a grcwa `Add_LayerGrid` case is added
here -- as an independent cross-check for rfx's future calibrated Floquet
S-parameters.

What this referee CANNOT answer: the rfx drive-isolation question above. An
external RCWA reference is moot until that is solved.

Tool chosen: `grcwa` 0.1.2 (https://github.com/weiliangjinca/grcwa, pip
installable, pure-Python + numpy/autograd, no compiler toolchain). Installed
cleanly via `pip install grcwa` in well under the task's 30-minute
environment budget. Alternatives considered: `S4` (build-heavy C++
extension -- avoided per the task's explicit steer away from build-heavy
tools); a standalone `rcwa` PyPI package was not evaluated because grcwa
installed and ran cleanly on the first attempt.

Step 1 -- REPRODUCE-GATE (`docs/agent-memory/task_recipes/external_solver_comparator.md`
discipline: reproduce the tool's own documented known-good result before any
comparison use). `_reproduce_grcwa_s4_regression()` below replicates,
parameter-for-parameter, grcwa's OWN committed regression test
(`tests/test_rcwa.py::test_rcwa` in the grcwa 0.1.2 source distribution,
https://github.com/weiliangjinca/grcwa/blob/master/tests/test_rcwa.py) --
verified two ways: (i) the grcwa GitHub repo carries NO tags (`git tag -l`
on a fresh clone of `master` returns empty), so "the grcwa 0.1.2 source
distribution" here means the PyPI sdist, confirmed by downloading it
(`pip download grcwa==0.1.2 --no-deps --no-binary :all:`) and inspecting
the tarball: `tests/test_rcwa.py` and `tests/test_kbloch.py` are both
present inside it; (ii) running that cloned repo's bundled suite
(`tests/test_rcwa.py` + `tests/test_kbloch.py`) against the installed `pip
install grcwa` package used here gave **17 passed** in this environment
(autograd is installed as a grcwa dependency here, which enables 7
additional gradient-check tests in `test_rcwa.py` that are skipped/undefined
without it -- an autograd-less environment would see 10 passed across the
same two files; state whichever count your own environment actually
produces rather than assuming one). The reproduce-gate itself
(`test_rcwa`) is a single patterned layer (100x100-pixel circular hole,
radius 0.4 of the unit cell, eps=12 fill / eps=1 background), nG=101,
oblique incidence theta=pi/18, phi=pi/9, compared against the two
DOCUMENTED expected transmittances the grcwa authors themselves
cross-validated against the independent RCWA implementation S4
(https://github.com/victorliu/S4): T_p = 0.85249901083265 and
T_s = 0.83900479939861, both to grcwa's own relative tolerance
`tolS4 = 1e-3`. Source name, documented numbers, reproduced numbers, and the
committed run log are recorded in the JSON artifact this script writes (see
`main()`); the log path is `validation/research/floquet/rcwa_referee_run.log`
(committed alongside this script -- generated by `python
validation/research/floquet/rcwa_referee.py > rcwa_referee_run.log 2>&1`).

Step 2 -- REFEREE-CALIBRATION SLAB CHECK (NOT independent evidence of RCWA
fidelity -- see the rescope below; instrument-plumbing pin only).
`run_referee_calibration_slab_check()` runs a uniform eps_r=4.0, d=10 mm
dielectric slab -- the SAME fixture rfx's own cv04
(`validation/crossval/04_multilayer_fresnel.py`) validates its TFSF/Fresnel
extraction against -- through grcwa at broadside (theta=phi=0) and one
retained order (nG=2, grcwa's minimum truncation that resolves to a single
diffraction order; see the nG gotcha note below) and gates the result
against the SAME closed-form Airy/Fresnel transfer-matrix formula cv04 uses
(`fresnel_slab_RT`, ported here with attribution, unmodified).

RESCOPE (adversarial review, PR #537): at one retained order, uniform
layers, and normal incidence, RCWA algebraically degenerates to the exact
same transfer-matrix problem the closed-form oracle solves --
`RT_Solve(normalize=1)`'s normalization factor `sqrt(eps0)/cos(theta)` is a
multiply-by-one at theta=0; the nondimensionalization's length unit L0
cancels identically since `freq_g * d_slab_g == (f*L0/C0) * (d/L0) ==
f*d/C0`; and the result is IDENTICAL (to ~1e-16) whether nG is 2, 9, 21, or
101 (verified) -- because there is no second order for a Fourier
factorization to act on. So the measured "~1e-15 max deviation" is expected
BY CONSTRUCTION, not evidence of RCWA's Fourier-factorization fidelity, and
must not be read as such. What this check actually pins: grcwa's absolute
R/T power scale (the internal "2R/2T" convention documented in
`RT_Solve`'s docstring -- a stray factor-of-2 error would show up here as a
clean 2x miss, not as noise), its layer ordering and semi-infinite-layer
handling, its forward-direction/eps sign conventions, and this script's
freq*thickness nondimensionalization -- all pinned against independent
analytic truth. This check exercises NO Fourier factorization, NO
multi-order diffraction bookkeeping, and NO oblique-incidence path; that
coverage lives in the Step 1 reproduce-gate instead (nG=101, `Epsilon_fft`
patterned-layer Fourier factorization, oblique theta=pi/18/phi=pi/9,
~8e-4 relative error vs S4 -- a real, non-trivial RCWA computation).

Hand-ported sanity checks (external scripts get no rfx preflight -- per
`external_solver_comparator.md` step 3, listing which checks were hand-ported
here): (a) energy conservation R+T=1 is checked at every referee-calibration
frequency (lossless dielectric, no loss channel available to absorb a
discrepancy); (b) the reproduce-gate's tolerance is grcwa's own committed
`tolS4`, not a loosened one; (c) the nG truncation was probed against
`grcwa.kbloch.Lattice_getG` before use: nG=1 pathologically truncates to
ZERO diffraction orders in grcwa 0.1.2 (`IndexError` at excitation time) --
nG=2 is the minimum that truncates to exactly one (zeroth) order and is what
"one retained order" means operationally in this script. This is a probed
gotcha an external tool does not warn about on its own, not an rfx preflight
check.

Run:
    python validation/research/floquet/rcwa_referee.py

Requires: grcwa (`pip install grcwa`). Skips cleanly with exit code 2 if
unavailable -- an unavailable optional dependency is inconclusive, not a
failure of the referee's own logic (mirrors the exit-code convention used by
`validation/crossval/*.py`, even though this script is intentionally outside
that registry -- see the governance note above).
"""

from __future__ import annotations

import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8  # m/s -- matches validation/crossval/04_multilayer_fresnel.py's C0


def fresnel_slab_RT(freqs_hz, eps_r, d_m):
    """Closed-form transfer-matrix R(f)/T(f) for a normal-incidence dielectric slab.

    Ported from `validation/crossval/04_multilayer_fresnel.py::fresnel_slab_RT`
    (unmodified physics) -- the SAME analytic oracle rfx's cv04 gates its own
    TFSF/Fresnel extraction against. Reused here as the referee-calibration
    check's oracle, not for any rfx comparison.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n = np.sqrt(eps_r)
    R = np.zeros_like(freqs_hz)
    T = np.zeros_like(freqs_hz)
    for i, f in enumerate(freqs_hz):
        if f <= 0:
            T[i] = 1.0
            continue
        delta = 2 * np.pi * f * n * d_m / C0
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        M00 = cos_d
        M01 = 1j * sin_d / n
        M10 = 1j * n * sin_d
        M11 = cos_d
        num = M00 + M01 - M10 - M11
        den = M00 + M01 + M10 + M11
        r = num / den
        t = 2.0 / den
        R[i] = np.abs(r) ** 2
        T[i] = np.abs(t) ** 2
    return R, T


def rcwa_slab_RT(freq_hz, eps_slab, d_slab_m, L0_m=1.0e-3, nG=2):
    """One-retained-order, broadside R/T of a uniform dielectric slab via grcwa.

    Nondimensionalizes with length unit L0_m (grcwa's natural units: c=1,
    eps0=1, mu0=1). nG=2 is grcwa's minimum truncation that resolves to
    exactly one (the zeroth) diffraction order for a square lattice at this
    order (`grcwa.kbloch.Lattice_getG`) -- nG=1 pathologically truncates to
    ZERO orders in grcwa 0.1.2; probed and avoided (see module docstring).
    NOTE: at one order + normal incidence this degenerates algebraically to
    the transfer-matrix problem `fresnel_slab_RT` solves -- see the module
    docstring's RESCOPE note for what this function does and does not prove.
    """
    import grcwa

    freq_g = freq_hz * L0_m / C0
    d_slab_g = d_slab_m / L0_m
    obj = grcwa.obj(nG, [1.0, 0.0], [0.0, 1.0], freq_g, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)  # input vacuum (semi-infinite)
    obj.Add_LayerUniform(d_slab_g, eps_slab)  # finite dielectric slab
    obj.Add_LayerUniform(1.0, 1.0)  # output vacuum (semi-infinite)
    obj.Init_Setup()
    obj.MakeExcitationPlanewave(1, 0.0, 0.0, 0.0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    return float(np.real(R)), float(np.real(T))


def run_referee_calibration_slab_check(freqs_hz, eps_slab=4.0, d_slab_m=10.0e-3):
    """Gate grcwa's broadside one-order slab R/T against the closed-form oracle.

    Fixture matches rfx's own cv04 slab (eps_r=4.0, d=10 mm) -- chosen so a
    future rfx-side comparison already shares one reference point with this
    referee's own calibration case. This is instrument-plumbing calibration
    (power scale, layer ordering, unit conventions), NOT evidence of RCWA's
    Fourier-factorization fidelity -- see the module docstring's RESCOPE note.
    """
    R_ref, T_ref = fresnel_slab_RT(freqs_hz, eps_slab, d_slab_m)
    rows = []
    max_dev_R = 0.0
    max_dev_T = 0.0
    max_energy_dev = 0.0
    for i, f in enumerate(freqs_hz):
        R_g, T_g = rcwa_slab_RT(f, eps_slab, d_slab_m)
        dev_R = abs(R_g - float(R_ref[i]))
        dev_T = abs(T_g - float(T_ref[i]))
        energy_dev = abs(R_g + T_g - 1.0)
        max_dev_R = max(max_dev_R, dev_R)
        max_dev_T = max(max_dev_T, dev_T)
        max_energy_dev = max(max_energy_dev, energy_dev)
        rows.append(
            {
                "freq_hz": float(f),
                "R_rcwa": R_g,
                "T_rcwa": T_g,
                "R_analytic": float(R_ref[i]),
                "T_analytic": float(T_ref[i]),
                "dev_R": dev_R,
                "dev_T": dev_T,
                "R_plus_T_rcwa": R_g + T_g,
            }
        )
    gate_threshold = 1e-6
    return {
        "eps_slab": eps_slab,
        "d_slab_m": d_slab_m,
        "rows": rows,
        "max_dev_R": max_dev_R,
        "max_dev_T": max_dev_T,
        "max_energy_conservation_dev": max_energy_dev,
        "gate_threshold": gate_threshold,
        "passed": bool(
            max_dev_R < gate_threshold
            and max_dev_T < gate_threshold
            and max_energy_dev < gate_threshold
        ),
    }


def _reproduce_grcwa_s4_regression() -> dict:
    """Reproduce grcwa's own S4-cross-validated regression case.

    Parameter-for-parameter from `tests/test_rcwa.py::test_rcwa` in the
    grcwa 0.1.2 source distribution (cloned from
    https://github.com/weiliangjinca/grcwa; verified to pass, unmodified,
    against the `pip install grcwa` package used here): a single
    100x100-pixel circular hole (radius 0.4 of the unit cell, eps=12.0 fill
    / eps=1.0 background) in a 0.1x0.1 square lattice, illuminated obliquely
    (theta=pi/18, phi=pi/9) at the tool's normalized freq=1.0. The two
    expected transmittances below are the grcwa authors' own S4
    cross-validation numbers (`tolS4 = 1e-3`, relative).
    """
    import grcwa

    nG = 101
    L1 = [0.1, 0.0]
    L2 = [0.0, 0.1]
    Nx = Ny = 100
    freq = 1.0
    theta = np.pi / 18
    phi = np.pi / 9
    thick0 = 1.0
    thickN = 1.0
    pthick = 0.2
    radius = 0.4

    x0 = np.linspace(0.0, 1.0, Nx)
    y0 = np.linspace(0.0, 1.0, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    epgrid = np.ones((Nx, Ny), dtype=float)
    epgrid[(x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2] = 12.0
    epgrid = epgrid.flatten()

    def solve(p_amp, s_amp):
        obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
        obj.Add_LayerUniform(thick0, 1.0)
        obj.Add_LayerGrid(pthick, Nx, Ny)
        obj.Add_LayerUniform(thickN, 1.0)
        obj.Init_Setup(Gmethod=0)
        obj.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
        obj.GridLayer_geteps(epgrid)
        _, T = obj.RT_Solve(normalize=0)
        return float(np.real(T))

    T_p = solve(1, 0)
    T_s = solve(0, 1)

    expected_T_p = 0.85249901083265
    expected_T_s = 0.83900479939861
    tolS4 = 1e-3

    rel_err_p = abs(T_p - expected_T_p) / expected_T_p
    rel_err_s = abs(T_s - expected_T_s) / expected_T_s

    return {
        "source": (
            "grcwa tests/test_rcwa.py::test_rcwa (grcwa 0.1.2, "
            "https://github.com/weiliangjinca/grcwa/blob/master/tests/test_rcwa.py)"
        ),
        "documented_reference": (
            "S4 (https://github.com/victorliu/S4), per grcwa's own tolS4=1e-3 comparison"
        ),
        "geometry": (
            "single 100x100 grid circular hole, r=0.4 of unit cell, eps 12/1, "
            "L=[0.1,0.1], theta=pi/18, phi=pi/9, freq=1.0, nG=101"
        ),
        "p_pol": {
            "expected_T": expected_T_p,
            "reproduced_T": T_p,
            "rel_err": rel_err_p,
            "tol": tolS4,
        },
        "s_pol": {
            "expected_T": expected_T_s,
            "reproduced_T": T_s,
            "rel_err": rel_err_s,
            "tol": tolS4,
        },
        "passed": bool(rel_err_p < tolS4 and rel_err_s < tolS4),
    }


def main() -> int:
    try:
        import grcwa
    except ImportError:
        print(
            "SKIP: grcwa is not installed (`pip install grcwa`). This "
            "referee cannot run without it -- exit 2 (inconclusive), NOT a "
            "failure of the referee's own logic."
        )
        return 2

    print("=" * 70)
    print("External RCWA referee for rfx issue #491 -- COMPARATOR LEG ONLY")
    print("No rfx code is exercised by this script. See module docstring")
    print("for the R2-STOP scope fence (rfx drive-isolation is NOT in scope).")
    print("=" * 70)

    print("\n--- Step 1: reproduce-gate (grcwa's own S4 regression) ---")
    reproduce = _reproduce_grcwa_s4_regression()
    print(f"  source: {reproduce['source']}")
    print(f"  documented reference: {reproduce['documented_reference']}")
    print(
        f"  p-pol T: reproduced={reproduce['p_pol']['reproduced_T']:.11f} "
        f"expected={reproduce['p_pol']['expected_T']:.11f} "
        f"rel_err={reproduce['p_pol']['rel_err']:.2e} (tol {reproduce['p_pol']['tol']:.0e})"
    )
    print(
        f"  s-pol T: reproduced={reproduce['s_pol']['reproduced_T']:.11f} "
        f"expected={reproduce['s_pol']['expected_T']:.11f} "
        f"rel_err={reproduce['s_pol']['rel_err']:.2e} (tol {reproduce['s_pol']['tol']:.0e})"
    )
    print(f"  reproduce-gate PASSED: {reproduce['passed']}")

    print(
        "\n--- Step 2: referee-calibration slab check (uniform slab vs "
        "closed-form Fresnel; instrument-plumbing pin, NOT RCWA-fidelity "
        "evidence -- see module docstring RESCOPE) ---"
    )
    freqs_hz = np.linspace(1.0e9, 20.0e9, 20).tolist()
    calibration = run_referee_calibration_slab_check(freqs_hz)
    for row in calibration["rows"]:
        print(
            f"  {row['freq_hz']/1e9:6.2f} GHz  R_rcwa={row['R_rcwa']:.6f} "
            f"R_analytic={row['R_analytic']:.6f}  T_rcwa={row['T_rcwa']:.6f} "
            f"T_analytic={row['T_analytic']:.6f}  R+T={row['R_plus_T_rcwa']:.6f}"
        )
    print(f"  max|R_rcwa-R_analytic| = {calibration['max_dev_R']:.3e}")
    print(f"  max|T_rcwa-T_analytic| = {calibration['max_dev_T']:.3e}")
    print(
        f"  max energy-conservation deviation |R+T-1| = "
        f"{calibration['max_energy_conservation_dev']:.3e}"
    )
    print(f"  referee-calibration slab check PASSED: {calibration['passed']}")

    overall_passed = bool(reproduce["passed"] and calibration["passed"])

    artifact = {
        "issue": 491,
        "scope": "comparator leg only -- no rfx Floquet comparison performed by this script",
        "r2_stop_pointer": (
            "docs/agent-memory/rfx-known-issues.md, 'RCWA FEASIBILITY SCOPED "
            "2026-07-05' entry -- rfx drive-isolation is the blocker, not RCWA "
            "availability; naive two-run reference subtraction FAILED there and "
            "must not be retried"
        ),
        "tool": {"name": "grcwa", "version": grcwa.__version__},
        "log_path": "validation/research/floquet/rcwa_referee_run.log",
        "reproduce_gate": reproduce,
        "referee_calibration_slab_check": calibration,
        "overall_passed": overall_passed,
    }

    artifact_path = os.path.join(SCRIPT_DIR, "rcwa_referee_artifact.json")
    with open(artifact_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print("\nArtifact written: validation/research/floquet/rcwa_referee_artifact.json")

    print("\n" + "=" * 70)
    if overall_passed:
        print(
            "ALL CHECKS PASSED (referee instrument calibration only -- "
            "this is NOT an rfx Floquet comparison)"
        )
    else:
        print("SOME CHECKS FAILED")
    print("=" * 70)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
