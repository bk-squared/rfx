"""Producer for the PEC-sphere three-way RCS fixture (campaign Lane 1).

Emits the COMPLETE ``fixture.json`` mechanically (Bempp BEM column + exact-Mie
column re-derived from scipy + rfx-fine value sourced from the sibling
``rcs_sphere_mie/fixture.json`` + provenance), so regeneration needs no hand
editing. Independent surface-integral-equation (BEM) cross-check of monostatic
PEC-sphere backscatter RCS: Bempp meshes the TRUE curved surface with triangles
(no FDTD staircase), so it confirms exact Mie with a *different error class* than
any FDTD code (Meep/openEMS).

Bempp is NOT a CI/runtime dependency of rfx; the emitted numbers are frozen into
fixture.json and gated by ``tests/test_rcs_sphere_three_way_gates.py`` (which
re-derives Mie from scipy.special, so this producer cannot self-certify).

Environment (measured, not assumed):
  * ``pip install bempp-cl gmsh`` — the import name is ``bempp_cl`` in 0.4.x
    (NOT ``bempp``); pyopencl absent + numba present -> Numba CPU backend.
  * On a many-core box set ``OPENBLAS_NUM_THREADS<=64`` (OpenBLAS core-dumps
    above its thread cap): ``OPENBLAS_NUM_THREADS=32 python generate_bempp.py``.

Convention (aligned with rfx/rcs.py + the Mie oracle): incidence +x, E||z,
|E_inc|=1; monostatic bin = -x backscatter; sigma = 4*pi*|E_inf|^2/|E_inc|^2,
reported as sigma/(pi a^2). (RCS is a magnitude, so it is convention-insensitive;
harness correctness is established by the two discriminating witnesses the gate
checks: the 4-ka zero-straddle and the monotone h-refinement convergence.)
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn, spherical_yn

import bempp_cl.api as bem
from bempp_cl.api.linalg import gmres

A = 0.015                          # sphere radius (m); sigma/(pi a^2) depends only on ka
D = np.array([1.0, 0.0, 0.0])      # incidence +x
POL = np.array([0.0, 0.0, 1.0])    # E || z, |E_inc| = 1
KA_LADDER = (0.8, 1.0, 1.5, 2.0)   # E4 ladder (matches rcs_mie_e4)

_HERE = Path(__file__).resolve().parent
_RFX_FINE = _HERE.parent / "rcs_sphere_mie" / "fixture.json"


def mie_ratio(ka: float) -> float:
    """Exact PEC-sphere backscatter sigma/(pi a^2) (Ruck 1970) via scipy."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp_, ynp_ = spherical_jn(n, x, derivative=True), spherical_yn(n, x, derivative=True)
    hn, hnp_ = jn + 1j * yn, jnp_ + 1j * ynp_
    a_n = jn / hn
    b_n = (jn + x * jnp_) / (hn + x * hnp_)
    return float(np.abs(np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))) ** 2 / x ** 2)


def bempp_sigma_over_pi_a2(ka: float, h: float):
    """Backscatter sigma/(pi a^2) via EFIE at mesh element size h."""
    k = ka / A
    grid = bem.shapes.sphere(r=A, h=h)
    rwg = bem.function_space(grid, "RWG", 0)
    snc = bem.function_space(grid, "SNC", 0)
    efie = bem.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k)

    @bem.complex_callable
    def tangential_trace(x, n, domain_index, result):
        E = POL * np.exp(1j * k * np.dot(D, x))
        result[:] = np.cross(E, n)

    trace = bem.GridFunction(rwg, fun=tangential_trace, dual_space=snc)
    sol, _ = gmres(efie, trace, tol=1e-6)
    ff = bem.operators.far_field.maxwell.electric_field(
        space=rwg, points=(-D).reshape(3, 1), wavenumber=k)
    e_inf = ff.evaluate(sol)
    sigma = 4.0 * np.pi * np.sum(np.abs(e_inf) ** 2)   # |E_inc| = 1
    return float(sigma / (np.pi * A ** 2)), int(rwg.global_dof_count)


def main():
    t0 = time.time()
    ladder = []
    for ka in KA_LADDER:
        lam = 2 * np.pi * A / ka
        h = min(A / 6.0, lam / 10.0)                 # small ka -> geometry-limited
        s, n = bempp_sigma_over_pi_a2(ka, h)
        mie = mie_ratio(ka)
        ladder.append({"ka": ka, "h_m": h, "N_dofs": n,
                       "sigma_bempp_over_pi_a2": s, "sigma_mie_over_pi_a2": mie,
                       "dB_bempp_vs_mie": 10.0 * np.log10(s / mie)})
        print(f"ka={ka}: N={n} bempp={s:.4f} mie={mie:.4f} dB={10*np.log10(s/mie):+.3f}")

    convergence = []
    for frac in (3, 5, 8):                            # h -> h/2 witness at ka=1
        s, n = bempp_sigma_over_pi_a2(1.0, A / frac)
        convergence.append({"h_m": A / frac, "N_dofs": n,
                            "sigma_bempp_over_pi_a2": s,
                            "dB_vs_mie": 10.0 * np.log10(s / mie_ratio(1.0))})
        print(f"conv ka=1 h=a/{frac}: N={n} bempp={s:.4f}")

    rfx_fine = json.loads(_RFX_FINE.read_text())
    rfx_val = rfx_fine["monostatic"]["rfx_sigma_over_pi_a2"]
    mie1 = mie_ratio(1.0)
    bempp1 = next(r["sigma_bempp_over_pi_a2"] for r in ladder if r["ka"] == 1.0)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()

    fixture = {
        "schema": "rfx.rcs_sphere_three_way", "schema_version": 1,
        "issue": "validation-campaign lane-1",
        "claim_scope": (
            "Independent surface-integral-equation (Bempp-cl EFIE, RWG/SNC) cross-check "
            "of monostatic PEC-sphere BACKSCATTER RCS vs the exact Mie series, across the "
            "E4 ka ladder {0.8,1.0,1.5,2.0}, plus a three-way point (exact-Mie / "
            "rfx-FDTD-fine / Bempp-BEM) at ka~1. Bempp meshes the TRUE CURVED surface with "
            "triangles (no FDTD staircase), so it is an independent-METHOD confirmation of "
            "the Mie reference and a non-FDTD witness that rfx's coarse-ladder error is "
            "rfx-side resolution, not a comparator/extraction artifact. This is NOT a "
            "bistatic validation and does not re-open or weaken any existing RCS gate."),
        "radius_m": A,
        "convention": {"incidence": "+x", "pol": "E||z", "backscatter": "-x",
                       "rcs": "sigma=4pi|E_inf|^2/|E_inc|^2, reported /(pi a^2)",
                       "far_field_op": "operators.far_field.maxwell.electric_field",
                       "note": "RCS is a magnitude -> convention-insensitive; harness "
                               "correctness rests on the 4-ka zero-straddle + monotone "
                               "h-convergence witnesses, not on any sign/conjugation check."},
        "bempp": {
            "solver": "bempp-cl EFIE (RWG domain/range, SNC dual), Numba CPU backend",
            "version": bem.__version__,
            "ladder": ladder,
            "convergence_ka1": convergence,
            "floor_measured_db": max(abs(r["dB_bempp_vs_mie"]) for r in ladder),
            "wall_time_s": time.time() - t0,
        },
        "three_way_ka1": {
            "ka": 1.0,
            "sigma_mie_over_pi_a2": mie1,
            "rfx_fine_over_pi_a2": rfx_val,
            "rfx_fine_source": (
                "tests/fixtures/rcs_sphere_mie/fixture.json monostatic "
                f"(ka={rfx_fine['geometry']['ka']:.4f}, "
                f"dx=lambda/{rfx_fine['geometry']['resolution_cells_per_lambda']}, "
                f"{rfx_fine['geometry']['cells_per_radius']:.2f} cells/radius; the "
                "0.0004 dB ka=0.9997-vs-1.0 mismatch is below any rounded figure)"),
            "bempp_over_pi_a2": bempp1,
            "spread_db": {
                "rfx_vs_mie": 10 * np.log10(rfx_val / mie1),
                "bempp_vs_mie": 10 * np.log10(bempp1 / mie1),
                "rfx_vs_bempp": 10 * np.log10(rfx_val / bempp1),
            },
        },
        "rfx_coarse_ladder_note": (
            "rfx's E4 COARSE ladder (dx=lambda/10-15, only 1-5 cells/radius; "
            "tests/fixtures/rcs_mie_e4/, gate 13.914 dB under a 15 dB GO floor) carries "
            "4.7-9.3 dB error. Bempp reproduces exact Mie to <=0.15 dB at the SAME ka with a "
            "staircase-free curved mesh, so that coarse-ladder gap is rfx-side resolution "
            "(staircasing + near-field NTFF box; the sibling fixture also records a large "
            "domain-size swing), independently confirmed by a non-FDTD method -- consistent "
            "with the two-regime finding (rcs_sphere_mie fine ~0.06 dB). Stated as an "
            "rfx-centric distance; no solver is framed as wrong."),
        "provenance": {
            "rfx_commit": sha, "bempp_version": bem.__version__,
            "producer": "tests/fixtures/rcs_sphere_three_way/generate_bempp.py",
            "env": "OPENBLAS_NUM_THREADS<=64 required (192-core box core-dumps otherwise); "
                   "import name is bempp_cl (not bempp) in 0.4.x",
        },
    }
    (_HERE / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote fixture.json in {fixture['bempp']['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
