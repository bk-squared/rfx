#!/usr/bin/env python3
"""Build the exact-Mie-series cross-method REFERENCE for rfx's RCS pipeline.

THE REFERENCE QUESTION
----------------------
rfx ships an FDTD RCS pipeline (``rfx/rcs.py``, TFSF illumination + NTFF
transform, Taflove & Hagness Ch. 8-9). Its committed tests
(``tests/test_rcs.py``) gate the PEC sphere only against the geometrical-optics
(GO) large-sphere limit ``sigma = pi a^2`` with a generous +/-15 dB tolerance,
and the PEC plate against a physical-optics (PO) estimate. Neither is an exact
solution: at ``ka ~ 1`` the true monostatic RCS oscillates well away from the GO
limit (the first Mie maximum sits ~5.6 dB above ``pi a^2``). There is no exact
analytic cross-method check in the RCS lane.

The exact conducting-sphere Mie series IS that check. It is a closed-form
analytic solution (no FDTD, no FEM, no mesh), so it is the strongest possible
arbiter for the sphere. This script:

  (a) ``mie_pec_backscatter_*`` -- the exact series (pure scipy/numpy), the
      oracle. Validated against two closed-form anchors: the Rayleigh limit
      ``sigma/(pi a^2) -> 9 (ka)^4`` as ``ka -> 0`` and the optical limit
      ``sigma/(pi a^2) -> 1`` as ``ka -> inf``.
  (b) runner mode -- computes rfx monostatic RCS for a small ``ka`` ladder at
      TWO grid resolutions (lambda/10 and lambda/15) for a convergence witness,
      matching the committed ``test_rcs.py`` sphere geometry (radius 15 mm,
      0.10 m cubic domain, 8 CPML, ez pol, 37 observation angles).
  (c) fixture-builder mode -- writes
      ``tests/fixtures/rcs_mie_e4/rcs_pec_sphere_mie.json`` with the raw per-ka
      rfx values + exact Mie values + errors + meta, so the gate test survives a
      clean checkout WITHOUT re-running FDTD (the committed reference column is
      re-derivable from the series by ``tests/test_rcs_mie_reference_gates.py``).

HONEST-ENVELOPE POSTURE
-----------------------
A staircased curved-PEC sphere on a Cartesian grid carries an O(dx) surface
error that refinement narrows but does NOT close (conformal PEC is not wired on
this path). The deliverable is therefore the MEASURED agreement envelope across
the ka ladder, not a pre-chosen dB target. We do not enter a refinement loop
chasing a number; we report the distance rfx sits from the exact series.

Mie backscatter convention (Ruck, Radar Cross Section Handbook, 1970):

    sigma / (pi a^2) = (1/(ka)^2) | sum_{n=1}^inf (-1)^n (2n+1) (a_n - b_n) |^2

with x = ka and PEC Mie coefficients

    a_n = j_n(x) / h_n(x)
    b_n = [x j_n(x)]' / [x h_n(x)]'        (Riccati-Bessel derivative)

h_n = j_n + i y_n (spherical Hankel, 1st kind); |a_n - b_n| is invariant to the
Hankel-kind choice, so the RCS magnitude is convention-independent.

Usage::

    python scripts/diagnostics/build_rcs_mie_reference.py                    # verdict from fixture
    python scripts/diagnostics/build_rcs_mie_reference.py --run --commit SHA # rerun FDTD ladder + rebuild fixture
    python scripts/diagnostics/build_rcs_mie_reference.py --anchors          # self-validate the Mie oracle

Pair ``--run`` with ``--commit <sha>``: the default stamp is "uncommitted",
which the fixture meta-integrity gate rejects on purpose (forces a real SHA).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import spherical_jn, spherical_yn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests/fixtures/rcs_mie_e4"
_FIXTURE = "rcs_pec_sphere_mie.json"

# Committed-test sphere geometry (tests/test_rcs.py::TestRCSPECSphere).
_RADIUS_M = 0.015
_DOMAIN_M = 0.10
_CPML = 8
_N_THETA = 37
_POL = "ez"

# ka ladder + the two convergence-witness resolutions (cells per wavelength).
_KA_LADDER = (0.8, 1.0, 1.5, 2.0)
_RESOLUTIONS = {"coarse": 10, "fine": 15}

# Domain-robustness witness (R5): at test-scale domains the NTFF box sits only
# 1-2 wavelengths from the sphere, so the extracted monostatic magnitude is not
# a converged quantity. We sweep the domain at two ka to record the swing as
# committed evidence -- a single dBsm number is NOT sufficient (CLAUDE.md R5).
_DOMAIN_WITNESS_KA = (1.0, 1.5)
_DOMAIN_WITNESS_M = (0.10, 0.15, 0.20)


# --------------------------------------------------------------------------- #
# (a) The exact Mie oracle (pure scipy/numpy -- no FDTD).
# --------------------------------------------------------------------------- #
def mie_pec_backscatter_ratio(ka: float, n_max: int | None = None) -> float:
    """Exact monostatic RCS of a PEC sphere, normalised: sigma / (pi a^2)."""
    x = float(ka)
    if n_max is None:
        n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)

    jn = spherical_jn(n, x)
    jnp_ = spherical_jn(n, x, derivative=True)
    yn = spherical_yn(n, x)
    ynp_ = spherical_yn(n, x, derivative=True)

    hn = jn + 1j * yn
    hnp_ = jnp_ + 1j * ynp_

    a_n = jn / hn                      # electric (TM) coefficient
    b_n = (jn + x * jnp_) / (hn + x * hnp_)  # magnetic (TE), Riccati-Bessel'

    series = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))
    return float(np.abs(series) ** 2 / x ** 2)


def mie_pec_backscatter_sigma_m2(ka: float, a_radius: float) -> float:
    """Exact monostatic RCS in m^2."""
    return mie_pec_backscatter_ratio(ka) * np.pi * a_radius ** 2


def mie_pec_backscatter_dbsm(ka: float, a_radius: float) -> float:
    """Exact monostatic RCS in dBsm."""
    return float(10.0 * np.log10(mie_pec_backscatter_sigma_m2(ka, a_radius)))


def _self_anchor_report() -> None:
    """Validate the oracle against its two closed-form limits (CHECK-MWE-FIRST)."""
    print("Rayleigh limit  sigma/(pi a^2) -> 9 (ka)^4 :")
    for ka in (0.01, 0.02, 0.05):
        got, want = mie_pec_backscatter_ratio(ka), 9.0 * ka ** 4
        print(f"  ka={ka:.3f}: got {got:.6e}  target {want:.6e}  ratio {got/want:.5f}")
    print("Optical limit   sigma/(pi a^2) -> 1 :")
    for ka in (20.0, 50.0, 100.0):
        print(f"  ka={ka:6.1f}: {mie_pec_backscatter_ratio(ka):.5f}")
    print("Canonical curve (first Mie max near ka~1) :")
    for ka in _KA_LADDER:
        r = mie_pec_backscatter_ratio(ka)
        print(f"  ka={ka:.3f}: sigma/(pi a^2)={r:.4f}  ({10*np.log10(r):+.3f} dB rel pi a^2)")


# --------------------------------------------------------------------------- #
# (b) rfx runner -- monostatic RCS at a given ka and resolution.
# --------------------------------------------------------------------------- #
def run_rfx_mono(
    ka: float, cells_per_lambda: int, domain_m: float = _DOMAIN_M
) -> dict[str, Any]:
    """Run rfx's RCS pipeline on a PEC sphere at this ka / resolution / domain.

    Fixed physical geometry (radius, CPML, angles) matches the committed
    ``test_rcs.py`` sphere; ka is swept by frequency, f0 = ka c / (2 pi a). The
    grid step is lambda/cells_per_lambda; n_steps scales to hold the physical
    sim duration roughly constant across ka, resolution, and domain size.
    """
    import jax.numpy as jnp
    from rfx.grid import Grid, C0
    from rfx.geometry.csg import Sphere, rasterize
    from rfx.core.yee import MaterialArrays
    from rfx.rcs import compute_rcs

    f0 = ka * C0 / (2.0 * np.pi * _RADIUS_M)
    lam = C0 / f0
    dx = lam / cells_per_lambda

    grid = Grid(
        freq_max=f0 * 1.5,
        domain=(domain_m, domain_m, domain_m),
        dx=dx,
        cpml_layers=_CPML,
    )
    center = (domain_m / 2, domain_m / 2, domain_m / 2)
    sphere = Sphere(center=center, radius=_RADIUS_M)
    eps_r, sigma = rasterize(grid, [(sphere, 1.0, 1e7)])  # PEC_SIGMA
    materials = MaterialArrays(
        eps_r=eps_r, sigma=sigma, mu_r=jnp.ones(grid.shape, dtype=jnp.float32)
    )

    # Hold physical duration ~ constant: the committed test uses 400 steps at
    # f0=3 GHz, lambda/10, 0.10 m. dt ~ dx, so steps scale with f0, resolution,
    # and (to keep the same number of domain crossings) the domain size.
    n_steps = int(round(
        400 * (f0 / 3.0e9) * (cells_per_lambda / 10.0) * (domain_m / _DOMAIN_M)
    ))

    theta_obs = np.linspace(0.01, np.pi - 0.01, _N_THETA)
    phi_obs = np.array([0.0])

    t0 = time.time()
    result = compute_rcs(
        grid, materials, n_steps,
        f0=f0, bandwidth=0.5, theta_inc=0.0, polarization=_POL,
        theta_obs=theta_obs, phi_obs=phi_obs, freqs=np.array([f0]),
        boundary="cpml", cpml_layers=_CPML,
    )
    wall = time.time() - t0

    mie_dbsm = mie_pec_backscatter_dbsm(ka, _RADIUS_M)
    rfx_dbsm = float(result.monostatic_rcs[0])
    return {
        "ka": float(ka),
        "f0_hz": float(f0),
        "lambda_m": float(lam),
        "dx_m": float(dx),
        "cells_per_lambda": int(cells_per_lambda),
        "domain_m": float(domain_m),
        "radius_cells": float(_RADIUS_M / dx),
        "grid_shape": [int(s) for s in grid.shape],
        "n_steps": int(n_steps),
        "rfx_mono_dbsm": rfx_dbsm,
        "mie_dbsm": mie_dbsm,
        "err_db": float(rfx_dbsm - mie_dbsm),
        "wall_s": round(wall, 1),
    }


# --------------------------------------------------------------------------- #
# (c) Fixture builder / verdict.
# --------------------------------------------------------------------------- #
def _meta(commit: str) -> dict[str, Any]:
    import rfx
    return {
        "reference": "exact-mie-series-pec-sphere",
        "method": "analytic conducting-sphere Mie backscatter series "
        "(Ruck 1970), pure scipy.special -- no FDTD/FEM/mesh",
        "solver_under_test": "rfx.rcs.compute_rcs (TFSF + NTFF, Taflove Ch.8-9)",
        "mie_convention": "sigma/(pi a^2) = (1/(ka)^2)|sum (-1)^n (2n+1)(a_n-b_n)|^2; "
        "a_n=j_n/h_n, b_n=[x j_n]'/[x h_n]'",
        "radius_m": _RADIUS_M,
        "domain_m": _DOMAIN_M,
        "cpml_layers": _CPML,
        "polarization": _POL,
        "n_theta_obs": _N_THETA,
        "resolutions_cells_per_lambda": _RESOLUTIONS,
        "ka_ladder": list(_KA_LADDER),
        "rfx_version": rfx.__version__,
        "commit": commit,
        "date": "2026-07-07",
        "committed_test_point": {
            "test": "tests/test_rcs.py::TestRCSPECSphere::test_rcs_pec_sphere_mie",
            "ka": 0.9431,
            "f0_hz": 3.0e9,
            "note": "one-off R2 go/no-go falsifier at the committed test point "
            "(f0=3 GHz, lambda/10, ka=0.9431 -- NOT on the ladder) measured "
            "|rfx - Mie| ~ 4.9 dB. This was the falsifier check, NOT a committed "
            "gate: the committed sphere gate is +/-15 dB vs the GO limit, and the "
            "exact-Mie envelope here is +/-10-15 dB (see meta.finding).",
        },
        "note": "raw arrays hold rfx monostatic dBsm + exact Mie dBsm + errors; "
        "the Mie column is re-derivable from the series (scipy) by the gate test.",
        "finding": "The exact Mie series CONFIRMS the committed test's +/-15 dB GO "
        "tolerance rather than tightening it. rfx's monostatic RCS magnitude does "
        "not converge with mesh refinement (convergence_delta mostly positive) or "
        "with domain size (domain_robustness swings several dB) at these test-scale "
        "1-2 wavelength domains, where the NTFF box sits close to the sphere. The "
        "ka~1 error (~4.9 dB) is within 5 dB but is not a converged agreement. A "
        "dedicated RCS diagnostic session (larger domains, NTFF-box placement, "
        "source-spectrum SNR at f0) is the recommended follow-up; this fixture does "
        "not chase the residual.",
    }


def build_fixture(run: bool, commit: str) -> dict[str, Any]:
    """Build (or reload) the ka-ladder fixture dict.

    If ``run`` is False and the fixture exists, the committed rfx values are
    reused (no FDTD); only the meta/commit is refreshed if asked.
    """
    fixtures = _FIXTURES
    fixtures.mkdir(parents=True, exist_ok=True)
    out = fixtures / _FIXTURE

    if run:
        rungs: list[dict[str, Any]] = []
        for ka in _KA_LADDER:
            row: dict[str, Any] = {"ka": float(ka)}
            for name, cpl in _RESOLUTIONS.items():
                res = run_rfx_mono(ka, cpl)
                print(
                    f"  ka={ka:.2f} {name:6s} (lam/{cpl}): grid={res['grid_shape']} "
                    f"steps={res['n_steps']} rfx={res['rfx_mono_dbsm']:+.2f} "
                    f"mie={res['mie_dbsm']:+.2f} err={res['err_db']:+.2f}dB "
                    f"[{res['wall_s']}s]"
                )
                row[name] = {k: v for k, v in res.items() if k != "wall_s"}
            row["mie_dbsm"] = mie_pec_backscatter_dbsm(ka, _RADIUS_M)
            rungs.append(row)

        witness: list[dict[str, Any]] = []
        print("  domain-robustness witness (lam/10):")
        for ka in _DOMAIN_WITNESS_KA:
            sweep = []
            for dom in _DOMAIN_WITNESS_M:
                res = run_rfx_mono(ka, 10, domain_m=dom)
                print(
                    f"    ka={ka:.2f} domain={dom*100:.0f}cm: grid={res['grid_shape']} "
                    f"rfx={res['rfx_mono_dbsm']:+.2f} err={res['err_db']:+.2f}dB"
                )
                sweep.append({k: v for k, v in res.items() if k != "wall_s"})
            witness.append({
                "ka": float(ka),
                "mie_dbsm": mie_pec_backscatter_dbsm(ka, _RADIUS_M),
                "sweep": sweep,
            })

        fixture = {"meta": _meta(commit), "ka_ladder": rungs, "domain_robustness": witness}
        fixture["envelope"] = _envelope(rungs, witness)
        out.write_text(json.dumps(fixture, indent=2) + "\n")
        return fixture

    fixture = json.loads(out.read_text())
    return fixture


def _envelope(
    rungs: list[dict[str, Any]], witness: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Measured agreement envelope across the ladder (per resolution)."""
    env: dict[str, Any] = {}
    for name in _RESOLUTIONS:
        errs = [abs(r[name]["err_db"]) for r in rungs]
        env[name] = {
            "max_abs_err_db": round(max(errs), 4),
            "mean_abs_err_db": round(float(np.mean(errs)), 4),
        }
    # Convergence witness: fine minus coarse |err| per rung (negative = improved).
    # A positive value means refinement did NOT close the gap.
    env["convergence_delta_db"] = [
        round(abs(r["fine"]["err_db"]) - abs(r["coarse"]["err_db"]), 4) for r in rungs
    ]
    # Domain-robustness swing: max-min monostatic dBsm across the domain sweep at
    # each witness ka. A large swing means the single-frequency magnitude is not
    # a converged quantity at test-scale domains.
    if witness:
        env["domain_swing_db"] = [
            round(max(s["rfx_mono_dbsm"] for s in w["sweep"])
                  - min(s["rfx_mono_dbsm"] for s in w["sweep"]), 4)
            for w in witness
        ]
    # Gate the committed rfx column at measured max * 1.5, CLAMPED to be never
    # looser than the existing GO test's +/-15 dB tolerance. Reported, not
    # chased: this is the honest wide envelope, not a tightening of the bound.
    measured_max = max(env["coarse"]["max_abs_err_db"], env["fine"]["max_abs_err_db"])
    env["measured_max_abs_err_db"] = round(measured_max, 4)
    env["gate_db"] = round(min(15.0, measured_max * 1.5), 4)
    env["gate_floor_db"] = 15.0
    return env


def _print_verdict(fixture: dict[str, Any]) -> None:
    env = fixture["envelope"]
    print("\nEXACT-MIE RCS REFERENCE (PEC sphere, monostatic)")
    print(f"  {'ka':>5} {'mie dBsm':>9} {'rfx coarse':>11} {'err':>7} "
          f"{'rfx fine':>10} {'err':>7} {'conv d':>7}")
    for r, cd in zip(fixture["ka_ladder"], env["convergence_delta_db"]):
        print(f"  {r['ka']:5.2f} {r['mie_dbsm']:9.2f} "
              f"{r['coarse']['rfx_mono_dbsm']:11.2f} {r['coarse']['err_db']:+7.2f} "
              f"{r['fine']['rfx_mono_dbsm']:10.2f} {r['fine']['err_db']:+7.2f} {cd:+7.2f}")
    print(f"  measured max|err|: coarse={env['coarse']['max_abs_err_db']} dB "
          f"fine={env['fine']['max_abs_err_db']} dB  -> gate={env['gate_db']} dB "
          f"(floor {env['gate_floor_db']} dB, NOT a tightening)")
    if "domain_robustness" in fixture:
        print("  domain-robustness witness (mono dBsm vs domain, lam/10):")
        for w, swing in zip(fixture["domain_robustness"], env.get("domain_swing_db", [])):
            vals = " ".join(f"{s['domain_m']*100:.0f}cm={s['rfx_mono_dbsm']:+.1f}"
                            for s in w["sweep"])
            print(f"    ka={w['ka']:.2f} mie={w['mie_dbsm']:+.1f}: {vals}  swing={swing:.1f}dB")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors", action="store_true",
                   help="self-validate the Mie oracle against Rayleigh/optical limits")
    p.add_argument("--run", action="store_true",
                   help="rerun the rfx FDTD ladder and rebuild the committed fixture")
    p.add_argument("--commit", default="uncommitted",
                   help="commit SHA to stamp into the rebuilt fixture meta")
    args = p.parse_args(argv)

    if args.anchors:
        _self_anchor_report()
        return 0

    fixture = build_fixture(run=args.run, commit=args.commit)
    _print_verdict(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
