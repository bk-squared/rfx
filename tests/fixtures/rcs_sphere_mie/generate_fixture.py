"""Generate fixture.json for the PEC-sphere monostatic RCS comparison.

Runs the committed-resolution rfx RCS pipeline (PEC sphere, ka~1,
dx=lambda/40) once on CPU and records:

  * geometry + mesh metadata,
  * the rfx monostatic (backscatter) RCS from ``RCSResult.monostatic_rcs``
    (the shipped extraction fixed in issue #276),
  * the exact-Mie oracle value and the measured delta in dB,
  * the full H-plane bistatic trace (rfx vs Mie) for R5 inspection —
    clearly labeled NON-GATED,
  * a claim_scope string bounding what this fixture validates.

Usage (from the repo root, CPU is sufficient — wall ~25 s at CPML_LAYERS = 24):

    JAX_PLATFORMS=cpu python tests/fixtures/rcs_sphere_mie/generate_fixture.py

The lock ``tests/locks/test_rcs_sphere_backscatter_window.py`` recomputes
six bins at ka = 0.75, 0.80, ..., 1.00 and pins the signed rfx - Mie
difference in a window per bin. Its docstring describes the window's
derivation. The stored rfx value is historical, not the lock's reference;
the independent oracle witnesses stay in ``tests/oracle/test_rcs_mie_fixture.py``.
The absorber depth is 24 layers because an earlier depth ladder with the second-order NTFF rule measured successive answer changes of 0.221, 0.018, 0.015 and 0.013 dB at 16, 24, 32 and 40 layers.
"""

import json
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from mie_oracle import (  # noqa: E402
    backscatter_rcs_over_pi_a2,
    bistatic_over_pi_a2,
    validate_oracle,
)

F0_HZ = 3e9
F0 = F0_HZ
SPHERE_RADIUS_M = 0.0159
RADIUS = SPHERE_RADIUS_M
RESOLUTION = 40
CUBIC_DOMAIN_SIZE_M = 0.10
DOMAIN_SIZE = CUBIC_DOMAIN_SIZE_M
CPML_LAYERS = 24
N_STEPS = 700
BANDWIDTH = 0.5
POLARIZATION = "ez"
PEC_APPROXIMATION_SIGMA_S_PER_M = 1e7
PEC_SIGMA = PEC_APPROXIMATION_SIGMA_S_PER_M
H_PLANE_PHI_SAMPLES_INCLUDING_ENDPOINTS = 37
N_PHI = H_PLANE_PHI_SAMPLES_INCLUDING_ENDPOINTS


def run_rfx():
    """Run the committed-resolution rfx RCS simulation. Returns raw pieces.
    The fixture refuses an rfx import outside the repo tree so a stale installed build cannot silently supply the extraction."""
    import rfx
    _rfx_root = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
    if _rfx_root != _REPO_ROOT:
        raise RuntimeError(
            f"import rfx resolved outside this repo tree ({rfx.__file__}); "
            "refusing to generate a fixture against a different rfx build."
        )
    import jax.numpy as jnp
    from rfx.grid import Grid, C0
    from rfx.geometry.csg import Sphere, rasterize
    from rfx.core.yee import MaterialArrays
    from rfx.measurement.rcs import compute_rcs

    lam = C0 / F0
    dx = lam / RESOLUTION
    ka = 2 * np.pi * RADIUS / lam

    grid = Grid(
        freq_max=F0 * 1.5,
        domain=(DOMAIN_SIZE,) * 3,
        dx=dx,
        cpml_layers=CPML_LAYERS,
    )
    center = (DOMAIN_SIZE / 2,) * 3
    sphere = Sphere(center=center, radius=RADIUS)
    eps_r, sigma = rasterize(grid, [(sphere, 1.0, PEC_SIGMA)])
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    theta_obs = np.array([np.pi / 2])
    phi_obs = np.linspace(0.0, np.pi, N_PHI)

    t0 = time.time()
    result = compute_rcs(
        grid, materials, N_STEPS,
        f0=F0, bandwidth=BANDWIDTH, theta_inc=0.0,
        polarization=POLARIZATION,
        theta_obs=theta_obs, phi_obs=phi_obs, freqs=np.array([F0]),
        boundary="cpml", cpml_layers=CPML_LAYERS,
    )
    wall = time.time() - t0
    return grid, ka, dx, phi_obs, result, wall


def main():
    """A regeneration preserves the record_* notes because they explain why the stored numbers moved."""
    witnesses = validate_oracle()
    print("Mie oracle witnesses PASS:", witnesses)

    grid, ka, dx, phi_obs, result, wall = run_rfx()
    pi_a2 = np.pi * RADIUS ** 2

    mono_dbsm = float(result.monostatic_rcs[0])
    rfx_back_over = float(10.0 ** (mono_dbsm / 10.0) / pi_a2)

    mie_back_over = float(backscatter_rcs_over_pi_a2(ka, n_max=20))
    mie_back_dbsm = float(10.0 * np.log10(mie_back_over * pi_a2))
    delta_db = abs(mono_dbsm - mie_back_dbsm)

    rcs_over = np.asarray(result.rcs_linear[0, 0, :]) / pi_a2
    trace = []
    for ph, ro in zip(phi_obs, rcs_over):
        mv = float(bistatic_over_pi_a2(ka, float(ph), "H"))
        d = abs(10 * np.log10(max(float(ro), 1e-30))
                - 10 * np.log10(max(mv, 1e-30)))
        trace.append({
            "scattering_angle_deg": round(float(np.degrees(ph)), 2),
            "rfx_sigma_over_pi_a2": float(ro),
            "mie_sigma_over_pi_a2": mv,
            "abs_delta_db": round(d, 3),
        })

    fwd_delta_db = trace[0]["abs_delta_db"]

    fixture = {
        "schema": "rfx.rcs_sphere_mie_monostatic_fixture",
        "schema_version": 1,
        "issue": 276,
        "claim_scope": (
            "MONOSTATIC (backscatter) RCS MAGNITUDE of a staircased PEC "
            "sphere at ka~1, at the committed resolution "
            f"(dx=lambda/{RESOLUTION}, ~{RADIUS / (299792458.0 / F0 / RESOLUTION):.1f} "
            "cells per radius), vs the exact Mie series. This is NOT a "
            "bistatic validation: the same run shows a spurious "
            "forward-oblique lobe at scattering angles 25-55 deg (~10 dB "
            "high vs Mie; TFSF/NTFF forward-face contamination suspected) "
            f"and a forward-scatter delta of {fwd_delta_db:.2f} dB. Those "
            "are recorded in bistatic_trace_non_gated for inspection and "
            "are deliberately NOT gated."
        ),
        "geometry": {
            "f0_hz": F0,
            "radius_m": RADIUS,
            "ka": float(ka),
            "dx_m": float(dx),
            "resolution_cells_per_lambda": RESOLUTION,
            "cells_per_radius": float(RADIUS / dx),
            "grid_shape": list(grid.shape),
            "domain_m": [DOMAIN_SIZE] * 3,
            "cpml_layers": CPML_LAYERS,
            "n_steps": N_STEPS,
            "bandwidth": BANDWIDTH,
            "polarization": POLARIZATION,
            "pec_sigma_s_per_m": PEC_SIGMA,
        },
        "monostatic": {
            "rfx_sigma_over_pi_a2": rfx_back_over,
            "rfx_dbsm": mono_dbsm,
            "mie_sigma_over_pi_a2": mie_back_over,
            "mie_dbsm": mie_back_dbsm,
            "abs_delta_db": round(delta_db, 4),
            "extraction": (
                "RCSResult.monostatic_rcs — far field evaluated exactly at "
                "the backscatter direction (theta=pi/2, phi=pi for +x "
                "incidence), post issue-#276 fix."
            ),
        },
        "mie_oracle_witnesses": {
            "rayleigh_rel_err": witnesses["rayleigh_rel_err"],
            "go_window_mean": witnesses["go_window_mean"],
            "convergence_abs_change": {
                str(k): v for k, v in witnesses["convergence_abs_change"].items()
            },
            "bistatic_bridge_value": witnesses["bistatic_bridge_value"],
        },
        "bistatic_trace_non_gated": {
            "note": (
                "H-plane cut (theta=pi/2, scattering angle == phi). For R5 "
                "inspection only — NOT gated. Known deviations: spurious "
                "forward-oblique lobe 25-55 deg (~10 dB high vs Mie), "
                "forward-scatter delta ~1.6 dB."
            ),
            "trace": trace,
        },
        "provenance": {
            "generator": "tests/fixtures/rcs_sphere_mie/generate_fixture.py",
            "wall_seconds_cpu": round(wall, 1),
        },
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixture.json")
    if os.path.exists(out):
        with open(out) as f:
            fixture.update({k: v for k, v in json.load(f).items()
                            if k.startswith("record_")})
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"\nwall={wall:.1f}s  grid={tuple(grid.shape)}  ka={ka:.4f}")
    print(f"rfx  monostatic: sigma/pi_a2={rfx_back_over:.4f}  ({mono_dbsm:+.2f} dBsm)")
    print(f"Mie  monostatic: sigma/pi_a2={mie_back_over:.4f}  ({mie_back_dbsm:+.2f} dBsm)")
    print(f"|delta| = {delta_db:.3f} dB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
