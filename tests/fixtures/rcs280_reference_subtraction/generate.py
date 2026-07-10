"""Producer for the issue-#280 two-run reference-subtraction fixture.

Validates ``compute_rcs(subtract_incident_reference=True)`` against the EXACT Mie
bistatic series on a PEC sphere (an exact analytic reference at every angle -- no
external solver needed), plus the empty-domain leakage isolation that diagnosed
the mechanism. Emits fixture.json mechanically. Pure rfx + scipy; no CI deps.

Mechanism (issue #280): the discrete TFSF boundary leaks residual incident field
into the scattered-field region; the NTFF box integrates it into a spurious
forward-oblique far-field lobe. An EMPTY-domain run (no scatterer) reproduces the
same lobe -> pure leakage. It is target-independent, so a two-run subtraction
E_scat = E_far[target] - E_far[vacuum] cancels it exactly (standard TF/SF
normalization). Backscatter leakage ~0, so the validated monostatic bin is
unchanged (default subtract_incident_reference=False is byte-identical).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "tests/fixtures/rcs_sphere_mie"))
from mie_oracle import bistatic_over_pi_a2  # noqa: E402

from rfx.grid import Grid, C0  # noqa: E402
from rfx.geometry.csg import Sphere, rasterize  # noqa: E402
from rfx.core.yee import MaterialArrays  # noqa: E402
from rfx.rcs import compute_rcs  # noqa: E402

F0 = 3e9
LAM = C0 / F0
A = 0.0159                     # sphere radius (m); rcs_sphere_mie geometry
RES = 40                       # dx = lam/40 (6.4 cells/radius)
DX = LAM / RES
DOMAIN = 0.10
CPML = 8
N_PHI = 37
N_STEPS = 700
KA = 2 * np.pi * A / LAM
phi = np.linspace(0.0, np.pi, N_PHI)


def _sphere_bistatic(subtract):
    grid = Grid(freq_max=F0 * 1.5, domain=(DOMAIN,) * 3, dx=DX, cpml_layers=CPML)
    c = DOMAIN / 2
    eps_s, sig_s = rasterize(grid, [(Sphere(center=(c,) * 3, radius=A), 1.0, 1e7)])
    mats = MaterialArrays(eps_s, sig_s, jnp.ones(grid.shape, jnp.float32))
    r = compute_rcs(grid, mats, N_STEPS, f0=F0, bandwidth=0.5, theta_inc=0.0,
                    polarization="ez", theta_obs=np.array([np.pi / 2]), phi_obs=phi,
                    freqs=np.array([F0]), boundary="cpml", cpml_layers=CPML,
                    subtract_incident_reference=subtract)
    return np.asarray(r.rcs_linear[0, 0, :]) / (np.pi * A ** 2)   # sigma/(pi a^2)


def _empty_leakage():
    """Empty domain, NO subtraction -> the pure leakage far-field (the diagnosis)."""
    grid = Grid(freq_max=F0 * 1.5, domain=(DOMAIN,) * 3, dx=DX, cpml_layers=CPML)
    vac = MaterialArrays(jnp.ones(grid.shape, jnp.float32),
                         jnp.zeros(grid.shape, jnp.float32),
                         jnp.ones(grid.shape, jnp.float32))
    r = compute_rcs(grid, vac, N_STEPS, f0=F0, bandwidth=0.5, theta_inc=0.0,
                    polarization="ez", theta_obs=np.array([np.pi / 2]), phi_obs=phi,
                    freqs=np.array([F0]), boundary="cpml", cpml_layers=CPML)
    return np.asarray(r.rcs_linear[0, 0, :]) / (np.pi * A ** 2)


def main():
    mie = np.array([bistatic_over_pi_a2(KA, float(p), "H") for p in phi])
    uncorr = _sphere_bistatic(False)
    corr = _sphere_bistatic(True)
    leak = _empty_leakage()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()

    def _db(x):
        return 10 * np.log10(np.maximum(x, 1e-30))
    fo = (np.degrees(phi) >= 15) & (np.degrees(phi) <= 90)
    dist_u = np.abs(_db(uncorr) - _db(mie))
    dist_c = np.abs(_db(corr) - _db(mie))

    fixture = {
        "schema": "rfx.rcs280_reference_subtraction", "schema_version": 1,
        "issue": "280",
        "claim_scope": (
            "Validates compute_rcs(subtract_incident_reference=True) against the EXACT "
            "Mie bistatic series on a PEC sphere (analytic reference at every angle). "
            "The two-run reference subtraction removes the TFSF-leakage forward-oblique "
            "lobe (issue #280): H-plane forward-oblique vs exact Mie collapses from "
            f"{dist_u[fo].max():.1f} dB to {dist_c[fo].max():.1f} dB, backscatter stays "
            f"{_db(corr)[-1] - _db(mie)[-1]:+.2f} dB. The empty-domain run isolates the "
            "leakage (a spurious far-field with NO scatterer). Default OFF is byte-"
            "identical on the validated monostatic path."),
        "geometry": {"shape": "pec_sphere", "radius_m": A, "f0_hz": F0, "ka": KA,
                     "dx_m": DX, "res_cells_per_lambda": RES,
                     "cells_per_radius": A / DX, "n_steps": N_STEPS, "domain_m": DOMAIN},
        "phi_deg": [round(float(np.degrees(p)), 2) for p in phi],
        "mie_bistatic_over_pi_a2": [float(x) for x in mie],
        "rfx_uncorrected_over_pi_a2": [float(x) for x in uncorr],
        "rfx_corrected_over_pi_a2": [float(x) for x in corr],
        "empty_domain_leakage_over_pi_a2": [float(x) for x in leak],
        "metrics": {
            "forward_oblique_15_90_max_db": {
                "uncorrected_vs_mie": float(dist_u[fo].max()),
                "corrected_vs_mie": float(dist_c[fo].max())},
            "backscatter_corrected_vs_mie_db": float(_db(corr)[-1] - _db(mie)[-1]),
            "full_curve_mean_abs_db_corrected": float(dist_c.mean()),
            "shape_correlation_db_corrected": float(np.corrcoef(_db(corr), _db(mie))[0, 1]),
            "empty_leakage_peak_over_pi_a2": float(leak.max()),
            "empty_leakage_backscatter_over_pi_a2": float(leak[-1]),
        },
        "provenance": {"rfx_commit": sha, "producer": str(_HERE.name) + "/generate.py"},
    }
    (_HERE / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    m = fixture["metrics"]
    print(f"forward-oblique vs Mie: uncorr {m['forward_oblique_15_90_max_db']['uncorrected_vs_mie']:.2f} "
          f"-> corrected {m['forward_oblique_15_90_max_db']['corrected_vs_mie']:.2f} dB")
    print(f"backscatter corrected-vs-Mie: {m['backscatter_corrected_vs_mie_db']:+.2f} dB; "
          f"mean {m['full_curve_mean_abs_db_corrected']:.2f} dB; corr {m['shape_correlation_db_corrected']:.4f}")
    print(f"empty-domain leakage: peak {10*np.log10(m['empty_leakage_peak_over_pi_a2']):.1f} vs "
          f"backscatter {10*np.log10(max(m['empty_leakage_backscatter_over_pi_a2'],1e-30)):.1f} (dB/pi a^2)")
    print("wrote fixture.json")


if __name__ == "__main__":
    main()
