"""Lock the conductivity-filled sphere's monostatic difference from exact Mie.

An Ez plane wave illuminates a centred sphere of radius 15.9 mm in a 0.1 m
cube, with a lambda/40 mesh at 3 GHz, 24-cell CPML and 700 time steps. One
run samples backscatter at six frequencies spanning ka = 0.75 through 1.00.
The windows pin rfx's own lambda/40 difference from Mie: a result exactly
on Mie fails every bin. This is a regression lock, not an accuracy oracle
against the closed form; no bistatic accuracy is asserted.

The bounds were set from the lambda/40, 60, 80 ladder's trend at lambda/40
plus or minus w, where w = max(0.10 dB, 2 * max(trend misfit, distance from
the trend, record and absorber sensitivity, quarter- and half-cell
staircase-position sensitivity)). Restoring the #820 normalization shifts
these bins by -1.135 to -1.147 dB and falls outside by at least 0.725 dB.
The band stops at ka = 1.00 because above it the meshes cross and the
staircase-position sensitivity exceeds half the #820 shift.

Here "PEC" is a 1e7 S/m sigma fill, not a realized PEC body. A raw
``rasterize(grid, [(shape, eps, sigma)])`` writes ``MaterialArrays.sigma``;
it does not pass through ``pec_mask``, ``add_thin_conductor`` or
``realized_pec_edge_masks``. Fields decay inside conductive cells instead
of being zeroed on PEC edges. ``Simulation._PEC_SIGMA_THRESHOLD`` (1e6 S/m)
is applied in ``rfx/api/_compile.py`` to a material on a geometry entry,
not to this raw rasterize call. The operator distinction is checked by
``tests/contracts/test_lattice_ownership_contract.py::test_a_sigma_fill_conductor_is_not_a_pec_body``.
Routing this fill through PEC realization would require new measurements.
"""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Sphere, rasterize
from rfx.grid import C0, Grid
from rfx.measurement.rcs import compute_rcs


LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/rcs_sphere_mie/fixture.json, tests/fixtures/rcs_sphere_mie/mie_oracle.py",
    "generator": "measurement script outside the tree (λ/40, 60, 80 ladder, staircase-position, record and absorber variants); record path in the PR body",
    "commit": "13c45814d",
    "date": "2026-11-15",
    "run_id": "remi rfx-rcsbar-j1-qmrf7, rfx-rcsbar-j2-hht9f, rfx-rcsbar-j3-4xvjm, rfx-rcsbar-j4-l9sh2",
    "host": "Linux cluster CPU (remilab-fs), JAX 0.10.2, float32",
    "pinned_until": "2027-05-15",
}

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "rcs_sphere_mie"
WINDOWS = (
    (0.75, 0.80, 1.54),
    (0.80, 0.74, 1.45),
    (0.85, 0.64, 1.35),
    (0.90, 0.50, 1.24),
    (0.95, 0.33, 1.13),
    (1.00, 0.15, 1.00),
)


@pytest.fixture(scope="module")
def sphere_backscatter():
    """Share one live six-frequency run and the exact Mie comparator."""
    geo = json.loads((FIXTURE_DIR / "fixture.json").read_text())["geometry"]
    spec = importlib.util.spec_from_file_location(
        "rcs_sphere_mie_oracle", FIXTURE_DIR / "mie_oracle.py",
    )
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    f0 = geo["f0_hz"]
    radius = geo["radius_m"]
    dx = C0 / f0 / geo["resolution_cells_per_lambda"]
    domain = tuple(geo["domain_m"])
    cpml = geo["cpml_layers"]
    grid = Grid(freq_max=f0 * 1.5, domain=domain, dx=dx, cpml_layers=cpml)
    assert tuple(grid.shape) == tuple(geo["grid_shape"]), (
        "Committed grid metadata drifted from the live Grid construction; "
        "regenerate the fixture."
    )

    sphere = Sphere(center=tuple(d / 2 for d in domain), radius=radius)
    eps_r, sigma = rasterize(grid, [(sphere, 1.0, geo["pec_sigma_s_per_m"])])
    materials = MaterialArrays(
        eps_r=eps_r, sigma=sigma,
        mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
    )
    ka_bins = np.array([ka for ka, _, _ in WINDOWS])
    freqs = ka_bins * C0 / (2 * np.pi * radius)
    result = compute_rcs(
        grid, materials, geo["n_steps"],
        f0=f0, bandwidth=geo["bandwidth"], theta_inc=0.0,
        polarization=geo["polarization"],
        theta_obs=np.array([np.pi / 2]),
        phi_obs=np.linspace(0.0, np.pi, 37), freqs=freqs,
        boundary="cpml", cpml_layers=cpml,
    )
    mie_dbsm = np.array([
        10.0 * np.log10(
            oracle.backscatter_rcs_over_pi_a2(ka, n_max=20) * np.pi * radius ** 2
        )
        for ka in ka_bins
    ])
    return result, mie_dbsm


@pytest.mark.parametrize("index", range(len(WINDOWS)), ids=[f"ka={ka:.2f}" for ka, _, _ in WINDOWS])
def test_monostatic_backscatter_window(sphere_backscatter, index):
    """Pin each signed difference and witness the exact-direction extraction."""
    result, mie_dbsm = sphere_backscatter
    ka, lower, upper = WINDOWS[index]
    mono_dbsm = float(result.monostatic_rcs[index])
    grid_back_dbsm = float(result.rcs_dbsm[index, 0, -1])
    delta_db = mono_dbsm - float(mie_dbsm[index])

    print(
        f"\nRCS sphere lock: ka={ka:.2f} | rfx {mono_dbsm:+.6f} dBsm | "
        f"Mie {mie_dbsm[index]:+.6f} dBsm | delta {delta_db:+.6f} dB | "
        f"window [{lower:.2f}, {upper:.2f}] dB"
    )
    assert abs(mono_dbsm - grid_back_dbsm) < 1e-6, (
        f"ka={ka:.2f}: monostatic_rcs ({mono_dbsm:.6f} dBsm) != "
        f"observation-grid sample at (pi/2, pi) ({grid_back_dbsm:.6f} dBsm)"
    )
    assert lower <= delta_db <= upper, (
        f"ka={ka:.2f}: rfx - Mie = {delta_db:+.6f} dB outside "
        f"[{lower:.2f}, {upper:.2f}] dB"
    )
