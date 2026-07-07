"""Gate test: PEC-sphere monostatic RCS vs exact Mie series (issue #276).

Recomputes the committed-resolution sphere run live and gates the SHIPPED
monostatic extraction (``RCSResult.monostatic_rcs``) against the exact Mie
oracle committed in ``tests/fixtures/rcs_sphere_mie/mie_oracle.py``.

Claim scope (see the fixture README): MONOSTATIC (backscatter) magnitude
only, at the committed resolution. The bistatic trace in the fixture is
NON-GATED — the same run shows a spurious forward-oblique lobe (25-55 deg,
~10 dB high vs Mie) and a ~1.6 dB forward-scatter delta; do not add
bistatic gates here without root-causing those first.

Runtime: one 58^3 x 700-step CPU run, ~7 s.
"""

import importlib.util
import json
import os

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.geometry.csg import Sphere, rasterize
from rfx.core.yee import MaterialArrays
from rfx.rcs import compute_rcs

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "rcs_sphere_mie",
)


def _load_mie_oracle():
    spec = importlib.util.spec_from_file_location(
        "rcs_sphere_mie_oracle", os.path.join(FIXTURE_DIR, "mie_oracle.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_fixture():
    with open(os.path.join(FIXTURE_DIR, "fixture.json")) as f:
        return json.load(f)


def test_mie_oracle_witnesses():
    """The oracle must pass its four rfx-independent validation witnesses."""
    oracle = _load_mie_oracle()
    witnesses = oracle.validate_oracle()  # raises AssertionError on failure
    # Rayleigh witness measured at 1.8e-3 rel err; keep the committed bound.
    assert witnesses["rayleigh_rel_err"] < 0.02
    assert abs(witnesses["go_window_mean"] - 1.0) < 0.1


def test_monostatic_backscatter_matches_exact_mie():
    """Live recompute at the committed resolution: |rfx - Mie| <= 1.0 dB.

    Also asserts (a) the exact-direction monostatic value agrees with the
    observation-grid sample at (theta=pi/2, phi=pi) — internal consistency
    witness for the #276 extraction — and (b) the live value matches the
    committed fixture.json within a cross-machine float tolerance
    (anti-drift).
    """
    oracle = _load_mie_oracle()
    fixture = _load_fixture()
    geo = fixture["geometry"]

    f0 = geo["f0_hz"]
    radius = geo["radius_m"]
    lam = C0 / f0
    dx = lam / geo["resolution_cells_per_lambda"]
    ka = 2 * np.pi * radius / lam
    domain = tuple(geo["domain_m"])
    cpml = geo["cpml_layers"]

    grid = Grid(
        freq_max=f0 * 1.5, domain=domain, dx=dx, cpml_layers=cpml,
    )
    assert tuple(grid.shape) == tuple(geo["grid_shape"]), (
        "Committed grid metadata drifted from the live Grid construction; "
        "regenerate the fixture."
    )

    center = tuple(d / 2 for d in domain)
    sphere = Sphere(center=center, radius=radius)
    eps_r, sigma = rasterize(
        grid, [(sphere, 1.0, geo["pec_sigma_s_per_m"])],
    )
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # H-plane observation cut; phi grid contains the exact backscatter
    # angle (phi=pi) so the grid sample can witness the exact-direction
    # extraction.
    theta_obs = np.array([np.pi / 2])
    phi_obs = np.linspace(0.0, np.pi, 37)

    result = compute_rcs(
        grid, materials, geo["n_steps"],
        f0=f0, bandwidth=geo["bandwidth"], theta_inc=0.0,
        polarization=geo["polarization"],
        theta_obs=theta_obs, phi_obs=phi_obs, freqs=np.array([f0]),
        boundary="cpml", cpml_layers=cpml,
    )

    mono_dbsm = float(result.monostatic_rcs[0])

    # (a) Internal witness: exact-direction evaluation == grid sample at
    # (theta=pi/2, phi=pi). Same NTFF data, same float64 post-processing.
    grid_back_dbsm = float(result.rcs_dbsm[0, 0, -1])
    assert abs(mono_dbsm - grid_back_dbsm) < 1e-6, (
        f"monostatic_rcs ({mono_dbsm:.6f} dBsm) != observation-grid sample "
        f"at (pi/2, pi) ({grid_back_dbsm:.6f} dBsm) — the exact-direction "
        "extraction disagrees with the grid path."
    )

    # (b) THE gate: shipped monostatic vs exact Mie series.
    pi_a2 = np.pi * radius ** 2
    mie_over = float(oracle.backscatter_rcs_over_pi_a2(ka, n_max=20))
    mie_dbsm = 10.0 * np.log10(mie_over * pi_a2)
    delta_db = abs(mono_dbsm - mie_dbsm)
    assert delta_db <= 1.0, (
        f"Monostatic RCS {mono_dbsm:.2f} dBsm is {delta_db:.2f} dB from the "
        f"exact Mie value {mie_dbsm:.2f} dBsm (gate 1.0 dB; measured 0.06 dB "
        "on the 2026-07-06 falsifier run and at fixture generation). "
        "A regression here means the monostatic extraction or the "
        "TFSF/NTFF/RCS chain drifted — see tests/fixtures/rcs_sphere_mie/."
    )

    # (c) Anti-drift: live recompute must match the committed fixture value.
    # Tolerance covers cross-machine float32 FDTD accumulation differences
    # only — a real physics/extraction change exceeds it.
    fix_dbsm = fixture["monostatic"]["rfx_dbsm"]
    assert abs(mono_dbsm - fix_dbsm) < 0.25, (
        f"Live monostatic {mono_dbsm:.4f} dBsm drifted from committed "
        f"fixture value {fix_dbsm:.4f} dBsm by "
        f"{abs(mono_dbsm - fix_dbsm):.4f} dB (>= 0.25). If an intentional "
        "physics change caused this, regenerate the fixture with "
        "tests/fixtures/rcs_sphere_mie/generate_fixture.py and re-review "
        "the delta vs Mie."
    )

    # R5 breadcrumb for -s runs.
    print(
        f"\nMie fixture gate: rfx {mono_dbsm:+.3f} dBsm | Mie "
        f"{mie_dbsm:+.3f} dBsm | |delta| {delta_db:.3f} dB | "
        f"fixture {fix_dbsm:+.3f} dBsm"
    )


def test_fixture_claim_scope_is_monostatic_only():
    """The committed fixture must keep its honesty markers.

    The claim scope is monostatic-only; the known bistatic deviations
    (forward-oblique lobe, forward delta) must stay documented and the
    bistatic trace must stay labeled non-gated.
    """
    fixture = _load_fixture()
    scope = fixture["claim_scope"]
    assert "MONOSTATIC" in scope
    assert "NOT a bistatic validation" in scope
    assert "bistatic_trace_non_gated" in fixture
    trace = fixture["bistatic_trace_non_gated"]["trace"]
    # Trace spans forward (0 deg) to backscatter (180 deg) for inspection.
    assert trace[0]["scattering_angle_deg"] == 0.0
    assert trace[-1]["scattering_angle_deg"] == 180.0
    # The recorded backscatter delta (the gated quantity) is small ...
    assert trace[-1]["abs_delta_db"] <= 1.0
    # ... while the documented forward-oblique deviation is real and large;
    # this assertion guards against silently "cleaning up" the evidence.
    oblique = [
        p["abs_delta_db"] for p in trace
        if 25.0 <= p["scattering_angle_deg"] <= 55.0
    ]
    assert max(oblique) > 5.0, (
        "The committed bistatic trace no longer shows the documented "
        "forward-oblique deviation (25-55 deg). If the lobe was actually "
        "fixed, update the fixture README + claim_scope alongside the "
        "regenerated trace instead of leaving stale honesty text."
    )
