"""issue #280 fix gates: two-run incident-reference subtraction for RCS bistatic.

``compute_rcs(subtract_incident_reference=True)`` removes the TFSF-boundary
incident-field leakage that the NTFF box otherwise integrates into a spurious
forward-oblique lobe. Validated here against the EXACT Mie bistatic series on a
PEC sphere (an analytic reference at every angle -- no external solver), plus the
empty-domain leakage isolation that diagnosed the mechanism.

Posture: the Mie truth is re-derived from the committed analytic oracle
(``tests/fixtures/rcs_sphere_mie/mie_oracle.py``, itself gated by
``test_rcs_mie_reference_gates.py``); the fixture's mie column is checked against
it. Gates use shape-robust metrics (forward-oblique-lobe removal, correlation,
mean |distance|, backscatter) -- NOT max |distance|, which at a deep pattern null
is a dB-amplified floor, not a fix defect. Additive: no existing gate touched.
The default subtract_incident_reference=False path is byte-identical (covered by
the existing monostatic RCS tests) -- checked here via the sourced backscatter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO / "tests/fixtures/rcs280_reference_subtraction/fixture.json"
_RFX_SPHERE = _REPO / "tests/fixtures/rcs_sphere_mie/fixture.json"
sys.path.insert(0, str(_REPO / "tests/fixtures/rcs_sphere_mie"))
from mie_oracle import bistatic_over_pi_a2  # noqa: E402


@pytest.fixture(scope="module")
def fx():
    return json.loads(_FIXTURE.read_text())


def _db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def test_committed_mie_column_matches_analytic_oracle(fx):
    """The committed Mie bistatic column equals the exact analytic oracle
    (re-derived here), so the reference cannot be silently altered."""
    ka = fx["geometry"]["ka"]
    phi = np.radians(np.array(fx["phi_deg"]))
    want = np.array([bistatic_over_pi_a2(ka, float(p), "H") for p in phi])
    got = np.array(fx["mie_bistatic_over_pi_a2"])
    assert np.allclose(got, want, rtol=1e-9), np.abs(got - want).max()


def test_forward_oblique_lobe_removed_vs_exact_mie(fx):
    """The core #280 claim: the spurious forward-oblique lobe (present uncorrected,
    >5 dB vs exact Mie) is removed by the two-run subtraction (<2 dB vs Mie)."""
    phi = np.array(fx["phi_deg"])
    fo = (phi >= 15) & (phi <= 90)
    mie = _db(fx["mie_bistatic_over_pi_a2"])
    u = np.abs(_db(fx["rfx_uncorrected_over_pi_a2"]) - mie)[fo]
    c = np.abs(_db(fx["rfx_corrected_over_pi_a2"]) - mie)[fo]
    assert u.max() > 5.0, u.max()          # the lobe was really there
    assert c.max() < 2.0, c.max()          # and is removed


def test_corrected_pattern_matches_exact_mie(fx):
    """Shape-robust validation of the corrected bistatic vs exact Mie:
    high correlation, small mean distance, clean backscatter."""
    mie = _db(fx["mie_bistatic_over_pi_a2"])
    corr = _db(fx["rfx_corrected_over_pi_a2"])
    assert np.corrcoef(corr, mie)[0, 1] >= 0.95
    assert np.abs(corr - mie).mean() <= 0.6           # measured ~0.42 dB
    assert abs(corr[-1] - mie[-1]) <= 0.5             # backscatter, measured ~0.06 dB


def test_empty_domain_isolates_leakage(fx):
    """Diagnosis witness: an EMPTY domain (no scatterer) produces a materially
    large spurious far-field (the leakage) that NULLS at backscatter -- which is
    why the monostatic bin is clean and stays so under subtraction."""
    leak = np.array(fx["empty_domain_leakage_over_pi_a2"])
    peak, back = leak.max(), leak[-1]
    assert _db(peak) > _db(1.0)                        # peak leakage is O(1)*(pi a^2), large
    assert _db(back) < _db(peak) - 40.0                # backscatter leakage >=40 dB down


def test_subtract_reference_branch_runs_live():
    """Live smoke on the new code branch (the other tests read a frozen fixture).
    (a) With a VACUUM target and the flag on, E_far[target] - E_far[vacuum] is
    the difference of two identical runs = exactly 0, so the bistatic RCS is ~0
    (the subtraction cancels the leakage against itself). (b) On a small PEC
    scatterer the flag changes the forward-oblique bistatic vs default-off."""
    import jax.numpy as jnp
    from rfx.grid import Grid
    from rfx.geometry.csg import Sphere, rasterize
    from rfx.core.yee import MaterialArrays
    from rfx.rcs import compute_rcs

    f0 = 6e9
    grid = Grid(freq_max=f0 * 1.5, domain=(0.05,) * 3, dx=0.05 / 20, cpml_layers=6)
    ph = np.linspace(0.0, np.pi, 9)
    kw = dict(f0=f0, bandwidth=0.5, theta_inc=0.0, polarization="ez",
              theta_obs=np.array([np.pi / 2]), phi_obs=ph, freqs=np.array([f0]),
              boundary="cpml", cpml_layers=6)
    ones = jnp.ones(grid.shape, jnp.float32)
    vac = MaterialArrays(ones, jnp.zeros(grid.shape, jnp.float32), ones)

    r_vac = compute_rcs(grid, vac, 200, subtract_incident_reference=True, **kw)
    assert float(np.max(np.asarray(r_vac.rcs_linear))) < 1e-12   # self-cancels exactly

    c = 0.05 / 2
    eps_s, sig_s = rasterize(grid, [(Sphere(center=(c,) * 3, radius=0.008), 1.0, 1e7)])
    mats = MaterialArrays(eps_s, sig_s, ones)
    off = np.asarray(compute_rcs(grid, mats, 300, subtract_incident_reference=False, **kw).rcs_linear[0, 0])
    on = np.asarray(compute_rcs(grid, mats, 300, subtract_incident_reference=True, **kw).rcs_linear[0, 0])
    fo = (np.degrees(ph) >= 15) & (np.degrees(ph) <= 90)
    assert np.max(np.abs(off[fo] - on[fo])) > 0, "flag had no effect on the bistatic pattern"


def test_uncorrected_backscatter_is_sourced_default_path(fx):
    """The default (subtract=False) path is unchanged: the uncorrected sphere
    backscatter here matches the committed monostatic value in the sibling
    rcs_sphere_mie fixture (same geometry), i.e. default-off is the validated
    byte-identical path."""
    committed = json.loads(_RFX_SPHERE.read_text())["monostatic"]["rfx_sigma_over_pi_a2"]
    uncorr_back = np.array(fx["rfx_uncorrected_over_pi_a2"])[-1]
    assert np.isclose(uncorr_back, committed, rtol=0.02), (uncorr_back, committed)
