"""RCS PEC-sphere — exact-Mie cross-method REFERENCE gates (WP 1-A).

Locks the first analytic (non-FDTD / non-FEM) cross-method reference for rfx's
RCS pipeline, committed under
``tests/fixtures/rcs_mie_e4/rcs_pec_sphere_mie.json``. The load-bearing gate is
the exact conducting-sphere Mie backscatter series (Ruck, Radar Cross Section
Handbook 1970), RE-DERIVED INDEPENDENTLY here from ``scipy.special`` — it does
not import the producer's Mie function, so a producer-side error cannot
self-certify. No FDTD runs here; the committed rfx monostatic values are frozen
evidence.

MEASURED-ENVELOPE POSTURE (honest, not tuned)
---------------------------------------------
Regenerated 2026-07-07 after the #276 monostatic-extraction fix (PR #279:
true backscatter). At the committed test-scale geometry (0.10 m cubic domain =
1-2 wavelengths, lambda/10-lambda/15) across the ka ladder {0.8, 1.0, 1.5, 2.0}:

  * measured max |rfx - Mie| = 9.28 dB (ka=1.5, lambda/15);
  * refinement (lambda/10 -> lambda/15) is mixed (2 of 4 rungs improve) -- the
    staircased curved-PEC surface + close NTFF box dominate at these
    resolutions;
  * the monostatic magnitude swings up to 8.4 dB (ka=1.0) across domain size.

``gate_db`` = measured_max * 1.5 = 13.9 dB, under the pre-existing 15 dB GO
ceiling. The finer-resolution regime is documented by the companion fixture
``tests/fixtures/rcs_sphere_mie/`` (PR #279), where the fixed extraction
reaches ~0.06 dB at ka~1, lambda/40 -- resolution, not extraction direction, is
now the driver of this envelope. Residual coarse-regime diagnostic: issue #280.
These gates lock the honest test-scale record; they must not be re-tuned to
look tighter than the physics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO_ROOT / "tests/fixtures/rcs_mie_e4/rcs_pec_sphere_mie.json"


# --------------------------------------------------------------------------- #
# Independent exact-Mie re-derivation (does NOT import the producer).
# --------------------------------------------------------------------------- #
def _mie_ratio(ka: float) -> float:
    """sigma/(pi a^2) for a PEC sphere, backscatter (Ruck 1970)."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp_, ynp_ = spherical_jn(n, x, derivative=True), spherical_yn(n, x, derivative=True)
    hn, hnp_ = jn + 1j * yn, jnp_ + 1j * ynp_
    a_n = jn / hn
    b_n = (jn + x * jnp_) / (hn + x * hnp_)
    series = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))
    return float(np.abs(series) ** 2 / x ** 2)


def _mie_dbsm(ka: float, radius_m: float) -> float:
    return float(10.0 * np.log10(_mie_ratio(ka) * np.pi * radius_m ** 2))


@pytest.fixture(scope="module")
def fixture():
    return json.loads(_FIXTURE.read_text())


# --------------------------------------------------------------------------- #
# Oracle self-anchors (the reference must pass its own closed-form limits).
# --------------------------------------------------------------------------- #
def test_mie_oracle_rayleigh_limit():
    """sigma/(pi a^2) -> 9 (ka)^4 as ka -> 0 (electrically small sphere)."""
    for ka in (0.01, 0.02, 0.05):
        assert np.isclose(_mie_ratio(ka), 9.0 * ka ** 4, rtol=2e-3), ka


def test_mie_oracle_optical_limit():
    """sigma/(pi a^2) -> 1 as ka -> inf (geometrical-optics / GO limit)."""
    assert _mie_ratio(50.0) == pytest.approx(1.0, abs=0.01)
    assert _mie_ratio(100.0) == pytest.approx(1.0, abs=0.005)


# --------------------------------------------------------------------------- #
# Fixture integrity: the committed Mie column is the exact series.
# --------------------------------------------------------------------------- #
def test_committed_mie_column_is_exact_series(fixture):
    """LOAD-BEARING: every committed mie_dbsm equals the independently re-derived
    exact series (rel 1e-9). Tampering with the reference fails here."""
    radius = fixture["meta"]["radius_m"]
    for rung in fixture["ka_ladder"]:
        want = _mie_dbsm(rung["ka"], radius)
        assert np.isclose(rung["mie_dbsm"], want, rtol=1e-9), (rung["ka"], want)
        for res in (rung["coarse"], rung["fine"]):
            assert np.isclose(res["mie_dbsm"], want, rtol=1e-9), (rung["ka"], "res")
    for w in fixture["domain_robustness"]:
        want = _mie_dbsm(w["ka"], radius)
        assert np.isclose(w["mie_dbsm"], want, rtol=1e-9), (w["ka"], want)


def test_errors_self_consistent(fixture):
    """err_db == rfx_mono_dbsm - mie_dbsm for every stored rfx value."""
    for rung in fixture["ka_ladder"]:
        for res in (rung["coarse"], rung["fine"]):
            assert np.isclose(
                res["err_db"], res["rfx_mono_dbsm"] - res["mie_dbsm"], atol=1e-9
            ), (rung["ka"], res["cells_per_lambda"])
    for w in fixture["domain_robustness"]:
        for s in w["sweep"]:
            assert np.isclose(s["err_db"], s["rfx_mono_dbsm"] - s["mie_dbsm"], atol=1e-9)


# --------------------------------------------------------------------------- #
# Measured-envelope gates (honest, no tightening).
# --------------------------------------------------------------------------- #
def test_ladder_within_measured_envelope(fixture):
    """Every committed rfx-vs-Mie error is within the measured envelope gate.
    Measured max |err| = 13.17 dB; gate_db = 15 dB (clamped to the GO floor)."""
    gate = fixture["envelope"]["gate_db"]
    for rung in fixture["ka_ladder"]:
        for res in (rung["coarse"], rung["fine"]):
            assert abs(res["err_db"]) <= gate + 1e-9, (rung["ka"], res["err_db"], gate)


def test_envelope_no_regression(fixture):
    """gate_db is never LOOSER than the existing GO tolerance (15 dB), and equals
    min(15, measured_max * 1.5). This adds a cross-method gate without relaxing
    any existing bound."""
    env = fixture["envelope"]
    assert env["gate_floor_db"] == 15.0
    assert env["gate_db"] <= env["gate_floor_db"] + 1e-9
    measured_max = max(
        abs(res["err_db"])
        for rung in fixture["ka_ladder"]
        for res in (rung["coarse"], rung["fine"])
    )
    # Stored envelope numbers are rounded to 4 decimals by the producer.
    assert np.isclose(env["measured_max_abs_err_db"], measured_max, atol=1e-3)
    assert np.isclose(env["gate_db"], min(15.0, measured_max * 1.5), atol=1e-3)


def test_convergence_witness(fixture):
    """Convergence witness: refinement (lambda/10 -> lambda/15) does not blow the
    error up beyond a measured margin. Recorded convergence_delta == |fine err| -
    |coarse err|; measured max is +5.76 dB (refinement does NOT close the gap --
    an honest curved-PEC staircase floor), gated <= 6.5 dB so a future regression
    that made refinement diverge would trip."""
    deltas = fixture["envelope"]["convergence_delta_db"]
    assert len(deltas) == len(fixture["ka_ladder"])
    for rung, d in zip(fixture["ka_ladder"], deltas):
        recomputed = abs(rung["fine"]["err_db"]) - abs(rung["coarse"]["err_db"])
        assert np.isclose(d, recomputed, atol=1e-3), (rung["ka"], d, recomputed)
        assert d <= 6.5, (rung["ka"], d)


def test_domain_robustness_witness(fixture):
    """R5 evidence lock: the monostatic magnitude does not converge with domain
    size at test-scale (swing recorded per witness ka). Swing == max-min of the
    swept monostatic dBsm; committed swings are ~9.2 dB (ka=1.0) and ~2.7 dB
    (ka=1.5) -- a single dBsm number is not a converged quantity here."""
    swings = fixture["envelope"]["domain_swing_db"]
    assert len(swings) == len(fixture["domain_robustness"])
    for w, swing in zip(fixture["domain_robustness"], swings):
        vals = [s["rfx_mono_dbsm"] for s in w["sweep"]]
        assert len(vals) >= 3, w["ka"]
        assert np.all(np.isfinite(vals))
        assert np.isclose(swing, max(vals) - min(vals), atol=1e-3), (w["ka"], swing)
    # The witness must actually witness non-convergence (largest swing > 2 dB),
    # else it is not evidence. This locks the finding, not a tuned pass.
    assert max(swings) > 2.0


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #
def test_meta_integrity(fixture):
    """Provenance guard: exact-Mie reference, geometry matches the committed
    test_rcs.py sphere, the honest 'finding' note + falsifier record are kept."""
    meta = fixture["meta"]
    assert meta["reference"] == "exact-mie-series-pec-sphere"
    assert meta["radius_m"] == 0.015
    assert meta["domain_m"] == 0.10
    assert meta["cpml_layers"] == 8
    assert meta["polarization"] == "ez"
    assert meta["n_theta_obs"] == 37
    assert meta["ka_ladder"] == [0.8, 1.0, 1.5, 2.0]
    assert meta["commit"] and meta["commit"] != "uncommitted"
    assert "finding" in meta and "does not" in meta["finding"].lower()
    fp = meta["committed_test_point"]
    assert 0.9 <= fp["ka"] <= 1.0
    assert "4.8" in fp["note"] or "4.9" in fp["note"], fp["note"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
