"""PEC-sphere Mie ka-sweep (0.5-4.0) — frozen-fixture gates (campaign item 1).

Locks the committed measurement record of
``validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture``
(``tests/fixtures/rcs_mie_ka_sweep/fixture.json``) against an INDEPENDENT
in-test re-derivation of the exact conducting-sphere Mie backscatter series
(Ruck 1970, from ``scipy.special`` — not the producer's oracle module, so a
producer-side error cannot self-certify).

Two-tier posture (measured 2026-07-27, three domain realizations):
  * GATED: coarse ka <= 1.25 (3-domain envelope 2.1 dB -> gate 3.2 dB) and
    fine-rung ka {2.0, 4.0} (envelope 2.35 dB -> gate 3.5 dB).
  * NOT GATED: coarse ka >= 1.5 and fine ka=3.0 — near the deep Mie nulls the
    monostatic value swings up to 6.3-8.3 dB under a domain-size-only change
    and the rfx local-minimum positions move with domain size. This module
    PINS that fence (claim-scope docpin + witness-presence assertions) so it
    cannot be silently converted into a gate, and conversely so the unconverged
    bins cannot be silently quoted as validated.

No FDTD runs here — the fixture is frozen evidence; live regeneration is the
crossval script's job. These gates must not be re-tuned to look tighter than
the recorded physics (no-silent-gate-loosening rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO_ROOT / "tests/fixtures/rcs_mie_ka_sweep/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_16_ka_sweep_results/rfx.json"


def _mie_backscatter_over_pi_a2(ka: float) -> float:
    """sigma/(pi a^2), PEC sphere backscatter (Ruck 1970) — independent."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp_ = spherical_jn(n, x, derivative=True)
    ynp_ = spherical_yn(n, x, derivative=True)
    hn, hnp_ = jn + 1j * yn, jnp_ + 1j * ynp_
    a_n = jn / hn
    b_n = (jn + x * jnp_) / (hn + x * hnp_)
    series = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))
    return float(np.abs(series) ** 2 / x ** 2)


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def test_fixture_and_artifact_are_the_same_record(fixture):
    """The public crossval artifact and the test fixture must not drift apart."""
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture


def test_gated_coarse_bins_within_envelope_gate(fixture):
    """ka <= 1.25 coarse: |rfx - Mie| <= 3.2 dB, Mie re-derived independently."""
    gate = float(fixture["gates"]["coarse_gate_db"])
    assert gate == 3.2  # anti-loosening pin; regenerate + root-cause to change
    rows = fixture["gated_coarse"]
    assert [r["ka"] for r in rows] == [0.5, 0.75, 1.0, 1.25]
    for r in rows:
        mie_over = _mie_backscatter_over_pi_a2(r["ka"])
        # independent Mie must agree with the recorded Mie leg (oracle check)
        assert abs(10 * np.log10(mie_over / r["mie_sigma_over_pi_a2"])) < 0.01
        delta = 10 * np.log10(r["rfx_sigma_over_pi_a2"] / mie_over)
        assert abs(delta) <= gate, (r["ka"], delta)


def test_gated_fine_rung_within_envelope_gate(fixture):
    """Fine rung (12.8 cells/radius) ka {2.0, 4.0}: |rfx - Mie| <= 3.5 dB."""
    gate = float(fixture["gates"]["fine_gate_db"])
    assert gate == 3.5
    rows = fixture["gated_fine"]
    assert [r["ka"] for r in rows] == [2.0, 4.0]
    for r in rows:
        mie_over = _mie_backscatter_over_pi_a2(r["ka"])
        assert abs(10 * np.log10(mie_over / r["mie_sigma_over_pi_a2"])) < 0.01
        delta = 10 * np.log10(r["rfx_sigma_over_pi_a2"] / mie_over)
        assert abs(delta) <= gate, (r["ka"], delta)


def test_null_region_is_fenced_not_gated(fixture):
    """The unconverged bins must be present as diagnostics AND fenced in prose.

    This is the docpin half: if someone converts the null region into a gate
    (or deletes the diagnostics to hide it), this test goes red first.
    """
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "not gated" in scope
    assert "domain-size-only change" in scope
    assert "local-minimum positions move" in scope
    # the diagnostic curve actually covers the fenced region
    kas = [r["ka"] for r in fixture["diagnostic_curve_clear20"]]
    assert set(np.arange(6, 17) * 0.25) <= set(kas)  # 1.5 .. 4.0 present
    # gates dict names the fence explicitly
    assert "never gated" in fixture["gates"]["posture"]


def test_attribution_witnesses_are_recorded(fixture):
    """Truncation + domain-realization witnesses must ride with the record."""
    trunc = fixture["truncation_witness"]
    assert {t["ka"] for t in trunc} == {2.0, 3.0}
    for t in trunc:
        assert abs(t["delta_1x_db"] - t["delta_2x_db"]) <= 0.3, t
    # three domain realizations of the coarse curve exist (20 canonical + 30/40)
    assert {"30", "40"} <= set(fixture["domain_realizations"])
    assert len(fixture["domain_realizations"]["30"]) == 15
    # and the measured envelopes quoted in the gates were not understated:
    for clear in ("30", "40"):
        for r in fixture["domain_realizations"][clear]:
            if r["ka"] <= 1.25:
                assert abs(r["delta_db"]) <= fixture["gates"]["coarse_gate_db"]


def test_operating_point_is_the_derived_one(fixture):
    """Config pins: cells-per-radius floors, not lambda-resolution floors."""
    cfg = fixture["config"]
    assert cfg["coarse_cells_per_radius"] == 6.4
    assert cfg["fine_cells_per_radius"] == 12.8
    for r in fixture["gated_coarse"] + fixture["diagnostic_curve_clear20"]:
        assert r["a_over_dx"] >= 6.3, r  # the sphere is never cell-starved
