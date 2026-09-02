"""Dielectric-sphere Mie ka-sweep (0.5-2.5) — frozen-fixture gates (item 6).

Locks the committed measurement record of
``validation/crossval/17_dielectric_sphere_mie.py --write-fixture``
(``tests/fixtures/rcs_dielectric_sphere_mie/fixture.json``) against an
INDEPENDENT in-test re-implementation of the Bohren-Huffman dielectric
Mie backscatter series (real lossless m, from ``scipy.special`` — not the
producer's in-script oracle, so a shared oracle bug cannot self-certify;
both legs share the B&H convention and scipy, which is stated rather than
overclaimed as full independence).

Single-tier posture (all PR #475 lessons applied from day one):
  * GATED: coarse (cpr 6.4) ka <= 1.25 at 6.3 dB — hard-pinned AND
    recomputed from the committed 7-point clearance scan + three domain
    realizations AND bound to the script's live constant, so the gate,
    the envelope field, the data, and the enforcing constant cannot
    drift pairwise.
  * NO fine rung is gated — a measured decision witnessed by the
    committed fine_rung_witness (cpr-12.8 scan envelopes 3.75/3.04 dB,
    barely below the coarse 4.18; the clear=20-only apparent convergence
    was single-sample aliasing, caught before commit).
  * NOT GATED: every bin with ka >= 1.5 — domain-size unconverged
    (spreads 11.7 dB at ka=1.75 and 29.0 dB at ka=2.5 across clearance
    20/30/40). Pinned by docpins + a data tripwire so the fence can be
    neither silently gated nor silently dropped.

No FDTD runs here — the fixture is frozen evidence; live regeneration is
the crossval script's job. Gates must not be re-tuned to look tighter than
the recorded physics (no-silent-gate-loosening rule).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from tests._gate_policy import gate_from_envelope

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/rcs_dielectric_sphere_mie/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_17_dielectric_results/rfx.json"

M_IDX = float(np.sqrt(2.56))
KA_GATED_COARSE = [0.5, 0.75, 1.0, 1.25]
KA_FINE_WITNESS = [0.75, 1.0]
KA_FENCED = [1.5, 1.75, 2.0, 2.25, 2.5]


def _mie_backscatter_over_pi_a2(m: float, ka: float) -> float:
    """Lossless dielectric sphere backscatter (Bohren-Huffman) — re-implemented."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    mx = m * x
    jx = spherical_jn(n, x)
    jpx = spherical_jn(n, x, derivative=True)
    yx = spherical_yn(n, x)
    ypx = spherical_yn(n, x, derivative=True)
    jmx = spherical_jn(n, mx)
    jpmx = spherical_jn(n, mx, derivative=True)
    psi_x, psi_px = x * jx, jx + x * jpx
    chi_x, chi_px = -x * yx, -(yx + x * ypx)
    xi_x, xi_px = psi_x - 1j * chi_x, psi_px - 1j * chi_px
    psi_mx, psi_pmx = mx * jmx, jmx + mx * jpmx
    a = (m * psi_mx * psi_px - psi_x * psi_pmx) / (m * psi_mx * xi_px - xi_x * psi_pmx)
    b = (psi_mx * psi_px - m * psi_x * psi_pmx) / (psi_mx * xi_px - m * xi_x * psi_pmx)
    # unitarity witness (lossless coefficients sit on the unitarity circle)
    assert max(float(np.max(np.abs(a.real - np.abs(a) ** 2))),
               float(np.max(np.abs(b.real - np.abs(b) ** 2)))) < 1e-10
    s = np.sum((2 * n + 1) * ((-1.0) ** n) * (a - b))
    return float(np.abs(s) ** 2 / x ** 2)


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def _gated_coarse_deltas(fixture) -> list[float]:
    out = [abs(r["delta_db"]) for r in fixture["gated_coarse"]]
    for c in ("30", "40"):
        out += [abs(r["delta_db"]) for r in fixture["domain_realizations"][c]
                if r["ka"] <= max(KA_GATED_COARSE)]
    for ka in KA_GATED_COARSE:
        out += [abs(r["delta_db"])
                for r in fixture["clearance_scan"]["coarse"][str(ka)]]
    return out


def test_fixture_and_artifact_are_the_same_record(fixture):
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture


def test_script_claim_scope_literal_matches_fixture(fixture):
    """Binds the script-source claim_scope to the committed fixture copy
    (AST-extracted), so a hand-patched fixture cannot silently revert at the
    next --write-fixture and a script edit cannot leave the fixture stale —
    the prose analogue of the D2 constant binding (PR #476 review)."""
    import ast
    mod = ast.parse((_REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"
                     ).read_text(encoding="utf-8"))
    lits = {k.value: ast.literal_eval(v)
            for node in ast.walk(mod) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
            and k.value in ("claim_scope", "offline_probes_2026_07_27")}
    assert set(lits) == {"claim_scope", "offline_probes_2026_07_27"}
    assert " ".join(lits["claim_scope"].split()) == " ".join(fixture["claim_scope"].split())
    # R1 (review): the provenance audit prose is bound the same way.
    assert (" ".join(lits["offline_probes_2026_07_27"].split())
            == " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()))


def test_f1_retraction_is_content_pinned(fixture):
    """R2 (review): the tautology retraction must be CONTENT-pinned, not just
    consistent — a coherent script+fixture edit removing it must go red."""
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "same-array tautology" in scope
    prov = " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()).lower()
    assert "retracted" in prov


def test_gate_is_hard_pinned_and_equals_recomputed_envelope(fixture):
    """PR #475 D1 lesson baked in: hard ceiling AND derived relation, both."""
    g = fixture["gates"]
    env = max(_gated_coarse_deltas(fixture))
    assert abs(g["coarse_measured_envelope_db"] - env) < 5e-3
    assert g["coarse_gate_db"] == pytest.approx(
        gate_from_envelope(env, quantum=10), abs=1e-9)
    # HARD pin — widening requires editing this line with a root-cause.
    assert g["coarse_gate_db"] == 6.3


def test_script_live_gate_constant_matches_fixture(fixture):
    """PR #475 D2 lesson: bind the constant CI actually enforces."""
    src = (_REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"
           ).read_text(encoding="utf-8")
    m = re.search(r"^GATE_COARSE_DB = ([0-9.]+)", src, re.MULTILINE)
    assert m, "gate constant not found in script source"
    assert float(m.group(1)) == fixture["gates"]["coarse_gate_db"]
    # and no second fine-gate constant may quietly reappear ungated:
    assert re.search(r"^GATE_FINE_DB", src, re.MULTILINE) is None


def test_gated_coarse_bins_within_envelope_gate(fixture):
    gate = float(fixture["gates"]["coarse_gate_db"])
    rows = fixture["gated_coarse"]
    assert [r["ka"] for r in rows] == KA_GATED_COARSE
    for r in rows:
        mie_over = _mie_backscatter_over_pi_a2(M_IDX, r["ka"])
        assert abs(10 * np.log10(mie_over / r["mie_sigma_over_pi_a2"])) < 0.01
        delta = 10 * np.log10(r["rfx_sigma_over_pi_a2"] / mie_over)
        assert abs(delta) <= gate, (r["ka"], delta)
    assert max(_gated_coarse_deltas(fixture)) <= gate


def test_fine_rung_is_witnessed_not_gated(fixture):
    """The measured no-fine-rung decision must stay auditable AND un-gated."""
    fw = fixture["fine_rung_witness"]
    assert fw["cells_per_radius"] == 12.8
    for ka in KA_FINE_WITNESS:
        assert len(fw["fine"][str(ka)]) == len(fw["clearances"]) >= 7
    env_w = max(abs(r["delta_db"]) for ka in KA_FINE_WITNESS
                for r in fw["fine"][str(ka)])
    assert abs(fixture["gates"]["fine_rung_witness_envelope_db"] - env_w) < 5e-3
    # the witness must keep showing WHY there is no fine tier: within ~15%
    # of the coarse envelope (if a future regeneration collapses it far
    # below, that is a real improvement — promote it with a root-cause,
    # do not silently keep the old posture prose)
    assert env_w > 0.7 * fixture["gates"]["coarse_measured_envelope_db"]
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "no fine rung is gated" in scope
    assert "aliasing" in scope
    assert "fine_ka" not in fixture["gates"]


def test_fenced_region_is_fenced_not_gated(fixture):
    """Docpins + data tripwire on the ka >= 1.5 fence."""
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "not gated" in scope
    assert "11.7 db (ka=1.75" in scope
    assert "29.0 db (ka=2.5" in scope
    assert "never gated" in fixture["gates"]["posture"]
    kas = [r["ka"] for r in fixture["diagnostic_curve_clear20"]]
    assert len(kas) == len(set(kas)) == 9
    assert set(KA_FENCED) <= set(kas)
    # data tripwire: the fenced region's committed record must EXCEED the
    # gate somewhere — if it stops doing so, promote with a root-cause,
    # do not silently delete the fence.
    fenced = [abs(r["delta_db"]) for r in fixture["diagnostic_curve_clear20"]
              if r["ka"] >= 1.5]
    for c in ("30", "40"):
        fenced += [abs(r["delta_db"]) for r in fixture["domain_realizations"][c]
                   if r["ka"] >= 1.5]
    assert max(fenced) > fixture["gates"]["coarse_gate_db"]


def test_attribution_witnesses_are_recorded(fixture):
    trunc = fixture["truncation_witness"]
    assert {t["ka"] for t in trunc} == set(KA_GATED_COARSE)
    for t in trunc:
        assert abs(t["delta_1x_db"] - t["delta_2x_db"]) <= 0.3, t
    assert {"30", "40"} <= set(fixture["domain_realizations"])
    assert len(fixture["domain_realizations"]["30"]) == 9
    scan = fixture["clearance_scan"]
    assert len(scan["clearances"]) >= 7
    for ka in KA_GATED_COARSE:
        assert len(scan["coarse"][str(ka)]) == len(scan["clearances"])
    prov = " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()).lower()
    assert "not committed as data" in prov


def _all_rows(fixture):
    """Every committed measurement row, across all families (review F2/F3)."""
    rows = list(fixture["gated_coarse"]) + list(fixture["diagnostic_curve_clear20"])
    for fam in fixture["domain_realizations"].values():
        rows += list(fam)
    for ka_rows in fixture["clearance_scan"]["coarse"].values():
        rows += list(ka_rows)
    for ka_rows in fixture["fine_rung_witness"]["fine"].values():
        rows += list(ka_rows)
    return rows


def test_operating_point_is_the_derived_one_on_every_row(fixture):
    """Review F2: the floors must hold on ALL row families — the 33 rows that
    set the gate envelope, not just the 40 that display it."""
    cfg = fixture["config"]
    assert cfg["eps_r"] == 2.56
    assert cfg["resolution_floor"] == 24            # lambda_internal / 15
    assert cfg["coarse_cells_per_radius"] == 6.4
    # F1 (review): the two-run flag is a no-op for monostatic_rcs (always
    # computed from the raw run, rfx/rcs.py) — it must stay OFF, and the
    # config must carry the note saying why.
    assert cfg["subtract_incident_reference"] is False
    assert "no-op" in cfg["subtract_note"]
    rows = _all_rows(fixture)
    assert len(rows) >= 70
    for r in rows:
        assert r["a_over_dx"] >= 6.3, r
        assert r["resolution"] >= 24, r
        expected_res = max(24, math.ceil(2 * math.pi * r["cells_per_radius"] / r["ka"]))
        assert r["resolution"] == expected_res, r


def test_every_row_is_internally_consistent_and_oracle_checked(fixture):
    """Review F3: cross-check ALL rows against the independent Mie leg and
    assert the three redundant encodings agree, so no delta_db anywhere in
    the envelope population can be edited independently of its row."""
    mie_cache = {}
    for r in _all_rows(fixture):
        ka = r["ka"]
        if ka not in mie_cache:
            mie_cache[ka] = _mie_backscatter_over_pi_a2(M_IDX, ka)
        # recorded Mie leg vs independent re-implementation
        assert abs(10 * np.log10(mie_cache[ka] / r["mie_sigma_over_pi_a2"])) < 0.01, r
        # delta_db == rfx_dbsm - mie_dbsm (rounding tolerance)
        assert abs(r["delta_db"] - (r["rfx_monostatic_dbsm"] - r["mie_dbsm"])) < 2e-3, r
        # sigma fields consistent with dBsm fields (radius exact from ka)
        lam = 299792458.0 / 3e9
        radius = ka * lam / (2 * math.pi)
        pi_a2 = math.pi * radius ** 2
        # tolerance sits above the recorded-precision floor (the 1e-6
        # sigma quantum is 2.5e-4 dB at ka=0.5; worst observed residual
        # 0.0009 dB) so a regen's thread-order noise cannot flake it.
        assert abs(r["rfx_monostatic_dbsm"]
                   - 10 * np.log10(r["rfx_sigma_over_pi_a2"] * pi_a2)) < 1e-2, r
        assert abs(r["mie_dbsm"]
                   - 10 * np.log10(r["mie_sigma_over_pi_a2"] * pi_a2)) < 1e-2, r


def test_scan_populations_carry_their_own_provenance(fixture):
    """Review F4: a degenerate scan (one clearance run seven times) must not
    be able to masquerade as the anti-aliasing population."""
    scan = fixture["clearance_scan"]
    clearances = scan["clearances"]
    assert len(set(clearances)) == len(clearances) >= 7
    for ka, rows in scan["coarse"].items():
        assert [r["clear_cells"] for r in rows] == clearances, ka
        assert all(r["cells_per_radius"] == 6.4 for r in rows), ka
    fw = fixture["fine_rung_witness"]
    assert len(set(fw["clearances"])) == len(fw["clearances"]) >= 7
    for ka, rows in fw["fine"].items():
        assert [r["clear_cells"] for r in rows] == fw["clearances"], ka
        assert all(r["cells_per_radius"] == 12.8 for r in rows), ka
    for c, fam in fixture["domain_realizations"].items():
        assert all(r["clear_cells"] == int(c) for r in fam), c
