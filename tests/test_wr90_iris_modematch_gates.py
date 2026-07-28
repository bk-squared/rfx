"""WR-90 single inductive iris vs mode-matching — frozen-fixture gates (item 3 S1).

Locks the committed record of
``validation/crossval/18_wr90_iris_modematch.py --write-fixture``
(``tests/fixtures/wr90_iris_modematch/fixture.json``) against an INDEPENDENT
in-test re-implementation of the TEn0 mode-matching cascade oracle (same
formulation class re-typed from the physics, sharing only numpy — a shared
producer bug in the overlap/junction algebra would still be caught by the
oracle's own unitarity/Marcuvitz witnesses, which this test re-runs).

Posture after the PR #480 review rework (all #475/#476 lessons plus #480's):
  * GATED: fine rung (dx=a/60, flux) |S11 - oracle| <= 0.04 abs over 8
    committed configs (3 apertures x {centred, off-centre iris} + 2 guide
    lengths); Richardson 2*fine - coarse on the oracle <= 0.01 abs at EVERY
    one of those 8 pairs (not just the canonical one — #480 B1).
  * REPORTED: coarse rung, raw extraction, residual ripple, first-order
    ratios — committed data, recomputed here, never gated.
  * RETRACTED and content-pinned: the modal-extraction fence (its 1.112-1.164
    evidence did not survive the footprint/absorber fixes; modal extraction
    is passivity-clean on the corrected setup). Structures beyond one
    symmetric inductive iris stay fenced.
  * Every prose number in claim_scope is RECOMPUTED from committed rows here
    (envelopes, ratios, ripple, raw-vs-flux) — #480 N1/N3 class.

No FDTD runs here; regeneration is the crossval script's job. Gates must not
be re-tuned to look tighter than the recorded physics.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO_ROOT / "tests/fixtures/wr90_iris_modematch/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_18_wr90_iris_results/rfx.json"
_SCRIPT = _REPO_ROOT / "validation/crossval/18_wr90_iris_modematch.py"

C0 = 299792458.0
MU0 = 4e-7 * np.pi
A = 22.86e-3
T = 1.524e-3


# --------------------------------------------------------------------------- #
# Independent re-implementation of the mode-matching oracle.
# --------------------------------------------------------------------------- #
def _gam(n, w, k):
    return np.sqrt(complex((n * np.pi / w) ** 2 - k * k))


def _ovl(a, d, n, m):
    x0 = (a - d) / 2
    al, be = n * np.pi / a, m * np.pi / d

    def iss(p, q, L):
        if abs(p - q) < 1e-30:
            return L / 2 - np.sin(2 * p * L) / (4 * p)
        return (np.sin((p - q) * L) / (p - q) - np.sin((p + q) * L) / (p + q)) / 2

    def ics(p, q, L):
        if abs(p - q) < 1e-30:
            return (1 - np.cos(2 * q * L)) / (4 * q) if q > 0 else 0.0
        return ((1 - np.cos((q + p) * L)) / (q + p)
                + (1 - np.cos((q - p) * L)) / (q - p)) / 2

    return (np.sqrt(2 / a) * np.sqrt(2 / d)
            * (np.cos(al * x0) * iss(al, be, d) + np.sin(al * x0) * ics(al, be, d)))


def _iris_s11(a, d, t, f, n_a=40):
    k = 2 * np.pi * f / C0
    n_b = max(4, int(round(n_a * d / a)))
    Na = np.arange(1, 2 * n_a, 2)
    Nb = np.arange(1, 2 * n_b, 2)
    gA = np.array([_gam(n, a, k) for n in Na])
    gB = np.array([_gam(m, d, k) for m in Nb])
    w = k * C0
    YA, YB = gA / (1j * w * MU0), gB / (1j * w * MU0)
    Cm = np.array([[_ovl(a, d, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + Cm.T @ YAd @ Cm)
    T_ba = 2 * Minv @ Cm.T @ YAd
    R_aa = Cm @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - Cm.T @ YAd @ Cm)
    T_ab = Cm @ (np.eye(n_b) + R_bb)
    sYA, sYB = np.sqrt(YA), np.sqrt(YB)
    S = [(sYA[:, None] * R_aa) / sYA[None, :],
         (sYA[:, None] * T_ab) / sYB[None, :],
         (sYB[:, None] * T_ba) / sYA[None, :],
         (sYB[:, None] * R_bb) / sYB[None, :]]
    P = np.diag(np.exp(-gB * t))
    z = np.zeros((n_b, n_b), dtype=complex)

    def star(sa, sb):
        A11, A12, A21, A22 = sa
        B11, B12, B21, B22 = sb
        n = A22.shape[0]
        i1 = np.linalg.inv(np.eye(n) - A22 @ B11)
        i2 = np.linalg.inv(np.eye(n) - B11 @ A22)
        return (A11 + A12 @ B11 @ i1 @ A21, A12 @ i2 @ B12,
                B21 @ i1 @ A21, B22 + B21 @ A22 @ i2 @ B12)

    rev = (S[3], S[2], S[1], S[0])
    tot = star(star((S[0], S[1], S[2], S[3]), (z, P, P, z)), rev)
    s11, s21 = tot[0][0, 0], tot[2][0, 0]
    # witnesses ride with every evaluation
    assert abs(abs(s11) ** 2 + abs(s21) ** 2 - 1) < 1e-9   # lossless unitarity
    return abs(s11)


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def _ripple_pp(s11):
    """Quadratic-detrended peak-to-peak of a |S11| trace."""
    x = np.arange(len(s11))
    y = np.asarray(s11, dtype=float)
    r = y - np.polyval(np.polyfit(x, y, 2), x)
    return float(r.max() - r.min())


def _rows(fixture):
    return (list(fixture["gated_fine"]) + list(fixture["coarse_diagnostic"])
            + list(fixture["raw_extraction_record"]))


def test_fixture_and_artifact_are_the_same_record(fixture):
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture


def test_script_prose_literals_match_fixture(fixture):
    """AST-binds claim_scope AND the modal-fence retraction entry (the prose
    analogue of the D2 constant binding; PR #476 pattern)."""
    mod = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    lits = {k.value: ast.literal_eval(v)
            for node in ast.walk(mod) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
            and k.value in ("claim_scope", "modal_fence_retraction_2026_07_28")}
    assert set(lits) == {"claim_scope", "modal_fence_retraction_2026_07_28"}
    assert " ".join(lits["claim_scope"].split()) == " ".join(fixture["claim_scope"].split())
    assert (" ".join(lits["modal_fence_retraction_2026_07_28"].split())
            == " ".join(
                fixture["provenance"]["modal_fence_retraction_2026_07_28"].split()))


def test_gates_are_hard_pinned_and_equal_recomputed_envelopes(fixture):
    """D1: hard ceilings AND the derived x1.5 relation, both, from data."""
    g = fixture["gates"]
    env_fine = max(r["max_gap_abs"] for r in fixture["gated_fine"])
    env_rich = max(r["richardson_dev_abs"] for r in fixture["coarse_diagnostic"])
    assert abs(g["fine_measured_envelope_abs"] - env_fine) < 5e-4
    assert abs(g["richardson_measured_envelope_abs"] - env_rich) < 5e-4
    assert g["fine_gate_abs"] == pytest.approx(
        math.ceil(env_fine * 1.5 * 100) / 100, abs=1e-9)
    assert g["richardson_gate_abs"] == pytest.approx(
        math.ceil(env_rich * 1.5 * 100) / 100, abs=1e-9)
    assert g["fine_gate_abs"] == 0.04       # hard pin — root-cause to change
    assert g["richardson_gate_abs"] == 0.01


def test_script_live_gate_constants_match_fixture(fixture):
    """D2: bind the constants CI actually enforces."""
    src = _SCRIPT.read_text(encoding="utf-8")
    m_f = re.search(r"^GATE_FINE_ABS = ([0-9.]+)", src, re.MULTILINE)
    m_r = re.search(r"^GATE_RICH_ABS = ([0-9.]+)", src, re.MULTILINE)
    assert m_f and m_r, "gate constants not found in script source"
    assert float(m_f.group(1)) == fixture["gates"]["fine_gate_abs"]
    assert float(m_r.group(1)) == fixture["gates"]["richardson_gate_abs"]
    # #480: the write-fixture self-check must demand EXACT ceil(x1.5) equality,
    # not an interval that would accept a stale provisional gate.
    assert "abs(gate - required) > 1e-9" in src


def test_gated_fine_rows_within_gate_against_independent_oracle(fixture):
    """Every committed fine row within the gate, oracle re-derived HERE."""
    gate = fixture["gates"]["fine_gate_abs"]
    freqs = fixture["config"]["freqs_hz"]
    rows = fixture["gated_fine"]
    assert len(rows) == 8      # 3 apertures x {centred, off-centre} + 2 glen
    seen = {(r["d_mm"], r["glen_m"], r["iris_frac"]) for r in rows}
    assert len(seen) == 8
    for r in rows:
        assert r["cells_per_a"] == fixture["config"]["fine_cells_per_a"]
        assert r["normalize"] == "flux"
        d = r["d_mm"] * 1e-3
        orc = [_iris_s11(A, d, T, f) for f in freqs]
        assert max(abs(a - b) for a, b in zip(orc, r["oracle_s11"])) < 1e-3, r["d_mm"]
        gap = max(abs(a - b) for a, b in zip(r["s11"], orc))
        assert gap <= gate + 1e-6, (r["d_mm"], r["glen_m"], r["iris_frac"], gap)
        assert abs(gap - r["max_gap_abs"]) < 2e-3
        assert r["max_colpow"] <= 1.02


def test_richardson_cross_confirms_oracle_at_every_pair(fixture):
    """#480 B1: the Richardson witness is domain-scanned like the fine gate."""
    gate = fixture["gates"]["richardson_gate_abs"]
    freqs = fixture["config"]["freqs_hz"]
    coarse = fixture["coarse_diagnostic"]
    assert len(coarse) == 8
    for cr in coarse:
        key = (cr["d_mm"], cr["glen_m"], cr["iris_frac"])
        fr = next(r for r in fixture["gated_fine"]
                  if (r["d_mm"], r["glen_m"], r["iris_frac"]) == key)
        orc = [_iris_s11(A, cr["d_mm"] * 1e-3, T, f) for f in freqs]
        rich = [2 * f_ - c_ for f_, c_ in zip(fr["s11"], cr["s11"])]
        dev = max(abs(a - b) for a, b in zip(rich, orc))
        assert dev <= gate + 1e-6, (key, dev)
        assert abs(dev - cr["richardson_dev_abs"]) < 2e-3
    # first-order ratios RECOMPUTED from the rows (#480 N3), ideal 0.5
    recomputed = []
    for fr in fixture["gated_fine"]:
        for cr in coarse:
            if (fr["d_mm"], fr["glen_m"], fr["iris_frac"]) == (
                    cr["d_mm"], cr["glen_m"], cr["iris_frac"]):
                recomputed.append(round(fr["max_gap_abs"] / cr["max_gap_abs"], 3))
    assert recomputed == fixture["gates"]["first_order_ratios"]
    assert all(0.40 <= x <= 0.70 for x in recomputed), recomputed


def test_prose_numbers_are_recomputed_from_rows(fixture):
    """#480 N1: every quantitative claim in claim_scope must come from data."""
    scope = " ".join(fixture["claim_scope"].split())
    freqs = fixture["config"]["freqs_hz"]
    # ripple envelopes per tier
    rip_fine = max(_ripple_pp(r["s11"]) for r in fixture["gated_fine"])
    rip_coarse = max(_ripple_pp(r["s11"]) for r in fixture["coarse_diagnostic"])
    assert rip_fine <= 0.0116 + 5e-4 and f"{rip_fine:.4f}"[:6] == "0.0116"
    assert rip_coarse <= 0.0166 + 5e-4 and f"{rip_coarse:.4f}"[:6] == "0.0166"
    assert "fine <= 0.0116" in scope and "coarse <= 0.0166" in scope
    # pointwise raw-vs-flux difference (NOT the max_gap statistic)
    diffs = []
    for raw in fixture["raw_extraction_record"]:
        flux = next(c for c in fixture["coarse_diagnostic"]
                    if c["d_mm"] == raw["d_mm"]
                    and (c["glen_m"], c["iris_frac"])
                    == (fixture["config"]["canonical_glen_m"],
                        fixture["config"]["canonical_iris_frac"]))
        diffs.append(max(abs(a - b) for a, b in zip(raw["s11"], flux["s11"])))
    assert max(diffs) == pytest.approx(0.033, abs=1e-3)
    assert "up to 0.033" in scope
    # coarse-rung range and the frequency count
    coarse_gaps = [r["max_gap_abs"] for r in fixture["coarse_diagnostic"]]
    assert f"{min(coarse_gaps):.3f}" == "0.018" and f"{max(coarse_gaps):.3f}" == "0.043"
    assert "0.018-0.043 abs" in scope
    assert len(freqs) == 29 and "29 frequency" in scope


def test_modal_fence_is_retracted_with_data(fixture):
    """The withdrawn fence must stay auditable AND supported by measurement."""
    scope = " ".join(fixture["claim_scope"].split())
    assert "RETRACTED" in scope
    assert "passivity-CLEAN" in scope
    assert "1.112-1.164" in scope          # the withdrawn evidence, named
    mw = fixture["modal_extraction_witness"]
    assert len(mw["rows"]) == 4
    assert {r["cells_per_a"] for r in mw["rows"]} == {30, 60}
    for r in mw["rows"]:
        assert r["max_colpow"] <= 1.05, r      # clean on the corrected setup
        assert r["extractor_warnings"] == []
    assert max(r["max_colpow"] for r in mw["rows"]) == pytest.approx(1.0207, abs=1e-3)
    assert "RETRACTED" in " ".join(mw["note"].split())
    prov = " ".join(
        fixture["provenance"]["modal_fence_retraction_2026_07_28"].split())
    assert "RETRACTED" in prov and "setup symptom" in prov
    assert "no longer fenced" in fixture["gates"]["posture"]


def test_setup_defects_and_scope_fence_are_content_pinned(fixture):
    """The three rasterization/absorber defects and the iris scope fence."""
    scope = " ".join(fixture["claim_scope"].split())
    assert "parasitic wall-slot" in scope
    assert "half-ulp fragile" in scope and "+/-0.07" in scope
    assert "d + 2*dx" in scope and "4-6x" in scope
    assert "0.75*lambda_g" in scope
    assert "ONE symmetric inductive iris" in scope
    assert "EXPERIMENTAL" in scope
    assert "never gated" in fixture["gates"]["posture"]


def test_diagnostics_and_witnesses_are_recorded(fixture):
    assert len(fixture["raw_extraction_record"]) == 3
    for r in fixture["raw_extraction_record"]:
        assert r["normalize"] == "False"
        assert r["max_colpow"] <= 1.02
    trunc = fixture["truncation_witness"]
    assert len(trunc) == 4                       # 3 apertures + asymmetric
    assert any(t["iris_frac"] == 0.42 for t in trunc)   # #480 B2
    assert all(t["shift_abs"] <= 0.001 for t in trunc)
    prov = " ".join(fixture["provenance"]["no_preflight_note"].split()).lower()
    assert "no sim.preflight()" in prov
    assert "coarse_domain_scan" not in fixture   # folded into coarse_diagnostic


def test_operating_point_is_grid_exact_on_every_row(fixture):
    """Floors/conventions on ALL row families (#476 F2 class)."""
    cfg = fixture["config"]
    assert "0.75" in cfg["cpml_layers_rule"]
    rows = _rows(fixture)
    assert len(rows) == 19
    for r in rows:
        cells = r["cells_per_a"]
        assert cells in (cfg["coarse_cells_per_a"], cfg["fine_cells_per_a"])
        assert r["dx_mm"] == pytest.approx(22.86 / cells, abs=1e-3)
        d_c = round(r["d_mm"] / r["dx_mm"])
        # fins cover nodes 0..fin_c so the electrical aperture equals nominal d
        assert r["aperture_cells"] == d_c - 1, r
        assert r["thickness_cells"] == round(1.524 / r["dx_mm"]), r
        assert len(r["s11"]) == len(cfg["freqs_hz"]) == 29
