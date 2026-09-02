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
import re
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import gate_from_envelope

_REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _residual_ripple_pp(row):
    """Quadratic-detrended peak-to-peak of the RESIDUAL |S11| - oracle.

    PR #480 R1: detrending the raw trace leaves the oracle's own curvature in
    the number (0.0096 of it at the wide aperture), so a "residual ripple"
    claim measured that way overstates the absorber artefact by ~50-80%.
    """
    y = np.asarray(row["s11"], dtype=float) - np.asarray(row["oracle_s11"], dtype=float)
    x = np.arange(len(y))
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
        gate_from_envelope(env_fine, quantum=100), abs=1e-9)
    assert g["richardson_gate_abs"] == pytest.approx(
        gate_from_envelope(env_rich, quantum=100), abs=1e-9)
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
    rip_fine = max(_residual_ripple_pp(r) for r in fixture["gated_fine"])
    rip_coarse = max(_residual_ripple_pp(r) for r in fixture["coarse_diagnostic"])
    assert f"{rip_fine:.4f}" == "0.0077", rip_fine
    assert f"{rip_coarse:.4f}" == "0.0158", rip_coarse
    assert "fine <= 0.0077, coarse <= 0.0158" in scope
    assert "MINUS the oracle" in scope       # the metric is named, not implied
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
    # PR #480 R2: accuracy rides with the retraction — modal must be recorded
    # AND be comparable to flux (a little worse, which is why flux gates).
    deltas = []
    for r in mw["rows"]:
        assert len(r["s11"]) == len(fixture["config"]["freqs_hz"])
        fam = (fixture["gated_fine"] if r["cells_per_a"]
               == fixture["config"]["fine_cells_per_a"]
               else fixture["coarse_diagnostic"])
        flux = next(x for x in fam if x["d_mm"] == r["d_mm"]
                    and (x["glen_m"], x["iris_frac"])
                    == (fixture["config"]["canonical_glen_m"],
                        fixture["config"]["canonical_iris_frac"]))
        delta = r["max_gap_abs"] - flux["max_gap_abs"]
        # Per-row: modal must stay CLOSE to flux (that is what makes the
        # retraction safe on accuracy grounds). The lower bound carries a
        # small negative tolerance because the tightest committed row has
        # only 0.0002 of margin (modal 0.0099 vs flux 0.0097) — a benign
        # regen shift must not red CI with "modal is better than flux"
        # (PR #480 re-review, optional robustness note).
        deltas.append(delta)
        assert -0.002 <= delta <= 0.01, (r["d_mm"], r["cells_per_a"], delta)
    # The "flux is the better extractor" claim is pinned on the MAXIMUM
    # delta, which is a robust 0.0055 rather than a 0.0002 knife-edge.
    assert max(deltas) >= 0.003, deltas
    assert "ACCURACY" in scope and "little worse" in scope
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


# --------------------------------------------------------------------------- #
# issue #812 re-gate — the aperture-sensitivity gates.
#
# The audit of issue #812 measured a one-cell aperture error, the smallest the
# grid-snapped geometry can express, passing BOTH gated observables: at
# d = 7.620 mm one fine cell moved the fine gap 0.0097 -> 0.0265 (gate 0.04)
# and the Richardson deviation 0.0010 -> 0.0030 (gate 0.01).  Both numbers are
# reproduced below, and the mechanism is named:
#
#   * the pooled fine gate is set by the WORST of eight configurations and
#     spent at all eight, so d = 7.620 (the least sensitive aperture, because
#     |S11| -> 1 saturates there) carried 4x the slack its own data earns;
#   * the Richardson witness cancels the defect BY CONSTRUCTION -- an aperture
#     error of one cell AT EACH RUNG is proportional to dx, which is exactly
#     what 2*S(a/60) - S(a/30) is built to remove.  No tightening of the
#     Richardson gate can catch that class, and none is attempted.
#
# Pre-declared with its derivation in
# docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md
# (sections 2.1-2.5) in a commit preceding the measurement that judges it.
# --------------------------------------------------------------------------- #

_DX_FINE = A / 60
_DX_COARSE = A / 30
_DECLARED_APERTURES_MM = (18.288, 12.192, 7.620)


def _cfg_key(d_mm, glen, frac):
    return f"{d_mm:.3f}|{glen:.2f}|{frac:.2f}"


def _script_per_config_gates() -> dict:
    """AST-extract GATE_FINE_ABS_PER_CONFIG from the script source.

    Extraction, not import: the crossval script imports rfx and builds a
    Simulation at module scope-adjacent call sites, and this file must stay a
    no-FDTD frozen-fixture lane.
    """
    mod = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(mod):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name)
                        and t.id == "GATE_FINE_ABS_PER_CONFIG"
                        for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("GATE_FINE_ABS_PER_CONFIG not found in script source")


def _one_cell_defect(fine_row, coarse_row, sign, freqs):
    """The audit's defect, modelled on the committed rows.

    Aperture one cell too wide (sign=+1) or narrow (-1) AT EACH RUNG, with the
    record and the oracle still at the nominal d.  The rfx trace is displaced
    by the oracle's own response to that aperture change, i.e. the FDTD's
    discretization error is held fixed and only the geometry moves; this is the
    first-order model, and it is confirmed against a real FDTD pair in the
    lane's report.  Returns (fine_gap, richardson_dev) against the NOMINAL
    oracle.
    """
    d = fine_row["d_mm"] * 1e-3
    base = np.array([_iris_s11(A, d, T, f) for f in freqs])
    shift_f = np.array([_iris_s11(A, d + sign * _DX_FINE, T, f) for f in freqs]) - base
    shift_c = np.array([_iris_s11(A, d + sign * _DX_COARSE, T, f) for f in freqs]) - base
    f_def = np.array(fine_row["s11"]) + shift_f
    c_def = np.array(coarse_row["s11"]) + shift_c
    return (float(np.max(np.abs(f_def - base))),
            float(np.max(np.abs(2 * f_def - c_def - base))))


def test_per_config_fine_gates_are_derived_bound_and_strictly_tighter(fixture):
    """G18-A: gate = round-up(that config's OWN envelope x 1.5) at quantum
    1000, bound to the script constant, never above the pooled ceiling."""
    script_gates = _script_per_config_gates()
    rows = fixture["gated_fine"]
    assert len(rows) == 8
    assert set(script_gates) == {
        _cfg_key(r["d_mm"], r["glen_m"], r["iris_frac"]) for r in rows}
    pooled = fixture["gates"]["fine_gate_abs"]
    for r in rows:
        key = _cfg_key(r["d_mm"], r["glen_m"], r["iris_frac"])
        required = gate_from_envelope(r["max_gap_abs"], quantum=1000)
        assert script_gates[key] == pytest.approx(required, abs=1e-12), key
        # never widened: the per-config gate is <= the pre-#812 pooled gate
        assert script_gates[key] <= pooled + 1e-12, key
        # (A): the committed row still sits inside its own tighter gate
        assert r["max_gap_abs"] <= script_gates[key] + 1e-9, (key, r["max_gap_abs"])
    # and the fixture carries the same table, so a script-only edit goes red
    assert fixture["gates"]["fine_gate_abs_per_config"] == script_gates
    # the tightening is real everywhere and large where it matters: every
    # configuration is strictly tighter than the pooled ceiling, and the three
    # that carried the most unearned slack (both d = 7.620 rows, whose one-cell
    # sensitivity is the smallest, and d = 18.288 canonical) gain >= 2x.
    assert max(script_gates.values()) < pooled
    assert sum(1 for v in script_gates.values() if v <= pooled / 2) == 3
    assert script_gates[_cfg_key(7.62, 0.2, 0.5)] == 0.015
    assert script_gates[_cfg_key(7.62, 0.2, 0.42)] == 0.015


def test_the_audit_one_cell_defect_fails_the_new_gate_and_passed_the_old(fixture):
    """Criterion (B), on the audit's own measured configuration.

    d = 7.620 mm canonical, aperture one fine cell too wide at each rung:
    the audit measured fine 0.0097 -> 0.0265 (pooled gate 0.04, PASS) and
    Richardson 0.0010 -> 0.0030 (gate 0.01, PASS).  Both reproduce here; the
    per-config gate 0.015 turns the fine leg red, and the Richardson leg stays
    green for the reason stated above (dx-proportional errors cancel).
    """
    freqs = fixture["config"]["freqs_hz"]
    fr = next(r for r in fixture["gated_fine"]
              if r["d_mm"] == 7.62 and r["iris_frac"] == 0.5 and r["glen_m"] == 0.2)
    cr = next(r for r in fixture["coarse_diagnostic"]
              if r["d_mm"] == 7.62 and r["iris_frac"] == 0.5 and r["glen_m"] == 0.2)
    assert fr["max_gap_abs"] == pytest.approx(0.0097, abs=5e-4)
    assert cr["richardson_dev_abs"] == pytest.approx(0.0010, abs=5e-4)
    gap, rich = _one_cell_defect(fr, cr, +1, freqs)
    # the audit's two numbers, reproduced
    assert gap == pytest.approx(0.0265, abs=5e-4), gap
    assert rich == pytest.approx(0.0030, abs=5e-4), rich
    # what the OLD gates did with them
    assert gap <= fixture["gates"]["fine_gate_abs"]          # 0.0265 <= 0.04
    assert rich <= fixture["gates"]["richardson_gate_abs"]   # 0.0030 <= 0.01
    # what the NEW gate does with them
    cfg_gate = _script_per_config_gates()[_cfg_key(7.62, 0.2, 0.5)]
    assert cfg_gate == 0.015
    assert gap > cfg_gate, (gap, cfg_gate)
    assert gap / cfg_gate >= 1.7, gap / cfg_gate


def test_one_cell_aperture_resolution_is_declared_and_pinned(fixture):
    """G18-B: the case's aperture RESOLUTION is itself the claim, so it is
    gated -- a future regeneration that loses detection goes red instead of
    quietly re-scoping.  Declared in the pre-declaration note section 2.4."""
    freqs = fixture["config"]["freqs_hz"]
    script_gates = _script_per_config_gates()
    detected = {+1: [], -1: []}
    rich_detected = 0
    for fr in fixture["gated_fine"]:
        key = _cfg_key(fr["d_mm"], fr["glen_m"], fr["iris_frac"])
        cr = next(c for c in fixture["coarse_diagnostic"]
                  if (c["d_mm"], c["glen_m"], c["iris_frac"])
                  == (fr["d_mm"], fr["glen_m"], fr["iris_frac"]))
        for sign in (+1, -1):
            gap, rich = _one_cell_defect(fr, cr, sign, freqs)
            if gap > script_gates[key]:
                detected[sign].append(gap / script_gates[key])
            rich_detected += rich > fixture["gates"]["richardson_gate_abs"]
    # over-aperture: every configuration, with margin above the repo's x1.5
    assert len(detected[+1]) == 8, detected
    assert min(detected[+1]) == pytest.approx(1.77, abs=0.02), min(detected[+1])
    # under-aperture: NOT resolved with margin anywhere -- the honest limit
    assert len(detected[-1]) == 2, detected
    assert max(detected[-1]) < 1.5, detected[-1]
    # Richardson is blind to the whole class, both signs, all configs
    assert rich_detected == 0
    scope = " ".join(fixture["claim_scope"].split())
    assert "one-cell" in scope and "under-aperture" in scope


def test_declared_apertures_are_pinned_and_grid_exact(fixture):
    """G18-C: the aperture set is a CLAIM, not a free parameter.

    Nothing before this checked it: the oracle is evaluated at whatever d the
    run was handed, so a silently relabelled aperture moves both sides together
    and every residual stays nominal.  The pin is geometric (a = 22.86 mm and
    the two declared rungs) and needs no tolerance: a one-fine-cell relabel
    (7.620 -> 8.001 mm) is 21 fine cells (odd, so the symmetric two-fin
    construction cannot realise it) and 10.5 coarse cells.
    """
    for r in _rows(fixture) + [x for x in fixture["modal_extraction_witness"]["rows"]]:
        assert r["d_mm"] in _DECLARED_APERTURES_MM, r["d_mm"]
    for d_mm in _DECLARED_APERTURES_MM:
        for dx in (_DX_COARSE, _DX_FINE):
            n = d_mm * 1e-3 / dx
            assert abs(n - round(n)) < 1e-9, (d_mm, dx)
            assert round(n) % 2 == 0, (d_mm, dx, n)   # symmetric-fin parity
    # falsifier: the one-cell relabel this pin exists to reject
    n_fine = 8.001e-3 / _DX_FINE
    n_coarse = 8.001e-3 / _DX_COARSE
    assert abs(n_fine - round(n_fine)) < 1e-9 and round(n_fine) % 2 == 1
    assert abs(n_coarse - round(n_coarse)) > 1e-3
    assert 8.001 not in _DECLARED_APERTURES_MM
    # and the script enforces it on every run_point call
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "def assert_declared_aperture(d_phys):" in src
    assert "assert_declared_aperture(d_phys)   # issue #812 G18-C" in src


# --------------------------------------------------------------------------- #
# issue #812 ROUND 2 — numeric provenance for the aperture-resolution claim.
#
# Round 1 shipped, into both committed evidence JSONs and the script's
# claim_scope literal, the assertion that the committed fine trace sits CLOSER
# to the oracle at d MINUS one fine cell than to the oracle at the declared d,
# quoting 0.0035 as that distance.  0.0035 is not a distance: it is the
# one-cell UNDER-aperture DEFECT metric, which carries the oracle shift with
# the opposite sign.  The two quantities are computed separately below and the
# retracted claim is refuted mechanically, at every configuration.
#
# Everything the corrected prose points at lives in
# validation/crossval/_18_wr90_iris_results/aperture_resolution.json, built by
# scripts/diagnostics/build_cv18_aperture_resolution.py (no FDTD) and
# re-derived here from the committed traces with THIS file's independent
# oracle re-implementation.
# --------------------------------------------------------------------------- #

_APERTURE_RES = (_REPO_ROOT
                 / "validation/crossval/_18_wr90_iris_results/aperture_resolution.json")
_OFFSET_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


@pytest.fixture(scope="module")
def aperture_resolution() -> dict:
    with open(_APERTURE_RES) as f:
        return json.load(f)


def _oracle_vec(d_m, freqs):
    return np.array([_iris_s11(A, d_m, T, f) for f in freqs])


def test_aperture_resolution_artifact_is_rederived_from_committed_traces(
        fixture, aperture_resolution):
    """Every emitted number, recomputed from the committed rows.

    Tolerance 2e-3 abs is this file's standing oracle-agreement budget (the
    same one the gated-row test uses against ``oracle_s11``); the artifact
    rounds to 1e-4, so a real regeneration drift would still show.
    """
    art = aperture_resolution
    freqs = fixture["config"]["freqs_hz"]
    gates = _script_per_config_gates()
    rich_gate = fixture["gates"]["richardson_gate_abs"]
    assert art["schema"] == "rfx.wr90_iris_aperture_resolution"
    assert art["runs_fdtd"] is False
    assert tuple(art["offset_grid_fine_cells"]) == _OFFSET_GRID
    assert len(art["pairs"]) == len(fixture["gated_fine"]) == 8
    # pair order is the fixture's row order, so pairs[i] is a stable citation
    assert [p["config"] for p in art["pairs"]] == [
        _cfg_key(r["d_mm"], r["glen_m"], r["iris_frac"])
        for r in fixture["gated_fine"]]
    assert art["pairs"][2]["config"] == "7.620|0.20|0.50"

    for p, fr in zip(art["pairs"], fixture["gated_fine"]):
        cr = next(c for c in fixture["coarse_diagnostic"]
                  if (c["d_mm"], c["glen_m"], c["iris_frac"])
                  == (fr["d_mm"], fr["glen_m"], fr["iris_frac"]))
        d = fr["d_mm"] * 1e-3
        s_f = np.asarray(fr["s11"], dtype=float)
        assert p["fine_gate_abs"] == gates[p["config"]]
        assert p["committed_fine_gap_abs"] == fr["max_gap_abs"]

        # (i) distance from the trace AS MEASURED to a shifted oracle
        for g in _OFFSET_GRID:
            want = float(np.max(np.abs(s_f - _oracle_vec(d + g * _DX_FINE, freqs))))
            assert p["oracle_distance_abs"][f"{g:+.1f}"] == pytest.approx(
                want, abs=2e-3), (p["config"], g)
        nearest = min(_OFFSET_GRID,
                      key=lambda g: p["oracle_distance_abs"][f"{g:+.1f}"])
        assert p["nearest_offset_fine_cells"] == nearest

        # (ii) the injected-defect metric — a DIFFERENT quantity
        for sign, name in ((+1, "over"), (-1, "under")):
            gap, rich = _one_cell_defect(fr, cr, sign, freqs)
            rec = p["one_cell_defect"][name]
            assert rec["fine_gap_abs"] == pytest.approx(gap, abs=2e-3), p["config"]
            assert rec["richardson_dev_abs"] == pytest.approx(rich, abs=2e-3)
            assert rec["detected_by_fine_gate"] is bool(gap > p["fine_gate_abs"])
            assert rec["detected_by_richardson_gate"] is bool(rich > rich_gate)
            assert rec["fine_margin_x"] == pytest.approx(
                gap / p["fine_gate_abs"], abs=0.2)
            assert rec["scores_better_than_undefected"] is bool(
                gap < fr["max_gap_abs"])

    s = art["summary"]
    over = [p["one_cell_defect"]["over"] for p in art["pairs"]]
    under = [p["one_cell_defect"]["under"] for p in art["pairs"]]
    assert s["n_pairs"] == 8
    assert s["over_aperture_detected"] == sum(
        d["detected_by_fine_gate"] for d in over) == 8
    assert s["under_aperture_detected"] == sum(
        d["detected_by_fine_gate"] for d in under) == 2
    assert s["over_aperture_min_margin_x"] == pytest.approx(
        min(d["fine_margin_x"] for d in over), abs=1e-9)
    assert s["under_aperture_max_margin_x"] == pytest.approx(
        max(d["fine_margin_x"] for d in under), abs=1e-9)
    assert s["under_aperture_max_margin_x"] < 1.5      # the honest limit
    assert s["over_aperture_min_margin_x"] >= 1.5      # the repo's own margin
    assert s["richardson_detected_either_sign"] == 0
    assert s["under_aperture_detected_configs"] == [
        p["config"] for p in art["pairs"]
        if p["one_cell_defect"]["under"]["detected_by_fine_gate"]]
    assert s["under_aperture_scores_better_configs"] == [
        p["config"] for p in art["pairs"]
        if p["one_cell_defect"]["under"]["scores_better_than_undefected"]]


def test_the_round1_narrow_oracle_claim_is_refuted_at_every_configuration(
        aperture_resolution):
    """The retracted claim, stated as its own falsifier.

    Round 1 asserted the committed fine trace is CLOSER to the oracle one fine
    cell NARROW than to the oracle at the declared d.  It is farther at all
    eight configurations, and the nearest oracle on the declared offset grid is
    WIDER (+0.5 fine cells) at all eight — which is exactly why injecting a
    one-cell UNDER-aperture cancels rather than adds, and why the two d = 7.620
    configurations score BETTER defective than nominal.
    """
    pairs = aperture_resolution["pairs"]
    for p in pairs:
        dist = p["oracle_distance_abs"]
        assert dist["-1.0"] > dist["+0.0"], p["config"]     # farther, not closer
        assert p["nearest_offset_fine_cells"] > 0, p["config"]
    assert aperture_resolution["summary"][
        "nearest_offset_fine_cells_values"] == [0.5]
    assert aperture_resolution["summary"][
        "nearest_offset_is_positive_at_all_pairs"] is True
    # the sole place the trace IS closer to a shifted oracle than to its own
    # is on the WIDE side, at the strong aperture
    closer_than_nominal_wide = [p["config"] for p in pairs
                                if p["oracle_distance_abs"]["+1.0"]
                                < p["oracle_distance_abs"]["+0.0"]]
    assert closer_than_nominal_wide == ["7.620|0.20|0.50", "7.620|0.20|0.42"]
    assert (aperture_resolution["summary"]["under_aperture_scores_better_configs"]
            == closer_than_nominal_wide)


def test_claim_scope_cites_the_artifact_and_not_the_retracted_sentence(fixture):
    """The prose must POINT at the artifact, not restate this class of digit."""
    scope = " ".join(fixture["claim_scope"].split())
    assert "aperture_resolution.json" in scope
    assert "summary.under_aperture_scores_better_configs" in scope
    assert "summary.nearest_offset_fine_cells_values" in scope
    assert "CORRECTION (issue #812 round 2)" in scope
    # the withdrawn assertion, in every form it was written
    assert "CLOSER to the oracle at d minus one fine cell" not in scope
    assert "-0.6 to -1 cell of effective aperture" not in scope


def test_live_one_cell_defect_is_caught_by_the_per_config_gate_and_not_the_old_ones():
    """#812 round 2, VESSL 369367257708: the audit's defect (upper fin one
    cell short at each rung at d = 7.620 mm, i.e. an aperture one cell too
    WIDE) solved for real. It PASSES the pre-#812 pooled 0.04 gate and the
    Richardson 0.01 gate -- the measured blindness -- and FAILS the new
    per-configuration 0.015 gate. Pinned so the committed live artifact
    cannot drift from what the manifest cites; the first-order model row it
    sits beside is aperture_resolution.json::pairs[2].one_cell_defect.over."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "validation/crossval/_18_wr90_iris_results"
    live = json.loads((root / "one_cell_defect_live.json").read_text())
    model = json.loads((root / "aperture_resolution.json").read_text())["pairs"][2]
    assert live["config"]["config_key"] == model["config"] == "7.620|0.20|0.50"
    assert live["config"]["fin_cells_delta"] == -1
    m = live["measured"]
    assert m["fine_gap_abs"] == pytest.approx(0.02842, abs=5e-6)
    assert m["richardson_dev_abs"] == pytest.approx(0.00588, abs=5e-6)
    assert m["fine_gap_abs"] <= live["config"]["pooled_fine_gate_abs"] == 0.04
    assert m["fine_gap_abs"] > live["config"]["fine_gate_abs_per_config"] == 0.015
    assert m["richardson_dev_abs"] <= live["config"]["richardson_gate_abs"] == 0.01
    assert m["passes_pooled_fine_gate"] and m["fails_per_config_fine_gate"] and m["passes_richardson_gate"]
    assert m["per_config_margin_x"] == pytest.approx(m["fine_gap_abs"] / 0.015, abs=1e-3)
    # the first-order model predicted the same verdicts, 7 % low on the gated leg
    assert model["one_cell_defect"]["over"]["fine_gap_abs"] == pytest.approx(0.0265, abs=1e-4)
    assert model["one_cell_defect"]["over"]["detected_by_fine_gate"] is True
    assert model["one_cell_defect"]["over"]["detected_by_richardson_gate"] is False
    assert abs(model["one_cell_defect"]["over"]["fine_gap_abs"] / m["fine_gap_abs"] - 1.0) < 0.10
