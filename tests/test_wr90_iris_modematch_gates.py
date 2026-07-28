"""WR-90 single inductive iris vs mode-matching — frozen-fixture gates (item 3 S1).

Locks the committed record of
``validation/crossval/18_wr90_iris_modematch.py --write-fixture``
(``tests/fixtures/wr90_iris_modematch/fixture.json``) against an INDEPENDENT
in-test re-implementation of the TEn0 mode-matching cascade oracle (same
formulation class re-typed from the physics, sharing only numpy — a shared
producer bug in the overlap/junction algebra would still be caught by the
oracle's own unitarity/Marcuvitz witnesses, which this test re-runs).

Posture (all PR #475/#476 lessons applied from day one):
  * GATED: fine rung (dx=a/60, flux) |S11 - oracle| <= 0.22 abs over all
    committed fine rows (three apertures + the worst-aperture domain scan —
    the envelope is domain-scanned BEFORE gating because the canonical
    config alone understates it 0.110 -> 0.143); Richardson
    2*fine - coarse on the oracle <= 0.04 abs per aperture.
  * REPORTED: coarse rung, raw extraction, first-order ratios — committed
    data, asserted present and internally consistent, never gated.
  * FENCED: modal extraction on strong reflectors (measured colpow 1.112)
    and anything beyond one symmetric inductive iris — content-pinned.

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


def test_fixture_and_artifact_are_the_same_record(fixture):
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture


def test_script_prose_literals_match_fixture(fixture):
    """AST-binds claim_scope AND the modal-fence provenance entry (the prose
    analogue of the D2 constant binding; PR #476 pattern)."""
    mod = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    lits = {k.value: ast.literal_eval(v)
            for node in ast.walk(mod) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
            and k.value in ("claim_scope", "modal_fence_record_2026_07_28")}
    assert set(lits) == {"claim_scope", "modal_fence_record_2026_07_28"}
    assert " ".join(lits["claim_scope"].split()) == " ".join(fixture["claim_scope"].split())
    assert (" ".join(lits["modal_fence_record_2026_07_28"].split())
            == " ".join(fixture["provenance"]["modal_fence_record_2026_07_28"].split()))


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
    assert g["fine_gate_abs"] == 0.22       # hard pin — root-cause to change
    assert g["richardson_gate_abs"] == 0.06


def test_script_live_gate_constants_match_fixture(fixture):
    """D2: bind the constants CI actually enforces."""
    src = _SCRIPT.read_text(encoding="utf-8")
    m_f = re.search(r"^GATE_FINE_ABS = ([0-9.]+)", src, re.MULTILINE)
    m_r = re.search(r"^GATE_RICH_ABS = ([0-9.]+)", src, re.MULTILINE)
    assert m_f and m_r, "gate constants not found in script source"
    assert float(m_f.group(1)) == fixture["gates"]["fine_gate_abs"]
    assert float(m_r.group(1)) == fixture["gates"]["richardson_gate_abs"]


def test_gated_fine_rows_within_gate_against_independent_oracle(fixture):
    """Every committed fine row within the gate, oracle re-derived HERE."""
    gate = fixture["gates"]["fine_gate_abs"]
    freqs = fixture["config"]["freqs_hz"]
    rows = fixture["gated_fine"]
    assert len(rows) >= 6      # 3 apertures + >=3 domain-scan configs
    for r in rows:
        assert r["cells_per_a"] == fixture["config"]["fine_cells_per_a"]
        assert r["normalize"] == "flux"
        d = r["d_mm"] * 1e-3
        orc = [_iris_s11(A, d, T, f) for f in freqs]
        # recorded oracle leg vs independent re-implementation
        assert max(abs(a - b) for a, b in zip(orc, r["oracle_s11"])) < 1e-3, r["d_mm"]
        gap = max(abs(a - b) for a, b in zip(r["s11"], orc))
        assert gap <= gate + 1e-6, (r["d_mm"], r["glen_m"], r["iris_frac"], gap)
        assert abs(gap - r["max_gap_abs"]) < 2e-3
        assert r["max_colpow"] <= 1.02     # passivity-clean gated rows


def test_richardson_cross_confirms_oracle_and_first_order(fixture):
    gate = fixture["gates"]["richardson_gate_abs"]
    freqs = fixture["config"]["freqs_hz"]
    canon = (fixture["config"]["canonical_glen_m"],
             fixture["config"]["canonical_iris_frac"])
    for cr in fixture["coarse_diagnostic"]:
        assert cr["richardson_dev_abs"] <= gate + 1e-6, cr["d_mm"]
        fr = next(r for r in fixture["gated_fine"]
                  if r["d_mm"] == cr["d_mm"]
                  and (r["glen_m"], r["iris_frac"]) == canon)
        d = cr["d_mm"] * 1e-3
        orc = [_iris_s11(A, d, T, f) for f in freqs]
        rich = [2 * f_ - c_ for f_, c_ in zip(fr["s11"], cr["s11"])]
        dev = max(abs(a - b) for a, b in zip(rich, orc))
        assert abs(dev - cr["richardson_dev_abs"]) < 2e-3
    # first-order ratio witness committed and in the measured band
    ratios = fixture["gates"]["first_order_ratios"]
    assert len(ratios) >= 6
    assert all(0.30 <= x <= 0.75 for x in ratios), ratios


def test_fences_are_content_pinned(fixture):
    """Modal fence, slot-bug fence, F1 domain-scan rationale, iris scope."""
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "column power 1.112" in scope                 # modal fence measured
    assert "fenced" in scope and "modal" in scope
    assert "parasitic wall-slot bug" in scope            # slot-bug record
    assert "contiguous-aperture raster assert" in scope
    assert "understates it at 0.110" in scope            # F1 rationale w/ number
    assert "one symmetric inductive iris" in scope       # scope fence
    assert "experimental" in scope
    prov = " ".join(fixture["provenance"]["modal_fence_record_2026_07_28"].split()).lower()
    assert "1.112" in prov and "fenced" in prov
    assert "never gated" in fixture["gates"]["posture"]


def test_diagnostics_and_witnesses_are_recorded(fixture):
    assert len(fixture["coarse_diagnostic"]) == 3
    assert len(fixture["coarse_domain_scan"]) >= 3
    assert len(fixture["raw_extraction_record"]) == 3
    for r in fixture["raw_extraction_record"]:
        assert r["normalize"] == "False"
        assert r["max_colpow"] <= 1.02
    trunc = fixture["truncation_witness"]
    assert len(trunc) == 3
    assert all(t["shift_abs"] <= 0.01 for t in trunc)
    prov = " ".join(fixture["provenance"]["no_preflight_note"].split()).lower()
    assert "no sim.preflight()" in prov


def test_operating_point_is_grid_exact_on_every_row(fixture):
    """F2 class: floors/conventions on ALL row families."""
    cfg = fixture["config"]
    rows = (list(fixture["gated_fine"]) + list(fixture["coarse_diagnostic"])
            + list(fixture["coarse_domain_scan"])
            + list(fixture["raw_extraction_record"]))
    assert len(rows) >= 13
    for r in rows:
        cells = r["cells_per_a"]
        assert cells in (cfg["coarse_cells_per_a"], cfg["fine_cells_per_a"])
        assert r["dx_mm"] == pytest.approx(22.86 / cells, abs=1e-3)
        d_c = round(r["d_mm"] / r["dx_mm"])
        # deterministic footprint from the half-cell-offset corners
        assert r["aperture_cells"] == d_c + 1, r
        assert r["thickness_cells"] == round(1.524 / r["dx_mm"]), r
        assert len(r["s11"]) == len(cfg["freqs_hz"]) == 8
