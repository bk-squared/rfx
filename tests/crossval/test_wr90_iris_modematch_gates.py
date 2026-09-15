"""WR-90 single inductive iris vs mode-matching — frozen-fixture gates (item 3 S1).

Locks the committed record of
``validation/crossval/18_wr90_iris_modematch.py --write-fixture``
(``tests/fixtures/wr90_iris_modematch/fixture.json``) against an INDEPENDENT
in-test re-implementation of the TEn0 mode-matching cascade oracle (same
formulation class re-typed from the physics, sharing only numpy — a shared
producer bug in the overlap/junction algebra would still be caught by the
oracle's own unitarity/Marcuvitz witnesses, which this test re-runs).

Posture after #931 regeneration (retaining the #475/#476/#480 gate rules):
  * GATED: fine rung (dx=a/60, flux), per-configuration gates 0.006-0.016
    with pooled |S11 - oracle| <= 0.02 abs over 8
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


# --------------------------------------------------------------------------- #
# #931 lattice ownership — the migration gate for this case
# --------------------------------------------------------------------------- #
# cv18 and cv19 draw the SAME physical object, a WR-90 inductive iris, and
# carried different conventions for it: cv18 had ``aperture_cells = d_c - 1``
# with no correction on thickness, cv19 had ``t_c = round(t/dx) + 1`` and
# ``L_c = round(L/dx) - 1``. Both compensations existed because the pre-#931
# rule stood one wall per masked cell at its LOWER node plane and never zeroed
# the far face, so a fin's inner face was a wall and its outer face was not.
# Under the ownership contract both faces are walls (design note §1.2), the
# clear aperture is the gap between the two fins' inner walls, and the ``- 1``
# is gone. One rule now covers both cases -- which is the cross-case
# inconsistency the contract removes, and cv19's own gate file imports this
# file's oracle, so the two must move together.
#
# The fixture is regenerated by the crossval-D migration
# (``18_wr90_iris_modematch.py --write-fixture``, VESSL run named
# ``rfx-931-post-cv18``). Until it is ingested the committed rows still carry
# the compensated aperture, so the identity test below skips and says so.
_MIGRATION_RUN = ("VESSL rfx-931-post-cv18 — "
                  "validation/crossval/18_wr90_iris_modematch.py "
                  "--write-fixture on the migrated builder (crossval-D)")


def _aperture_cells(r) -> int:
    """Clear-aperture cell count under either fixture schema.

    schema_version 1 spelled it ``aperture_cells`` (an OPEN-node count, d_c - 1
    under the pre-#931 rule); the crossval-D regeneration (schema_version 2)
    renamed it ``realized_aperture_cells`` because its value changed -- a
    same-named key with a new meaning is read wrong exactly once.
    """
    if "realized_aperture_cells" in r:
        return int(r["realized_aperture_cells"])
    return int(r["aperture_cells"])


def _thickness_cells(r) -> int:
    if "realized_thickness_cells" in r:
        return int(r["realized_thickness_cells"])
    return int(r["thickness_cells"])


def _fixture_is_post_931(fixture) -> bool:
    """True when the aperture rows no longer carry the ``- 1`` compensation."""
    for r in _rows(fixture):
        d_c = round(r["d_mm"] / r["dx_mm"])
        if _aperture_cells(r) == d_c - 1:
            return False
    return True


def _require_post_931(fixture) -> None:
    if not _fixture_is_post_931(fixture):
        pytest.skip(
            "committed cv18 fixture still carries the pre-#931 aperture "
            f"compensation (aperture_cells = round(d/dx) - 1); {_MIGRATION_RUN}")


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
    # Hard pins, one per realization — root-cause to change either. #931
    # re-derived the pooled fine gate DOWN from the corrected-thickness
    # envelope (0.0232 -> 0.0106, ceil(0.0106 x 1.5) at quantum 100 = 0.02);
    # the Richardson gate came out unchanged (0.0051 -> 0.0046, still 0.01).
    assert g["fine_gate_abs"] == (0.02 if _fixture_is_post_931(fixture) else 0.04)
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
    post = _fixture_is_post_931(fixture)
    rip_fine = max(_residual_ripple_pp(r) for r in fixture["gated_fine"])
    rip_coarse = max(_residual_ripple_pp(r) for r in fixture["coarse_diagnostic"])
    want_fine, want_coarse = ("0.0076", "0.0152") if post else ("0.0077", "0.0158")
    assert f"{rip_fine:.4f}" == want_fine, rip_fine
    assert f"{rip_coarse:.4f}" == want_coarse, rip_coarse
    assert f"fine <= {want_fine}, coarse <= {want_coarse}" in scope
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
    want_rawflux = 0.0068 if post else 0.033
    assert max(diffs) == pytest.approx(want_rawflux, abs=1e-3)
    assert f"up to {want_rawflux:g}" in scope
    # coarse-rung range and the frequency count
    coarse_gaps = [r["max_gap_abs"] for r in fixture["coarse_diagnostic"]]
    want_lo, want_hi = ("0.008", "0.025") if post else ("0.018", "0.043")
    assert f"{min(coarse_gaps):.3f}" == want_lo and f"{max(coarse_gaps):.3f}" == want_hi
    assert f"{want_lo}-{want_hi} abs" in scope
    assert len(freqs) == 29 and "29 frequency" in scope


def test_case_docstring_quotes_fixture_derived_numbers(fixture):
    """The script's visible summary must track the regenerated fixture too."""
    doc = " ".join(ast.get_docstring(ast.parse(_SCRIPT.read_text())).split())
    gates = fixture["gates"]
    assert (f"envelope {gates['fine_measured_envelope_abs']:.4f} -> "
            f"pooled gate {gates['fine_gate_abs']:.2f}") in doc
    per_config = gates["fine_gate_abs_per_config"].values()
    assert f"({min(per_config):.3f}-{max(per_config):.3f})" in doc
    assert (f"{gates['richardson_measured_envelope_abs']:.4f} -> "
            f"gate {gates['richardson_gate_abs']:.2f}") in doc
    ratios = gates["first_order_ratios"]
    assert f"gap ratios {min(ratios):.3f}-{max(ratios):.3f}" in doc
    coarse = fixture["coarse_diagnostic"]
    gaps = [r["max_gap_abs"] for r in coarse]
    assert f"{min(gaps):.3f}-{max(gaps):.3f} abs" in doc
    raw_gaps = [r["max_gap_abs"] for r in fixture["raw_extraction_record"]]
    assert f"gaps {min(raw_gaps):.3f}-{max(raw_gaps):.3f}" in doc
    ripple = [max(_residual_ripple_pp(r) for r in fixture[tier])
              for tier in ("gated_fine", "coarse_diagnostic")]
    assert f"fine <= {ripple[0]:.4f}, coarse <= {ripple[1]:.4f}" in doc
    differences = []
    for raw in fixture["raw_extraction_record"]:
        flux = next(r for r in coarse if all(r[k] == raw[k] for k in
                    ("d_mm", "glen_m", "iris_frac")))
        differences.append(max(abs(a - b) for a, b in zip(raw["s11"], flux["s11"])))
    assert f"up to {max(differences):.4f} at the wide aperture" in doc


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
    want_colpow = 1.0200 if _fixture_is_post_931(fixture) else 1.0207
    assert max(r["max_colpow"] for r in mw["rows"]) == pytest.approx(
        want_colpow, abs=1e-3)
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
    # #931: the corner recipe INVERTED (cell-centre sampling makes a
    # node-plane corner the well-defined one) and the oracle's thickness
    # input became correct for the first time. Both are load-bearing history
    # and must stay in the record.
    assert "#931" in scope and "cell CENTRES" in scope
    assert "thickness deficit" in scope


def test_one_cell_volume_witness_is_recorded_and_passing(fixture):
    """#931 design note section 5: a one-cell PEC volume stands TWO walls.

    The contract's own claim at t = 1 cell had no independent witness before
    this: the thin-limit anchor is a t -> 0 statement, and every assert in the
    case counted masked planes, which agree with the drawing by construction.
    Here the lattice-blind mode-matching oracle is run against rfx across
    t = 1..8 cells and every row's realized thickness must equal its drawn one.

    THE CRITERION CHANGED, and the change is recorded rather than swapped in.
    The witness first read "the t = 1 residual lies inside the range t = 2..8
    spans". VESSL 369367259159 measured the residual MONOTONE decreasing in t,
    and for a monotone family t = 1 is the extremum for every possible outcome
    -- a perfect 0.0000 fails it too -- so the criterion could not pass and
    said nothing in either direction. It is retired for vacuity, its verdict
    kept in the record, and replaced by an identification test with no tunable
    constant: the oracle at t-1, t and t+1 cells, argmin on t, every rung. At
    t = 1 the t-1 alternative IS the pre-#931 realization (one wall, a
    zero-thickness screen), so this is a direct discriminator between the two
    rules at the one place they disagree.
    """
    if "one_cell_volume_witness" not in fixture:
        pytest.skip("the one-cell volume witness is written by the cv18 "
                    f"regeneration; the committed fixture predates it ({_MIGRATION_RUN})")
    w = fixture["one_cell_volume_witness"]
    rows = w["rows"]
    assert [r["t_cells"] for r in rows] == [1, 2, 3, 4, 5, 6, 8]
    for r in rows:
        assert r["realized_thickness_cells"] == r["t_cells"], r
        lo, hi = r["iris_wall_nodes"]
        assert hi - lo == r["t_cells"], r
        assert r["t_mm"] == pytest.approx(
            r["t_cells"] * 22.86 / w["cells_per_a"], abs=1e-3), r
    one = next(r["max_gap_abs"] for r in rows if r["t_cells"] == 1)
    multi = [r["max_gap_abs"] for r in rows if r["t_cells"] >= 2]
    assert w["one_cell_gap_abs"] == one
    assert list(w["multi_cell_gap_range_abs"]) == [min(multi), max(multi)]
    if "identification" not in rows[0]:
        # the retired criterion, on a record that predates the replacement
        assert min(multi) <= one <= max(multi), (one, multi)
        assert w["passed"] is True
        return
    # the retired criterion is kept as evidence, not as a gate, and it is
    # recorded as having failed on a monotone residual
    assert w["monotone_range_criterion"]["status"].startswith("RETIRED")
    gaps = [r["max_gap_abs"] for r in rows]
    assert gaps == sorted(gaps, reverse=True), (
        "the residual is no longer monotone in t, so the retired criterion "
        "was not vacuous on this record and the retirement needs re-arguing",
        gaps)
    for r in rows:
        idn = r["identification"]
        assert idn["argmin_t_cells"] == r["t_cells"], (
            "a rung does not identify its own realized thickness", r["t_cells"], idn)
        assert idn["identified_own_thickness"] is True
        assert idn["gap_at_t"] < idn["gap_at_t_minus_1"], r["t_cells"]
        assert idn["gap_at_t"] < idn["gap_at_t_plus_1"], r["t_cells"]
        assert idn["margin_vs_runner_up_x"] > 1.0
    one_idn = next(r["identification"] for r in rows if r["t_cells"] == 1)
    # the #931 claim itself: at one cell the two-wall oracle beats the
    # one-wall (zero-thickness) alternative by a clear factor, not a hair
    assert w["one_cell_two_wall_vs_one_wall_x"] == pytest.approx(
        one_idn["gap_at_t_minus_1"] / one_idn["gap_at_t"], rel=1e-3)
    assert w["one_cell_two_wall_vs_one_wall_x"] > 3.0, w
    assert w["identified_every_thickness"] is True
    assert w["passed"] is True


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
        # #931 §1.2: both fins realize walls on both of their faces, so the
        # clear aperture is the gap between the two INNER walls and equals the
        # drawn gap; the iris stands walls at BOTH its faces, so the realized
        # thickness is the drawn cell count. The old ``d_c - 1`` counted one
        # fin face the pre-#931 rule never zeroed; it is deleted, not re-tuned.
        # The regenerated fixture (schema_version 2, crossval-D) renames the
        # keys rather than reusing them -- ``realized_aperture_cells`` /
        # ``realized_thickness_cells`` -- because the aperture VALUE changed,
        # and adds the wall-node pairs; both spellings are read here.
        if _fixture_is_post_931(fixture):
            assert _aperture_cells(r) == d_c, r
        else:
            assert _aperture_cells(r) == d_c - 1, r
        # thickness never carried a compensation here (cv19's did) and does not
        # acquire one: drawn cells = realized cells on both cases now.
        t_mm = float(r.get("t_mm", fixture["config"]["t_m"] * 1e3))
        assert _thickness_cells(r) == round(t_mm / r["dx_mm"]), r
        if "iris_wall_nodes" in r:
            lo, hi = r["iris_wall_nodes"]
            assert hi - lo == _thickness_cells(r), r
        if "aperture_wall_nodes" in r:
            ylo, yhi = r["aperture_wall_nodes"]
            assert yhi - ylo == _aperture_cells(r), r
        assert len(r["s11"]) == len(cfg["freqs_hz"]) == 29


def test_the_two_wr90_iris_cases_share_one_thickness_convention(fixture):
    """cv18 and cv19 must not describe one object with two conventions.

    Before #931 the same WR-90 inductive iris was ``round(t/dx)`` cells thick
    in this case and ``round(t/dx) + 1`` in cv19, and cv19's own fixture
    recorded that its electrical thickness matched NEITHER rule (four FDTD runs
    measured ``(t_c - 0.68)*dx``). Under the contract the electrical thickness
    is ``t_c*dx`` in both. This is the cross-case check; cv19's gate file
    imports this file's oracle, so a divergence here would silently reduce
    that N=1 inheritance to a comparison of two different structures.
    """
    _require_post_931(fixture)
    for r in _rows(fixture):
        assert _thickness_cells(r) == round(
            fixture["config"]["t_m"] * 1e3 / r["dx_mm"]), r


def test_the_one_cell_iris_evidence_lives_in_the_witness_block(fixture):
    """Design note §5: one independent witness that the two-wall rule is right
    at ONE cell — and a statement of WHERE that witness is, so the duty cannot
    be silently unmet.

    This test used to look for a ``t_c = 1`` row among the GATED rows and skip
    when it found none, saying the cv18 regeneration would add one. The
    regeneration has landed and did not, because it cannot: the gated
    population is the three declared apertures at a/30 and a/60, whose iris is
    2 or 4 cells thick by construction. The t = 1 evidence is a separate
    coarse-rung sweep, recorded under ``one_cell_volume_witness``, and the
    coarse rung is REPORTED and never gated in this case — so comparing that
    row to ``gates.fine_gate_abs`` was a category error as well as unreachable.

    Its original criterion ("the t = 1 residual sits on the same curve as the
    t = 2..8 rows") is the monotone-range criterion, retired for vacuity by
    VESSL 369367259159 and kept in the record as
    ``one_cell_volume_witness.monotone_range_criterion``. The live criterion is
    thickness identification, asserted in full by
    ``test_one_cell_volume_witness_is_recorded_and_passing``. What is asserted
    here, and nowhere else, is that the two facts stay joined: no gated row
    carries the thickness in dispute, and the block that does carry it exists
    and passed.
    """
    gated_t1 = [r for r in _rows(fixture) if _thickness_cells(r) == 1]
    assert gated_t1 == [], (
        "a gated row now realizes a one-cell iris; the witness block is no "
        "longer the only place the contested thickness is measured, and this "
        "test must gate that row instead of deferring to the witness", gated_t1)
    if "one_cell_volume_witness" not in fixture:
        pytest.skip("the one-cell volume witness is written by the cv18 "
                    f"regeneration; the committed fixture predates it ({_MIGRATION_RUN})")
    w = fixture["one_cell_volume_witness"]
    one = [r for r in w["rows"] if r["realized_thickness_cells"] == 1]
    assert len(one) == 1, w["rows"]
    assert w["one_cell_gap_abs"] == one[0]["max_gap_abs"]
    assert w["passed"] is True, (
        "design note §5 leaves the two-wall rule at one cell unwitnessed", w)


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
    post = _fixture_is_post_931(fixture)
    # The two d = 7.620 rows carry the smallest one-cell sensitivity, so they
    # are the ones the pooled ceiling over-served most; they are still the
    # tightest after #931 (0.006 against a 0.02 pooled ceiling). The count of
    # rows at or under half the ceiling is 2 post-#931 and was 3 before, which
    # is arithmetic on a ceiling that itself halved, not a loosening.
    assert sum(1 for v in script_gates.values() if v <= pooled / 2) == (2 if post else 3)
    want_weak = 0.006 if post else 0.015
    assert script_gates[_cfg_key(7.62, 0.2, 0.5)] == want_weak
    assert script_gates[_cfg_key(7.62, 0.2, 0.42)] == want_weak


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
    post = _fixture_is_post_931(fixture)
    # The audit's numbers were measured on the pre-#931 geometry, whose iris
    # was one cell thinner than drawn; closing that shrinks every residual.
    # The SHAPE of the finding is what this test locks, and it survives: the
    # defect passes the pooled and Richardson gates and fails the per-config
    # one, by a wider margin than before.
    want = dict(own=0.0034, rich_own=0.0012, gap=0.0134, rich=0.0027) if post \
        else dict(own=0.0097, rich_own=0.0010, gap=0.0265, rich=0.0030)
    assert fr["max_gap_abs"] == pytest.approx(want["own"], abs=5e-4)
    assert cr["richardson_dev_abs"] == pytest.approx(want["rich_own"], abs=5e-4)
    gap, rich = _one_cell_defect(fr, cr, +1, freqs)
    assert gap == pytest.approx(want["gap"], abs=5e-4), gap
    assert rich == pytest.approx(want["rich"], abs=5e-4), rich
    # what the OLD gates did with them
    assert gap <= fixture["gates"]["fine_gate_abs"]          # 0.0265 <= 0.04
    assert rich <= fixture["gates"]["richardson_gate_abs"]   # 0.0030 <= 0.01
    # what the NEW gate does with them
    cfg_gate = _script_per_config_gates()[_cfg_key(7.62, 0.2, 0.5)]
    assert cfg_gate == (0.006 if post else 0.015)
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
    if _fixture_is_post_931(fixture):
        # BOTH signs resolved at every configuration once the thickness
        # deficit is closed. The pre-#931 asymmetry was the deficit reading
        # out on the aperture axis, not an aperture property.
        # The margin is a RATIO against a gate that #931 shrank 2-3x, so this
        # file's standing 2e-3 oracle-agreement budget buys a much wider band
        # on the ratio than it did before; the exact margins are pinned where
        # they are exact, against the artifact, in
        # test_aperture_resolution_artifact_is_rederived_from_committed_traces.
        # What is asserted here is the claim: detected everywhere, in both
        # signs, above the repo's own 1.5x margin.
        assert min(detected[+1]) == pytest.approx(1.62, abs=0.2), min(detected[+1])
        assert min(detected[+1]) >= 1.5, detected[+1]
        assert len(detected[-1]) == 8, detected
        assert min(detected[-1]) >= 1.5, detected[-1]
    else:
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
    post_art = "under_aperture_min_margin_x" in s
    assert s["n_pairs"] == 8
    assert s["over_aperture_detected"] == sum(
        d["detected_by_fine_gate"] for d in over) == 8
    assert s["under_aperture_detected"] == sum(
        d["detected_by_fine_gate"] for d in under) == (8 if post_art else 2)
    assert s["over_aperture_min_margin_x"] == pytest.approx(
        min(d["fine_margin_x"] for d in over), abs=1e-9)
    assert s["under_aperture_max_margin_x"] == pytest.approx(
        max(d["fine_margin_x"] for d in under), abs=1e-9)
    assert s["over_aperture_min_margin_x"] >= 1.5      # the repo's own margin
    if post_art:
        # with every configuration detecting, the binding number is the WORST
        # margin, so the artifact emits it and it too clears the repo margin
        assert s["under_aperture_min_margin_x"] == pytest.approx(
            min(d["fine_margin_x"] for d in under), abs=1e-9)
        assert s["under_aperture_min_margin_x"] >= 1.5
    else:
        assert s["under_aperture_max_margin_x"] < 1.5  # the honest limit
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
    summary = aperture_resolution["summary"]
    post_art = "under_aperture_min_margin_x" in summary
    pairs = aperture_resolution["pairs"]
    for p in pairs:
        dist = p["oracle_distance_abs"]
        # the retracted claim, refuted at every configuration under BOTH
        # realizations: the trace is FARTHER from the narrow oracle, not closer
        assert dist["-1.0"] > dist["+0.0"], p["config"]
    closer_than_nominal_wide = [p["config"] for p in pairs
                                if p["oracle_distance_abs"]["+1.0"]
                                < p["oracle_distance_abs"]["+0.0"]]
    if post_art:
        # #931: the half-cell WIDE bias is gone with the thickness deficit --
        # the nearest oracle is the declared aperture at every configuration,
        # so nothing scores better defective than nominal any more.
        for p in pairs:
            assert p["nearest_offset_fine_cells"] == 0.0, p["config"]
        assert summary["nearest_offset_fine_cells_values"] == [0.0]
        assert summary["nearest_offset_is_positive_at_all_pairs"] is False
        assert closer_than_nominal_wide == []
        assert summary["under_aperture_scores_better_configs"] == []
    else:
        for p in pairs:
            assert p["nearest_offset_fine_cells"] > 0, p["config"]
        assert summary["nearest_offset_fine_cells_values"] == [0.5]
        assert summary["nearest_offset_is_positive_at_all_pairs"] is True
        # the sole place the trace IS closer to a shifted oracle than to its
        # own is on the WIDE side, at the strong aperture
        assert closer_than_nominal_wide == ["7.620|0.20|0.50", "7.620|0.20|0.42"]
        assert (summary["under_aperture_scores_better_configs"]
                == closer_than_nominal_wide)


def test_claim_scope_cites_the_artifact_and_not_the_retracted_sentence(fixture):
    """The prose must POINT at the artifact, not restate this class of digit."""
    scope = " ".join(fixture["claim_scope"].split())
    assert "aperture_resolution.json" in scope
    assert "summary.under_aperture_scores_better_configs" in scope
    assert "summary.nearest_offset_fine_cells_values" in scope
    assert ("CORRECTION (issue #812 round 2)" in scope
            or "CORRECTION HISTORY (issue #812 round 2)" in scope)
    # the withdrawn assertion, in every form it was written
    assert "CLOSER to the oracle at d minus one fine cell" not in scope
    assert "-0.6 to -1 cell of effective aperture" not in scope


def test_live_one_cell_defect_is_caught_by_the_per_config_gate_and_not_the_old_ones():
    """#812 round 2, re-solved under the #931 contract: the audit's defect
    (upper fin one cell short at each rung at d = 7.620 mm, i.e. an aperture
    one cell too WIDE) measured for real. It PASSES the pre-#812 pooled gate
    and the Richardson gate -- the measured blindness -- and FAILS the
    per-configuration gate. Pinned so the committed live artifact cannot drift
    from what the manifest cites; the first-order model row it sits beside is
    aperture_resolution.json::pairs[2].one_cell_defect.over.

    #931 re-pin. Both artifacts this reads were regenerated on the migrated
    geometry and are the ones committed here: the probe run
    ``issue931-post-cv18-followups-20260907T230853Z`` (rc 0, source aa66bed2,
    "CRITERION (B) LIVE: CONFIRMED") wrote one_cell_defect_live.json, and the
    same run rebuilt aperture_resolution.json from the pass-2 record
    (``issue931-post-cv18-20260907T195222Z``, rc 0). Every digit below moved,
    and none of it was hand-entered:

      fine_gap_abs           0.02842 -> 0.01246   (measured, probe run)
      richardson_dev_abs     0.00588 -> 7e-05     (measured, probe run)
      pooled_fine_gate_abs   0.04    -> 0.02      = ceil(0.0106 x 1.5) @ 1/100
      fine_gate_abs_per_cfg  0.015   -> 0.006     = ceil(0.0034 x 1.5) @ 1/1000
      richardson_gate_abs    0.01    -> 0.01      = ceil(0.0046 x 1.5) @ 1/100
      model over.fine_gap    0.0265  -> 0.0134    (rebuilt artifact)

    No gate is loosened here: both fine gates moved DOWN, by the fixture's own
    round-UP(envelope x 1.5) rule against the corrected-thickness envelopes,
    and the defect is caught with more margin than before (1.895x -> 2.077x).
    """
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "validation/crossval/_18_wr90_iris_results"
    live = json.loads((root / "one_cell_defect_live.json").read_text())
    model = json.loads((root / "aperture_resolution.json").read_text())["pairs"][2]
    assert live["config"]["config_key"] == model["config"] == "7.620|0.20|0.50"
    assert live["config"]["fin_cells_delta"] == -1
    m = live["measured"]
    assert m["fine_gap_abs"] == pytest.approx(0.01246, abs=5e-6)
    assert m["richardson_dev_abs"] == pytest.approx(7e-05, abs=5e-6)
    assert m["fine_gap_abs"] <= live["config"]["pooled_fine_gate_abs"] == 0.02
    assert m["fine_gap_abs"] > live["config"]["fine_gate_abs_per_config"] == 0.006
    assert m["richardson_dev_abs"] <= live["config"]["richardson_gate_abs"] == 0.01
    assert m["passes_pooled_fine_gate"] and m["fails_per_config_fine_gate"] and m["passes_richardson_gate"]
    assert m["per_config_margin_x"] == pytest.approx(m["fine_gap_abs"] / 0.006, abs=1e-3)
    # the three gates the artifact carries are the committed ones, not a copy
    # that drifted: the probe reads them from the fixture and the script source
    gates = json.loads((Path(__file__).resolve().parents[1]
                        / "fixtures/wr90_iris_modematch/fixture.json").read_text())["gates"]
    assert live["config"]["pooled_fine_gate_abs"] == gates["fine_gate_abs"]
    assert live["config"]["richardson_gate_abs"] == gates["richardson_gate_abs"]
    assert (live["config"]["fine_gate_abs_per_config"]
            == gates["fine_gate_abs_per_config"]["7.620|0.20|0.50"])
    # the first-order model predicted the same verdicts, 7.5 % HIGH on the
    # gated leg (pre-#931 it was 7 % low; the sign flipped, the size did not)
    assert model["one_cell_defect"]["over"]["fine_gap_abs"] == pytest.approx(0.0134, abs=1e-4)
    assert model["one_cell_defect"]["over"]["detected_by_fine_gate"] is True
    assert model["one_cell_defect"]["over"]["detected_by_richardson_gate"] is False
    assert abs(model["one_cell_defect"]["over"]["fine_gap_abs"] / m["fine_gap_abs"] - 1.0) < 0.10
