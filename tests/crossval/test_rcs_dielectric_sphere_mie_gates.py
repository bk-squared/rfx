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


# --------------------------------------------------------------------------- #
# issue #812 re-gate — the permittivity gets a channel of its own.
#
# The audit of issue #812 measured this case's dB gate passing for a
# rasterized permittivity wrong by a factor. How wide that blind window is
# lives in validation/crossval/_17_dielectric_results/material_blind_window.json
# (no FDTD: committed gated_coarse deltas + the Mie oracle), is re-derived
# below, and reproduces the live defect runs' pass/fail verdict at every
# probed permittivity. That window is not a defect of the threshold — it is
# the sensitivity of the observable:
# d(sigma_dB)/d(eps/eps) is 9.816/10.202/9.978/7.134 dB per unit RELATIVE
# permittivity at ka = 0.50/0.75/1.00/1.25, so 6.3 dB IS a factor-wide window
# and the gate is already round-up(envelope x 1.5), i.e. as tight as the repo
# rule permits. The fix is a second channel, not a smaller number.
#
# Pre-declared with its derivation in
# docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md
# section 1.4, in a commit preceding the measurement that judges it.
# --------------------------------------------------------------------------- #

_SCRIPT_17 = _REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"


def _load_script_module():
    """Import the crossval script by path (its name is not an identifier).

    Imported lazily, inside the tests that need it, so this frozen-fixture
    lane keeps costing nothing at collection time.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cv17_script", _SCRIPT_17)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_material_gate_constants_are_pinned_and_bound(fixture):
    """D2 pattern: bind the constants CI actually enforces, and hard-pin the
    derived value so widening needs this line edited with a root cause."""
    src = _SCRIPT_17.read_text(encoding="utf-8")
    m = re.search(r"^EPS_REALIZED_TOL = ([0-9.]+)", src, re.MULTILINE)
    n = re.search(r"^N_DISTINCT_EPS_EXPECTED = ([0-9]+)", src, re.MULTILINE)
    assert m and n, "material gate constants not found in script source"
    # HARD pin: 0.05 dB (half the gate's 0.1 dB quantum) / 10.202 dB per unit
    # relative eps (the worst gated-ka Mie sensitivity) = 0.0049 -> 0.005.
    assert float(m.group(1)) == 0.005
    assert int(n.group(1)) == 2
    assert fixture["gates"]["eps_realized_tol"] == float(m.group(1))
    assert fixture["gates"]["n_distinct_eps_expected"] == int(n.group(1))
    # the gate must actually be spent in the gated loop, not merely defined
    assert "material_gate_ok(" in src
    assert "MATERIAL FAIL" in src


def test_the_declared_permittivity_sensitivity_is_the_one_recorded(fixture):
    """The window in claim_scope is a physical statement; re-derive it here
    from the independent Mie leg rather than trusting the prose."""
    scope = " ".join(fixture["claim_scope"].split())
    expected = {0.5: 9.816, 0.75: 10.202, 1.0: 9.978, 1.25: 7.134}
    h = 1e-3
    for ka, want in expected.items():
        base = _mie_backscatter_over_pi_a2(M_IDX, ka)
        up = _mie_backscatter_over_pi_a2(float(np.sqrt(2.56 * (1 + h))), ka)
        slope = 10 * np.log10(up / base) / h
        assert slope == pytest.approx(want, abs=5e-3), (ka, slope)
    assert "9.816/10.202/9.978/7.134 dB per unit RELATIVE permittivity" in scope
    # and the sensitivity-derived tolerance follows from the worst of them
    assert fixture["gates"]["eps_realized_tol"] == pytest.approx(
        math.ceil(0.05 / max(expected.values()) * 1000) / 1000, abs=1e-12)


def test_material_gate_rejects_the_permittivity_the_db_gate_cannot_see():
    """Criterion (B). The material actually rasterized is read back out of the
    array and judged; a permittivity at the measured edge of the dB gate's
    blind window must fail, and must fail for the RIGHT reason."""
    mod = _load_script_module()
    # positive control: what the binary rasterize path really delivers
    ok_arr = np.where(np.arange(64).reshape(4, 4, 4) < 20,
                      np.float32(2.56), np.float32(1.0))
    stats = mod.check_realized_material(ok_arr)
    assert stats["n_distinct_eps"] == 2
    assert stats["eps_rel_dev"] < 1e-6, stats      # float32 round-trip only
    assert mod.material_gate_ok(stats)

    # (B) the defect: a run whose rasterized permittivity sits at the edge of
    # the window the 6.3 dB gate tolerates (summary.blind_window_bracket_eps
    # in material_blind_window.json -- both edges PASS that gate).
    for bad_eps in (5.5, 2.0):
        bad = np.where(np.arange(64).reshape(4, 4, 4) < 20,
                       np.float32(bad_eps), np.float32(1.0))
        st = mod.check_realized_material(bad)
        assert st["n_distinct_eps"] == 2                  # structurally fine
        assert st["eps_realized"] == pytest.approx(bad_eps, rel=1e-6)
        assert st["eps_rel_dev"] == pytest.approx(
            abs(bad_eps / 2.56 - 1), rel=1e-6)            # the RIGHT reason
        assert st["eps_rel_dev"] > mod.EPS_REALIZED_TOL
        assert not mod.material_gate_ok(st)

    # (B) the other half of the headline claim: sub-cell interface averaging
    # would leave a third value in the array. Nothing checked this before.
    smoothed = np.where(np.arange(64).reshape(4, 4, 4) < 20,
                        np.float32(2.56), np.float32(1.0))
    smoothed[0, 0, 3] = np.float32(1.78)
    st = mod.check_realized_material(smoothed)
    assert st["n_distinct_eps"] == 3
    assert st["eps_rel_dev"] <= mod.EPS_REALIZED_TOL      # G17-A alone is blind
    assert not mod.material_gate_ok(st)                   # G17-B catches it


def test_realized_material_is_recorded_and_will_be_gated_on_the_frozen_leg(fixture):
    """Forward-guard for the frozen leg.

    The committed fixture predates #812, so no row carries the realized
    permittivity yet and this leg is protected only indirectly (the script's
    live gate exits 1, so a fixture regenerated at the wrong material cannot
    be produced from a green run, and ``config.eps_r`` is pinned at 2.56).
    That indirection is written down here rather than left implicit, and the
    moment a regeneration carries the fields they become gated -- all of them,
    never a silent subset.
    """
    src = (_REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"
           ).read_text(encoding="utf-8")
    assert '"eps_realized": round(material["eps_realized"], 9)' in src
    assert '"n_distinct_eps": material["n_distinct_eps"]' in src
    rows = _all_rows(fixture)
    carrying = [r for r in rows if "eps_realized" in r]
    if carrying:
        assert len(carrying) == len(rows), (
            "a regeneration recorded the realized permittivity on some rows "
            "but not others -- a partial record cannot be gated")
        tol = fixture["gates"]["eps_realized_tol"]
        for r in carrying:
            assert r["n_distinct_eps"] == fixture["gates"]["n_distinct_eps_expected"], r
            assert abs(r["eps_realized"] / 2.56 - 1.0) <= tol, r
    else:
        # pre-#812 record: the absence is a dated fact, not a silent gap.
        assert fixture["schema_version"] == 1
        assert fixture["config"]["eps_r"] == 2.56


# --------------------------------------------------------------------------- #
# issue #812 ROUND 2 — numeric provenance for the blind-window claim.
#
# Round 1 wrote the width of the dB gate's permittivity blind window as four
# FDTD-measured digits that no committed artifact carried. The window is now
# re-derived with no FDTD from the committed gated_coarse deltas plus this
# file's INDEPENDENT Mie leg, emitted by
# scripts/diagnostics/build_cv17_material_blind_window.py, and checked here.
# --------------------------------------------------------------------------- #

_BLIND_WINDOW = (_REPO_ROOT
                 / "validation/crossval/_17_dielectric_results/material_blind_window.json")


@pytest.fixture(scope="module")
def blind_window() -> dict:
    with open(_BLIND_WINDOW) as f:
        return json.load(f)


def test_material_blind_window_is_rederived_from_the_committed_deltas(
        fixture, blind_window):
    """Every emitted number, recomputed against the independent Mie leg.

    The model holds the solver's discretization error fixed and moves only the
    material, with the ORACLE at the declared 2.56 -- which is what the dB gate
    compares against. Nothing here runs FDTD.
    """
    art = blind_window
    rows = fixture["gated_coarse"]
    gate = fixture["gates"]["coarse_gate_db"]
    assert art["schema"] == "rfx.rcs_mie_material_blind_window"
    assert art["runs_fdtd"] is False
    assert art["config"]["declared_eps_r"] == fixture["config"]["eps_r"] == 2.56
    assert art["config"]["coarse_gate_db"] == gate
    assert art["config"]["coarse_ka"] == [r["ka"] for r in rows]
    assert [s["eps_r"] for s in art["scan"]] == art["eps_grid"]

    def mie_db(eps, ka):
        return 10 * math.log10(
            _mie_backscatter_over_pi_a2(float(np.sqrt(eps)), ka))

    for s in art["scan"]:
        want = [abs(r["delta_db"] + mie_db(s["eps_r"], r["ka"]) - mie_db(2.56, r["ka"]))
                for r in rows]
        assert s["per_bin_abs_delta_db"] == pytest.approx(want, abs=2e-3), s["eps_r"]
        assert s["max_abs_delta_db"] == pytest.approx(max(want), abs=2e-3)
        assert s["inside_db_gate"] is bool(max(want) <= gate)
        assert s["eps_rel_dev"] == pytest.approx(
            abs(s["eps_r"] / 2.56 - 1.0), abs=1e-6)

    summ = art["summary"]
    inside = [s["eps_r"] for s in art["scan"] if s["inside_db_gate"]]
    assert summ["blind_window_eps_grid_values"] == inside
    lo, hi = summ["blind_window_bracket_eps"]
    assert lo in inside and hi in inside
    # the bracket is contiguous on the declared grid and contains the declared eps
    grid = art["eps_grid"]
    assert all(art["scan"][i]["inside_db_gate"]
               for i in range(grid.index(lo), grid.index(hi) + 1))
    assert lo < 2.56 < hi
    assert summ["first_failing_eps_below"] == grid[grid.index(lo) - 1]
    assert summ["first_failing_eps_above"] == grid[grid.index(hi) + 1]
    assert not art["scan"][grid.index(lo) - 1]["inside_db_gate"]
    assert not art["scan"][grid.index(hi) + 1]["inside_db_gate"]


def test_blind_window_is_why_the_material_channel_exists(fixture, blind_window):
    """The re-scope, restated as an inequality instead of a paragraph.

    The dB gate tolerates a permittivity error two orders of magnitude wider
    than the material channel does, which is the whole argument for G17-A
    being a separate channel rather than a tighter dB threshold.
    """
    summ = blind_window["summary"]
    tol = fixture["gates"]["eps_realized_tol"]
    assert summ["material_gate_rel_tol"] == tol
    assert summ["blind_window_max_rel_dev"] == pytest.approx(
        max(abs(e / 2.56 - 1.0)
            for e in summ["blind_window_eps_grid_values"]), abs=1e-4)
    assert summ["blind_window_over_material_gate_x"] == pytest.approx(
        summ["blind_window_max_rel_dev"] / tol, abs=0.2)
    assert summ["blind_window_over_material_gate_x"] > 100
    scope = " ".join(fixture["claim_scope"].split())
    assert "material_blind_window.json" in scope
    assert "summary.blind_window_bracket_eps" in scope
    # the withdrawn prose-only digits are gone from the live claim
    for digits in ("6.07 dB", "5.50 dB", "7.70 dB", "8.55 dB"):
        assert digits not in scope


def test_island_probe_records_that_the_live_window_is_wider_than_the_model(fixture):
    """#812 round 2 (VESSL 369367257712): the first-order blind-window model
    predicts a FAIL island at eps 4.6-4.9; the live solver, with the
    rasterizer delivering eps and the oracle at the declared 2.56, PASSES the
    6.3 dB gate at 4.6 / 4.7 / 4.8 and fails only at 4.9. Pin the committed
    probe so the disagreement cannot be softened by regeneration, and
    re-derive each run's verdict from its own per-bin deltas."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "validation/crossval/_17_dielectric_results/cv17_permittivity_island.json"
    art = json.loads(path.read_text())
    gate = art["coarse_gate_db"]
    assert gate == fixture["gates"]["coarse_gate_db"]
    assert art["oracle_eps_r"] == 2.56 and art["ka_gated"] == [0.5, 0.75, 1.0, 1.25]
    by_eps = {}
    for r in art["runs"]:
        worst = max(abs(x) for x in r["per_bin_delta_db"])
        assert r["max_abs_delta_db"] == pytest.approx(worst, abs=1e-3)
        assert r["inside_db_gate"] is bool(worst <= gate)
        assert r["verdict_agrees_with_model"] is (r["inside_db_gate"] == r["model_inside_db_gate"])
        assert all(abs(e / r["eps_r"] - 1.0) < 1e-6 for e in r["eps_realized_per_bin"])
        by_eps[r["eps_r"]] = r
    assert sorted(by_eps) == [4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
    assert [e for e in sorted(by_eps) if not by_eps[e]["inside_db_gate"]] == [4.9]
    assert [e for e in sorted(by_eps) if by_eps[e]["model_inside_db_gate"] is False] == [4.6, 4.7, 4.8, 4.9]
    assert art["summary"]["n_verdicts_agree_with_model"] == 3
    assert art["summary"]["live_fail_eps"] == [4.9]
    # the island's worst live bin is the ka = 1.25 resonance bin, as the model said
    assert all(abs(by_eps[e]["per_bin_delta_db"][3]) == by_eps[e]["max_abs_delta_db"]
               for e in (4.7, 4.8, 4.9))
