"""cv06b -- public-carrier / docstring number correctness (issue #723).

Tracked carriers quoted `06b_msl_notch_filter_uniform.py`'s dx=80um run as
current MSL evidence: `validation/README.md`, `docs/guides/sparameter_
support_matrix.md` + `.json` (the support contract),
`docs/agent/port-selection.mdx`, and `docs/public/guide/benchmarks.mdx`.
Nothing coupled them to the script, so the #723 mesh change (dx 80um ->
63.5um = h_sub/4) would have left all of them stale with CI green -- that is
exactly how it was found, by hand, in review.

This test couples them. It locks:

  1. the script's own mesh convention (`DX == H_SUB / 4`) -- if the mesh
     moves again, this reds first;
  2. the realized board that convention buys, measured live on the real
     build (no time stepping): substrate exactly 254.0um, and the two
     numbers #931 separated -- the GEOMETRIC realized width of each sheet
     (571.5um, the node span the contract owns and `fidelity_report`
     prints) and the ELECTRICAL width the analytic reference takes
     (635.0um = 10 node rows * dx, unchanged by #931 because the strip's
     row count is unchanged). Both must be measured, not asserted from a
     formula.

     #931 history: under the pre-contract rule the 600um trace realized
     635.0um and `round(W_TRACE/DX)*DX = 571.5um` was demonstrably the WRONG
     answer, so this file asserted the inequality. Under the lattice
     ownership contract the trace is foil, i.e. a SHEET, and a sheet's
     footprint is the CLOSED node rectangle (design note §1.3): nodes 16..25
     at dx=63.5um, geometric width 571.5um -- which for this W/dx coincides
     with the old "naive" formula. The inequality is therefore deleted rather
     than inverted; what is pinned is that both numbers are READ off the
     build, and which one each consumer takes;
  3. the committed run log's headline numbers, parsed from the log rather
     than retyped;
  4. each carrier quotes the CURRENT numbers, and carries the superseded
     dx=80um ones only inside text that frames them as history.

Fail-closed: every regex assert requires a match, so a carrier rewording
that drops the number reds this test instead of passing quietly.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV06B = REPO_ROOT / "validation/crossval/06b_msl_notch_filter_uniform.py"
# The CURRENT committed run: the post-#931 re-solve (VESSL 369367259191,
# 2026-09-07), with the trace and stub declared as sheets.
RUN_LOG = (REPO_ROOT / "validation/crossval/_06b_notch_uniform_logs"
           / "20260907T124851Z_run.log")
# The pre-2.0 record of the SAME mesh: one-cell PEC Boxes, not sheets. The
# #723 anchor gate reads it for the leg that compares two pre-contract boards.
RUN_LOG_PRE931 = (REPO_ROOT / "validation/crossval/_06b_notch_uniform_logs"
                  / "20260827T131217Z_run.log")
RUN_LOG_DX80 = (REPO_ROOT / "validation/crossval/_06b_notch_uniform_logs"
                / "20260828T054132Z_dx80_origin_main_cdc38bc8_run.log")
README = REPO_ROOT / "validation/README.md"
MATRIX_MD = REPO_ROOT / "docs/guides/sparameter_support_matrix.md"
MATRIX_JSON = REPO_ROOT / "docs/guides/sparameter_support_matrix.json"
PORT_SELECTION = REPO_ROOT / "docs/agent/port-selection.mdx"
BENCHMARKS = REPO_ROOT / "docs/public/guide/benchmarks.mdx"

# The superseded dx=80um headline. Any carrier may mention these, but only
# in a sentence that marks them as the old mesh.
STALE_NUMBERS = ("1.63", "34.2", "57.9")


def _z0_median(log_text: str) -> float:
    """Parse ``Re(Z0) median = <x> Ω`` out of a committed cv06b run log."""
    m = re.search(r"Re\(Z0\) median\s*=\s*(-?[\d.]+)", log_text)
    assert m, "Re(Z0) median line missing from the run log"
    return float(m.group(1))


def _load_cv06b():
    spec = importlib.util.spec_from_file_location("_cv06b_carriers", CV06B)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv06b():
    return _load_cv06b()


@pytest.fixture(scope="module")
def log_text():
    return RUN_LOG.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# #931 lattice ownership — the migration gate for cv06b
# --------------------------------------------------------------------------- #
# cv06b draws its trace and stub as one-cell PEC Boxes from z = H_SUB to
# z = H_SUB + DX. Real copper on this board is ~35 um against a 63.5 um cell,
# i.e. foil, so under the contract they are SHEETS on the substrate-top node
# plane (design note §1.3) and the crossval-B migration declares them with
# add_thin_conductor. Measured here, on a stand-in build of the same board:
#
#   1-cell Box (V): z walls at 254.0 and 317.5 um  (the far face is NEW under
#                   the contract), realized y width 635.0 um
#   zero-thickness sheet (S): one wall at 254.0 um, realized y width 571.5 um
#
# So (S) keeps the electrical z-plane the board has today and moves the
# in-plane width; (V) keeps the width and turns the foil into a 63.5 um slab
# with a second wall. The width feeds Hammerstad-Jensen: 46.18 ohm at 635 um,
# 49.39 ohm at 571.5 um against a design-board 47.90 ohm. Every one of those
# numbers is downstream of the re-solve, so nothing here is re-typed.
_MIGRATION_RUN = ("VESSL rfx-931-post-cv06b — "
                  "validation/crossval/06b_msl_notch_filter_uniform.py "
                  "re-solved with the trace and stub declared as sheets "
                  "(crossval-B); the run log, W_realized, Z0 median and "
                  "F_NOTCH_AN all move with it")


def _is_sheet_declared(sim) -> bool:
    """True when cv06b's conductors are sheet declarations, not Boxes."""
    from tests._realized_pec import realize
    return len(realize(sim).sheets) >= 2


def test_mesh_convention_is_h_sub_over_four(cv06b):
    """dx = h_sub/4 is the whole point of #723 -- an aligned substrate."""
    assert cv06b.DX == pytest.approx(cv06b.H_SUB / 4.0, rel=1e-12)
    assert cv06b.DX == pytest.approx(63.5e-6, rel=1e-12)
    # h_sub/dx must be an exact integer: that is the alignment property, not
    # merely "a finer mesh".
    n = cv06b.H_SUB / cv06b.DX
    assert n == pytest.approx(round(n), abs=1e-9)


def test_realized_board_is_measured_not_assumed(cv06b):
    """Substrate 254.0um exactly; trace and stub both 571.5um GEOMETRIC,
    measured from the realized PEC EDGE set on the real build (no time
    stepping), and 635.0um ELECTRICAL from the script's own reader.

    The substrate is a dielectric and the contract does not touch dielectric
    sampling (design note §1.1), so 254.0um exact holds at every stage.

    #931: the metal is declared as two SHEETS (zero-thickness Boxes on the
    substrate-top node plane). A sheet's conductor is the set of edges
    BETWEEN its footprint nodes, so 10 node rows carry 9 edges = 571.5um.
    The pre-#931 neighbour rule zeroed one extra edge past the hi rim of
    each footprint and this case read 635.0um = 10 * dx -- a CELL count of
    a NODE mask. Both readings are recorded here so the change cannot be
    mistaken for a re-pin; ``_realized_trace_width`` must return what the
    realization actually did, and the analytic reference must be fed that.
    """
    sim = cv06b._build_sim()
    report = sim.fidelity_report(print_report=False)
    axes = {item["entity"]: {a["axis"]: a for a in item["axes"]}
            for item in report if "axes" in item}

    # Dielectric sampling is untouched by #931 (design note §1.8), so the
    # substrate row must be bit-identical to the pre-#931 one.
    sub_z = axes["geometry[0] 'ro4350b'"]["z"]
    assert sub_z["realized_extent_um"] == pytest.approx(254.0, abs=1e-6)
    assert max(sub_z["face_residual_um"]) == pytest.approx(0.0, abs=1e-9)

    # The metal rows are SHEET rows now: node bounds on the in-plane axes,
    # and a single plane on the normal axis.
    trace_y = axes["geometry[1] 'pec'"]["y"]
    stub_x = axes["geometry[2] 'pec'"]["x"]
    assert trace_y["realized_extent_um"] == pytest.approx(571.5, abs=1e-6)
    assert stub_x["realized_extent_um"] == pytest.approx(571.5, abs=1e-6)
    for entity in ("geometry[1] 'pec'", "geometry[2] 'pec'"):
        item = next(it for it in report if it["entity"] == entity)
        assert item["realized_plane"]["axis"] == "z"
        assert item["realized_plane"]["coordinate"] == pytest.approx(
            cv06b.H_SUB, rel=1e-12)

    # The ELECTRICAL width the analytic reference takes is n_rows * dx, a
    # different quantity from the geometric extent above and one cell
    # bigger. It is unchanged by #931 because the strip's row count is
    # unchanged. This tests the retained convention, not its physical
    # adjudication: the docstring records the pre-#931 46.48 ohm measurement
    # and the current 48.19 ohm result that falsified the width prediction.
    assert cv06b._realized_trace_width(sim) == pytest.approx(635.0e-6, rel=1e-12)

    # PRE-#931 this asserted the round() formula gives the WRONG answer for
    # the realized width. It gives the GEOMETRIC one, which the contract now
    # reports; it still does not give the electrical one.
    naive = round(cv06b.W_TRACE / cv06b.DX) * cv06b.DX
    assert naive == pytest.approx(571.5e-6, rel=1e-9)
    assert naive != pytest.approx(635.0e-6, rel=1e-6)


def test_trace_and_stub_are_foil_on_the_substrate_top_plane(cv06b):
    """#931 §1.3: 35 um copper against a 63.5 um cell is foil, i.e. a SHEET.

    Declared as a one-cell Box the same metal is a 63.5 um SLAB: the contract
    gives it a second wall at z = 317.5 um and shorts the normal edge through
    it, which changes the microstrip cross-section, Z0 and the notch
    frequency materially. So the declaration is what is checked — one wall at
    the substrate top, no cell owned, normal E live through the film — and
    the falsifier arm asserts there is no wall a cell above the laminate.
    """
    sim = cv06b._build_sim()
    if not _is_sheet_declared(sim):
        pytest.skip("cv06b still draws the trace and stub as one-cell PEC "
                    f"Boxes; {_MIGRATION_RUN}")
    from tests._realized_pec import (assert_no_wall_at, assert_normal_edge_live,
                                     assert_sheet_owns_no_cell,
                                     assert_sheet_planes, realize)
    realized = realize(sim)
    # two sheets (trace and stub), both on the substrate-top plane
    assert_sheet_planes(realized, 2, [cv06b.H_SUB, cv06b.H_SUB],
                        what="cv06b trace/stub")
    assert_sheet_owns_no_cell(realized, what="cv06b trace/stub")
    assert_normal_edge_live(realized, what="cv06b trace/stub")
    assert_no_wall_at(realized, 2, [cv06b.H_SUB + cv06b.DX],
                      what="cv06b board")


def test_shared_helper_agrees_with_the_case_gate(cv06b):
    """The case's own gate and the branch-wide helper must say the same
    thing. `tests/_realized_geometry` is the single shared spelling of the
    build-time check (#931); this case carries its own `realized_metal`
    because a crossval script cannot import from `tests/`, so the two are
    cross-checked here rather than left to drift."""
    from tests._realized_geometry import assert_sheet_planes, assert_wall_planes

    sim = cv06b._build_sim()
    # One tangential wall plane, at the substrate top, from the shared owner.
    assert_wall_planes(sim, 2, [cv06b.H_SUB], what="cv06b metal")
    # And it is there because two SHEETS were declared, not a volume.
    assert_sheet_planes(sim, 2, [cv06b.H_SUB, cv06b.H_SUB],
                        what="cv06b trace + stub")


def test_build_time_assertion_accepts_the_shipped_geometry(cv06b):
    """``assert_realized_metal`` is the #931 build gate: it must pass on the
    shipped declaration and return the measured sheet."""
    m = cv06b.assert_realized_metal(cv06b._build_sim())
    assert m["n_sheets"] == 2
    assert m["n_volume_cells"] == 0
    assert m["plane_z"] == pytest.approx(cv06b.H_SUB, rel=1e-12)
    assert m["trace_w"] == pytest.approx(m["stub_w"], abs=1e-12)
    assert m["n_rows"] == m["n_cols"] == 10
    assert m["trace_w"] == pytest.approx(571.5e-6, rel=1e-12)
    assert m["trace_w_elec"] == pytest.approx(635.0e-6, rel=1e-12)
    # Edge-to-edge the stub did not move; from the line CENTRE it lost one
    # cell, which is the quarter-wave length that sets the notch.
    assert m["stub_len"] == pytest.approx(12.0015e-3, abs=1e-9)
    assert m["stub_len_centreline"] == pytest.approx(12.28725e-3, abs=1e-9)


def test_z0_anchor_is_the_design_board_not_a_realized_one(cv06b):
    """#723 review, BLOCKING 1. What the fix buys is realized == declared on
    z, so the measured Z0 can be compared to the DESIGN's Hammerstad-Jensen.
    Evaluating HJ on each mesh's own realized board instead makes the
    dx=80um mesh look at least as good, so that framing must not be used to
    claim a port-accuracy improvement.

    #931, 2026-09-07: this gate RUNS on the shipped sheet-declared tree
    again. It had been skipped there on the premise that the realized trace
    width moves 635.0 -> 571.5 um and carries the HJ anchor with it. That
    premise names the wrong width. 571.5 um is the GEOMETRIC node span the
    contract reports; the analytic reference is fed the ELECTRICAL width
    ``_realized_trace_width`` returns, n_rows * dx = 635.0 um, which the
    contract does not move -- test_realized_board_is_measured_not_assumed
    pins both, and the pre-#931 and post-#931 run logs each print
    ``W_realized=635.0um`` in their header. What did move is the MEASURED
    median, 46.5 -> 48.2 ohm across the re-solve, so each leg below reads
    the log that belongs to the board it is comparing.
    """
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    z0_design, _ = hammerstad_jensen_z0_eps_eff(600e-6, 254e-6, cv06b.EPS_R)
    z0_realized_63, _ = hammerstad_jensen_z0_eps_eff(635e-6, 254e-6, cv06b.EPS_R)
    z0_realized_80, _ = hammerstad_jensen_z0_eps_eff(560e-6, 320e-6, cv06b.EPS_R)
    # The GEOMETRIC-width anchor, recorded because fidelity_report prints
    # that width and someone will reach for it. It is deliberately NOT used
    # in either leg: the analytic reference takes the electrical width.
    z0_geometric_63, _ = hammerstad_jensen_z0_eps_eff(571.5e-6, 254e-6,
                                                      cv06b.EPS_R)

    assert z0_design == pytest.approx(47.90, abs=0.02)
    assert z0_realized_63 == pytest.approx(46.18, abs=0.02)
    assert z0_realized_80 == pytest.approx(57.46, abs=0.02)
    assert z0_geometric_63 == pytest.approx(49.39, abs=0.02)

    # ---- Leg A: the #723 review's own comparison, on the two logs it was
    # made from. Both are the one-cell-PEC-Box realization, so the two sides
    # are the same kind of board and no post-contract number is spliced in.
    z0_meas_63_pre = _z0_median(RUN_LOG_PRE931.read_text(encoding="utf-8"))
    z0_meas_80 = _z0_median(RUN_LOG_DX80.read_text(encoding="utf-8"))
    assert z0_meas_63_pre == pytest.approx(46.5, abs=0.05)
    assert z0_meas_80 == pytest.approx(57.9, abs=0.05)
    dev_realized_63 = abs(z0_meas_63_pre - z0_realized_63) / z0_realized_63 * 100
    dev_realized_80 = abs(z0_meas_80 - z0_realized_80) / z0_realized_80 * 100
    assert dev_realized_63 == pytest.approx(0.69, abs=0.03)
    assert dev_realized_80 == pytest.approx(0.76, abs=0.03)
    # Both under 1% -- i.e. the realized-board comparison does not separate
    # the two meshes, which is why no "Nx bias reduction" may be claimed.
    assert dev_realized_63 < 1.0
    assert dev_realized_80 < 1.0
    assert dev_realized_80 / dev_realized_63 < 3.0

    # What IS true on that pair: on the design board the aligned mesh lands
    # within 3%.
    dev_design_63_pre = abs(z0_meas_63_pre - z0_design) / z0_design * 100
    assert dev_design_63_pre == pytest.approx(2.9, abs=0.15)

    # ---- Leg B: the board this tree actually ships, read from the current
    # committed post-#931 run. ONE mesh: there is no post-contract dx=80um
    # re-measurement, so nothing here compares meshes, and the leg-A verdict
    # above is not restated on this board.
    z0_meas_63 = _z0_median(RUN_LOG.read_text(encoding="utf-8"))
    assert z0_meas_63 == pytest.approx(48.2, abs=0.05)
    dev_realized_63_now = abs(z0_meas_63 - z0_realized_63) / z0_realized_63 * 100
    dev_design_63_now = abs(z0_meas_63 - z0_design) / z0_design * 100
    assert dev_realized_63_now == pytest.approx(4.37, abs=0.05)
    assert dev_design_63_now == pytest.approx(0.64, abs=0.05)
    # On the sheet board the measured median sits nearer the DESIGN board
    # than the realized-board anchor, so the framing #723 refused does not
    # favour this mesh under the contract either. Not an accuracy claim:
    # one mesh, one run, no coarse-mesh counterpart to compare it with.
    assert dev_design_63_now < dev_realized_63_now


def test_committed_log_reports_the_numbers_the_carriers_quote(log_text, cv06b):
    """Parse the log; do not retype it."""
    def grab(label: str) -> float:
        m = re.search(rf"{label}\s*=\s*(-?[\d.]+)", log_text)
        assert m, f"{label!r} missing from {RUN_LOG.name}"
        return float(m.group(1))

    assert grab("Notch frequency error") == pytest.approx(2.16, abs=0.005)
    assert grab(r"Notch depth \|S21\|") == pytest.approx(-39.4, abs=0.05)
    assert grab(r"Re\(Z0\) median") == pytest.approx(48.2, abs=0.05)
    # #931: the realized width is a reading, so read it out of the log and
    # check it against the build that produced it, instead of pinning the
    # literal 635.0 that only holds for the pre-contract realization.
    m = re.search(r"W_realized=([\d.]+)", log_text)
    assert m, "W_realized missing from the run log"
    logged_um = float(m.group(1))
    live_um = cv06b._realized_trace_width(cv06b._build_sim()) * 1e6
    assert logged_um == pytest.approx(live_um, abs=1e-6), (
        f"the committed run log records W_realized={logged_um} um but this "
        f"tree realizes {live_um} um — the log predates the geometry it "
        f"claims to describe ({_MIGRATION_RUN})")
    assert "mesh: dx=63.5µm, n_z_sub=4" in log_text
    assert "PASS: cv06b" in log_text


def test_committed_log_carries_the_warnings_the_contract_quotes(log_text):
    """R5: the headline is read inside a flagged band and after a passivity
    projection. Both must stay quotable from the same log."""
    assert "standing-wave null at the port plane: 9 bins in [3.7545, 7.0000] GHz" in log_text
    assert "63 of 100 frequency bins were non-passive as extracted" in log_text
    assert "worst sigma_max = 1.006" in log_text
    assert "'msl_0' = 33.02 ohm" in log_text
    assert "'msl_1' = 59.23 ohm" in log_text


@pytest.mark.parametrize(
    "carrier", [README, MATRIX_MD, MATRIX_JSON, PORT_SELECTION, BENCHMARKS])
def test_carrier_quotes_the_current_mesh(carrier):
    text = carrier.read_text(encoding="utf-8")
    assert re.search(r"63\.5\s*(µm|um)", text), (
        f"{carrier.relative_to(REPO_ROOT)} does not name the shipped "
        "dx=63.5um mesh")


@pytest.mark.parametrize("carrier", [MATRIX_MD, MATRIX_JSON, PORT_SELECTION])
def test_carrier_quotes_the_current_headline(carrier):
    """#931: the current headline is the post-contract re-solve's, and the
    pre-2.0 one may only appear framed as the realization 2.0 does not
    produce (checked by ``test_carrier_marks_the_pre931_headline_as_history``
    below)."""
    text = carrier.read_text(encoding="utf-8")
    for number in ("2.16", "39.4", "48.2"):
        assert number in text, (
            f"{carrier.relative_to(REPO_ROOT)} is missing the current cv06b "
            f"figure {number}")


@pytest.mark.parametrize(
    "carrier", [README, MATRIX_MD, MATRIX_JSON, PORT_SELECTION, BENCHMARKS])
def test_carrier_does_not_present_the_dx80_numbers_as_current(carrier):
    """The stale figures may appear, but only in a sentence that marks them
    as the superseded mesh."""
    text = carrier.read_text(encoding="utf-8")
    history_markers = ("history", "historical", "earlier", "superseded",
                       "through 2026-08", "was dx", "did not solve")
    for number in STALE_NUMBERS:
        for m in re.finditer(re.escape(number), text):
            window = text[max(0, m.start() - 700):m.end() + 700].lower()
            assert any(k in window for k in history_markers), (
                f"{carrier.relative_to(REPO_ROOT)} quotes the superseded "
                f"dx=80um figure {number} without marking it as history")


@pytest.mark.parametrize("carrier", [MATRIX_MD, MATRIX_JSON, PORT_SELECTION])
def test_carrier_marks_the_pre931_headline_as_history(carrier):
    """1.40 / -43.3 / 46.5 are the ONE-CELL-BOX realization's numbers. 2.0
    cannot produce them, so a carrier may keep them only inside a sentence
    that says so -- the same rule this file already applies to the dx=80um
    figures, one realization later."""
    text = carrier.read_text(encoding="utf-8")
    markers = ("pre-2.0", "PRE-2.0", "history", "historical", "one-cell PEC "
               "Box", "one-cell Box")
    for number in ("1.40", "43.3", "46.5"):
        for m in re.finditer(re.escape(number), text):
            window = text[max(0, m.start() - 700):m.end() + 700]
            assert any(k in window for k in markers), (
                f"{carrier.relative_to(REPO_ROOT)} quotes the pre-#931 cv06b "
                f"figure {number} without marking it as the superseded "
                "realization")


def test_e4_board_mismatch_is_disclosed_where_the_comparison_is_cited():
    """Post-#723 cv06b solves h_sub=254um and the E4/openEMS leg still solves
    300um. Every carrier that cites that comparison must say so."""
    for carrier in (README, MATRIX_MD, MATRIX_JSON, BENCHMARKS):
        text = carrier.read_text(encoding="utf-8")
        if "msl_notch_e4" in text or "dx=50 um" in text or "dx=50um" in text:
            assert "300" in text, (
                f"{carrier.relative_to(REPO_ROOT)} cites the dx=50um E4 "
                "comparison without disclosing its 300um board")
