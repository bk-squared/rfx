"""Header/record consistency checks for the openEMS coax two-port referee.

``validation/crossval/21_coax_two_port_referee.py`` (formerly
``validation/research/coax_two_port/openems_coax_two_port_referee.py``,
promoted 2026-08-04 -- PI decision, issue #489 comment
https://github.com/bk-squared/rfx/issues/489#issuecomment-5179893223) is a
COMPARATOR-leg-only, VESSL-only harness for rfx issue #489 stage 3 (see its
module docstring for the full scope fence): openEMS is not installed in
THIS test environment (this test does not need it -- it only loads the
module and inspects Python-level data). The referee itself HAS since run
on VESSL, three times (run-1 VESSL 369367251366, run-2 369367251627,
run-3 369367251629) -- ``REPRODUCE_GATE_RECORD`` reflects that (``status:
"RUN"``, run-3's own numbers and log path). This test checks that its
Stage B layout arithmetic is self-consistent (testable without openEMS --
see ``_stage_b_layout()``), and that the exit-code split between a
physics/self-check failure and an internal config bug is real (not just
documented).

Design (per the task's fail-loud-honest requirement, M1/M3 review fixes):
this test PASSES on EITHER the never-run placeholder state OR a
legitimately-filled one (run-3 fix, PR #548 review: a test must check the
record's CONTRACT shape, not cement one particular state -- UNRUN <=> no
numbers, no log path; RUN <=> numbers present AND a log path under
.omx/ or docs/research_notes/vessl_logs/ that actually exists on disk)
rather than pinning a specific finding as fact -- the record's own
content (which tutorial, which geometry, which run's numbers) is free to
evolve without this test needing to be rewritten each time.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
from types import ModuleType
from typing import Final

import numpy as np

REFEREE_DIR: Final = (
    pathlib.Path(__file__).resolve().parents[2]
    / "validation" / "crossval"
)
SCRIPT_PATH: Final = REFEREE_DIR / "21_coax_two_port_referee.py"
REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]


def _load_referee_module() -> ModuleType:
    """Load the referee script as a throwaway module (not sys.modules-registered).

    Must succeed WITHOUT openEMS installed -- the referee script defers its
    openEMS import into functions specifically so this works (see its
    ``_import_openems`` docstring).
    """
    assert SCRIPT_PATH.exists(), f"missing referee script {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("_coax_two_port_referee", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_openems():
    """The module itself must not require openEMS at import time."""
    module = _load_referee_module()
    assert hasattr(module, "REPRODUCE_GATE_RECORD")


def test_reproduce_gate_record_has_required_fields():
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "expected_zl_a_ohm", "gate", "status", "reproduced_zl_mean_ohm",
        "reproduced_zl_max_dev_ohm", "log_path", "vessl_run_id",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"


def test_tutorial_citation_is_verifiable():
    """The reproduce-gate must cite a REAL, checkable tutorial path -- not
    an assertion that no tutorial exists (M1 fix: the first version wrongly
    claimed no canonical openEMS coax tutorial exists; PR #540 review
    found matlab/examples/waveguide/Coax.m). This test checks the citation
    has the shape of something a reviewer could independently verify
    (repo + path + how it was checked + when), not that any PARTICULAR
    tutorial is named -- the record's own content should be free to
    evolve without this test needing a rewrite."""
    module = _load_referee_module()
    tutorial = module.REPRODUCE_GATE_RECORD["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".m") or tutorial["path"].endswith(".py"), (
        "tutorial citation should point at a real source file"
    )
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"


def test_do_not_repeat_cites_the_recorded_failure():
    """R1/R2 class: the rebuild must name the specific recorded failure it
    avoids repeating, not just assert a new approach in the abstract."""
    module = _load_referee_module()
    do_not_repeat = module.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "build_coaxial_line_openems_broad_comparison.py" in do_not_repeat
    assert "AddLumpedPort" in do_not_repeat


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path.

    This is the test that must go RED if someone later claims reproduced
    numbers without a log path pointing at the run that produced them --
    not a pinned number that rots after the first real VESSL run.

    TRACKED-PATH fix (PR #548 review, correcting PR #548's OWN first
    fill): a filled-in (``status != "UNRUN"``) record's ``log_path`` must
    live under a GIT-TRACKED prefix -- ``validation/crossval/
    _21_coax_two_port_referee_logs/`` (moved here on promotion,
    2026-08-04 -- formerly ``validation/research/coax_two_port/logs/``)
    -- not merely "exists on disk on the machine that ran the job". The
    original M3 fix's allowed
    prefixes, ``.omx/`` and ``docs/research_notes/vessl_logs/``, are
    BOTH gitignored in this repo -- accepting either one here let the
    VERY FIRST real fill (``status="RUN"``, ``log_path`` under ``.omx/``)
    go red in every clone/CI checkout except the one machine that
    happened to still have that VESSL job's local artifact
    (reviewer-reproduced: 23/24 in a fresh worktree). Gitignored
    prefixes are recognized below only as historical/documentation
    record of the pre-#548 design's intent -- REJECTED for an actually
    filled record, whose whole point is that "a reviewer outside this
    machine can open the log a claimed number came from". There is no
    parallel leniency for UNRUN: that branch requires ``log_path is
    None`` regardless, so "which prefix is OK" never arises for an
    unfilled placeholder -- the distinction that matters is FILLED (must
    be tracked) vs UNFILLED (no path at all, any prefix moot).
    """
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_zl_mean_ohm"] is None
        assert record["reproduced_zl_max_dev_ohm"] is None
        assert record["log_path"] is None
    else:
        assert record["reproduced_zl_mean_ohm"] is not None
        assert record["reproduced_zl_max_dev_ohm"] is not None
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"

        gitignored_prefixes = (".omx/", "docs/research_notes/vessl_logs/")
        tracked_prefixes = ("validation/crossval/_21_coax_two_port_referee_logs/",)
        assert not log_path_str.startswith(gitignored_prefixes), (
            f"log_path {log_path_str!r} lives under a GITIGNORED prefix -- "
            f"fine for the pre-#548 design's own history, but a FILLED "
            f"(status != 'UNRUN') record needs a log a reviewer OUTSIDE "
            f"this machine can open; use a tracked prefix instead: "
            f"{tracked_prefixes!r} (PR #548 review)"
        )
        assert log_path_str.startswith(tracked_prefixes), (
            f"log_path {log_path_str!r} must live under a TRACKED prefix "
            f"{tracked_prefixes!r} once status is RUN (PR #548 review)"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but its "
            f"log_path {log_path} does not exist -- a claimed number needs a "
            f"real log, per external_solver_comparator.md step 2"
        )


def test_mesh_refinement_predeclaration_has_required_fields():
    """N1 (review fix): MESH_REFINEMENT_PREDECLARATION claims REPRODUCE_
    GATE_RECORD's own fill-contract discipline -- give it the same
    required-fields coverage so its shape cannot silently rot."""
    module = _load_referee_module()
    record = module.MESH_REFINEMENT_PREDECLARATION
    required_fields = {
        "issue", "leg", "predeclared_on", "refinement_factor", "dx_scale",
        "dx_mm_before", "dx_mm_after", "annulus_cells_before",
        "annulus_cells_after", "measured_ratio_before", "excess_before",
        "predicted_excess_after", "predicted_ratio_after",
        "acceptance_band_ratio", "falsifier", "estimated_runtime_s",
        "estimated_runtime_h", "runtime_basis", "status",
        "measured_ratio_after", "vessl_run_id", "log_path", "verified_on",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"MESH_REFINEMENT_PREDECLARATION missing fields: {missing}"


def test_mesh_refinement_predeclaration_is_committed_unrun_and_self_consistent():
    """UNRUN <=> no numbers, no log path, no run id -- same fill-contract
    shape REPRODUCE_GATE_RECORD's own test enforces above."""
    module = _load_referee_module()
    record = module.MESH_REFINEMENT_PREDECLARATION
    if record["status"] == "UNRUN":
        assert record["measured_ratio_after"] is None
        assert record["vessl_run_id"] is None
        assert record["log_path"] is None
        assert record["verified_on"] is None
    else:
        assert record["measured_ratio_after"] is not None
        assert record["vessl_run_id"], "a filled record needs a vessl_run_id"
        assert record["log_path"], "a filled record needs a log_path"


def test_mesh_refinement_predeclaration_literal_pins():
    """N1 (review fix): pin the numbers a resubmission could otherwise
    reshape after the fact -- acceptance_band_ratio (post-B4: [lo, hi]
    with hi=1.11 the ONLY refuting edge) and predicted_ratio_after must
    not silently drift once a real measured_ratio_after arrives."""
    module = _load_referee_module()
    record = module.MESH_REFINEMENT_PREDECLARATION
    assert record["refinement_factor"] == 1.5
    assert abs(record["dx_scale"] - 2.0 / 3.0) < 1e-12
    assert record["acceptance_band_ratio"] == [1.05, 1.11]
    assert abs(record["excess_before"] - 0.1208) < 1e-9
    assert abs(record["predicted_ratio_after"] - (1.0 + 0.1208 / 1.5)) < 1e-9
    # N3 fix: the corrected (spatial-cell-ratio x timestep-ratio) runtime
    # estimate, not the wrong refinement_factor**4 figure (2796*1.5**4=
    # 14155s, 3.93h) -- pinned so a future edit cannot silently
    # reintroduce the R^4 error via estimated_runtime_s/_h.
    assert abs(record["estimated_runtime_h"] - 2.41) < 0.01
    assert abs(record["estimated_runtime_s"] - 2796 * 1.5 ** 4) > 1000, (
        "estimated_runtime_s matches the WRONG refinement_factor**4 figure"
    )
    # B4 fix: the falsifier text must name the one-sided direction
    # explicitly (not just "outside the band", which reads two-sided).
    assert "ABOVE 1.11" in record["falsifier"]
    assert "NOT a falsifier" in record["falsifier"] or "not a falsifier" in record["falsifier"].lower()


def test_mesh_refinement_gate_confirms_a_ratio_below_the_band():
    """B4 (BLOCKING, adversarial review 2026-08-04): the two-sided version
    of this check printed 'FALSIFIED, R2-STOP' for ANY ratio outside
    [1.05, 1.11] -- including a ratio like 1.03, which is a STRONGER
    confirmation of staircase-dispersion attribution (openEMS's own
    material averaging routinely exceeds the naive first-order O(dx) the
    predeclaration's excess-scaling estimate assumed), not a refutation.
    This is the falsifier for that misfire: a ratio well below
    acceptance_band_ratio[0] must NOT be reported as refuted."""
    module = _load_referee_module()
    evaluation = module._evaluate_mesh_refinement_ratio(
        1.03, module.MESH_REFINEMENT_PREDECLARATION
    )
    assert evaluation["refuted"] is False
    assert "CONFIRMED" in evaluation["verdict"]
    assert "REFUTED" not in evaluation["verdict"]


def test_mesh_refinement_gate_refutes_only_above_hi():
    """The ONLY refuting outcome: ratio strictly above acceptance_band_
    ratio[1] (1.11). At exactly hi, and anywhere below it, the gate must
    confirm, not refute (one-sided, per the B4 fix)."""
    module = _load_referee_module()
    predeclaration = module.MESH_REFINEMENT_PREDECLARATION
    hi = predeclaration["acceptance_band_ratio"][1]

    at_hi = module._evaluate_mesh_refinement_ratio(hi, predeclaration)
    assert at_hi["refuted"] is False, "exactly at the hard edge must still confirm"

    just_above = module._evaluate_mesh_refinement_ratio(hi + 1e-6, predeclaration)
    assert just_above["refuted"] is True

    unchanged_near_original = module._evaluate_mesh_refinement_ratio(1.1208, predeclaration)
    assert unchanged_near_original["refuted"] is True, (
        "a ratio unchanged near the pre-refinement 1.1208 (no convergence "
        "at all) must be the canonical refutation case"
    )


def test_mesh_refinement_gate_no_data_when_ratio_is_none():
    """A missing ratio (e.g. the Stage B witness itself raised before
    producing one) must be reported as NO-DATA -- distinct from both
    CONFIRMED and REFUTED, never silently treated as either."""
    module = _load_referee_module()
    evaluation = module._evaluate_mesh_refinement_ratio(
        None, module.MESH_REFINEMENT_PREDECLARATION
    )
    assert evaluation["refuted"] is None
    assert "NO-DATA" in evaluation["verdict"]


# ---------------------------------------------------------------------------
# Mesh-refinement witness RUN-845 fixture (issue #489 leg 1, VESSL
# 369367251845, 2026-08-05): mirrors the MSL phase referee's own "run-1
# pattern" (tests/crossval/test_msl_phase_referee_header.py's RUN1_RESULT_PATH /
# _load_run1_result / test_run1_stage_b_headline_facts_computed_from_the_
# fixture) -- the committed result JSON is the SOURCE OF TRUTH; every
# assertion below reads a value the fixture actually contains rather than
# re-typing arrays as fresh Python literals, so a future re-run's fixture
# swap cannot silently drift from what these tests claim to check.
# ---------------------------------------------------------------------------
RUN845_RESULT_PATH: Final = (
    REPO_ROOT / "validation" / "crossval" / "_21_coax_two_port_referee_logs"
    / "mesh_refinement_369367251845_result.json"
)
RUN845_LOG_PATH: Final = (
    REPO_ROOT / "validation" / "crossval" / "_21_coax_two_port_referee_logs"
    / "mesh_refinement_369367251845_run.log"
)


def _load_run845_result() -> dict:
    assert RUN845_RESULT_PATH.exists(), f"missing committed run-845 result {RUN845_RESULT_PATH}"
    return json.loads(RUN845_RESULT_PATH.read_text())


def test_run845_log_is_committed_and_matches_the_filled_predeclaration():
    """The committed log/fixture and the script's own FILLED
    MESH_REFINEMENT_PREDECLARATION must agree -- a drift here would mean
    the record was filled from a DIFFERENT run than the one whose
    evidence is actually committed."""
    assert RUN845_LOG_PATH.exists(), f"missing committed run-845 log {RUN845_LOG_PATH}"
    log_text = RUN845_LOG_PATH.read_text()
    assert "MESH-REFINEMENT PREDECLARATION CHECK" in log_text
    assert "CONFIRMED" in log_text
    assert "369367251845" not in log_text  # the run id is not printed inside the referee's own stdout
    module = _load_referee_module()
    record = module.MESH_REFINEMENT_PREDECLARATION
    assert record["status"] == "RUN"
    assert record["vessl_run_id"] == "369367251845"
    assert record["log_path"] == "validation/crossval/_21_coax_two_port_referee_logs/mesh_refinement_369367251845_run.log"


def test_run845_result_matches_the_filled_predeclaration():
    module = _load_referee_module()
    result = _load_run845_result()
    record = module.MESH_REFINEMENT_PREDECLARATION
    sb = result["stage_b"]
    assert result["overall_passed"] is True
    assert sb["sanity_passed"] is True
    assert sb["beta_ratio_measured_over_analytic_mean"] == record["measured_ratio_after"]
    assert abs(sb["dx_scale"] - record["dx_scale"]) < 1e-12


def test_run845_headline_facts_computed_from_the_fixture():
    """Assert the headline facts as COMPUTED FROM the fixture -- not
    independently declared -- per the run-1 pattern's own 'do not assert
    anything the fixture doesn't contain' discipline. Also re-derives the
    one-sided gate verdict and the implied convergence order from the raw
    ratio, rather than trusting the committed prose alone."""
    result = _load_run845_result()
    sb = result["stage_b"]

    ratio = sb["beta_ratio_measured_over_analytic_mean"]
    # B4's own acceptance band: informational lower edge 1.05, hard upper
    # (refuting) edge 1.11. The measured ratio must land inside this band
    # -- a CONFIRMATION, and specifically BELOW the pre-declared 1.0805
    # prediction (a stronger result than predicted, not merely adequate).
    assert 1.05 <= ratio <= 1.11, ratio
    predeclared_prediction = 1.0805333333333333
    assert ratio < predeclared_prediction, (
        "measured ratio should be a STRONGER confirmation than the "
        "first-order prediction, not merely inside the band"
    )

    module = _load_referee_module()
    evaluation = module._evaluate_mesh_refinement_ratio(
        ratio, module.MESH_REFINEMENT_PREDECLARATION
    )
    assert evaluation["refuted"] is False
    assert "CONFIRMED" in evaluation["verdict"]

    # Implied convergence order: a sane bracket (not a tight pin -- physical
    # staircase convergence order for this class of defect plausibly spans
    # naive-first-order (p=1) to super-quadratic; this just catches a gross
    # regression in the log/log formula itself, not a specific physics claim).
    excess_before = module.MESH_REFINEMENT_PREDECLARATION["excess_before"]
    excess_after = ratio - 1.0
    p = math.log(excess_before / excess_after) / math.log(1.5)
    assert 0.5 <= p <= 3.0, p
    assert abs(p - 1.4847707054524188) < 1e-9

    # Per-bin beta-ratio range at the refined mesh (R5 -- inspect the full
    # per-bin data, not just the headline mean): the gated central-band
    # array the mean above was itself averaged from.
    per_bin = sb["matched_through_witness"]["beta_ratio_measured_over_analytic"]
    assert len(per_bin) == 4
    assert min(per_bin) >= 1.05 and max(per_bin) <= 1.11
    assert abs(sum(per_bin) / len(per_bin) - ratio) < 1e-9

    # Witness legs beyond the headline ratio, all from the SAME fixture:
    # passivity, |S21| band, reciprocity, matched-through phase/group-delay
    # against the port's own MEASURED beta.
    assert sb["passivity_drive1"]["passed"] is True
    assert sb["passivity_drive2"]["passed"] is True
    assert sb["passivity_drive1"]["max_balance"] < 1.05
    assert sb["passivity_drive2"]["max_balance"] < 1.05
    assert min(sb["s21_mag"]) > 0.999 and max(sb["s21_mag"]) < 1.001
    assert sb["reciprocity_max_mag_dev"] < 1e-3
    assert sb["reciprocity_max_phase_dev_deg"] < 1.0
    assert sb["matched_through_witness"]["beta_source"] == "measured"
    assert sb["matched_through_witness"]["max_phase_dev_deg"] < 1.0
    assert sb["matched_through_witness_s12"]["max_phase_dev_deg"] < 1.0


def test_run845_does_not_disturb_default_scale_gates():
    """Default-scale invariant: filling the mesh-refinement witness must
    not touch Stage A's own REPRODUCE_GATE_RECORD (a DIFFERENT run,
    369367251629, gating the dx_scale=1.0 fixture's own reproduce-gate) --
    the two records are independent fill-contract objects and a fill of
    one must never silently overwrite the other."""
    module = _load_referee_module()
    assert module.REPRODUCE_GATE_RECORD["vessl_run_id"] == "369367251629"
    assert module.REPRODUCE_GATE_RECORD["status"] == "RUN"
    # The default (dx_scale=1.0) layout must still be byte-identical to
    # B_DX_MM/B_PML_DEPTH_MM -- unaffected by the dx_scale=0.6667 fixture
    # this run measured at.
    default_layout = module._stage_b_layout()
    assert default_layout["dx_mm"] == module.B_DX_MM
    assert default_layout["pml_mm"] == module.B_PML_DEPTH_MM


def test_declared_question_and_governance_notes_present():
    module = _load_referee_module()
    assert "reference-plane referral" in module.DECLARED_QUESTION
    assert "validation/crossval/" in module.MUST_MOVE_WHEN_VALIDATED
    assert "REPRODUCE_GATE_RECORD" in module.MUST_MOVE_WHEN_VALIDATED


def test_stage_a_matches_coaxm_tutorial_constants():
    """Regression lock on Coax.m's own literal parameters -- a future edit
    that silently drifts Stage A away from the tutorial (defeating the
    whole point of a reproduce-gate) should fail this test."""
    module = _load_referee_module()
    assert module.A_LENGTH_MM == 1000.0
    assert module.A_R_I_MM == 100.0
    assert module.A_R_O_MM == 230.0
    assert module.A_R_OS_MM == 240.0
    assert module.A_MESH_RES_MM == 5.0
    assert module.A_F0_HZ == module.A_FC_HZ == 0.5e9
    assert module.A_N_FREQS == 201
    assert module.A_NUM_TS == 5000
    assert module.A_END_CRITERIA == 1.0e-5
    assert module.A_FEEDSHIFT_MM == 50.0  # 10 * mesh_res(1)


def test_zl_a_analytic_matches_coaxm_formula():
    """Independently recompute Coax.m's own ZL_a = Z0/2/pi/sqrt(epsR)*log(r_o/r_i)
    (epsR=1, r_o=230, r_i=100) and check it matches the module's constant --
    a regression lock on the reproduce-gate's own oracle."""
    module = _load_referee_module()
    eta0 = 376.73031346177066
    expected = (eta0 / (2.0 * np.pi)) * np.log(230.0 / 100.0)
    assert abs(module.ZL_A_ANALYTIC_OHM - expected) < 1e-9
    assert 45.0 < module.ZL_A_ANALYTIC_OHM < 55.0  # sanity: coax Z0 is a ~50ohm-class number
    assert module.REPRODUCE_GATE_RECORD["expected_zl_a_ohm"] == float(module.ZL_A_ANALYTIC_OHM)


def _rebuild_referee_grid():
    """Rebuild the EXACT rfx grid the referee's Stage B docstring (lines
    ~122-134) claims its ``B_*`` constants were read off of -- not the
    literals themselves. This is the comparator-first check the #737
    audit found missing repo-wide: ``rfx`` is locally importable (only
    openEMS is VESSL-only), so there is no excuse for a test asserting
    committed geometry constants against themselves instead of against a
    freshly built ``Grid``.

    Cheap: ``sim._build_grid()`` does no time stepping (measured
    2026-08-27, PYTHONPATH=/root/rfx-sub/rfx: first call 0.0001s, repeats
    0.0000s -- the only real cost in this test file is importing ``rfx``
    itself, ~1s, paid once per process). Calling this from more than one
    test is free; no fixture/caching needed.
    """
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port(
        (0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
        waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2),
    )
    return sim._build_grid()


def test_referee_fixture_rebuilds_and_matches_the_declared_discretization():
    """Rebuilds the rfx fixture the header docstring cites provenance from
    and checks the DISCRETIZATION-LEVEL facts (shape, dx, cpml pad depth)
    match what ``B_DX_MM``/``B_CPML_CELLS`` claim -- the part of the B3
    review fix that was never wrong.

    This does NOT check the transverse/axial CLEAR-SPAN constants
    (``B_INTERIOR_*_CELLS``, ``B_CLEAR_*_MM``) -- see
    ``test_referee_interior_span_constants_match_rebuilt_grid_fencepost``,
    which is the one that pins the #739 fencepost fix."""
    module = _load_referee_module()
    grid = _rebuild_referee_grid()

    assert grid.shape == (55, 55, 194)
    assert abs(grid.dx - module.B_DX_MM * 1e-3) < 1e-12
    assert (grid.pad_x_lo, grid.pad_x_hi) == (module.B_CPML_CELLS, module.B_CPML_CELLS)
    assert (grid.pad_y_lo, grid.pad_y_hi) == (module.B_CPML_CELLS, module.B_CPML_CELLS)
    assert (grid.pad_z_lo, grid.pad_z_hi) == (module.B_CPML_CELLS, module.B_CPML_CELLS)

    # Non-geometric literals this constant block also carries -- these are
    # NOT touched by the #739 fencepost fix (they don't come off grid.shape)
    # and stay asserted here so the rewrite does not silently drop them.
    assert module.B_A_MM == 0.635
    assert module.B_B_MM == 2.055
    assert module.B_PTFE_EPS_R == 2.1
    assert abs(module.B_L12_MM - 58.4595293) < 1e-4
    # Z0 = sqrt(L'/C') closed form must land close to the standard SMA/PTFE
    # value (~48.6 ohm) this repo's other coax lanes all cite.
    assert 48.0 < module.B_Z0_OHM < 49.0


def test_referee_interior_span_constants_match_rebuilt_grid_fencepost():
    """The #739 defect, isolated: pins ``B_INTERIOR_*_CELLS``/
    ``B_CLEAR_*_MM`` against the REBUILT grid's own interior span instead
    of against each other. Before the #739 fix, the committed constants
    read grid.shape minus pad (a NODE count) and called it a cell count,
    then multiplied by dx -- double-counting one fencepost cell per axis
    (rfx's own grid.py, Grid.__init__, "+1 fence-post correction: N cells
    need N+1 nodes"; this repo's other two shape-derived crossval
    referees already get this right against a live sim._build_grid() --
    18_wr90_iris_modematch.py:317-318 and
    19_wr90_iris_filter_aghanim.py:509). This is the tautology the old
    ``test_geometry_constants_match_the_rasterized_rfx_fixture`` could
    not catch: its own anti-regression guard, ``abs(B_CLEAR_X_MM - 8.0) >
    0.1``, passes for BOTH the pre-#739 8.619mm and the corrected
    8.244mm, so it cannot distinguish them by construction. Now a plain
    passing assertion (not xfail): #746 landed this test strict-xfailing
    against the old 23/23/162 constants specifically so a future fix to
    those constants could not land silently; #739 items 2-3 IS that fix,
    so the lock converts to a regression assertion on the now-correct
    22/22/161."""
    module = _load_referee_module()
    grid = _rebuild_referee_grid()

    interior_x_nodes = grid.shape[0] - grid.pad_x_lo - grid.pad_x_hi
    interior_y_nodes = grid.shape[1] - grid.pad_y_lo - grid.pad_y_hi
    interior_z_nodes = grid.shape[2] - grid.pad_z_lo - grid.pad_z_hi
    interior_x_cells = interior_x_nodes - 1
    interior_y_cells = interior_y_nodes - 1
    interior_z_cells = interior_z_nodes - 1

    assert module.B_INTERIOR_X_CELLS == interior_x_cells
    assert module.B_INTERIOR_Y_CELLS == interior_y_cells
    assert module.B_INTERIOR_Z_CELLS == interior_z_cells
    assert abs(module.B_CLEAR_X_MM - interior_x_cells * grid.dx * 1e3) < 1e-9
    assert abs(module.B_CLEAR_Y_MM - interior_y_cells * grid.dx * 1e3) < 1e-9
    assert abs(module.B_CLEAR_Z_MM - interior_z_cells * grid.dx * 1e3) < 1e-9


def test_stage_b_layout_is_self_consistent_and_openems_free():
    """``_stage_b_layout()`` must be callable (and its own assertions must
    pass) WITHOUT openEMS -- this is the function whose failure should map
    to exit 3 (config bug), distinct from a physics-gate failure (exit 1)."""
    module = _load_referee_module()
    layout = module._stage_b_layout()  # must not raise

    required = {
        "lx_mm", "ly_mm", "lz_mm", "cx_mm", "cy_mm",
        "z_port1_start_mm", "z_port1_stop_mm", "z_port2_start_mm", "z_port2_stop_mm",
        "z_feed_bot_mm", "z_feed_top_mm", "z_split_mm",
        "ref_plane_shift_port1_mm", "ref_plane_shift_port2_mm",
        "measplane_port1_mm", "feedshift_port1_mm",
        "measplane_port2_mm", "feedshift_port2_mm",
    }
    assert required <= set(layout.keys())

    # H2 fix (restored 2026-08-03, PR #546 review after a diagnosis
    # reversal -- see the referee's "LINE TERMINATION TOPOLOGY" docstring
    # section): both ports' OUTER ends sit exactly at the domain edges,
    # INTO PML_16 -- a MUR z-boundary was tried in between and reverted
    # (MUR-on-PTFE is a RECORDED-unstable combination in this repo's own
    # do-not-repeat history, not a fix).
    assert layout["z_port1_start_mm"] == 0.0
    assert layout["z_port2_stop_mm"] == layout["lz_mm"]

    # The TARGET reference planes (not the bare conductor extent, which
    # legitimately reaches the domain edges, into the PML) must stay
    # safely inside the clear, non-PML region.
    assert layout["z_feed_bot_mm"] > module.B_PML_DEPTH_MM
    assert layout["z_feed_top_mm"] < layout["lz_mm"] - module.B_PML_DEPTH_MM
    assert layout["z_port1_start_mm"] < layout["z_split_mm"] < layout["z_port2_stop_mm"]

    # FeedShift/MeasPlaneShift must land strictly inside each port's own
    # span and stay separated from each other.
    port1_span = layout["z_port1_stop_mm"] - layout["z_port1_start_mm"]
    port2_span = layout["z_port2_stop_mm"] - layout["z_port2_start_mm"]
    assert 0.0 < layout["measplane_port1_mm"] < port1_span
    assert 0.0 < layout["feedshift_port1_mm"] < port1_span
    assert 0.0 < layout["measplane_port2_mm"] < port2_span
    assert 0.0 < layout["feedshift_port2_mm"] < port2_span
    assert abs(layout["measplane_port1_mm"] - layout["feedshift_port1_mm"]) > module.B_DX_MM
    assert abs(layout["measplane_port2_mm"] - layout["feedshift_port2_mm"]) > module.B_DX_MM


def test_stage_b_feed_measure_ordering_matches_coaxm(monkeypatch):
    """M1/M2/M3 fix (PR #546 review, BLOCKING): FeedShift must sit near
    the wall (past the PML's own inner edge) and MeasPlaneShift must sit
    FURTHER from the wall than FeedShift (Coax.m's own ordering) -- the
    REVERSE of the pre-review scheme, where MeasPlaneShift was only 2
    cells short of the target and FeedShift sat further out still. That
    backwards ordering is the sharper lead hypothesis for run-1's
    |S11|=280: CoaxialPort's own 3-line probe stencil, sitting between
    the wall and the feed, samples mostly boundary-reflected energy, a
    vanishing-denominator artifact as the absorber improves (module
    docstring "REFERENCE-PLANE REFERRAL" has the full derivation).

    This test pins the ordering NUMERICALLY (not just via
    ``_stage_b_layout()``'s own internal assertions, which a narrowly
    circumvented "fix" could still satisfy) by reading each port's own
    feed/measure distance from ITS OWN wall directly -- symmetric for
    both ports by construction, since port2's wall is at its `stop`, not
    its `start` (B4').
    """
    module = _load_referee_module()
    layout = module._stage_b_layout()
    cell = module.B_DX_MM
    pml_cells = module.B_CPML_CELLS

    port2_span = layout["z_port2_stop_mm"] - layout["z_port2_start_mm"]

    feed1_from_wall_cells = layout["feedshift_port1_mm"] / cell
    meas1_from_wall_cells = layout["measplane_port1_mm"] / cell
    # port2's wall is at `stop`, so distance-from-wall = span - offset-from-start.
    feed2_from_wall_cells = (port2_span - layout["feedshift_port2_mm"]) / cell
    meas2_from_wall_cells = (port2_span - layout["measplane_port2_mm"]) / cell

    for name, feed_cells, meas_cells in (
        ("port1", feed1_from_wall_cells, meas1_from_wall_cells),
        ("port2", feed2_from_wall_cells, meas2_from_wall_cells),
    ):
        # M2: FEED must clear the PML (never inside the absorber).
        assert feed_cells > pml_cells, (
            f"{name} FeedShift does not clear the PML -- got {feed_cells:.1f} "
            f"cells from wall, PML is {pml_cells} cells (M2 regression)"
        )
        # M1: MEASURE must be device-side of FEED (Coax.m's own ordering),
        # not the reverse.
        assert meas_cells > feed_cells, (
            f"{name} MeasPlaneShift ({meas_cells:.1f} cells from wall) is not "
            f"device-side of FeedShift ({feed_cells:.1f} cells) -- M1 regression, "
            f"the exact backwards ordering diagnosed as run-1's sharper cause"
        )
        # M3: MEASURE needs a stencil-safety margin past the PML edge so
        # CoaxialPort's own 3-line probe snap cannot land back on a
        # PML-adjacent mesh line.
        assert meas_cells > pml_cells + module.B_MEAS_STENCIL_MARGIN_CELLS, (
            f"{name} MeasPlaneShift stencil margin too thin -- got "
            f"{meas_cells:.1f} cells from wall (M3 regression)"
        )

    # Mutation check: this test must actually discriminate -- feeding it
    # the OLD (pre-review) backwards ordering must fail the M1 assertion.
    old_feed1_cells = 8.0   # pre-review: feedshift ~= ref_plane_shift + 5 cells
    old_meas1_cells = 1.0   # pre-review: measplane ~= ref_plane_shift - 2 cells
    assert not (old_meas1_cells > old_feed1_cells), (
        "old backwards ordering coincidentally passes -- test not discriminating"
    )


def test_stage_b_port_directions_are_both_positive():
    """B4' fix (round-2 review, BLOCKING, hole-closer): both Stage B ports
    must be built stop>start (direction=+1), matching Coax.m's own
    same-direction layout -- CoaxialPort's current probe is direction-
    signed, so a mirror-symmetric (direction=-1) port 2 makes
    |uf_inc2/uf_inc1| read ~0 for a genuine forward wave instead of ~1
    (verified by the reviewer against openEMS's own CalcPort formulas on
    a synthetic forward TEM wave). All 13 round-1 tests stayed GREEN
    after the reviewer mutated port 2's orientation back to direction=-1
    -- this test closes that hole by pinning the sign directly from the
    SAME layout fields ``_build_stage_b_drive`` consumes to build the
    real ``CoaxialPort`` objects (single source of truth, not a
    duplicated/independent computation that could itself drift)."""
    module = _load_referee_module()
    layout = module._stage_b_layout()

    direction_port1 = 1.0 if (layout["z_port1_stop_mm"] - layout["z_port1_start_mm"]) > 0 else -1.0
    direction_port2 = 1.0 if (layout["z_port2_stop_mm"] - layout["z_port2_start_mm"]) > 0 else -1.0
    assert direction_port1 == 1.0, "port1 direction must be +1"
    assert direction_port2 == 1.0, "port2 direction must be +1 (B4' regression)"

    # Mutation check: flipping port2's start/stop (the EXACT round-1
    # regression) must be caught -- both by _stage_b_layout()'s own
    # assertion (AssertionError, exit 3) and by this test reading the
    # same fields a different way.
    mutated_direction = (
        1.0 if (layout["z_port2_start_mm"] - layout["z_port2_stop_mm"]) > 0 else -1.0
    )
    assert mutated_direction == -1.0, "mutation sanity check itself is broken"


def test_extract_two_port_s_synthetic_forward_and_reverse_drive():
    """Round-3 fix (BLOCKING): the round-2 fix flipped only the ``s_thru``
    numerator for the reverse drive and left ``s_self`` as ``uf_ref_self/
    uf_inc_self`` unconditionally -- WRONG, since ``uf_inc_self`` is not
    the reverse drive's own launched wave. The reviewer verified this with
    openEMS's own ``CalcPort`` formulas on synthetic wave amplitudes,
    entirely WITHOUT openEMS installed (plain complex arithmetic): the
    round-2 code reported S22 as 1/S22_true and S12 as S21_true/S22_true
    (both traced to the same wrong denominator). This test imitates that
    approach as a real, permanent unit test feeding fake port objects
    with known ``uf_inc``/``uf_ref`` values through ``_extract_two_port_s``
    for BOTH drives (item (c) of the round-3 review -- the previous
    ``reverse_drive``/``thru_uses_ref`` flag was unpinned by any test;
    mutating it stayed green).

    Wave-amplitude construction (matches ``_extract_two_port_s``'s own
    docstring derivation): for a KNOWN reciprocal 2x2 S-matrix (S12=S21),
    normalize each drive's own launched-wave amplitude to 1 --
      forward (a1=1, a2=0): b1=S11*a1, b2=S21*a1 (b2 is +z at port2, so
        IS uf_inc2 under the shared +z convention).
      reverse (a2'=1, a1'=0): b2'=S22*a2', b1'=S12*a2' (a2' is -z at
        port2, so IS uf_ref2; b1' is -z at port1, so IS uf_ref1).
    """
    module = _load_referee_module()

    class _FakePort:
        def __init__(self, uf_inc, uf_ref):
            self.uf_inc = np.asarray(uf_inc, dtype=np.complex128)
            self.uf_ref = np.asarray(uf_ref, dtype=np.complex128)

    s11_true = 0.15 + 0.05j
    s21_true = 0.85 - 0.30j   # |s21_true| ~= 0.90, mirrors the reviewer's "0.9"
    s12_true = s21_true       # reciprocal
    s22_true = 0.03 + 0.04j   # |s22_true| == 0.05, mirrors the reviewer's "0.051"

    # forward drive: a1=1 (port1's own launched wave, +z=uf_inc1), a2=0
    uf_inc1_fwd = np.array([1.0 + 0j])
    uf_ref1_fwd = s11_true * uf_inc1_fwd
    uf_inc2_fwd = s21_true * uf_inc1_fwd
    uf_ref2_fwd = np.array([0.0 + 0j])
    port1_fwd = _FakePort(uf_inc1_fwd, uf_ref1_fwd)
    port2_fwd = _FakePort(uf_inc2_fwd, uf_ref2_fwd)

    # reverse drive: a2'=1 (port2's own launched wave, -z=uf_ref2), a1'=0
    uf_ref2_rev = np.array([1.0 + 0j])
    uf_inc2_rev = s22_true * uf_ref2_rev
    uf_ref1_rev = s12_true * uf_ref2_rev
    uf_inc1_rev = np.array([0.0 + 0j])
    port2_rev = _FakePort(uf_inc2_rev, uf_ref2_rev)
    port1_rev = _FakePort(uf_inc1_rev, uf_ref1_rev)

    fwd = module._extract_two_port_s(port1_fwd, port2_fwd, reverse_drive=False)
    rev = module._extract_two_port_s(port2_rev, port1_rev, reverse_drive=True)

    tol = 1e-9
    assert abs(fwd["s_self"][0] - s11_true) < tol, f"S11: got {fwd['s_self'][0]}, want {s11_true}"
    assert abs(fwd["s_thru"][0] - s21_true) < tol, f"S21: got {fwd['s_thru'][0]}, want {s21_true}"
    assert abs(rev["s_self"][0] - s22_true) < tol, f"S22: got {rev['s_self'][0]}, want {s22_true}"
    assert abs(rev["s_thru"][0] - s12_true) < tol, f"S12: got {rev['s_thru'][0]}, want {s12_true}"

    # Discrimination check: the OLD (round-2) buggy formula -- unconditional
    # uf_ref_self/uf_inc_self and thru/uf_inc_self -- must NOT match the
    # truth for the reverse drive (proving this test actually catches the
    # regression, not just tautologically confirms the current code).
    old_buggy_s22 = port2_rev.uf_ref[0] / port2_rev.uf_inc[0]
    old_buggy_s12 = port1_rev.uf_ref[0] / port2_rev.uf_inc[0]
    assert abs(old_buggy_s22 - (1.0 / s22_true)) < tol, "old-bug reconstruction itself is wrong"
    assert abs(old_buggy_s12 - (s21_true / s22_true)) < tol, "old-bug reconstruction itself is wrong"
    assert abs(old_buggy_s22 - s22_true) > 1.0, (
        "old buggy S22 formula coincidentally matches truth -- test not discriminating"
    )
    assert abs(old_buggy_s12 - s12_true) > 1.0, (
        "old buggy S12 formula coincidentally matches truth -- test not discriminating"
    )

    # reverse_drive is recorded so a reviewer can tell which branch a
    # given result came from without re-deriving it.
    assert fwd["reverse_drive"] is False
    assert rev["reverse_drive"] is True


def _fake_stage_b_drive_dict(drive: str, nrts: int) -> dict:
    """Shared fake ``_run_one_drive`` return value for the partial-data
    tests below. z0/beta use NONZERO imaginary parts deliberately -- a
    zero imaginary part could accidentally round-trip through a broken
    serializer (e.g. one that only handles real floats) and hide B2."""
    s_self = np.array([300.0 + 0j]) if drive == "port1" else np.array([0.05 + 0j])
    return {
        "extracted": {"s_self": s_self, "s_thru": np.array([0.9 + 0j])},
        "z0_port1": np.array([50.0 + 1.5j]), "z0_port2": np.array([48.0 - 2.0j]),
        "beta_port1": np.array([12.0 + 0.1j]), "beta_port2": np.array([12.0 - 0.1j]),
        "max_uf_inc": 1.0, "n_trace_samples": 100, "nrts_cap": nrts,
        "truncated_suspected": False, "elapsed_s": 1.0,
        "end_criteria_not_reached": True,
    }


def test_stage_b_physics_gate_failure_carries_partial_data(monkeypatch):
    """Forensics fix (2026-08-03, run-1 diagnosis): the FIRST live run
    (VESSL 369367251366) raised out of the non-physical-field guard with
    only a single scalar (max|S|=280.7) ever reaching the JSON artifact
    -- the per-bin S-matrices and per-drive diagnostics that would have
    let a reviewer distinguish "channel-inversion bug" from "genuine
    field blow-up" from run.log alone were never recorded anywhere. This
    test pins that a RuntimeError raised inside ``_run_stage_b`` (via the
    non-physical-field guard) now carries a ``partial_stage_b_data``
    attribute with the S-matrices computed before the guard tripped, so
    ``main()``'s RuntimeError handler can write them into the failure
    artifact instead of losing them.

    Scope note (B3, PR #546 review): this test checks the attribute IN
    MEMORY only -- it does not exercise ``json.dump`` at all, which is
    exactly how the B2 serialization bug (complex128 ndarrays in
    ``drive{1,2}_diagnostics``) slipped past it. See
    ``test_main_writes_valid_json_on_stage_b_physics_gate_failure`` below
    for the end-to-end check that actually calls ``json.load()`` on the
    written file.
    """
    module = _load_referee_module()

    def fake_run_one_drive(_CSX, _openems, _port_cls, *, drive, sim_root, threads, nrts, end_criteria, dx_scale=1.0):
        return _fake_stage_b_drive_dict(drive, nrts)

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_one_drive", fake_run_one_drive)

    try:
        module._run_stage_b(sim_root="/tmp/_unused_forensics_test", threads=1,
                            nrts=100, end_criteria=1e-4)
        assert False, "expected the non-physical-field guard to raise on s11=300"
    except RuntimeError as exc:
        assert "[s11]" in str(exc)
        partial = getattr(exc, "partial_stage_b_data", None)
        assert partial is not None, "RuntimeError must carry partial_stage_b_data"
        assert partial["s11_mag"] == [300.0]
        assert partial["s21_mag"] == [0.9]
        assert "drive1_diagnostics" in partial and "drive2_diagnostics" in partial

        # B2 fix: z0/beta must already be [re, im] pairs here (via
        # _json_safe_diagnostics), not raw complex128 ndarrays.
        z0_port1 = partial["drive1_diagnostics"]["z0_port1"]
        assert z0_port1 == [[50.0, 1.5]], f"z0_port1 not JSON-safe: {z0_port1!r}"
        assert isinstance(z0_port1, list) and isinstance(z0_port1[0], list)


def test_check_excitation_and_trace_uses_drive_correct_channel():
    """M1 fix (PR #546 review): ``_check_excitation_and_trace`` must
    check the channel the DRIVE actually launched -- ``uf_inc`` for the
    forward drive (port1), ``uf_ref`` for the reverse drive (port2, per
    ``_extract_two_port_s``'s own documented convention: "port 2's own
    launched wave into the device is its -z component, uf_ref_self (NOT
    uf_inc_self)"). Checking ``uf_inc`` unconditionally, as before,
    would silently verify the WRONG channel for the reverse drive --
    this test builds a fake port where uf_inc is healthy but uf_ref is
    dropped (exactly zero, the openEMS "port never coupled" signature)
    and checks the function catches it when told to look at uf_ref, and
    does NOT catch it (wrongly reports healthy) when defaulted to
    uf_inc -- proving the ``channel`` parameter actually changes
    behavior, not just exists.

    Run-3 rescope (PR #547 review): the guard is SCALE-FREE (see
    ``_check_excitation_and_trace``'s own docstring -- the ``[D5]``
    pattern, ``rfx/interop/emitters/openems.py``, and issue #465/PR #473
    already fixed this exact false-fire class in this repo once).
    Fake values are still at openEMS's own real scale (a healthy real
    run reads ~1.79e-13 to 1.93e-12, per
    ``docs/design_notes/geometry_setup_interop.md``) to keep the test
    honest about scale, but the "broken" channel is now an EXACT zero,
    not merely small -- a nonzero-but-small value like the previously
    used 6e-14 is not a failure signature at all once there is no floor,
    and asserting it would raise re-introduces the refuted invariant
    this round removes.
    """
    module = _load_referee_module()

    class _FakePort:
        U_filenames: list = []
        uf_inc = np.array([5.0e-13 + 0j])   # healthy, real openEMS scale
        uf_ref = np.array([0.0 + 0j])       # dropped -- the launched
                                             # channel for a reverse drive

    port = _FakePort()

    # Default (uf_inc) must NOT raise -- uf_inc is healthy at real scale.
    inc_peak, _ = module._check_excitation_and_trace(port, "/tmp/_unused", "test")
    assert inc_peak == 5.0e-13

    # channel="uf_ref" (the reverse-drive's own launched channel) MUST
    # raise -- this one is exactly zero (the openEMS dropped-port
    # signature), not merely small.
    try:
        module._check_excitation_and_trace(port, "/tmp/_unused", "test", channel="uf_ref")
        assert False, "expected zero uf_ref to raise when channel='uf_ref'"
    except RuntimeError as exc:
        assert "uf_ref" in str(exc)


def test_excitation_check_is_scale_free_not_an_absolute_floor():
    """Run-3 fix (2026-08-03, PR #547 review): TWO successive rounds of
    this module tuned an ABSOLUTE floor on ``inc_peak`` (1e-6, then a
    "corrected" 1e-13) -- both refuted by the SAME repo's own proven
    guard: ``[D5]`` in ``rfx/interop/emitters/openems.py`` (~line 1676)
    records "a verified-good small run on this install peaked at
    |uf_inc| ~ 3e-14" -- BELOW both tried floors AND below the
    do-not-repeat file's own recorded "failure" value (6e-14): the
    healthy and broken populations OVERLAP in openEMS's raw units, so no
    absolute floor separates them. This exact false-fire was already
    diagnosed and fixed once in this repo (issue #465, PR #473,
    ``8578fcd``) -- the fix that shipped is scale-free (exact-zero/
    non-finite only), which is what this test pins: a value SMALLER than
    every floor this module has ever tried (1e-14, well below the
    verified-good 3e-14 the D5 guard itself cites) must still PASS,
    because it is finite and nonzero -- only an exact zero (or NaN) is a
    defect signature.
    """
    module = _load_referee_module()

    class _FakePort:
        U_filenames: list = []

        def __init__(self, uf_inc):
            self.uf_inc = np.array([uf_inc + 0j])

    # Smaller than every floor this module has tried (1e-6, then 1e-13)
    # AND smaller than the D5 guard's own verified-good value (3e-14) --
    # must still PASS. This is the discriminating case: an absolute-floor
    # regression (reintroducing any positive threshold) fails here.
    tiny_but_real = _FakePort(1e-14)
    inc_peak, _ = module._check_excitation_and_trace(tiny_but_real, "/tmp/_unused", "tiny")
    assert inc_peak == 1e-14

    # Exact zero (the openEMS dropped-port signature) and NaN must still
    # raise -- scale-free does not mean "anything goes".
    for broken_value in (0.0, float("nan")):
        broken = _FakePort(broken_value)
        try:
            module._check_excitation_and_trace(broken, "/tmp/_unused", "broken")
            assert False, f"expected {broken_value!r} to raise"
        except RuntimeError:
            pass

    assert not hasattr(module, "_EXCITATION_ENERGY_FLOOR"), (
        "an absolute floor constant must not exist -- PR #547 review "
        "refuted it via the repo's own D5 guard and issue #465/PR #473"
    )


def test_main_writes_valid_json_on_stage_b_physics_gate_failure(monkeypatch, tmp_path):
    """B3 fix (PR #546 review, BLOCKING): the regression test above only
    checked ``partial_stage_b_data`` IN MEMORY -- it never called
    ``json.dump``/``json.load`` at all, which is exactly how B2 (complex
    ndarrays in ``drive{1,2}_diagnostics``) slipped past it (reviewer-
    reproduced: the failure-path writer crashed with ``TypeError``,
    leaving a truncated or missing artifact). This test drives
    ``main()`` all the way through a Stage B physics-gate failure and
    asserts the written file actually PARSES.
    """
    import json

    module = _load_referee_module()

    def fake_run_stage_a(*, sim_root, threads, use_pml):
        return {
            "passed": True, "zl_mean_ohm": 50.0, "zl_max_dev_ohm": 0.1,
            "passivity": {"passed": True}, "matched_through_witness": {"passed": True},
        }

    def fake_run_one_drive(_CSX, _openems, _port_cls, *, drive, sim_root, threads, nrts, end_criteria, dx_scale=1.0):
        return _fake_stage_b_drive_dict(drive, nrts)

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_stage_a_reproduce_gate", fake_run_stage_a)
    monkeypatch.setattr(module, "_run_one_drive", fake_run_one_drive)

    out_path = tmp_path / "out.json"
    rc = module.main(["--output", str(out_path)])

    assert rc == 1, f"expected exit 1 (physics-gate failure), got {rc}"
    assert out_path.exists(), "main() must write a JSON artifact even on failure"

    with open(out_path) as f:
        artifact = json.load(f)  # must not raise -- this is what B2 broke

    assert artifact["error"]
    assert "stage_b_partial" in artifact
    partial = artifact["stage_b_partial"]
    assert partial["s11_mag"] == [300.0]
    assert partial["drive1_diagnostics"]["z0_port1"] == [[50.0, 1.5]]
    assert partial["drive1_diagnostics"]["end_criteria_not_reached"] is True


def test_stage_b_matched_through_band_and_group_delay():
    """B5' fix (round-2 review, BLOCKING): Stage B's own matched-through
    band must differ from Stage A's ideal-lossless band (rfx's raw |S21|
    profile is 0.960->0.737, a real lossy line), and the expected group
    delay (L12*sqrt(eps_r)/c) must land near the reviewer's own
    independently-stated ~282 ps for L12=58.4595mm, eps_r=2.1."""
    module = _load_referee_module()
    assert module.B_S21_THRU_BAND == (0.5, 1.1)
    assert module.B_S21_THRU_BAND != module.REPRODUCE_GATE_RECORD["gate"]["s21_thru_band"]

    expected_gd_s = module.B_L12_MM * 1e-3 * (module.B_PTFE_EPS_R ** 0.5) / module._C0
    assert abs(expected_gd_s * 1e12 - 282.0) < 5.0


# Run-3 (VESSL 369367251629) committed forensics: freqs 4-12 GHz (9 pts),
# .omx/coax-two-port-referee/20260803T182559Z/openems_coax_two_port.json's
# stage_b_partial. Recorded here VERBATIM (not re-derived) as a permanent
# fixture -- this is the exact data that showed the analytic-beta
# assumption in _matched_through_witness fails (111.38 deg) while the
# port's own measured beta passes (<1 deg). Pins the diagnosis itself,
# not just "some fix exists".
_RUN3_FREQS_GHZ = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
_RUN3_S21 = [
    [-0.10178395789276987, -0.9947813873544716], [-0.866908124484904, 0.49853137662871505],
    [0.8093312852813679, 0.5874003767160384], [0.2039989433192543, -0.9789700021218559],
    [-0.9762025698046496, 0.21687781552564456], [0.6007351793240925, 0.7994437767482262],
    [0.4778275403413366, -0.8784484875365575], [-0.9977106504427633, -0.06775523370850421],
    [0.3569821660144234, 0.9341058676426305],
]
_RUN3_S12 = [
    [-0.10179691287104925, -0.9947924466056416], [-0.8668989701758875, 0.4985510803902553],
    [0.8093328993935945, 0.5873968466839313], [0.20397830945821815, -0.9789743050808357],
    [-0.9762034811684641, 0.21687144517256018], [0.6007391786857395, 0.7994415047250955],
    [0.4778280770003993, -0.8784497561822715], [-0.9977158509219143, -0.06773546248046698],
    [0.35699440259104104, 0.9341057748131218],
]
_RUN3_S11 = [
    [6.631303519201963e-05, -0.00011686804566331664], [-0.00012853357758483953, -0.00028856861831318066],
    [-0.00011674886873790862, -0.00032795537021504536], [-0.0005204981350035947, -1.0296241135269366e-05],
    [-0.0007915640876916811, -0.00016518731233492254], [-0.00034776276114460985, 0.0004830473362504877],
    [-0.0008939681254331274, 0.001165546278465178], [-0.00016507074822102292, 0.0005841232511762107],
    [0.0011058815814476504, 0.0018492473344146887],
]
_RUN3_S22 = [
    [5.898248024424526e-05, -0.00011777007616695023], [-0.00014034312784866302, -0.0002820860010893682],
    [-0.00011478998866623993, -0.00032805085647930147], [-0.0005210861014952454, -1.0788268666929396e-05],
    [-0.0007918717723981598, -0.0001647009475082112], [-0.0003488296360283009, 0.000487621749493795],
    [-0.0008956613465229802, 0.0011684382089198755], [-0.0001691399526852089, 0.0005825048113253377],
    [0.0011029870579533432, 0.001850566854975996],
]
_RUN3_BETA_PORT1_D1 = [
    [136.09101931196903, -0.01649078820414983], [170.13617981172283, -0.029272331708947892],
    [204.1918068030309, -0.030520265869223965], [238.26054521306463, -0.022009345540307406],
    [272.3573016004093, -0.020949968138124917], [306.475449749617, -0.02334136211816488],
    [340.6194897444045, -0.03815060788334049], [374.7888778145982, -0.05597846036549409],
    [408.99096252465364, -0.07097161612786418],
]
_RUN3_BETA_PORT2_D1 = [
    [136.0832834854087, 0.003682243954273083], [170.12782075071675, 0.012158515739560831],
    [204.18324294995574, 0.0027347741393589082], [238.25629260895758, 0.021911184077467078],
    [272.34648514048325, 0.007143288633369828], [306.44607486105457, 0.044185757537405945],
    [340.6142917311148, 0.05043285347080633], [374.7890595209023, 0.04710847900989794],
    [408.97181481282405, 0.07762984179484164],
]
_RUN3_BETA_PORT1_D2 = [
    [136.08384097183855, 0.000997997837746734], [170.12830603521184, 0.010401602868161566],
    [204.18352430176137, 0.004887881536652085], [238.257866163468, 0.021157897001576415],
    [272.34820939710164, 0.00840646357503059], [306.4468305265026, 0.04821469809538255],
    [340.6150052015737, 0.05006892983189138], [374.79382327086233, 0.051748372453706115],
    [408.97232688220157, 0.07866617148707125],
]
_RUN3_BETA_PORT2_D2 = [
    [136.0901663783743, -0.015448057441132108], [170.13771844640192, -0.025506094423440318],
    [204.19145635065618, -0.03086373789552074], [238.26167607086484, -0.02401851530670198],
    [272.35502171803535, -0.01904883706935545], [306.47555613016044, -0.022886628175141727],
    [340.618842528779, -0.03986300862429704], [374.78802568468524, -0.056164338000943434],
    [408.9928976396013, -0.07074324827722468],
]


def _run3_complex(pairs):
    return np.array([complex(re, im) for re, im in pairs])


def test_matched_through_witness_run3_regression_measured_vs_analytic_beta():
    """Run-3 fix (2026-08-03, PR #548 review): diagnosed OFFLINE from the
    committed run-3 forensics (VESSL 369367251629,
    .omx/coax-two-port-referee/20260803T182559Z/openems_coax_two_port.json's
    stage_b_partial -- the exact data below, recorded verbatim). Stage A
    passed, Stage B completed both drives with |S21|~=1.000, no blow-up
    (H2/M1 topology exonerated) -- but the matched-through witness's
    phase leg failed: max_phase_dev 111.38 deg vs the 30 deg tolerance,
    while magnitude and (loosely) group delay passed.

    Root cause: the witness's ``expected_phase = -beta*L`` used an
    IDEALIZED analytic beta (2*pi*f*sqrt(2.1)/c), but CalcPort's own
    MEASURED beta (already computed from the same field data, all 4
    available readings -- port1/port2, both drives -- agreeing to
    ~0.01%) is consistently ~1.12x larger: a real, reproducible ~12%
    Yee-staircasing bias from this coax's own coarse a-to-b PTFE annulus
    (~3.8 cells across (2.055-0.635mm)/0.375mm), not a solver or
    referral defect -- the SAME class of effect this repo's own MSL
    preflight documents for a comparably coarse dielectric gap.

    This test pins BOTH halves of that finding directly against the
    real run-3 numbers: the OLD (analytic-beta) path must still fail
    with (very close to) the EXACT reported deviation -- proving this
    fix targets the real defect, not a coincidentally-similar one -- and
    the NEW (measured-beta) path must pass with sub-degree phase
    deviation on the SAME data.
    """
    module = _load_referee_module()
    freqs_hz = np.array(_RUN3_FREQS_GHZ) * 1e9
    s21 = _run3_complex(_RUN3_S21)
    s12 = _run3_complex(_RUN3_S12)

    beta_measured = 0.25 * (
        _run3_complex(_RUN3_BETA_PORT1_D1) + _run3_complex(_RUN3_BETA_PORT2_D1)
        + _run3_complex(_RUN3_BETA_PORT1_D2) + _run3_complex(_RUN3_BETA_PORT2_D2)
    )

    # OLD path (beta=None, analytic): must reproduce the EXACT reported
    # failure -- this is the regression lock on the DIAGNOSIS, not just
    # on "some number is large".
    try:
        module._matched_through_witness(
            freqs_hz, s21, L_m=module.B_L12_MM * 1e-3, eps_r=module.B_PTFE_EPS_R,
            mag_band=module.B_S21_THRU_BAND, label="run3_s21_analytic")
        assert False, "expected the analytic-beta path to still fail on run-3's own data"
    except RuntimeError as exc:
        assert "111.38" in str(exc), f"expected ~111.38 deg deviation, got: {exc}"

    # NEW path (measured beta): must PASS, on BOTH S21 and S12, with
    # sub-degree phase deviation.
    result_s21 = module._matched_through_witness(
        freqs_hz, s21, L_m=module.B_L12_MM * 1e-3, eps_r=module.B_PTFE_EPS_R,
        mag_band=module.B_S21_THRU_BAND, label="run3_s21_measured", beta=beta_measured)
    result_s12 = module._matched_through_witness(
        freqs_hz, s12, L_m=module.B_L12_MM * 1e-3, eps_r=module.B_PTFE_EPS_R,
        mag_band=module.B_S21_THRU_BAND, label="run3_s12_measured", beta=beta_measured)

    assert result_s21["passed"] is True
    assert result_s12["passed"] is True
    assert result_s21["max_phase_dev_deg"] < 1.0
    assert result_s12["max_phase_dev_deg"] < 1.0
    assert result_s21["beta_source"] == "measured"

    # Non-blocking review fix (PR #548): the ratio must be VISIBLE per-bin
    # (not just implied by a passing phase check) and must land in the
    # ~12% band this diagnosis measured -- a future ~2x drift in the same
    # direction should trip this directly, without raw-array archaeology.
    ratio = result_s21["beta_ratio_measured_over_analytic"]
    assert ratio is not None and len(ratio) == 4  # the gated central band
    assert all(1.10 < r < 1.14 for r in ratio), f"beta ratio out of the recorded ~12% band: {ratio}"

    # The analytic path (beta=None) must not report a measured ratio at
    # all -- there is nothing "measured" to ratio against. Use a
    # synthetic, perfectly-matched-to-the-analytic-model S21 (zero
    # deviation by construction) so this passes cleanly and isolates the
    # field check from any real data's own imperfections.
    beta_analytic = 2.0 * np.pi * freqs_hz * np.sqrt(module.B_PTFE_EPS_R) / module._C0
    s21_synthetic = np.exp(-1j * beta_analytic * module.B_L12_MM * 1e-3)
    result_analytic = module._matched_through_witness(
        freqs_hz, s21_synthetic, L_m=module.B_L12_MM * 1e-3, eps_r=module.B_PTFE_EPS_R,
        mag_band=module.B_S21_THRU_BAND, label="ratio_none_check")
    assert result_analytic["passed"] is True
    assert result_analytic["beta_ratio_measured_over_analytic"] is None


def test_run_stage_b_passes_end_to_end_on_run3_data_with_the_fix(monkeypatch):
    """Run-3 fix, end-to-end: replays run-3's own recorded per-drive data
    through the REAL ``_run_stage_b`` (via a fake ``_run_one_drive``, the
    same pattern the earlier partial-data tests use) and confirms the
    fixed pipeline now reports ``sanity_passed=True`` on the EXACT data
    that failed on VESSL -- not just that the witness function alone
    passes in isolation, but that the wiring inside ``_run_stage_b``
    (computing and threading ``beta_measured`` through to both witness
    calls) is what actually gets exercised.
    """
    module = _load_referee_module()

    def fake_run_one_drive(_CSX, _openems, _port_cls, *, drive, sim_root, threads, nrts, end_criteria, dx_scale=1.0):
        if drive == "port1":
            extracted = {"s_self": _run3_complex(_RUN3_S11), "s_thru": _run3_complex(_RUN3_S21),
                        "s_thru_alternate_channel": _run3_complex(_RUN3_S21)}
            beta_port1, beta_port2 = _run3_complex(_RUN3_BETA_PORT1_D1), _run3_complex(_RUN3_BETA_PORT2_D1)
        else:
            extracted = {"s_self": _run3_complex(_RUN3_S22), "s_thru": _run3_complex(_RUN3_S12),
                        "s_thru_alternate_channel": _run3_complex(_RUN3_S12)}
            beta_port1, beta_port2 = _run3_complex(_RUN3_BETA_PORT1_D2), _run3_complex(_RUN3_BETA_PORT2_D2)
        return {
            "extracted": extracted,
            "z0_port1": np.zeros(9, dtype=np.complex128), "z0_port2": np.zeros(9, dtype=np.complex128),
            "beta_port1": beta_port1, "beta_port2": beta_port2,
            "max_uf_inc": 1.0, "n_trace_samples": 100, "nrts_cap": nrts,
            "truncated_suspected": False, "elapsed_s": 1.0,
            "end_criteria_not_reached": False,
        }

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_one_drive", fake_run_one_drive)
    monkeypatch.setattr(module, "B_FREQS_HZ", np.array(_RUN3_FREQS_GHZ) * 1e9)
    monkeypatch.setattr(module, "B_FREQS_GHZ", np.array(_RUN3_FREQS_GHZ))

    result = module._run_stage_b(sim_root="/tmp/_unused_run3_e2e", threads=1, nrts=200000, end_criteria=1e-4)
    assert result["sanity_passed"] is True
    assert result["matched_through_witness"]["max_phase_dev_deg"] < 1.0
    assert result["matched_through_witness_s12"]["max_phase_dev_deg"] < 1.0


def test_stage_a_wires_its_own_num_ts_and_end_criteria_into_openems(monkeypatch):
    """H1' fix (round-2 review, BLOCKING): A_NUM_TS/A_END_CRITERIA were
    DEFINED and pinned by test_stage_a_matches_coaxm_tutorial_constants
    (above) but never actually reached ``openEMS(...)`` -- Stage A silently
    ran whatever ``--nrts``/``--end-criteria`` Stage B's CLI flags carried
    (default 200000/1e-4) against Coax.m's MUR boundaries, where run
    length controls reflected-energy accumulation. That constants test
    passed before AND after the bug existed -- it only checked the
    constants' OWN values, never that anything downstream used them. This
    test asserts the WIRING: it captures what
    ``_build_stage_a_coax_tutorial`` is actually called with, via
    ``_run_stage_a_reproduce_gate`` (which, post-fix, no longer even
    accepts nrts/end_criteria as its own parameters -- there is no longer
    a code path for a caller-supplied CLI value to reach Stage A at all).
    """
    module = _load_referee_module()
    calls = []

    class _FakeFdtd:
        def Run(self, *args, **kwargs):
            pass  # no-op: _run_openems_capturing_stdout just needs this to not raise

    def fake_build(ContinuousStructure, openEMS, CoaxialPort, *, nrts, end_criteria, use_pml):
        calls.append({"nrts": nrts, "end_criteria": end_criteria})
        if len(calls) < 2:
            # let the smoke-run call through so the real (second) call --
            # the one this test actually cares about -- is reached too.
            return _FakeFdtd(), None, None
        raise RuntimeError("stub: stop here, only checking the wiring")

    monkeypatch.setattr(module, "_build_stage_a_coax_tutorial", fake_build)
    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))

    try:
        module._run_stage_a_reproduce_gate(sim_root="/tmp/_unused_h1prime", threads=1, use_pml=False)
    except RuntimeError:
        pass

    assert calls, "_build_stage_a_coax_tutorial was never called"
    assert any(c["end_criteria"] == module.A_END_CRITERIA for c in calls), (
        f"no call used A_END_CRITERIA={module.A_END_CRITERIA}; got {calls}"
    )
    assert any(c["nrts"] == module.A_NUM_TS for c in calls), (
        f"no call used A_NUM_TS={module.A_NUM_TS}; got {calls}"
    )
    # the smoke run must NOT silently use Stage B's 200000/1e-4 CLI
    # defaults either -- it is min(200, A_NUM_TS), always derived FROM
    # A_NUM_TS, never an independent value.
    assert all(c["nrts"] in (module.A_NUM_TS, min(200, module.A_NUM_TS)) for c in calls), (
        f"a call used an nrts unrelated to A_NUM_TS; got {calls}"
    )
    assert 200000 not in [c["nrts"] for c in calls], (
        "Stage A used Stage B's CLI --nrts default (200000) -- H1' regression"
    )
    assert 1e-4 not in [c["end_criteria"] for c in calls], (
        "Stage A used Stage B's CLI --end-criteria default (1e-4) -- H1' regression"
    )


def test_openems_unavailable_exits_2(monkeypatch):
    """main() must return exit code 2 (declared skip), not raise, when the
    openEMS Python bindings are absent -- lane convention (vessl_external_
    referee_lane.md): 0=self-checks passed, 1=self-check failed, 2=missing,
    3=internal config error."""
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("CSXCAD.CSXCAD", "openEMS.openEMS", "openEMS.ports", "CSXCAD", "openEMS"):
            raise ImportError("simulated: openEMS not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert module.main([]) == 2


def test_config_error_exits_3_distinct_from_physics_failure(monkeypatch):
    """M2 review fix: an internal config/layout bug must NOT be reported
    with the same exit code as a physics/self-check gate failure. This
    simulates a layout bug (by monkeypatching ``_stage_b_layout`` to raise
    AssertionError, as it would if rfx's geometry drifted and the
    hard-coded constants above were not updated) and checks main() reports
    exit 3, not 1 or a raw traceback."""
    module = _load_referee_module()

    def _broken_layout(*args, **kwargs):
        raise AssertionError("simulated: rfx geometry drifted")

    monkeypatch.setattr(module, "_stage_b_layout", _broken_layout)
    assert module.main([]) == 3


def test_freqs_overlap_the_rfx_band():
    """The referee's Stage B frequency grid must cover the rfx BAND points
    ([4,6,8,10,12] GHz, tests/unit/sparams/test_coax_two_port_smatrix.py) it will be
    compared against by hand."""
    module = _load_referee_module()
    freqs_ghz = set(round(float(f), 6) for f in module.B_FREQS_GHZ)
    for f in (4.0, 6.0, 8.0, 10.0, 12.0):
        assert f in freqs_ghz, f"{f} GHz missing from referee frequency grid"
