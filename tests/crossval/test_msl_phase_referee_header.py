"""Header/record/contract checks for the openEMS MSL phase referee (#490 lane 2).

``validation/crossval/20_msl_phase_referee.py`` (promoted 2026-08-04, #490
reviewer judgment; formerly ``validation/research/msl_phase_referee/
openems_msl_phase_referee.py``) is a COMPARATOR-leg-only, VESSL-only
harness (see its module docstring for the full scope fence): openEMS is
not installed in THIS test environment, and
this test does not need it -- it only loads the module and inspects
Python-level data, plus exercises the openEMS-free pure-arithmetic helpers
(``_stage_b_layout``, ``_self_consistency_witness``, ``_check_excitation_
and_trace``, ``_passivity_witness``, ``_non_physical_guard``) directly,
mirroring ``tests/crossval/test_coax_two_port_referee_header.py``'s design.

Design (fail-loud-honest, per the coax referee's own precedent): the
reproduce-gate-record tests pass on EITHER the never-run UNRUN placeholder
OR a legitimately-filled state -- they check the record's CONTRACT shape,
not one pinned finding.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import subprocess
from types import ModuleType
from typing import Final

import numpy as np
import pytest

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
# Promoted 2026-08-04 (#490 reviewer judgment): validation/research/
# msl_phase_referee/openems_msl_phase_referee.py -> validation/crossval/
# 20_msl_phase_referee.py, registered in validation/crossval/manifest.json.
SCRIPT_PATH: Final = REPO_ROOT / "validation" / "crossval" / "20_msl_phase_referee.py"
RFX_FIXTURE_PATH: Final = REPO_ROOT / "tests" / "fixtures" / "msl_phase_referee" / "msl_thru_rfx_dx50.json"


def _load_referee_module() -> ModuleType:
    """Load the referee script as a throwaway module (not sys.modules-registered).

    Must succeed WITHOUT openEMS installed -- ``_import_openems`` defers its
    openEMS import into a function specifically so this works.
    """
    assert SCRIPT_PATH.exists(), f"missing referee script {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("_msl_phase_referee", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module / reproduce-gate-record contract
# ---------------------------------------------------------------------------
def test_module_imports_without_openems():
    module = _load_referee_module()
    assert hasattr(module, "REPRODUCE_GATE_RECORD")


def test_reproduce_gate_record_has_required_fields():
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "expected_f_notch_an_hz", "gate", "status", "reproduced_f_notch_hz",
        "reproduced_f_notch_dev_pct", "log_path", "vessl_run_id",
        "settling_evidence", "measured_precision",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"


def test_tutorial_citation_is_verifiable():
    module = _load_referee_module()
    tutorial = module.REPRODUCE_GATE_RECORD["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".py") or tutorial["path"].endswith(".m"), (
        "tutorial citation should point at a real source file"
    )
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"
    assert "MSL_NotchFilter" in tutorial["path"]


def test_do_not_repeat_cites_the_dx80_mixed_cell_trap():
    """R1/R2 class: the rebuild must name the specific recorded failure it
    avoids repeating (the dx=80um mixed-cell openEMS non-physical result on
    the identical RO4350B substrate), not just assert dx=50um in the abstract.
    """
    module = _load_referee_module()
    do_not_repeat = module.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "build_msl_notch_openems_comparison.py" in do_not_repeat
    assert "3.175" in do_not_repeat or "80" in do_not_repeat
    assert "5.08" in do_not_repeat or "50" in do_not_repeat


def test_settling_evidence_field_documents_the_smoke_vs_real_positive_control():
    """#552 reviewer follow-up: settling_evidence must live ON the record
    (so it serializes into every future run's own artifact JSON), not only
    in the module docstring -- pinned by content, and cross-checked against
    the SAME log path the fill-contract test above already verifies is
    git-tracked and on disk."""
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    settling = record["settling_evidence"]
    assert isinstance(settling, str) and settling.strip()
    assert record["log_path"].split("/")[-1] in settling, (
        "settling_evidence should cite the SAME log file the record's own "
        "log_path points at"
    )
    assert "SMOKE" in settling and "REAL" in settling
    assert "-40dB" in settling and "-50dB" in settling
    assert "STDERR" in settling


def test_measured_precision_field_documents_the_s21_bias():
    """#552 reviewer follow-up: measured_precision must live ON the record,
    pinned by content (the same numbers the run-1 regression-fixture tests
    below independently recompute FROM the committed fixture)."""
    module = _load_referee_module()
    measured_precision = module.REPRODUCE_GATE_RECORD["measured_precision"]
    assert isinstance(measured_precision, str) and measured_precision.strip()
    assert "29" in measured_precision
    assert "1.00872" in measured_precision
    assert "74%" in measured_precision


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path. A
    filled (status != 'UNRUN') record's log_path must live under a
    GIT-TRACKED prefix (PR #548 lesson -- .omx/ and docs/research_notes/
    vessl_logs/ are both gitignored and unreadable by a reviewer outside
    the machine that ran the job).
    """
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_f_notch_hz"] is None
        assert record["reproduced_f_notch_dev_pct"] is None
        assert record["log_path"] is None
    else:
        assert record["reproduced_f_notch_hz"] is not None
        assert record["reproduced_f_notch_dev_pct"] is not None
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"

        gitignored_prefixes = (".omx/", "docs/research_notes/vessl_logs/")
        tracked_prefixes = ("validation/crossval/_20_msl_phase_referee_logs/",)
        assert not log_path_str.startswith(gitignored_prefixes), (
            f"log_path {log_path_str!r} lives under a GITIGNORED prefix -- "
            f"a FILLED record needs a log a reviewer OUTSIDE this machine "
            f"can open; use {tracked_prefixes!r} instead (PR #548 lesson)"
        )
        assert log_path_str.startswith(tracked_prefixes), (
            f"log_path {log_path_str!r} must live under a TRACKED prefix {tracked_prefixes!r}"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but its "
            f"log_path {log_path} does not exist"
        )

        # TRACKEDNESS fix (#490 reviewer, 2026-08-04 -- "the #548 failure one
        # layer down"): os.path.exists() alone passes on THIS machine even
        # for a file that was only ever written to the working tree and
        # never `git add`ed -- invisible to anyone who clones the repo
        # fresh. `git ls-files --error-unmatch` is the actual ground truth
        # for "is this path in the index/committed", not merely present on
        # disk here.
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(log_path)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert tracked.returncode == 0 and tracked.stdout.strip() == log_path_str, (
            f"reproduce_gate_record claims status={record['status']!r} with "
            f"log_path {log_path_str!r}, and the file exists on disk, but "
            f"`git ls-files --error-unmatch` does not confirm it is TRACKED "
            f"(rc={tracked.returncode}, stdout={tracked.stdout!r}, "
            f"stderr={tracked.stderr!r}) -- an untracked evidence file is "
            f"invisible to a reviewer outside this machine even though it "
            f"passes exists() here."
        )


def test_declared_question_and_governance_notes_present():
    module = _load_referee_module()
    assert "reference-plane" in module.DECLARED_QUESTION.lower()
    assert "validation/crossval/" in module.MUST_MOVE_WHEN_VALIDATED
    assert "REPRODUCE_GATE_RECORD" in module.MUST_MOVE_WHEN_VALIDATED


def test_f_notch_an_matches_cv06b_closed_form():
    """Independently recompute the Hammerstad-Jensen quarter-wave-notch
    closed form ``validation/crossval/06b_msl_notch_filter_uniform.py``
    uses, on the DECLARED 600um/254um board, and check the two land on the
    same value (5 sig figs) -- a regression lock on the reproduce-gate's
    own oracle, and a cross-check that this script did not silently
    diverge from the repo's existing validated formula.

    issue #723 (2026-08-27) CORRECTION: this is the SAME formula, but NOT
    the same board as cv06b's own runtime output any more. ``module.
    F_NOTCH_AN_HZ`` here is Stage A's value (A_MSL_WIDTH_UM=600,
    A_SUBSTRATE_THICKNESS_UM=254 -- realized EXACTLY, Stage A's z-mesh is
    an explicit ``linspace(0, 254, 5)``, not an off-lattice uniform
    arange), so ``w_trace``/``h_sub`` below correctly describe what Stage
    A solves. cv06b itself, under its own #723 fix, now computes ``u``
    from its REALIZED trace width (635.0um, not the declared 600um) --
    its analytic notch moved 3.6872 -> 3.6790 GHz. This test's hardcoded
    600e-6/254e-6 therefore reproduces STAGE A's board and this test's own
    ``expected``, not cv06b's current runtime ``F_NOTCH_AN`` -- the
    assertion below stays numerically green because it was never
    comparing against cv06b's live output (no import of cv06b exists
    here), only against this module's OWN Stage-A constant.
    """
    module = _load_referee_module()
    c0 = 2.998e8
    stub_len = 12.0e-3
    w_trace = 600e-6
    h_sub = 254e-6
    eps_r = 3.66
    u = w_trace / h_sub
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5
    expected = c0 / (4.0 * stub_len * np.sqrt(eps_eff))

    assert abs(module.F_NOTCH_AN_HZ - expected) / expected < 1e-9
    # sanity: this repo's own recorded "fringing-free analytic 3.69 GHz"
    # (docs/agent-memory/rfx-known-issues.md, cv06b/msl_notch_e4 entries)
    assert 3.6e9 < module.F_NOTCH_AN_HZ < 3.8e9
    assert module.REPRODUCE_GATE_RECORD["expected_f_notch_an_hz"] == float(module.F_NOTCH_AN_HZ)


def test_stage_a_gate_band_is_one_sided_low_biased_and_contains_expected():
    module = _load_referee_module()
    gate = module.REPRODUCE_GATE_RECORD["gate"]
    assert gate["f_notch_lo_hz"] < module.F_NOTCH_AN_HZ < gate["f_notch_hi_hz"]
    # one-sided-low-biased: the low margin must be wider than the high
    # margin (precedent: openEMS reads notch frequency LOW vs analytic).
    lo_margin = module.F_NOTCH_AN_HZ - gate["f_notch_lo_hz"]
    hi_margin = gate["f_notch_hi_hz"] - module.F_NOTCH_AN_HZ
    assert lo_margin > hi_margin


def test_stage_a_matches_tutorial_constants():
    """Regression lock on MSL_NotchFilter.py's own literal parameters."""
    module = _load_referee_module()
    assert module.A_MSL_LENGTH_UM == 50000.0
    assert module.A_MSL_WIDTH_UM == 600.0
    assert module.A_SUBSTRATE_THICKNESS_UM == 254.0
    assert module.A_SUBSTRATE_EPR == 3.66
    assert module.A_STUB_LENGTH_UM == 12.0e3
    assert module.A_F_MAX_HZ == 7.0e9
    assert module.A_N_FREQS == 1601


# ---------------------------------------------------------------------------
# rfx fixture loading + layout arithmetic (openEMS-free)
# ---------------------------------------------------------------------------
def test_load_rfx_fixture_missing_raises_filenotfound(tmp_path):
    module = _load_referee_module()
    with pytest.raises(FileNotFoundError):
        module._load_rfx_fixture(str(tmp_path / "does_not_exist.json"))


def test_rfx_fixture_is_committed_and_has_required_schema():
    """The committed, regenerable rfx-side reference fixture must exist and
    carry the fields Stage B's layout/comparison code reads."""
    assert RFX_FIXTURE_PATH.exists(), (
        f"missing committed rfx fixture {RFX_FIXTURE_PATH} -- regenerate "
        f"with scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py"
    )
    fixture = json.loads(RFX_FIXTURE_PATH.read_text())
    for key in ("meta", "reference_plane_geometry", "freqs_hz", "s11", "s21", "beta_first_port"):
        assert key in fixture, f"rfx fixture missing top-level key {key!r}"
    geom = fixture["reference_plane_geometry"]
    for port_name in ("msl_0", "msl_1"):
        assert port_name in geom
        for field in ("feed_x_m", "direction", "probe0_x_m"):
            assert field in geom[port_name]
    assert "_grid" in geom
    grid = geom["_grid"]
    for field in ("dx", "pad_x_lo", "pad_x_hi", "pad_y_lo", "pad_z_hi", "rasterized_clear_m"):
        assert field in grid


def test_assert_matches_rfx_fixture_accepts_the_committed_fixture():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    module._assert_matches_rfx_fixture(fixture)  # must not raise


def test_assert_matches_rfx_fixture_rejects_drift():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    drifted = json.loads(json.dumps(fixture))
    drifted["meta"]["dx_m"] = fixture["meta"]["dx_m"] * 2.0
    with pytest.raises(AssertionError):
        module._assert_matches_rfx_fixture(drifted)


def test_stage_b_layout_is_self_consistent_and_openems_free():
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)  # must not raise

    required = {
        "lx_clear_m", "ly_clear_m", "lz_clear_m", "pml_x_m", "pml_y_m", "pml_z_hi_m",
        "feed_x_port0_m", "feed_x_port1_m", "probe0_x_port0_m", "probe0_x_port1_m",
        "ref_plane_shift_port0_m", "ref_plane_shift_port1_m", "l12_m",
    }
    assert required <= set(layout.keys())
    assert layout["feed_x_port0_m"] < layout["feed_x_port1_m"]
    assert layout["probe0_x_port0_m"] < layout["probe0_x_port1_m"]
    assert layout["l12_m"] > 0.0
    # ref_plane_shift is a positive magnitude for BOTH ports (mirror-
    # symmetric layout, module docstring CONVENTION NOTE part (c)).
    assert layout["ref_plane_shift_port0_m"] > 0.0
    assert layout["ref_plane_shift_port1_m"] > 0.0


def test_ref_plane_shift_transform_independently_rederived():
    """The (target - start) * direction transform, independently re-derived
    against a SYNTHETIC geometry dict (not the real fixture) -- proves the
    formula itself, not merely that it runs on today's fixture numbers.
    Matches the discipline the waveguide Lane-1 phase gate's own convention
    note used ('independently re-derived in review to 1.1e-14 deg').
    """
    module = _load_referee_module()
    synthetic_fixture = {
        "meta": {
            "eps_r": module.B_EPS_R, "h_sub_m": module.B_H_SUB_M,
            "w_trace_m": module.B_W_TRACE_M, "l_line_m": module.B_L_LINE_M,
            "port_margin_m": module.B_PORT_MARGIN_M, "dx_m": module.B_DX_M,
            "f_max_hz": module.B_F_MAX_HZ,
            # issue #723: _stage_b_layout requires these unconditionally
            # (a production fixture missing them raises KeyError rather
            # than silently falling back to the declared board -- see
            # that function's own docstring/comment). This test is about
            # the ref_plane_shift TRANSFORM, not the realized-geometry
            # values, so these are placeholders satisfying the contract.
            "h_sub_realized_m": module.B_H_SUB_M,
            "w_trace_realized_m": module.B_W_TRACE_M,
            "trace_y_lo_realized_m": 0.0,
            "trace_y_hi_realized_m": module.B_W_TRACE_M,
            "n_z_sub_realized": round(module.B_H_SUB_M / module.B_DX_M),
        },
        "reference_plane_geometry": {
            "msl_0": {"feed_x_m": 0.002, "direction": "+x", "probe0_x_m": 0.0045},
            "msl_1": {"feed_x_m": 0.012, "direction": "-x", "probe0_x_m": 0.0095},
            "_grid": {
                "dx": module.B_DX_M, "pad_x_lo": 8, "pad_x_hi": 8,
                "pad_y_lo": 8, "pad_y_hi": 8, "pad_z_lo": 0, "pad_z_hi": 8,
                "rasterized_clear_m": {"x": 0.01405, "y": 0.0025, "z": 0.00185},
            },
        },
    }
    layout = module._stage_b_layout(synthetic_fixture)
    # port0: direction +1, start=2.0mm, target=4.5mm -> shift=(4.5-2.0)*1=2.5mm
    assert abs(layout["ref_plane_shift_port0_m"] - 0.0025) < 1e-12
    # port1: direction -1, start=12.0mm, target=9.5mm -> shift=(9.5-12.0)*-1=2.5mm
    assert abs(layout["ref_plane_shift_port1_m"] - 0.0025) < 1e-12
    assert abs(layout["l12_m"] - 0.005) < 1e-12

    # Discrimination check: a sign error in the transform must NOT
    # coincidentally reproduce these exact numbers.
    wrong_shift_port1 = (0.0095 - 0.012) * 1.0  # forgetting the direction sign
    assert abs(wrong_shift_port1 - 0.0025) > 1e-6, (
        "sign-error variant coincidentally matches the correct shift -- "
        "this discrimination check is not testing anything"
    )


# ---------------------------------------------------------------------------
# Sanity-check helpers (scale-free excitation guard, passivity, non-physical,
# self-consistency witness) -- all openEMS-free, synthetic inputs.
# ---------------------------------------------------------------------------
def test_check_excitation_and_trace_is_scale_free_not_an_absolute_floor():
    """Mirrors the coax lane's own PR #547 lesson: no absolute floor, only
    exact-zero/non-finite is a defect signature."""
    module = _load_referee_module()

    class _FakePort:
        U_filenames: list = []

        def __init__(self, uf_inc):
            self.uf_inc = np.array([uf_inc + 0j])

    tiny_but_real = _FakePort(1e-14)
    peak, _ = module._check_excitation_and_trace(tiny_but_real, "/tmp/_unused", "tiny")
    assert peak == 1e-14

    for broken_value in (0.0, float("nan")):
        broken = _FakePort(broken_value)
        with pytest.raises(RuntimeError):
            module._check_excitation_and_trace(broken, "/tmp/_unused", "broken")

    assert not hasattr(module, "_EXCITATION_ENERGY_FLOOR")


# ---------------------------------------------------------------------------
# GUARD CHANNEL-GAP FIX tests (run-1 forensics, review 2026-08-04): the
# pre-solve fail-fast scanner's allowlist and truncation-pattern scoping,
# plus the D3 truncation-detection helper. Excerpts below are copied
# VERBATIM from the committed run-1 log (``validation/crossval/
# _20_msl_phase_referee_logs/20260804T070702Z_run.log``, lines 20-30) --
# not paraphrased -- per the #529 resolving-power discipline: a guard
# must be proven to discriminate on REAL data, not a plausible-looking
# synthetic.
# ---------------------------------------------------------------------------
_RUN1_LOG_PATH: Final = (
    REPO_ROOT / "validation" / "crossval" / "_20_msl_phase_referee_logs" / "20260804T070702Z_run.log"
)

# Stage B SMOKE portion (NrTS=200, EndCriteria=0.0/"-infdB" -- committed log
# lines 20-27): truncation strings legitimately fire here, by construction
# (a 200-timestep budget cannot hold this script's own excitation pulse).
_RUN1_SMOKE_LOG_EXCERPT = (
    "Operator::CalcGaussianPulsExcitation: Requested excitation pusle would "
    "be 43004 timesteps or 6.74077e-10 s long. Cutting to max number of "
    "timesteps!\n"
    "openEMS::SetupFDTD: Warning, the timestep seems to be very small --> "
    "long simulation. Check your mesh!?\n"
    "openEMS::SetupFDTD: Warning, max. number of timesteps is smaller than "
    "three times the excitation. \n"
    "\tYou may want to choose a higher number of max. timesteps... \n"
    "Warning: Unused primitive (type: Box) detected in property: port0_metal!\n"
    "Warning: Unused primitive (type: Box) detected in property: port1_metal!\n"
    "RunFDTD: Warning: Max. number of timesteps was reached before the "
    "end-criteria of -infdB was reached... \n"
    "\tYou may want to choose a higher number of max. timesteps... \n"
)

# Stage B REAL portion (NrTS=300000, EndCriteria=1e-4 -- committed log lines
# 28-30): reached its own EndCriteria; no truncation strings present.
_RUN1_REAL_LOG_EXCERPT = (
    "openEMS::SetupFDTD: Warning, the timestep seems to be very small --> "
    "long simulation. Check your mesh!?\n"
    "Warning: Unused primitive (type: Box) detected in property: port0_metal!\n"
    "Warning: Unused primitive (type: Box) detected in property: port1_metal!\n"
)


def test_run1_log_excerpts_are_verbatim_substrings_of_the_committed_log():
    """Ties the two excerpts above to the ACTUAL committed evidence file --
    if a future edit to the committed log drifts from these hardcoded
    strings, this test (not just the ones that use the excerpts) goes red."""
    assert _RUN1_LOG_PATH.exists(), f"missing committed run-1 log {_RUN1_LOG_PATH}"
    full_log = _RUN1_LOG_PATH.read_text()
    for excerpt in (_RUN1_SMOKE_LOG_EXCERPT, _RUN1_REAL_LOG_EXCERPT):
        for line in excerpt.splitlines():
            assert line.strip() in full_log, (
                f"excerpt line not found verbatim in committed log: {line!r}"
            )


def test_scan_stdout_allowlists_port_metal_unused_primitive():
    """M3 topology's own port0_metal/port1_metal 'Unused primitive' warning
    (see ``_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES``'s own docstring) must
    NOT trip the fail-fast gate -- feeding it the REAL run-1 excerpt (which
    contains exactly these two lines) must not raise."""
    module = _load_referee_module()
    module._scan_stdout_for_bad_patterns(_RUN1_REAL_LOG_EXCERPT, "positive_control_real")  # must not raise


def test_scan_stdout_still_raises_on_non_allowlisted_unused_primitive():
    """Discrimination check: the allowlist must be scoped to the EXACT two
    property names -- any OTHER 'Unused primitive' (e.g. a genuinely
    dropped conductor, property: substrate!) must still trip the gate."""
    module = _load_referee_module()
    bad_log = "Warning: Unused primitive (type: Box) detected in property: substrate!\n"
    with pytest.raises(RuntimeError) as excinfo:
        module._scan_stdout_for_bad_patterns(bad_log, "bad")
    assert "substrate" in str(excinfo.value)


def test_scan_stdout_truncation_patterns_scoped_to_real_run_only():
    """TRUNCATION PATTERNS fix: ``check_truncation=False`` (the smoke-call
    default) must NOT raise on the smoke excerpt's own legitimate
    truncation strings -- proving the exemption actually exempts. The SAME
    text with ``check_truncation=True`` (the real-call setting) MUST raise
    -- proving the pattern set actually discriminates, not just exists."""
    module = _load_referee_module()
    module._scan_stdout_for_bad_patterns(
        _RUN1_SMOKE_LOG_EXCERPT, "positive_control_smoke", check_truncation=False)  # must not raise

    with pytest.raises(RuntimeError) as excinfo:
        module._scan_stdout_for_bad_patterns(
            _RUN1_SMOKE_LOG_EXCERPT, "positive_control_smoke_as_real", check_truncation=True)
    assert "Cutting to max number of timesteps" in str(excinfo.value)


def test_scan_stdout_real_run1_excerpt_passes_truncation_check():
    """The positive control's OTHER half: run-1's own REAL portion (which
    genuinely reached its own EndCriteria) must pass ``check_truncation=
    True`` -- same binary, same stream, only the smoke/real split differs,
    per the module docstring's own claimed positive control."""
    module = _load_referee_module()
    module._scan_stdout_for_bad_patterns(
        _RUN1_REAL_LOG_EXCERPT, "positive_control_real_truncation_check", check_truncation=True)  # must not raise


# ---------------------------------------------------------------------------
# D2/D3 fix: _log_indicates_truncation resolving-power tests.
# ---------------------------------------------------------------------------
def test_log_indicates_truncation_flips_on_synthetic_under_settled_input():
    """#529 resolving-power pattern: the D3 replacement (the pre-fix
    ``n_trace_samples >= nrts`` comparison was structurally unreachable --
    a probe-trace ROW count can never approach the raw timestep BUDGET)
    must be PROVEN to flip True on a genuinely under-settled run's own log
    text, not just read False on run-1's own (converged) data."""
    module = _load_referee_module()
    assert module._log_indicates_truncation(_RUN1_REAL_LOG_EXCERPT) is False
    assert module._log_indicates_truncation(_RUN1_SMOKE_LOG_EXCERPT) is True

    synthetic_under_settled = (
        "RunFDTD: Warning: Max. number of timesteps was reached before the "
        "end-criteria of -30dB was reached... \n"
    )
    assert module._log_indicates_truncation(synthetic_under_settled) is True
    assert module._log_indicates_truncation("no such warning anywhere in this log\n") is False


def test_run_stage_a_and_stage_b_wire_truncated_from_log_not_from_counts():
    """Regression lock on the D3 wiring itself (not just the pure helper):
    ``_run_stage_a_reproduce_gate``'s ``truncated_suspected`` and
    ``_run_stage_b``'s ``truncated``/``end_criteria_not_reached`` must be
    SOURCED from ``_log_indicates_truncation``, not re-implemented -- pinned
    by reading the source rather than only exercising the pure function,
    since a future edit could reintroduce a parallel, independent (and
    possibly inconsistent) count-based check without this test noticing."""
    module = _load_referee_module()
    import inspect
    src_a = inspect.getsource(module._run_stage_a_reproduce_gate)
    src_b = inspect.getsource(module._run_stage_b)
    assert "_log_indicates_truncation(real_log)" in src_a
    assert "_log_indicates_truncation(real_log)" in src_b
    assert "n_samples >= nrts" not in src_b, (
        "the structurally-unreachable probe-row-count comparison (D3 "
        "regression) must not come back"
    )


def test_stage_a_passed_is_symmetric_with_stage_b_on_truncation():
    """#552 reviewer follow-up: pre-fix, ``_run_stage_a_reproduce_gate``
    computed ``truncated_suspected`` but never consulted it in ``passed``
    (``passed = bool(f_notch_ok)``) -- a truncated Stage A real run would
    still report ``passed=True``, unlike Stage B's own ``sanity_passed``,
    which DOES gate on ``truncated``. Pinned by reading the source (Stage
    A's real run defaults to NrTS~=1e9, so a live truncated run is
    unreachable in ordinary practice -- this test is a structural
    regression lock, not a claim that run-1 was ever affected)."""
    module = _load_referee_module()
    import inspect
    src_a = inspect.getsource(module._run_stage_a_reproduce_gate)
    assert "passed = bool(f_notch_ok and not truncated_suspected)" in src_a, (
        "Stage A's own passed must consult truncated_suspected, matching "
        "Stage B's sanity_passed -- symmetry regression"
    )


def test_non_physical_guard_raises_above_two():
    module = _load_referee_module()
    module._non_physical_guard(np.array([0.1, 0.9, 1.5]), "ok")  # must not raise
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, 2.5]), "bad")
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, float("nan")]), "nan")


def test_passivity_witness_raises_above_tolerance():
    module = _load_referee_module()
    s11 = np.array([0.1, 0.1])
    s21_ok = np.array([0.9, 0.9])
    result = module._passivity_witness(s11, s21_ok, "ok")
    assert result["passed"] is True

    s21_bad = np.array([1.5, 1.5])
    with pytest.raises(RuntimeError):
        module._passivity_witness(s11, s21_bad, "bad")


def test_self_consistency_witness_passes_on_a_synthetic_matched_wave():
    """Feed S21 = exp(-j*beta*L) (a perfectly matched, lossless, non-
    dispersive line by construction) with beta EXACTLY equal to what the
    witness is told -- must pass with ~zero phase deviation, proving the
    witness's own sign convention matches the module's stated
    E(x,t)=Re[exp(j(wt-beta*x))] forward-wave convention (CONVENTION NOTE
    part (a))."""
    module = _load_referee_module()
    freqs_hz = np.linspace(3.0e9, 4.5e9, 9)
    l12_m = 5.0e-3
    beta = 2.0 * np.pi * freqs_hz * np.sqrt(2.9) / 2.998e8
    s21 = np.exp(-1j * beta * l12_m)

    result = module._self_consistency_witness(
        freqs_hz, s21, beta, l12_m=l12_m, mag_band=(0.5, 1.1),
        phase_tol_deg=1.0, gd_tol_ps=50.0, label="synthetic",
    )
    assert result["passed"] is True
    assert result["max_phase_dev_deg"] < 1e-6


def test_self_consistency_witness_fails_on_wrong_sign_beta():
    """Discrimination check: flipping the SIGN of beta (the exact class of
    bug a reference-plane/convention error would introduce) must fail the
    witness, proving it actually catches a convention regression rather
    than passing unconditionally."""
    module = _load_referee_module()
    freqs_hz = np.linspace(3.0e9, 4.5e9, 9)
    l12_m = 5.0e-3
    beta = 2.0 * np.pi * freqs_hz * np.sqrt(2.9) / 2.998e8
    s21 = np.exp(-1j * beta * l12_m)

    with pytest.raises(RuntimeError):
        module._self_consistency_witness(
            freqs_hz, s21, -beta, l12_m=l12_m, mag_band=(0.5, 1.1),
            phase_tol_deg=1.0, gd_tol_ps=50.0, label="wrong_sign",
        )


# ---------------------------------------------------------------------------
# main() exit-code contract
# ---------------------------------------------------------------------------
def test_openems_unavailable_exits_2():
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("CSXCAD.CSXCAD", "openEMS.openEMS", "openEMS.ports", "CSXCAD", "openEMS"):
            raise ImportError("simulated: openEMS not installed")
        return real_import(name, *args, **kwargs)

    orig = builtins.__import__
    builtins.__import__ = fake_import
    try:
        rc = module.main(["--rfx-fixture", str(RFX_FIXTURE_PATH)])
    finally:
        builtins.__import__ = orig
    assert rc == 2


def test_missing_rfx_fixture_exits_3():
    module = _load_referee_module()
    rc = module.main(["--rfx-fixture", "/tmp/_definitely_missing_msl_phase_fixture.json"])
    assert rc == 3


def test_config_error_exits_3_distinct_from_physics_failure(monkeypatch):
    module = _load_referee_module()

    def _broken_layout(fixture):
        raise AssertionError("simulated: rfx geometry drifted")

    monkeypatch.setattr(module, "_stage_b_layout", _broken_layout)
    rc = module.main(["--rfx-fixture", str(RFX_FIXTURE_PATH)])
    assert rc == 3


def test_main_writes_valid_json_on_stage_b_physics_gate_failure(monkeypatch, tmp_path):
    """Forensics pattern (coax lane B3 fix): a Stage B physics-gate failure
    must still leave a JSON artifact a reviewer can open, carrying whatever
    partial data was computed before the guard tripped."""
    module = _load_referee_module()

    def fake_stage_a(*, sim_root, threads):
        return {
            "passed": True, "f_notch_hz": module.F_NOTCH_AN_HZ, "notch_depth_db": -30.0,
            "f_notch_expected_hz": module.F_NOTCH_AN_HZ, "f_notch_dev_pct": 0.0,
            "f_notch_ok": True, "max_uf_inc": 1e-12, "n_trace_samples": 100,
            "truncated_suspected": False, "elapsed_s": 1.0,
        }

    def fake_stage_b(*, sim_root, threads, nrts, end_criteria, rfx_fixture):
        exc = RuntimeError("[stage_b_s11] simulated non-physical field")
        exc.partial_stage_b_data = {
            "s11_mag": [300.0], "s21_mag": [0.9], "drive_diagnostics": {"max_uf_inc": 1e-12},
        }
        raise exc

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_stage_a_reproduce_gate", fake_stage_a)
    monkeypatch.setattr(module, "_run_stage_b", fake_stage_b)

    out_path = tmp_path / "out.json"
    rc = module.main(["--output", str(out_path), "--rfx-fixture", str(RFX_FIXTURE_PATH)])

    assert rc == 1, f"expected exit 1 (physics-gate failure), got {rc}"
    assert out_path.exists()

    artifact = json.loads(out_path.read_text())
    assert artifact["error"]
    assert "stage_b_partial" in artifact
    assert artifact["stage_b_partial"]["s11_mag"] == [300.0]


def test_main_writes_valid_json_on_overall_pass(monkeypatch, tmp_path):
    module = _load_referee_module()

    def fake_stage_a(*, sim_root, threads):
        return {
            "passed": True, "f_notch_hz": module.F_NOTCH_AN_HZ, "notch_depth_db": -30.0,
            "f_notch_expected_hz": module.F_NOTCH_AN_HZ, "f_notch_dev_pct": 0.0,
            "f_notch_ok": True, "max_uf_inc": 1e-12, "n_trace_samples": 100,
            "truncated_suspected": False, "elapsed_s": 1.0,
        }

    def fake_stage_b(*, sim_root, threads, nrts, end_criteria, rfx_fixture):
        return {
            "s21_mag": [0.95, 0.96], "s11_mag": [0.05, 0.06],
            "self_consistency_openems": {"passed": True},
            "self_consistency_rfx": {"passed": True},
            "sanity_passed": True,
        }

    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_run_stage_a_reproduce_gate", fake_stage_a)
    monkeypatch.setattr(module, "_run_stage_b", fake_stage_b)

    out_path = tmp_path / "out_pass.json"
    rc = module.main(["--output", str(out_path), "--rfx-fixture", str(RFX_FIXTURE_PATH)])

    assert rc == 0
    artifact = json.loads(out_path.read_text())
    assert artifact["overall_passed"] is True
    assert artifact["stage_b"]["sanity_passed"] is True


def test_freqs_grid_shared_with_rfx_fixture_no_interpolation():
    """Stage B must evaluate openEMS's own CalcPort at the EXACT frequency
    grid the committed rfx fixture used (module docstring CONVENTION NOTE
    'UNWRAPPING') -- pinned here by checking the fixture's own freqs_hz
    against the constants Stage B is built from (F_MAX/N points match)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    freqs = np.asarray(fixture["freqs_hz"])
    assert freqs[-1] == pytest.approx(module.B_F_MAX_HZ, rel=1e-9)
    assert freqs.min() > 0.0
    assert np.all(np.diff(freqs) > 0), "fixture freqs_hz must be strictly increasing"


# ---------------------------------------------------------------------------
# M1 review fix: gate must be PROVEN to discriminate a realistic referral
# defect (the #529 resolving-power pattern), not just pass a clean run.
# ---------------------------------------------------------------------------
def _rfx_fixture_s21_beta_l12(module):
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    s21 = np.asarray([complex(re, im) for re, im in fixture["s21"]], dtype=np.complex128)
    beta = np.asarray([complex(re, im) for re, im in fixture["beta_first_port"]], dtype=np.complex128)
    layout = module._stage_b_layout(fixture)
    return freqs, s21, beta, layout


def test_gate_budget_is_derived_not_a_round_number():
    """Regression lock on the M1 fix itself: the phase gate must be the
    derived, tight value (3 deg), not the pre-review 30 deg (which admitted
    a 3.372mm/67%-of-L12 plane error and let a real referral-drop defect
    pass -- see the module docstring 'GATE-BUDGET DERIVATION')."""
    module = _load_referee_module()
    assert module.B_PHASE_TOL_DEG == pytest.approx(3.0)
    assert module.B_PHASE_TOL_DEG < 10.0, (
        "phase gate drifted back toward the refuted 30 deg round number"
    )


def test_self_consistency_witness_resolving_power_single_port_referral_drop():
    """The #529 resolving-power pattern: a gate must be PROVEN to
    discriminate a real defect on committed data, not just documented as
    doing so. Plants a synthetic single-port referral drop -- S21 rotated
    by exp(+j*beta*ref_shift) for ONE port's own ref_plane_shift (2.5mm on
    this fixture) -- on the REAL committed rfx fixture's own S21/beta, and
    asserts:
      (a) the UNPERTURBED fixture passes the current B_PHASE_TOL_DEG gate
          with wide margin (measured max_phase_dev_deg ~0.12, ~25x under
          the 3 deg gate);
      (b) the PERTURBED (defect-planted) fixture REDS the same gate.
    This is what makes "the gate resolves a referral-plane error of this
    size" a MEASURED claim, not an assertion -- the pre-review 30 deg gate
    passed this exact perturbation (measured ~16-22 deg, well under 30),
    which is the defect this review caught (module docstring 'GATE-BUDGET
    DERIVATION').
    """
    module = _load_referee_module()
    freqs, s21, beta, layout = _rfx_fixture_s21_beta_l12(module)

    # (a) nominal fixture: must PASS with real margin under the current gate.
    result_nominal = module._self_consistency_witness(
        freqs, s21, beta, l12_m=layout["l12_m"], mag_band=module.B_S21_MAG_BAND,
        phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
        label="resolving_power_nominal",
    )
    assert result_nominal["passed"] is True
    assert result_nominal["max_phase_dev_deg"] < module.B_PHASE_TOL_DEG / 5.0, (
        f"nominal fixture's own phase deviation "
        f"({result_nominal['max_phase_dev_deg']:.3f} deg) does not leave "
        f"the expected wide margin under the gate ({module.B_PHASE_TOL_DEG} deg) "
        f"-- the resolving-power claim needs BOTH a passing nominal case "
        f"AND a reding perturbed case to mean anything"
    )

    # (b) planted defect: ONE port's own ref_plane_shift (2.5mm) is
    # rotated back in as if it had never been applied.
    ref_shift_one_port_m = layout["ref_plane_shift_port0_m"]
    beta_re = np.real(beta)
    s21_perturbed = s21 * np.exp(1j * beta_re * ref_shift_one_port_m)

    with pytest.raises(RuntimeError) as excinfo:
        module._self_consistency_witness(
            freqs, s21_perturbed, beta, l12_m=layout["l12_m"], mag_band=module.B_S21_MAG_BAND,
            phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
            label="resolving_power_planted_defect",
        )
    assert "self-consistency witness failed" in str(excinfo.value)

    # Measure (not just assert-raises) the actual deviation the planted
    # defect produces, over the gate band, so the docstring's own cited
    # "15.88-22.21 deg" claim is a checked number, not folklore (was written
    # "15.99-22.33" -- the round-2 review re-measured 15.88 / 22.21).
    mask = (freqs >= module.B_GATE_F_LO_HZ) & (freqs <= module.B_GATE_F_HI_HZ)
    expected_phase = -beta_re * layout["l12_m"]
    measured_phase = np.unwrap(np.angle(s21_perturbed))
    phase_dev_deg = np.degrees(np.angle(np.exp(1j * (measured_phase - expected_phase))))
    max_dev = float(np.max(np.abs(phase_dev_deg[mask])))
    assert 10.0 < max_dev < 35.0, (
        f"planted single-port referral-drop deviation ({max_dev:.2f} deg) "
        f"fell outside the expected ballpark (10-35 deg) -- re-derive the "
        f"docstring's own cited numbers if this genuinely moved"
    )
    assert max_dev > 5.0 * module.B_PHASE_TOL_DEG, (
        f"planted defect ({max_dev:.2f} deg) is not comfortably (5x) past "
        f"the gate ({module.B_PHASE_TOL_DEG} deg) -- the resolving-power "
        f"margin this docstring claims ('~7-10x') would not hold"
    )


def test_gd_gate_is_honestly_inert_for_the_referral_drop_class():
    """M1 honesty fix: the group-delay gate (200ps) must NOT catch the same
    single-port referral-drop defect the phase gate catches -- pins the
    docstring's own 'GROUP-DELAY GATE HONESTY' claim (~14ps for a
    single-port 2.5mm drop) as a checked number instead of prose."""
    module = _load_referee_module()
    freqs, s21, beta, layout = _rfx_fixture_s21_beta_l12(module)

    ref_shift_one_port_m = layout["ref_plane_shift_port0_m"]
    beta_re = np.real(beta)
    s21_perturbed = s21 * np.exp(1j * beta_re * ref_shift_one_port_m)

    omega = 2.0 * np.pi * freqs
    mask = (freqs >= module.B_GATE_F_LO_HZ) & (freqs <= module.B_GATE_F_HI_HZ)
    expected_phase = -beta_re * layout["l12_m"]
    measured_phase = np.unwrap(np.angle(s21_perturbed))
    gd_measured = -np.gradient(measured_phase, omega)
    gd_expected = -np.gradient(expected_phase, omega)
    gd_dev_ps = np.abs(gd_measured[mask] - gd_expected[mask]) * 1e12

    assert float(np.max(gd_dev_ps)) < module.B_GD_TOL_PS, (
        "the planted single-port referral-drop defect unexpectedly TRIPPED "
        "the group-delay gate -- the docstring's inertness claim is wrong, "
        "fix the prose (or, if this is now doing useful work, say so)"
    )
    # Analytic cross-check: dl*sqrt(eps_eff)/c0 for this substrate.
    eps_eff = (module.B_EPS_R + 1.0) / 2.0 + (module.B_EPS_R - 1.0) / 2.0 * (
        1.0 + 12.0 / (module.B_W_TRACE_M / module.B_H_SUB_M)
    ) ** -0.5
    analytic_gd_ps = ref_shift_one_port_m * np.sqrt(eps_eff) / 2.998e8 * 1e12
    assert abs(float(np.mean(gd_dev_ps)) - analytic_gd_ps) < 1.0, (
        f"measured group-delay deviation ({float(np.mean(gd_dev_ps)):.2f} ps) "
        f"does not match the analytic dl*sqrt(eps_eff)/c0 prediction "
        f"({analytic_gd_ps:.2f} ps) -- the docstring's own physical "
        f"explanation would be wrong"
    )


# ---------------------------------------------------------------------------
# m6 review fix: substrate-top mesh line + rasterized cell count.
# ---------------------------------------------------------------------------
def test_stage_b_substrate_z_mesh_rasterized_cell_count():
    """#325 class: the substrate's ACTUAL rasterized cell count must be
    asserted, not left to an unguided uniform arange's arbitrary rounding.
    At this fixture's own h_sub=254um, dx=50um, 254/50=5.08 -> round=5."""
    module = _load_referee_module()
    z_lines, n_sub = module._stage_b_substrate_z_mesh(254.0, 50.0)  # um units
    assert n_sub == 5
    assert z_lines[0] == pytest.approx(0.0)
    assert z_lines[-1] == pytest.approx(254.0)
    assert len(z_lines) == n_sub + 1
    # Cell count must be derived from h_sub/dx, not hard-coded independent
    # of the inputs -- discrimination check with a different dx.
    _, n_sub_alt = module._stage_b_substrate_z_mesh(254.0, 80.0)
    assert n_sub_alt == 3  # round(254/80) = round(3.175) = 3
    assert n_sub_alt != n_sub


def test_build_stage_b_asserts_substrate_cell_count_against_realized_board():
    """The wiring inside _build_stage_b (not just the pure helper above)
    must cross-check n_sub against rfx's OWN realized substrate cell
    count -- pinned by reading the source, since a future edit could call
    the helper with the wrong arguments and still "pass" the isolated
    unit test above.

    issue #723 (2026-08-27): the assertion target moved from
    ``round(B_H_SUB_M / B_DX_M)`` (the DECLARED-board count, 5) to
    ``layout["n_z_sub_realized"]`` (rfx's REALIZED-board count, 6). The
    declared target would fail by construction post-fix even when Stage B
    and rfx agree with each other, since h_sub is now sourced from the
    fixture. This test pins the NEW target and rejects the old one -- an
    earlier revision accepted either via a bare ``"n_sub =="`` substring,
    which would have passed a silent revert to the declared board (#723
    review). The two targets are shown to be distinguishable below, so
    the pin is not vacuous."""
    module = _load_referee_module()
    import inspect
    src = inspect.getsource(module._build_stage_b)
    assert "_stage_b_substrate_z_mesh" in src
    assert 'n_sub == layout["n_z_sub_realized"]' in src, (
        "Stage B must assert its rasterized substrate cell count against "
        "the fixture's realized count, not a declared-board constant")
    assert "n_sub == round(B_H_SUB_M / B_DX_M)" not in src

    # Not vacuous: the two candidate targets genuinely differ on this
    # fixture, so asserting against the wrong one would fire.
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    _, n_realized = module._stage_b_substrate_z_mesh(
        layout["h_sub_realized_m"], module.B_DX_M)
    _, n_declared = module._stage_b_substrate_z_mesh(
        module.B_H_SUB_M, module.B_DX_M)
    assert n_realized == layout["n_z_sub_realized"] == 6
    assert n_declared == round(module.B_H_SUB_M / module.B_DX_M) == 5
    assert n_realized != n_declared


# ---------------------------------------------------------------------------
# M3/m5/NEW-1/NEW-3 review fixes: Stage B's trace span and MeasPlaneShift
# targets are now sourced from the PURE, openEMS-free
# _stage_b_port_placement -- _build_stage_b calls it directly (wired, not
# duplicated), so these are NUMERIC assertions on the actual values used,
# not inspect.getsource string-counting (the NEW-3 fix).
# ---------------------------------------------------------------------------
def test_stage_b_port_placement_trace_span_is_feed_to_feed_not_pml():
    """M3 fix, numeric: the trace span must be EXACTLY
    (feed_x_port0_m, feed_x_port1_m) -- NOT the pre-review version's
    domain-edge (PML-side) x0/x1 coordinates (the parallel-impedance
    defect this review caught: Feed_R=50 in parallel with a PML-matched
    continuation, predicted |S11|~=0.33)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert placement["trace_x_start_m"] == layout["feed_x_port0_m"]
    assert placement["trace_x_stop_m"] == layout["feed_x_port1_m"]
    # Must NOT be the PML-side domain-edge coordinates (x0 = -pml_x_m,
    # x1 = lx_clear_m + pml_x_m) -- a numeric, not textual, guard against
    # the M3 regression class.
    x0_pml_side = -layout["pml_x_m"]
    x1_pml_side = layout["lx_clear_m"] + layout["pml_x_m"]
    assert placement["trace_x_start_m"] != pytest.approx(x0_pml_side)
    assert placement["trace_x_stop_m"] != pytest.approx(x1_pml_side)
    # Sanity: the trace sits strictly inside the clear (non-PML) region.
    assert 0.0 < placement["trace_x_start_m"] < placement["trace_x_stop_m"] < layout["lx_clear_m"]


def test_stage_b_port_placement_measplaneshift_is_explicit_and_far_from_feed():
    """m5 fix, numeric: MeasPlaneShift's own TARGET must equal this
    fixture's own ref_plane_shift (2.5mm) -- NOT the class default (half
    the port's own 300um span, 150um/3 cells from the excitation+Feed_R,
    near-field contaminated)."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert placement["measplane_shift_target_port0_m"] == placement["ref_plane_shift_port0_m"]
    assert placement["measplane_shift_target_port1_m"] == placement["ref_plane_shift_port1_m"]
    # Comfortably (>3x) past the near-field-contaminated class-default
    # radius (half the port's own 300um span = 150um).
    half_port_span_m = 0.5 * module.B_PORT_W_CELLS * module.B_DX_M
    assert placement["measplane_shift_target_port0_m"] > 3.0 * half_port_span_m
    assert placement["measplane_shift_target_port1_m"] > 3.0 * half_port_span_m


def test_predict_measplane_snap_matches_uniform_grid_rounding():
    """_predict_measplane_snap must round to the nearest INTEGER multiple
    of dx -- pinned with both an exact (on-grid) and an off-grid target so
    the rounding behavior itself is checked, not just one lucky case."""
    module = _load_referee_module()
    dx = 50e-6
    assert module._predict_measplane_snap(2.5e-3, dx) == pytest.approx(2.5e-3)  # exact: 50 cells
    assert module._predict_measplane_snap(2.51e-3, dx) == pytest.approx(2.5e-3)  # rounds down
    assert module._predict_measplane_snap(2.53e-3, dx) == pytest.approx(2.55e-3)  # rounds up


def test_stage_b_port_placement_effective_shift_is_zero_by_construction():
    """NEW-1 fix (the one that matters, review 2026-08-04): on the
    COMMITTED fixture, the predicted effective CalcPort shift
    (ref_plane_shift - predicted measplane_shift) must be (numerically)
    ZERO -- proving, not merely asserting in prose, that the referral
    ROTATION is a no-op here and the referee's own phase-plane
    correctness comes from MeasPlaneShift's STENCIL PLACEMENT instead.
    See DECLARED_QUESTION / _stage_b_port_placement's own docstring for
    the full, non-overclaiming statement this test backs."""
    module = _load_referee_module()
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    layout = module._stage_b_layout(fixture)
    placement = module._stage_b_port_placement(layout)

    assert abs(placement["effective_shift_predicted_port0_m"]) < 1e-12
    assert abs(placement["effective_shift_predicted_port1_m"]) < 1e-12

    # Discrimination check: an OFF-GRID target (not this fixture's own
    # 2.5mm, which lands exactly on a mesh line) must NOT read zero --
    # proving this test can actually tell "no-op" from "real rotation".
    synthetic_layout = dict(layout)
    synthetic_layout["feed_x_port0_m"] = 0.002
    synthetic_layout["ref_plane_shift_port0_m"] = 2.517e-3  # deliberately off-grid
    off_grid_placement = module._stage_b_port_placement(synthetic_layout)
    assert abs(off_grid_placement["effective_shift_predicted_port0_m"]) > 1e-6, (
        "off-grid target unexpectedly read zero shift -- this test is not discriminating"
    )


def test_build_stage_b_asserts_port_start_matches_feed_x():
    """n8 fix: the silent coupling between _stage_b_layout's ref_plane_
    shift derivation (which assumes start[prop]==feed_x) and the actual
    port construction, ~250 lines away, must be an explicit assertion."""
    module = _load_referee_module()
    import inspect
    src = inspect.getsource(module._build_stage_b)
    assert "port0.start[0]" in src and "port1.start[0]" in src, (
        "expected explicit start[prop]==feed_x assertions on both ports (n8 fix)"
    )


# ---------------------------------------------------------------------------
# RUN-1 REGRESSION FIXTURE (2026-08-04, VESSL 369367251705): coax-lane
# precedent (tests/crossval/test_coax_two_port_referee_header.py's own committed
# run-3 forensics block), adapted here to load an ACTUAL committed JSON
# fixture (``validation/crossval/_20_msl_phase_referee_logs/
# 20260804T055009Z_result.json``, the full run-1 artifact, copied verbatim
# from the primary checkout) rather than re-typing arrays as Python
# literals -- avoiding a second, possibly-drifting hand transcription of
# the same numbers. Every assertion below reads a value the fixture
# actually contains; nothing here is asserted independently of it.
# ---------------------------------------------------------------------------
RUN1_RESULT_PATH: Final = (
    REPO_ROOT / "validation" / "crossval" / "_20_msl_phase_referee_logs" / "20260804T055009Z_result.json"
)


def _load_run1_result() -> dict:
    assert RUN1_RESULT_PATH.exists(), f"missing committed run-1 result {RUN1_RESULT_PATH}"
    return json.loads(RUN1_RESULT_PATH.read_text())


def test_run1_result_is_committed_and_matches_the_filled_reproduce_gate_record():
    """The committed fixture and the script's own FILLED REPRODUCE_GATE_
    RECORD must agree -- a drift here would mean the record was filled from
    a DIFFERENT run than the one whose evidence is actually committed."""
    module = _load_referee_module()
    result = _load_run1_result()
    record = module.REPRODUCE_GATE_RECORD
    assert record["status"] == "RUN"
    assert result["stage_a"]["f_notch_hz"] == record["reproduced_f_notch_hz"]
    assert result["stage_a"]["f_notch_dev_pct"] == record["reproduced_f_notch_dev_pct"]
    assert result["overall_passed"] is True


def test_run1_stage_a_summary_matches_the_fixture():
    result = _load_run1_result()
    sa = result["stage_a"]
    assert sa["f_notch_hz"] == 3671100625.0
    assert sa["f_notch_dev_pct"] == 0.4364433837294213
    assert sa["f_notch_expected_hz"] == 3687193135.4851503
    assert sa["passed"] is True
    assert sa["truncated_suspected"] is False


def test_run1_stage_b_headline_facts_computed_from_the_fixture():
    """Assert the headline facts as COMPUTED FROM the fixture -- not
    independently declared -- per the task's own 'do not assert anything
    the fixture doesn't contain' discipline."""
    result = _load_run1_result()
    sb = result["stage_b"]
    csr = sb["cross_solver_report"]
    raw = csr["raw_phase_diff_deg"]
    assert len(raw) == 30

    n_le_1 = sum(1 for x in raw if abs(x) <= 1.0)
    idx_gt_3 = [i for i, x in enumerate(raw) if abs(x) > 3.0]
    assert n_le_1 == 22, f"expected 22/30 bins <=1 deg, got {n_le_1}"
    assert idx_gt_3 == [0], f"expected only bin 0 to exceed 3 deg, got {idx_gt_3}"

    # Monotonic-decay signature over the first three bins.
    assert abs(raw[0]) > abs(raw[1]) > abs(raw[2])

    # Top bin (highest frequency, 5.0 GHz): |raw| < 0.2 deg.
    assert abs(raw[-1]) < 0.2

    # Conjugate-discriminator: the openEMS-side unwrapped S21 phase has
    # travelled a LOT over the band (>50 deg one-way, so >100 deg for
    # 2x) at the top bin, while the cross-solver raw phase DIFFERENCE at
    # that SAME bin stays under 0.2 deg -- ruling out a trivial "both
    # near zero" false-agreement (a conjugated/flipped convention on one
    # side would instead show up as roughly 2x the accumulated phase in
    # the diff, not a near-zero one).
    unwrapped_top_deg = math.degrees(sb["s21_phase_rad_unwrapped"][-1])
    assert 2.0 * abs(unwrapped_top_deg) > 100.0
    assert abs(raw[-1]) < 0.2

    assert sb["sanity_passed"] is True
    assert sb["passivity"]["passed"] is True
    assert sb["self_consistency_openems"]["passed"] is True
    assert sb["self_consistency_rfx"]["passed"] is True


def test_run1_implied_plane_error_sign_flip():
    """Reviewer's sharper attribution test: the implied reference-plane
    error ``Delta_d = Delta_phi / beta`` a CONSTANT plane-position defect
    would require, computed independently here from ``raw_phase_diff_deg``
    and ``beta_openems_real`` (both loaded straight from the fixture, not
    re-derived from anything else in this file) -- must show a SIGN FLIP
    across the band, with bin 0's own magnitude large (>1000 um). A
    genuine single constant-offset referral defect could not produce
    either fact (module docstring 'REPORTED (not gated) cross-solver
    comparison')."""
    result = _load_run1_result()
    sb = result["stage_b"]
    csr = sb["cross_solver_report"]
    raw_rad = [math.radians(x) for x in csr["raw_phase_diff_deg"]]
    beta = csr["beta_openems_real"]
    dd_um = [1e6 * (phi / b) for phi, b in zip(raw_rad, beta)]

    assert abs(dd_um[0]) > 1000.0, f"expected bin0 implied plane error > 1000 um, got {dd_um[0]:.1f}"
    assert (dd_um[0] > 0) != (dd_um[-1] > 0), (
        f"expected a sign flip between bin0 ({dd_um[0]:.1f} um) and bin29 "
        f"({dd_um[-1]:.1f} um) -- same sign would be consistent with a "
        f"genuine constant-offset referral defect, weakening the "
        f"attribution this test backs"
    )


def test_run1_beta_ratio_effective_shift_and_passivity_balance():
    """Pins beta_ratio_rfx_over_openems, effective_shift_real (both ports),
    and the full passivity balance array to the committed fixture."""
    result = _load_run1_result()
    sb = result["stage_b"]
    csr = sb["cross_solver_report"]

    beta_ratio = csr["beta_ratio_rfx_over_openems"]
    assert len(beta_ratio) == 30
    assert all(1.0 < r < 1.02 for r in beta_ratio), (
        f"beta ratio out of the recorded ~0.3-0.8% band: "
        f"min={min(beta_ratio)} max={max(beta_ratio)}"
    )

    port_info = sb["port_info"]
    assert abs(port_info["effective_shift_real_port0_m"]) < 1e-12
    assert abs(port_info["effective_shift_real_port1_m"]) < 1e-12

    balance = sb["passivity"]["balance"]
    assert len(balance) == 30
    assert max(balance) == sb["passivity"]["max_balance"]
    assert sb["passivity"]["max_balance"] < 1.05

    s21_mag_rfx = csr["s21_mag_rfx"]
    s21_mag_openems = csr["s21_mag_openems"]
    assert len(s21_mag_rfx) == 30 and len(s21_mag_openems) == 30
    assert all(0.99 < x < 1.0 for x in s21_mag_rfx)
    assert all(0.99 < x < 1.02 for x in s21_mag_openems)


def test_run1_measured_precision_s21_bias_and_balance_attribution():
    """Pins the module docstring's own 'MEASURED PRECISION' paragraph
    numbers (D-review addition): openEMS's own Stage B |S21| carries a
    small systematic bias above unity, and at bin 0 most of the passivity
    balance's excess over 1.0 traces to |S21|^2-1, not |S11|^2."""
    result = _load_run1_result()
    sb = result["stage_b"]
    s21_mag = sb["s21_mag"]
    s11_mag = sb["s11_mag"]
    balance = sb["passivity"]["balance"]
    freqs = sb["freqs_hz"]

    n_gt1 = sum(1 for x in s21_mag if x > 1.0)
    assert n_gt1 == 29

    gated = [x for f, x in zip(freqs, s21_mag) if 3.0e9 <= f <= 4.5e9]
    assert min(gated) == pytest.approx(1.0007550003423686)
    assert max(gated) == pytest.approx(1.0013482202192805)
    assert max(s21_mag) == pytest.approx(1.0087176793781167)
    assert s21_mag.index(max(s21_mag)) == 0

    excess0 = balance[0] - 1.0
    s21_sq_minus1 = s21_mag[0] ** 2 - 1.0
    s11_sq = s11_mag[0] ** 2
    assert s11_sq + s21_mag[0] ** 2 == pytest.approx(balance[0])
    frac = s21_sq_minus1 / excess0
    assert 0.73 < frac < 0.75, f"expected ~74% of balance[0] excess from |S21|^2-1, got {frac:.3f}"


# ---------------------------------------------------------------------------
# Issue #812 audit pattern P1 -- the self-referential phase gate, and the two
# independent-reference legs that replace its blind spot.
#
# Every measurement below REPLAYS committed artifacts through the real
# witness functions; openEMS is absent from this host, so no fresh run is
# available and none is needed. Two independent committed configurations are
# used, not one:
#   run-1  20260804T055009Z_result.json  -- the DECLARED board (openEMS
#          substrate 5 x 50.8um = 254um), VESSL 369367251705
#   run-2  20260827T102342Z_result.json  -- the #723 REALIZED board (openEMS
#          substrate 6 x 50.0um = 300um), VESSL 369367256520
# The rfx side is the SAME committed fixture in both, so its own realized
# board (300um) is the analytic reference for it in both.
# ---------------------------------------------------------------------------
_LOGS_DIR: Final = REPO_ROOT / "validation" / "crossval" / "_20_msl_phase_referee_logs"
_RUN1_RESULT_PATH: Final = _LOGS_DIR / "20260804T055009Z_result.json"
_RUN2_RESULT_PATH: Final = _LOGS_DIR / "20260827T102342Z_result.json"
_EVIDENCE_PATH: Final = (REPO_ROOT / "validation" / "crossval" / "_issue812_phase_identity"
                         / "regate_evidence.json")


def _cx(pairs):
    return np.array([complex(re, im) for re, im in pairs], dtype=np.complex128)


def _rfx_realized_eps_eff(module):
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    return module._hammerstad_jensen_eps_eff(
        fixture["meta"]["w_trace_realized_m"],
        fixture["meta"]["h_sub_realized_m"],
        module.B_EPS_R,
    )


def test_inlined_hammerstad_jensen_matches_rfx_microstrip():
    """The referee may not import rfx (module docstring SCOPE FENCE), so the
    closed form is duplicated here. Pin the duplicate against rfx's own
    implementation so the two cannot drift -- the duplication is the price
    of the fence, not a licence for a second, different formula."""
    module = _load_referee_module()
    from rfx.microstrip import microstrip_eps_eff

    for w, h in ((600e-6, 300e-6), (600e-6, 254e-6), (635e-6, 300e-6), (1e-3, 200e-6)):
        assert module._hammerstad_jensen_eps_eff(w, h, module.B_EPS_R) == pytest.approx(
            microstrip_eps_eff(w, h, module.B_EPS_R), rel=1e-12)

    # Stage A's own long-standing inline constant must be the same formula.
    assert module._A_EPS_EFF == pytest.approx(
        module._hammerstad_jensen_eps_eff(module._A_W_TRACE_M, module._A_H_SUB_M,
                                          module._A_EPS_R), rel=1e-12)


def test_external_phase_reference_budgets_are_recomputable_from_geometry():
    """Both new tolerances must be re-derivable from the board's DECLARED
    geometry alone -- that is what makes them pre-declarable. Every term in
    ``EXTERNAL_PHASE_REFERENCE_PREDECLARATION`` is recomputed here from the
    realized board, and the declared tolerance must SIT ABOVE the sum (a
    tolerance below its own budget would be a fitted number wearing a
    derivation)."""
    module = _load_referee_module()
    pre = module.EXTERNAL_PHASE_REFERENCE_PREDECLARATION
    er, w, h, t_cond = module.B_EPS_R, 600e-6, 300e-6, module.B_DX_M
    eps0 = module._hammerstad_jensen_eps_eff(w, h, er)

    # (i) Hammerstad-Jensen model accuracy, 1% in eps_eff -> 0.5% in beta.
    assert pre["analytic_tol_budget_frac"]["hammerstad_jensen_model"] == pytest.approx(
        0.5 * 0.01, abs=1e-6)
    # (ii) Bahl-Garg one-cell conductor thickness. The dict records the
    # budget terms to 3 significant figures (0.0121 vs the exact
    # 0.0120290), so compare at that resolution -- the DECLARED tolerance
    # asserted below is the number that binds, and it is unaffected.
    d_eps_t = -(er - 1.0) * (t_cond / h) / (4.6 * math.sqrt(w / h))
    assert abs(0.5 * d_eps_t / eps0) == pytest.approx(
        pre["analytic_tol_budget_frac"]["conductor_thickness_one_cell"], abs=1e-4)
    # (iii) Getsinger dispersion at the band top.
    u = w / h
    z0 = 120.0 * math.pi / (math.sqrt(eps0) * (u + 1.393 + 0.667 * math.log(u + 1.444)))
    f_p = z0 / (2.0 * 4.0e-7 * math.pi * h)
    g = 0.6 + 0.009 * z0
    eps_f = er - (er - eps0) / (1.0 + g * (module.B_GATE_F_HI_HZ / f_p) ** 2)
    assert 0.5 * (eps_f - eps0) / eps0 == pytest.approx(
        pre["analytic_tol_budget_frac"]["quasi_static_dispersion_at_band_top"], abs=5e-5)

    budget = pre["analytic_tol_budget_frac"]
    assert budget["sum"] == pytest.approx(
        budget["hammerstad_jensen_model"] + budget["conductor_thickness_one_cell"]
        + budget["quasi_static_dispersion_at_band_top"], abs=2e-4)
    assert module.B_BETA_ANALYTIC_TOL_FRAC >= budget["sum"]
    assert module.B_BETA_ANALYTIC_TOL_FRAC == 0.020

    # Cross-solver budget: two one-cell rasterization differences plus this
    # file's own committed +-4-cell reference-plane term.
    beta_max = 155.92
    l12 = 5.0e-3
    dh = abs(0.5 * (module._hammerstad_jensen_eps_eff(w, h - module.B_DX_M, er) - eps0) / eps0)
    dw = abs(0.5 * (module._hammerstad_jensen_eps_eff(w - module.B_DX_M, h, er) - eps0) / eps0)
    xs = pre["cross_solver_tol_budget_deg"]
    assert math.degrees(dh * beta_max * l12) == pytest.approx(xs["h_sub_one_cell"], abs=2e-3)
    assert math.degrees(dw * beta_max * l12) == pytest.approx(xs["w_trace_one_cell"], abs=2e-3)
    assert xs["reference_plane_four_cells"] == 1.787  # the file's own GATE-BUDGET term
    assert xs["sum"] == pytest.approx(
        xs["h_sub_one_cell"] + xs["w_trace_one_cell"] + xs["reference_plane_four_cells"],
        abs=2e-3)
    assert module.B_CROSS_SOLVER_PHASE_TOL_DEG >= xs["sum"]
    assert module.B_CROSS_SOLVER_PHASE_TOL_DEG == 3.0


def test_self_consistency_witness_is_blind_to_a_factor_two_phase_velocity_error():
    """The audit's own measurement, pinned as a permanent record of WHY the
    two independent witnesses exist.

    The audit reported 0.2414 deg for a factor-2 phase-velocity error
    against this file's 3.0 deg gate. That number comes from SCALING the
    de-embedded phase (which scales the extraction residual with it); the
    cleaner model -- rotate the through path by the extra propagation term
    and scale beta to match -- leaves the deviation BIT-IDENTICAL to its
    unperturbed value, 0.1207 deg. Both are pinned. Neither is a matter of
    tolerance: this witness's resolving power for the coherent-beta class
    is zero at ANY tolerance.
    """
    module = _load_referee_module()
    freqs, s21, beta, layout = _rfx_fixture_s21_beta_l12(module)
    l12 = layout["l12_m"]

    baseline = module._self_consistency_witness(
        freqs, s21, beta, l12_m=l12, mag_band=module.B_S21_MAG_BAND,
        phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
        label="baseline")
    assert baseline["max_phase_dev_deg"] == pytest.approx(0.12072012657087564, rel=1e-9)
    assert baseline["evidence_level"] == "E1 (intra-run self-consistency)"

    # (a) the audit's construction: scale the de-embedded phase itself.
    s21_scaled = np.abs(s21) * np.exp(2.0j * np.angle(s21))
    audit = module._self_consistency_witness(
        freqs, s21_scaled, beta * 2.0, l12_m=l12, mag_band=module.B_S21_MAG_BAND,
        phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
        label="audit_construction")
    assert audit["passed"] is True
    assert audit["max_phase_dev_deg"] == pytest.approx(0.2414, abs=5e-4)

    # (b) the propagation-only construction: deviation cannot move at all.
    for k in (2.0, 0.5):
        s21_k = s21 * np.exp(-1j * (k - 1.0) * np.real(beta) * l12)
        res = module._self_consistency_witness(
            freqs, s21_k, beta * k, l12_m=l12, mag_band=module.B_S21_MAG_BAND,
            phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
            label=f"coherent_k{k}")
        assert res["passed"] is True, k
        assert res["max_phase_dev_deg"] == pytest.approx(
            baseline["max_phase_dev_deg"], rel=1e-9), k
        assert res["group_delay_dev_ps"] == pytest.approx(
            baseline["group_delay_dev_ps"], rel=1e-9), k


def test_dispersion_corrected_residual_is_blind_for_the_same_reason():
    """The module docstring recommends the dispersion-corrected residual as
    the honest cross-solver number. It is not usable as a gate for THIS
    defect class: ``residual = raw_diff - (beta_openems - beta_rfx)*L12``
    subtracts a term built from ``beta_rfx``, so doubling ``beta_rfx`` and
    the rfx phase together leaves it bit-unchanged. Only the RAW difference
    moves -- which is why the raw one is what got gated."""
    module = _load_referee_module()
    stage_b = json.loads(_RUN2_RESULT_PATH.read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    l12 = stage_b["layout"]["l12_m"]
    s21_openems = _cx(stage_b["s21"])
    beta_openems = np.asarray(stage_b["cross_solver_report"]["beta_openems_real"], dtype=float)
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    s21_rfx = _cx(fixture["s21"])
    beta_rfx = np.real(_cx(fixture["beta_first_port"]))

    def raw_and_residual(s21_r, beta_r):
        raw = np.degrees(np.angle(np.exp(
            1j * (np.unwrap(np.angle(s21_r)) - np.unwrap(np.angle(s21_openems))))))
        resid = np.degrees(np.angle(np.exp(
            1j * (np.radians(raw) - (beta_openems - beta_r) * l12))))
        return raw, resid

    raw0, resid0 = raw_and_residual(s21_rfx, beta_rfx)
    s21_bad = s21_rfx * np.exp(-1j * beta_rfx * l12)          # phase velocity halved
    raw1, resid1 = raw_and_residual(s21_bad, 2.0 * beta_rfx)

    mask = module._gate_band_mask(freqs)
    assert np.allclose(resid0, resid1, atol=1e-12), "residual must be provably blind"
    assert np.max(np.abs(resid0[mask])) == pytest.approx(0.715345, abs=1e-5)
    assert np.max(np.abs(raw0[mask])) == pytest.approx(0.3418, abs=1e-3)
    assert np.max(np.abs(raw1[mask])) == pytest.approx(44.8146, abs=1e-3)


def _independent_legs_on(module, result_path, eps_eff_openems):
    stage_b = json.loads(pathlib.Path(result_path).read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    cross = stage_b["cross_solver_report"]
    eps_rfx = _rfx_realized_eps_eff(module)
    rfx = module._analytic_beta_witness(
        freqs, np.asarray(cross["beta_rfx_real"], dtype=float), eps_eff=eps_rfx,
        tol_frac=module.B_BETA_ANALYTIC_TOL_FRAC, label="replay", solver="rfx")
    oe = module._analytic_beta_witness(
        freqs, np.asarray(cross["beta_openems_real"], dtype=float),
        eps_eff=eps_eff_openems, tol_frac=module.B_BETA_ANALYTIC_TOL_FRAC,
        label="replay", solver="openems")
    xs = module._cross_solver_phase_witness(
        freqs, np.asarray(cross["raw_phase_diff_deg"], dtype=float),
        tol_deg=module.B_CROSS_SOLVER_PHASE_TOL_DEG, label="replay")
    return rfx, oe, xs


def test_independent_phase_legs_pass_on_both_committed_runs():
    """Criterion (A), on TWO independent committed configurations.

    The rfx fixture's realized board is 300um in both runs (it is the same
    committed fixture). openEMS's own board differs between them -- the
    declared 254um in run-1, the #723-matched 300um in run-2 -- so each
    run's openEMS leg is judged against ITS OWN board's closed form, which
    is exactly what the production wiring does (it reads the realized
    geometry from the fixture the run was built on).
    """
    module = _load_referee_module()
    eps_declared = module._hammerstad_jensen_eps_eff(600e-6, 254e-6, module.B_EPS_R)
    eps_realized = _rfx_realized_eps_eff(module)
    assert eps_realized == pytest.approx(2.8326927491022724, rel=1e-12)
    assert eps_declared == pytest.approx(2.8693862252597855, rel=1e-12)

    rfx2, oe2, xs2 = _independent_legs_on(module, _RUN2_RESULT_PATH, eps_realized)
    assert (rfx2["passed"], oe2["passed"], xs2["passed"]) == (True, True, True)
    assert rfx2["max_abs_dev_frac"] == pytest.approx(0.00938, abs=1e-5)
    assert oe2["max_abs_dev_frac"] == pytest.approx(0.00307, abs=1e-5)
    assert xs2["max_abs_raw_phase_diff_deg"] == pytest.approx(0.3418, abs=1e-3)

    rfx1, oe1, xs1 = _independent_legs_on(module, _RUN1_RESULT_PATH, eps_declared)
    assert (rfx1["passed"], oe1["passed"], xs1["passed"]) == (True, True, True)
    assert rfx1["max_abs_dev_frac"] == pytest.approx(0.00938, abs=1e-5)
    assert oe1["max_abs_dev_frac"] == pytest.approx(0.00494, abs=1e-5)
    assert xs1["max_abs_raw_phase_diff_deg"] == pytest.approx(0.3039, abs=1e-3)

    # Margins, so a future reader sees how much room criterion (A) has.
    assert module.B_BETA_ANALYTIC_TOL_FRAC / rfx2["max_abs_dev_frac"] > 2.0
    assert module.B_CROSS_SOLVER_PHASE_TOL_DEG / xs2["max_abs_raw_phase_diff_deg"] > 8.0

    assert rfx2["evidence_level"].startswith("E2")
    assert xs2["evidence_level"].startswith("E4")
    assert xs2["attributes_to_a_solver"] is False


def test_independent_phase_legs_fire_on_the_factor_two_phase_velocity_error():
    """Criterion (B): the defect the audit measured this case blind to must
    RED both new gates, for the right reasons -- the analytic leg naming
    the Hammerstad-Jensen comparison and attributing to the rfx side, the
    cross-solver leg naming the two solvers' de-embedded phases and
    explicitly declining to attribute."""
    module = _load_referee_module()
    stage_b = json.loads(_RUN2_RESULT_PATH.read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    l12 = stage_b["layout"]["l12_m"]
    s21_openems = _cx(stage_b["s21"])
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    s21_rfx = _cx(fixture["s21"])
    beta_rfx = np.real(_cx(fixture["beta_first_port"]))
    eps_rfx = _rfx_realized_eps_eff(module)

    for k, dev, xs_deg in ((2.0, 1.018764, 44.8146), (0.5, 0.495643, 22.1883)):
        beta_bad = beta_rfx * k
        s21_bad = s21_rfx * np.exp(-1j * (k - 1.0) * beta_rfx * l12)

        with pytest.raises(RuntimeError) as excinfo:
            module._analytic_beta_witness(
                freqs, beta_bad, eps_eff=eps_rfx,
                tol_frac=module.B_BETA_ANALYTIC_TOL_FRAC,
                label="perturbed", solver="rfx")
        message = str(excinfo.value)
        assert "analytic-beta witness failed for solver 'rfx'" in message
        assert "Hammerstad-Jensen quasi-static" in message
        assert f"by {dev:.6f} > 0.020000" in message, (k, message[:400])

        raw = np.degrees(np.angle(np.exp(
            1j * (np.unwrap(np.angle(s21_bad)) - np.unwrap(np.angle(s21_openems))))))
        with pytest.raises(RuntimeError) as excinfo:
            module._cross_solver_phase_witness(
                freqs, raw, tol_deg=module.B_CROSS_SOLVER_PHASE_TOL_DEG,
                label="perturbed")
        message = str(excinfo.value)
        assert "cross-solver phase witness failed" in message
        assert f"= {xs_deg:.4f} deg > 3.0000 deg" in message, (k, message[:400])
        assert "does\nNOT say which" in message or "NOT say which" in message

        # ...while the E1 witness the audit measured still passes.
        old = module._self_consistency_witness(
            freqs, s21_bad, beta_bad, l12_m=l12, mag_band=module.B_S21_MAG_BAND,
            phase_tol_deg=module.B_PHASE_TOL_DEG, gd_tol_ps=module.B_GD_TOL_PS,
            label="old")
        assert old["passed"] is True


def test_independent_phase_legs_are_wired_into_sanity_passed():
    """Revert-proof: both new gates must be inside ``sanity_passed`` and
    inside the forensics try-block, not reported-only fields a later edit
    can drop silently."""
    module = _load_referee_module()
    import inspect
    src = inspect.getsource(module._run_stage_b)
    assert "analytic_beta_openems = _analytic_beta_witness(" in src
    assert "analytic_beta_rfx = _analytic_beta_witness(" in src
    assert "cross_solver_phase = _cross_solver_phase_witness(" in src
    assert 'and analytic_beta_openems["passed"] and analytic_beta_rfx["passed"]' in src
    assert 'and cross_solver_phase["passed"]' in src
    assert "exc.partial_stage_b_data = partial_data" in src
    # The analytic reference must be the REALIZED board (issue #723), not
    # the declared constants -- a regression to B_H_SUB_M here would make
    # the E2 leg judge a board that is not simulated.
    assert 'layout["w_trace_realized_m"], layout["h_sub_realized_m"]' in src


def _stage_b_replay_fakes(module, monkeypatch, *, openems_k: float = 1.0):
    """Fake every openEMS seam ``_run_stage_b`` touches and replay the
    committed run-2 stage-B data through the REAL function (round-2 review
    of #812 P1: the wiring was pinned by a source-string test only).

    ``openems_k`` scales the openEMS side's phase velocity coherently
    (``beta`` and the through-path phase together) so the attribution of
    the E2 leg can be measured on that side too.
    """
    stage_b = json.loads(_RUN2_RESULT_PATH.read_text())["stage_b"]
    freqs = np.asarray(stage_b["freqs_hz"], dtype=float)
    l12 = stage_b["layout"]["l12_m"]
    s11 = _cx(stage_b["s11"])
    s21 = _cx(stage_b["s21"])
    beta0 = _cx(stage_b["beta_port0"])
    beta1 = _cx(stage_b["beta_port1"])
    if openems_k != 1.0:
        beta_mean = 0.5 * np.real(beta0 + beta1)
        s21 = s21 * np.exp(-1j * (openems_k - 1.0) * beta_mean * l12)
        beta0, beta1 = beta0 * openems_k, beta1 * openems_k

    class _FakePort:
        def __init__(self, uf_ref, beta):
            self.uf_inc = np.ones_like(freqs, dtype=np.complex128)
            self.uf_ref = np.asarray(uf_ref, dtype=np.complex128)
            self.beta = np.asarray(beta, dtype=np.complex128)

        def CalcPort(self, sim_dir, freqs_hz, *, ref_plane_shift):  # noqa: N802 (openEMS API)
            assert np.allclose(freqs_hz, freqs)

    port0, port1 = _FakePort(s11, beta0), _FakePort(s21, beta1)
    monkeypatch.setattr(module, "_import_openems", lambda: (object, object, object))
    monkeypatch.setattr(module, "_build_stage_b",
                        lambda *a, **k: (None, port0, port1, dict(stage_b["port_info"])))
    monkeypatch.setattr(module, "_run_openems_capturing_stdout", lambda *a, **k: "")
    monkeypatch.setattr(module, "_scan_stdout_for_bad_patterns", lambda *a, **k: None)
    monkeypatch.setattr(module, "_log_indicates_truncation", lambda *a, **k: False)
    monkeypatch.setattr(module, "_check_excitation_and_trace", lambda *a, **k: (1.0, 227))


def _perturbed_rfx_fixture(module, k: float) -> dict:
    fixture = module._load_rfx_fixture(str(RFX_FIXTURE_PATH))
    s21 = _cx(fixture["s21"])
    beta = _cx(fixture["beta_first_port"])
    l12 = module._stage_b_layout(fixture)["l12_m"]
    s21_k = s21 * np.exp(-1j * (k - 1.0) * np.real(beta) * l12)
    beta_k = beta * k
    out = dict(fixture)
    out["s21"] = [[float(c.real), float(c.imag)] for c in s21_k]
    out["beta_first_port"] = [[float(c.real), float(c.imag)] for c in beta_k]
    return out


def test_run_stage_b_reds_end_to_end_on_a_coherent_rfx_beta_error(monkeypatch):
    """Criterion (B) through the REAL ``_run_stage_b`` wiring, not the
    witnesses in isolation -- the cv21 ``_run_one_drive`` replay pattern.

    (A) run-2's committed data passes and the three independent legs equal
    the artifact's ``cv20.run2_realized_board`` keys. (B) a coherent
    factor-2 / factor-0.5 error on the rfx side raises naming solver
    ``rfx`` and still carries the partial forensics; the same error on the
    openEMS side raises naming solver ``openems`` -- the E2 leg attributes
    to whichever side is out of envelope, including openEMS.
    """
    module = _load_referee_module()
    evidence = json.loads(_EVIDENCE_PATH.read_text())["cv20"]["run2_realized_board"]

    _stage_b_replay_fakes(module, monkeypatch)
    result = module._run_stage_b(sim_root="/tmp/_unused_812_cv20_ok", threads=1, nrts=200000,
                                 end_criteria=1e-4,
                                 rfx_fixture=module._load_rfx_fixture(str(RFX_FIXTURE_PATH)))
    assert result["sanity_passed"] is True
    report = result["cross_solver_report"]
    assert report["analytic_beta_witness_rfx"]["max_abs_dev_frac"] == pytest.approx(
        evidence["analytic_beta_rfx_max_abs_dev_frac"], rel=1e-9)
    assert report["analytic_beta_witness_openems"]["max_abs_dev_frac"] == pytest.approx(
        evidence["analytic_beta_openems_max_abs_dev_frac"], rel=1e-9)
    assert report["cross_solver_phase_witness"]["max_abs_raw_phase_diff_deg"] == pytest.approx(
        evidence["cross_solver_max_abs_raw_phase_diff_deg"], rel=1e-9)

    for k in (2.0, 0.5):
        with pytest.raises(RuntimeError) as excinfo:
            module._run_stage_b(sim_root="/tmp/_unused_812_cv20_bad", threads=1, nrts=200000,
                                end_criteria=1e-4, rfx_fixture=_perturbed_rfx_fixture(module, k))
        message = str(excinfo.value)
        assert "stage_b_analytic_beta" in message and "solver 'rfx'" in message, k
        assert hasattr(excinfo.value, "partial_stage_b_data")
        assert "cross_solver_partial" in excinfo.value.partial_stage_b_data

    _stage_b_replay_fakes(module, monkeypatch, openems_k=2.0)
    with pytest.raises(RuntimeError) as excinfo:
        module._run_stage_b(sim_root="/tmp/_unused_812_cv20_oe", threads=1, nrts=200000,
                            end_criteria=1e-4,
                            rfx_fixture=module._load_rfx_fixture(str(RFX_FIXTURE_PATH)))
    assert "solver 'openems'" in str(excinfo.value)
