"""The openEMS MSL notch tutorial gate still tells a bad solver run from a good one.

``tests/crossval/_openems_tutorial_gate.py`` holds the reproduce gate that runs
before any openEMS reference is recorded: the MSL notch filter and Sheen
low-pass filter reference makers use it, and so does the probe-fed microstrip
referee's A1 leg. CI never runs openEMS, so these tests are what notices when
an edit stops the gate from refusing a bad run: one that stopped before its
fields decayed, one that dropped a primitive, one whose |S| blew up.

They were written against the MSL thru-line phase case's copies of the same
helpers (``tests/crossval/test_msl_phase_referee_header.py``) and moved here
when that case was removed on 2026-09-23. The shared module's copies were
byte-identical to the case's. None of these tests needs openEMS.
"""
from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType
from typing import Final

import numpy as np
import pytest

from tests._git_tracked import git_available, is_tracked

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
GATE_PATH: Final = REPO_ROOT / "tests" / "crossval" / "_openems_tutorial_gate.py"


def _load_gate() -> ModuleType:
    """Load the shared gate as a throwaway module, without openEMS installed."""
    spec = importlib.util.spec_from_file_location("_openems_tutorial_gate_under_test", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The tutorial and its gate band
# ---------------------------------------------------------------------------
def test_stage_a_gate_band_is_one_sided_low_biased_and_contains_expected():
    module = _load_gate()
    gate = module.STAGE_A_GATE
    assert gate["f_notch_lo_hz"] < module.F_NOTCH_AN_HZ < gate["f_notch_hi_hz"]
    # one-sided-low-biased: the low margin must be wider than the high
    # margin (precedent: openEMS reads notch frequency LOW vs analytic).
    lo_margin = module.F_NOTCH_AN_HZ - gate["f_notch_lo_hz"]
    hi_margin = gate["f_notch_hi_hz"] - module.F_NOTCH_AN_HZ
    assert lo_margin > hi_margin


def test_stage_a_matches_tutorial_constants():
    """Regression lock on MSL_NotchFilter.py's own literal parameters."""
    module = _load_gate()
    assert module.A_MSL_LENGTH_UM == 50000.0
    assert module.A_MSL_WIDTH_UM == 600.0
    assert module.A_SUBSTRATE_THICKNESS_UM == 254.0
    assert module.A_SUBSTRATE_EPR == 3.66
    assert module.A_STUB_LENGTH_UM == 12.0e3
    assert module.A_F_MAX_HZ == 7.0e9
    assert module.A_N_FREQS == 1601


# ---------------------------------------------------------------------------
# Sanity-check helpers -- openEMS-free, synthetic inputs.
# ---------------------------------------------------------------------------
def test_check_excitation_and_trace_is_scale_free_not_an_absolute_floor():
    """The coax lane's PR #547 lesson: no absolute floor, only
    exact-zero/non-finite is a defect signature."""
    module = _load_gate()

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


def test_non_physical_guard_raises_above_two():
    module = _load_gate()
    module._non_physical_guard(np.array([0.1, 0.9, 1.5]), "ok")  # must not raise
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, 2.5]), "bad")
    with pytest.raises(RuntimeError):
        module._non_physical_guard(np.array([0.1, float("nan")]), "nan")


def test_passivity_witness_raises_above_tolerance():
    module = _load_gate()
    s11 = np.array([0.1, 0.1])
    s21_ok = np.array([0.9, 0.9])
    result = module._passivity_witness(s11, s21_ok, "ok")
    assert result["passed"] is True

    s21_bad = np.array([1.5, 1.5])
    with pytest.raises(RuntimeError):
        module._passivity_witness(s11, s21_bad, "bad")


# ---------------------------------------------------------------------------
# The pre-solve scanner's allowlist and truncation scoping, and the truncation
# reader, on REAL openEMS output: the excerpts below are copied verbatim from
# the recorded run-1 log (lines 20-30), which moved beside the MSL notch
# filter's reference when the thru-line case was removed. A guard must be shown
# to discriminate on real data, not on a plausible-looking synthetic.
# ---------------------------------------------------------------------------
_RUN1_LOG_PATH: Final = (
    REPO_ROOT / "tests" / "crossval" / "msl_notch_filter" / "reference"
    / "tutorial_reproduction_20260804T070702Z_run.log"
)

# SMOKE portion (NrTS=200, EndCriteria=0.0/"-infdB" -- log lines 20-27): the
# truncation strings legitimately fire here, by construction (a 200-timestep
# budget cannot hold the excitation pulse).
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

# REAL portion (NrTS=300000, EndCriteria=1e-4 -- log lines 28-30): reached its
# own EndCriteria; no truncation strings present.
_RUN1_REAL_LOG_EXCERPT = (
    "openEMS::SetupFDTD: Warning, the timestep seems to be very small --> "
    "long simulation. Check your mesh!?\n"
    "Warning: Unused primitive (type: Box) detected in property: port0_metal!\n"
    "Warning: Unused primitive (type: Box) detected in property: port1_metal!\n"
)


def test_run1_log_excerpts_are_verbatim_substrings_of_the_committed_log():
    """Ties the two excerpts above to the committed evidence file: if the log
    drifts from these strings, this test (not only the ones using them) goes
    red."""
    assert _RUN1_LOG_PATH.exists(), f"missing committed run-1 log {_RUN1_LOG_PATH}"
    full_log = _RUN1_LOG_PATH.read_text()
    for excerpt in (_RUN1_SMOKE_LOG_EXCERPT, _RUN1_REAL_LOG_EXCERPT):
        for line in excerpt.splitlines():
            assert line.strip() in full_log, (
                f"excerpt line not found verbatim in committed log: {line!r}"
            )


def test_the_recorded_reproduction_log_is_committed_and_carries_the_number():
    """The gate's REPRODUCE_GATE_RECORD cites a log and the lines in it; both
    must hold for a reader who clones the repository. ``exists()`` alone passes
    on a machine where the file was written and never added (the PR #548
    lesson), so the file must also be tracked."""
    record = _load_gate().REPRODUCE_GATE_RECORD
    log_path = REPO_ROOT / record["log_path"]
    assert log_path == _RUN1_LOG_PATH, record["log_path"]
    assert log_path.exists(), f"the recorded reproduction log is missing: {log_path}"
    if git_available(REPO_ROOT):
        assert is_tracked(record["log_path"], REPO_ROOT), (
            f"{record['log_path']} exists here but git does not track it -- a "
            f"clone would not have it"
        )
    first, last = (int(n) for n in record["log_lines"].split("-"))
    cited = "\n".join(log_path.read_text().splitlines()[first - 1:last])
    assert f"measured={record['reproduced_f_notch_hz'] / 1e9:.4f} GHz" in cited, cited


def test_scan_stdout_allowlists_port_metal_unused_primitive():
    """The port0_metal/port1_metal 'Unused primitive' warning (see
    ``_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES``) must NOT trip the fail-fast
    gate: the real run-1 excerpt, which carries exactly those two lines, must
    not raise."""
    module = _load_gate()
    module._scan_stdout_for_bad_patterns(_RUN1_REAL_LOG_EXCERPT, "positive_control_real")  # must not raise


def test_scan_stdout_still_raises_on_non_allowlisted_unused_primitive():
    """The allowlist is scoped to the EXACT two property names: any OTHER
    'Unused primitive' (a genuinely dropped substrate) must still trip it."""
    module = _load_gate()
    bad_log = "Warning: Unused primitive (type: Box) detected in property: substrate!\n"
    with pytest.raises(RuntimeError) as excinfo:
        module._scan_stdout_for_bad_patterns(bad_log, "bad")
    assert "substrate" in str(excinfo.value)


def test_scan_stdout_truncation_patterns_scoped_to_real_run_only():
    """``check_truncation=False`` (the smoke-call default) must NOT raise on the
    smoke excerpt's own legitimate truncation strings; the SAME text with
    ``check_truncation=True`` (the real-call setting) MUST raise."""
    module = _load_gate()
    module._scan_stdout_for_bad_patterns(
        _RUN1_SMOKE_LOG_EXCERPT, "positive_control_smoke", check_truncation=False)  # must not raise

    with pytest.raises(RuntimeError) as excinfo:
        module._scan_stdout_for_bad_patterns(
            _RUN1_SMOKE_LOG_EXCERPT, "positive_control_smoke_as_real", check_truncation=True)
    assert "Cutting to max number of timesteps" in str(excinfo.value)


def test_scan_stdout_real_run1_excerpt_passes_truncation_check():
    """The positive control's other half: run-1's REAL portion, which reached
    its own EndCriteria, must pass ``check_truncation=True``."""
    module = _load_gate()
    module._scan_stdout_for_bad_patterns(
        _RUN1_REAL_LOG_EXCERPT, "positive_control_real_truncation_check", check_truncation=True)  # must not raise


def test_log_indicates_truncation_flips_on_synthetic_under_settled_input():
    """The truncation reader must be shown to flip True on an under-settled
    run's own log text, not only to read False on run-1's converged data."""
    module = _load_gate()
    assert module._log_indicates_truncation(_RUN1_REAL_LOG_EXCERPT) is False
    assert module._log_indicates_truncation(_RUN1_SMOKE_LOG_EXCERPT) is True

    synthetic_under_settled = (
        "RunFDTD: Warning: Max. number of timesteps was reached before the "
        "end-criteria of -30dB was reached... \n"
    )
    assert module._log_indicates_truncation(synthetic_under_settled) is True
    assert module._log_indicates_truncation("no such warning anywhere in this log\n") is False
