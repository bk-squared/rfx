"""Consistency checks for the external RCWA referee (rfx issue #491, comparator leg).

`validation/research/floquet/rcwa_referee.py` is a COMPARATOR-leg-only harness
(see its module docstring for the full scope fence): it builds and calibrates
an external RCWA reference instrument (`grcwa`), and makes no claim about rfx
Floquet physics. This test does NOT depend on that comparison ever landing.

Two tiers of check:

1. Artifact-consistency (always runs, no `grcwa` dependency): the committed
   `rcwa_referee_artifact.json` is internally consistent with the tolerances
   the referee script itself declares -- a regression lock on the committed
   evidence, independent of whether `grcwa` is installed in this environment.
2. Live-tool opportunistic re-check (`pytest.importorskip("grcwa")` --
   skips cleanly, is NOT a CI dependency): if `grcwa` happens to be
   available, re-run a fast subset of the referee's own checks to catch
   drift between the committed artifact and the currently installed grcwa.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from types import ModuleType
from typing import Final

import pytest

FLOQUET_DIR: Final = (
    pathlib.Path(__file__).resolve().parent.parent / "validation" / "research" / "floquet"
)
ARTIFACT_PATH: Final = FLOQUET_DIR / "rcwa_referee_artifact.json"
SCRIPT_PATH: Final = FLOQUET_DIR / "rcwa_referee.py"

# These are the ONLY externally-sourced numbers in this harness -- grcwa's own
# S4-cross-validated regression constants (tests/test_rcwa.py::test_rcwa in
# the grcwa 0.1.2 sdist). Pinned here so a silent edit of the anchor (e.g.
# regenerating the gate against a different, undocumented reference) fails
# loudly instead of passing via a self-consistent but wrong rel_err.
EXPECTED_T_P_S4: Final = 0.85249901083265
EXPECTED_T_S_S4: Final = 0.83900479939861


def _load_artifact() -> dict:
    assert ARTIFACT_PATH.exists(), (
        f"missing committed artifact {ARTIFACT_PATH} -- regenerate with "
        f"`python {SCRIPT_PATH}` and commit the result"
    )
    with open(ARTIFACT_PATH) as fh:
        return json.load(fh)


def _load_referee_module() -> ModuleType:
    """Load the referee script as a throwaway module (not registered in sys.modules)."""
    spec = importlib.util.spec_from_file_location("_rcwa_referee", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_scope_is_comparator_leg_only():
    artifact = _load_artifact()
    assert artifact["issue"] == 491
    assert "comparator leg only" in artifact["scope"]
    assert "rfx-known-issues.md" in artifact["r2_stop_pointer"]
    assert "drive-isolation" in artifact["r2_stop_pointer"]


def test_artifact_reproduce_gate_pins_the_s4_anchor_constants():
    """The two S4 reference numbers must match grcwa's own documented values.

    These are the only externally-sourced numbers in the PR; nothing else
    binds them, so a drifted anchor (e.g. `expected_T` regenerated against a
    different, undocumented reference) would otherwise pass silently as long
    as `reproduced_T` drifted the same way.
    """
    artifact = _load_artifact()
    gate = artifact["reproduce_gate"]
    assert gate["p_pol"]["expected_T"] == EXPECTED_T_P_S4
    assert gate["s_pol"]["expected_T"] == EXPECTED_T_S_S4


def test_artifact_reproduce_gate_is_internally_consistent():
    artifact = _load_artifact()
    gate = artifact["reproduce_gate"]
    for pol in ("p_pol", "s_pol"):
        rec = gate[pol]
        assert rec["rel_err"] < rec["tol"], f"{pol} reproduce-gate exceeds its own tolerance"
        # the reproduced number should be close to the documented S4 value in
        # absolute terms too, not just pass by a loose relative tolerance
        assert abs(rec["reproduced_T"] - rec["expected_T"]) < rec["tol"] * rec["expected_T"]
    assert gate["passed"] is True


def test_artifact_referee_calibration_slab_check_is_internally_consistent():
    artifact = _load_artifact()
    calibration = artifact["referee_calibration_slab_check"]
    assert calibration["rows"], "referee-calibration slab table must not be empty"
    threshold = calibration["gate_threshold"]
    assert calibration["max_dev_R"] < threshold
    assert calibration["max_dev_T"] < threshold
    assert calibration["max_energy_conservation_dev"] < threshold
    assert calibration["passed"] is True

    for row in calibration["rows"]:
        # lossless slab: energy conservation must hold at every frequency,
        # not just in the aggregate max
        assert abs(row["R_plus_T_rcwa"] - 1.0) < threshold
        assert abs(row["R_rcwa"] - row["R_analytic"]) < threshold
        assert abs(row["T_rcwa"] - row["T_analytic"]) < threshold


def test_artifact_overall_passed_matches_both_gates():
    artifact = _load_artifact()
    assert artifact["overall_passed"] == (
        artifact["reproduce_gate"]["passed"]
        and artifact["referee_calibration_slab_check"]["passed"]
    )


def test_committed_log_path_exists():
    artifact = _load_artifact()
    log_path = pathlib.Path(__file__).resolve().parent.parent / artifact["log_path"]
    assert log_path.exists(), f"referee run log {log_path} must be committed alongside the artifact"


def test_grcwa_unavailable_skips_cleanly(monkeypatch):
    """`main()` must return exit code 2 (inconclusive), not raise, when grcwa is absent."""
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "grcwa":
            raise ImportError("simulated: grcwa not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert module.main() == 2


def test_live_grcwa_reproduces_committed_numbers():
    """Opportunistic live re-check -- NOT a CI dependency, skips if grcwa is absent."""
    pytest.importorskip("grcwa", reason="grcwa required for the live RCWA referee re-check")
    module = _load_referee_module()

    reproduce = module._reproduce_grcwa_s4_regression()
    assert reproduce["passed"] is True
    assert reproduce["p_pol"]["expected_T"] == EXPECTED_T_P_S4
    assert reproduce["s_pol"]["expected_T"] == EXPECTED_T_S_S4

    calibration = module.run_referee_calibration_slab_check([6.0e9, 15.0e9])
    assert calibration["passed"] is True
    assert calibration["max_dev_R"] < 1e-9
    assert calibration["max_dev_T"] < 1e-9
