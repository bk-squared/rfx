"""The committed #812 P1 evidence artifact must still be what the harness emits.

Round 2 of #812 (issue retrospective, 2026-09-01): numbers live in a committed
JSON the harness writes, and prose references an artifact key instead of
restating digits. This test is what makes that artifact trustworthy -- it
re-runs ``scripts/diagnostics/build_issue812_phase_identity_evidence.py`` into a
scratch path and asserts the committed file still equals what the current
referees produce from the committed field data. Edit a witness or a threshold
without rebuilding and this fails.

It runs no FDTD and asserts no physics; the physics gates live in
``tests/test_msl_phase_referee_header.py`` and
``tests/test_coax_two_port_referee_header.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ARTIFACT = _REPO / "validation/crossval/_issue812_phase_identity/regate_evidence.json"
_BUILDER = _REPO / "scripts/diagnostics/build_issue812_phase_identity_evidence.py"


def _builder():
    spec = importlib.util.spec_from_file_location("issue812_evidence_builder", _BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed() -> dict:
    assert _ARTIFACT.is_file(), f"missing committed evidence artifact {_ARTIFACT}"
    return json.loads(_ARTIFACT.read_text())


def _leaves_beyond_tolerance(fresh, committed, path="", rel=1e-9, abs_=1e-12):
    """Every leaf where the fresh replay and the committed artifact disagree:
    floats beyond ``rel``/``abs_`` (the committed file was written on macOS and
    CI replays it on Linux, where last-digit rounding of the same arithmetic
    differs -- a byte-equal comparison red on CI for that reason on
    2026-09-02), everything else exactly, structure exactly."""
    bad = []
    if isinstance(fresh, dict) and isinstance(committed, dict):
        for k in sorted(set(fresh) | set(committed)):
            if k in fresh and k in committed:
                bad += _leaves_beyond_tolerance(fresh[k], committed[k], f"{path}.{k}", rel, abs_)
            else:
                bad.append(f"{path}.{k}: present on one side only")
    elif isinstance(fresh, list) and isinstance(committed, list):
        if len(fresh) != len(committed):
            bad.append(f"{path}: length {len(fresh)} vs {len(committed)}")
        else:
            for i, (a, b) in enumerate(zip(fresh, committed)):
                bad += _leaves_beyond_tolerance(a, b, f"{path}[{i}]", rel, abs_)
    elif (isinstance(fresh, float) or isinstance(committed, float)) and not (
            isinstance(fresh, bool) or isinstance(committed, bool)):
        if not math.isclose(fresh, committed, rel_tol=rel, abs_tol=abs_):
            bad.append(f"{path}: fresh {fresh!r} vs committed {committed!r}")
    elif fresh != committed:
        bad.append(f"{path}: fresh {fresh!r} vs committed {committed!r}")
    return bad


def test_committed_artifact_equals_a_fresh_replay(tmp_path, committed):
    module = _builder()
    out = tmp_path / "regate_evidence.json"
    assert module.main(["--output", str(out)]) == 0
    bad = _leaves_beyond_tolerance(json.loads(out.read_text()), committed)
    assert not bad, "committed artifact is stale or platform-divergent beyond 1e-9:\n" + "\n".join(bad)


def test_cv20_records_that_the_cross_solver_difference_is_now_gated(committed):
    """Round-1 blocker: the lane inverted this decision and left the old one
    standing in the manifest and in a source comment. The decision now has a
    machine-readable home, so the two cannot silently disagree again."""
    cv20 = committed["cv20"]
    assert cv20["cross_solver_raw_phase_difference_is_gated"] is True
    assert cv20["evidence_levels_supported_by_a_leg_in_this_case"] == ["E1", "E2", "E4"]
    assert cv20["e4_supporting_leg_count"] == 1
    for run in ("run1_declared_board", "run2_realized_board"):
        assert cv20[run]["all_three_passed"] is True                    # criterion (A)
        assert cv20[run]["cross_solver_margin_x"] > 8.0
    for arm in cv20["criterion_b"]:                                     # criterion (B)
        assert arm["e2_fired"] and arm["e4_fired"]
        assert arm["e1_self_consistency_fired"] is False
        assert arm["e4_attributes_to_a_solver"] is False


def test_cv21_carries_a_two_sided_detection_floor(committed):
    """Reviewer nonblocking: a one-sided summary of this floor is a false
    summary -- the gate is blind on BOTH sides of k = 1."""
    floor = committed["cv21"]["detection_floor"]
    for key in ("declared_registered", "measured_registered", "measured_refined"):
        assert floor[key]["k_lo"] < 1.0 < floor[key]["k_hi"], key
    assert floor["declared_registered"]["k_hi"] == pytest.approx(
        floor["predeclared_k_hi"], abs=1e-6)
    assert floor["declared_registered"]["k_lo"] == pytest.approx(
        floor["predeclared_k_lo"], abs=1e-6)
    # The floor tightens with the mesh, on both sides, with no gate change.
    assert floor["measured_refined"]["k_hi"] < floor["measured_registered"]["k_hi"]
    assert floor["measured_refined"]["k_lo"] > floor["measured_registered"]["k_lo"]


def test_cv21_margin_is_the_declared_headroom_by_construction_at_both_meshes(committed):
    """Reviewer nonblocking: the refined-mesh criterion-(A) check is
    arithmetically the registered-mesh statement, not an independent one. The
    committed convergence order IS the two-point fit through the two committed
    excesses, so the envelope at the refined mesh is the headroom times that
    mesh's own committed excess -- exactly."""
    b = committed["cv21"]["margin_is_the_declared_headroom_by_construction"]
    assert b["order_p_recovery_abs_error"] == 0.0
    assert b["bound_at_n_after_minus_headroom_times_excess_after"] == pytest.approx(
        0.0, abs=1e-15)
    for mesh in ("registered_mesh", "refined_mesh"):
        assert committed["cv21"][mesh]["beta_margin_x"] == pytest.approx(
            b["headroom_declared"], rel=4e-3), mesh


def test_cv21_registers_no_e4_because_no_leg_supports_one(committed):
    """Reviewer nonblocking, with the refuting search run rather than skipped:
    the referee imports no rfx module and reads no rfx fixture, so nothing in
    the case compares an rfx quantity against the external solver."""
    source = (_REPO / "validation/crossval/21_coax_two_port_referee.py").read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[2] if code.count('"""') >= 2 else code
    assert "import rfx" not in body and "from rfx" not in body
    assert "tests/fixtures" not in source
    assert committed["cv21"]["evidence_levels_supported_by_a_leg_in_this_case"] == ["E1", "E2"]
    assert committed["cv21"]["e4_supporting_leg_count"] == 0

    manifest = json.loads((_REPO / "validation/crossval/manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["id"] == "21_coax_two_port_referee")
    assert case["evidence_levels"] == ["E1", "E2"]
