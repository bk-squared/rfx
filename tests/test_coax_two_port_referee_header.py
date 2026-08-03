"""Header/record consistency checks for the openEMS coax two-port referee.

``validation/research/coax_two_port/openems_coax_two_port_referee.py`` is a
COMPARATOR-leg-only, VESSL-only harness for rfx issue #489 stage 3 (see its
module docstring for the full scope fence): openEMS is not installed in
this environment, so it has never actually run. This test does NOT require
openEMS -- it checks that the script's reproduce-gate record is committed
in an honest, fail-loud state: fields exist and are explicitly marked
UNRUN, not silently populated with unverifiable numbers.

Design (per the task's fail-loud-honest requirement): this test PASSES on
the current, never-run placeholder state. It is designed to go RED only if
someone later claims reproduced numbers (``status`` != "UNRUN") without
also supplying a real, existing log path -- i.e. it enforces the
invariant, not a specific pinned number that would need updating after
every VESSL run.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType
from typing import Final

REFEREE_DIR: Final = (
    pathlib.Path(__file__).resolve().parent.parent
    / "validation" / "research" / "coax_two_port"
)
SCRIPT_PATH: Final = REFEREE_DIR / "openems_coax_two_port_referee.py"
REPO_ROOT: Final = pathlib.Path(__file__).resolve().parent.parent


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
        "stage", "no_canonical_tutorial_found", "tutorials_checked",
        "checked_on", "fallback", "nearest_starting_point", "oracle",
        "gate", "status", "reproduced_short_mag", "reproduced_matched_mag",
        "log_path", "vessl_run_id",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"


def test_no_canonical_tutorial_finding_is_declared_and_dated():
    """The 'no canonical openEMS coax tutorial exists' finding is an audit
    artifact, not prose -- it must name what was checked and when."""
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD
    assert record["no_canonical_tutorial_found"] is True
    checked = record["tutorials_checked"]
    assert "thliebig/openEMS python/Tutorials" in checked
    assert len(checked["thliebig/openEMS python/Tutorials"]) >= 5
    assert "thliebig/openEMS matlab/examples/transmission_lines" in checked
    assert record["checked_on"]  # non-empty date string
    assert "coax" not in " ".join(checked["thliebig/openEMS python/Tutorials"]).lower()


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path.

    This is the test that must go RED if someone later claims reproduced
    numbers without a log path pointing at the run that produced them --
    not a pinned number that rots after the first real VESSL run.
    """
    module = _load_referee_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_short_mag"] is None
        assert record["reproduced_matched_mag"] is None
        assert record["log_path"] is None
    else:
        # Once a real run fills this in, it must be backed by an artifact
        # that actually exists on disk -- a claimed number with no log is
        # worse than an honest UNRUN placeholder (feedback_summary_
        # compression_audit.md).
        assert record["reproduced_short_mag"] is not None
        assert record["reproduced_matched_mag"] is not None
        assert record["log_path"], "a filled-in reproduce_gate_record needs a log_path"
        log_path = REPO_ROOT / record["log_path"]
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but its "
            f"log_path {log_path} does not exist -- a claimed number needs a "
            f"real log, per external_solver_comparator.md step 2"
        )


def test_declared_question_and_governance_notes_present():
    module = _load_referee_module()
    assert "reference-plane referral" in module.DECLARED_QUESTION
    assert "validation/crossval/" in module.MUST_MOVE_WHEN_VALIDATED
    assert "REPRODUCE_GATE_RECORD" in module.MUST_MOVE_WHEN_VALIDATED


def test_geometry_constants_match_the_rfx_fixture():
    """Pins the numbers this script's header claims were read off the live
    rfx grid-construction path -- a regression lock so a future edit can't
    silently drift the target geometry away from the rfx fixture without
    the test noticing."""
    module = _load_referee_module()
    assert module.A_MM == 0.635
    assert module.B_MM == 2.055
    assert module.PTFE_EPS_R == 2.1
    assert abs(module.DX_MM - 3.7474057249999997e-4 * 1e3) < 1e-9
    assert module.CPML_CELLS == 16
    assert abs(module.L12_MM - 58.4595293) < 1e-4
    # Z0 = sqrt(L'/C') closed form must land close to the standard SMA/PTFE
    # value (~48.6 ohm) this repo's other coax lanes all cite.
    assert 48.0 < module.Z0_OHM < 49.0


def test_openems_unavailable_exits_2(monkeypatch):
    """main() must return exit code 2 (declared skip), not raise, when the
    openEMS Python bindings are absent -- lane convention (vessl_external_
    referee_lane.md): 0=self-checks passed, 1=self-check failed, 2=missing."""
    module = _load_referee_module()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("CSXCAD.CSXCAD", "openEMS.openEMS", "CSXCAD", "openEMS"):
            raise ImportError("simulated: openEMS not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert module.main([]) == 2


def test_freqs_overlap_the_rfx_band():
    """The referee's frequency grid must cover the rfx BAND points
    ([4,6,8,10,12] GHz, tests/test_coax_two_port_fdtd.py) it will be
    compared against by hand."""
    module = _load_referee_module()
    freqs_ghz = set(round(float(f), 6) for f in module.FREQS_GHZ)
    for f in (4.0, 6.0, 8.0, 10.0, 12.0):
        assert f in freqs_ghz, f"{f} GHz missing from referee frequency grid"
