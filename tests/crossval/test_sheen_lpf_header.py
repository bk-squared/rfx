"""Header/record consistency checks for cv07's openEMS tutorial reproduce-gate.

``validation/crossval/07_sheen_lpf.py`` compares rfx against openEMS on the
Sheen 1990 microstrip low-pass filter (see that module's own docstring for
the doublet-comparator caveats -- unrelated to this test). Before trusting
that comparator's own openEMS leg, ``run_openems_tutorial()`` reproduces
openEMS's own ``MSL_NotchFilter.py`` tutorial -- the same tutorial
``validation/crossval/20_msl_phase_referee.py``'s Stage A already
reproduces, ported independently here per this repo's comparator-first
convention (``docs/agent-memory/task_recipes/external_solver_comparator.md``).

This test loads the module WITHOUT openEMS installed (this test environment
does not have it -- ``07_sheen_lpf.py`` defers its openEMS import into
functions specifically so this works) and checks the record's CONTRACT
shape, not one pinned finding: UNRUN <=> no numbers, no log path; RUN <=>
numbers present AND a log path under a git-tracked prefix that actually
exists on disk. Mirrors ``tests/crossval/test_coax_two_port_referee_header.py``
's ``test_reproduce_gate_record_is_committed_unrun_and_self_consistent``,
the established pattern for this repo's REPRODUCE_GATE_RECORD convention.

issue #971: this case's own reproduce-gate record used to be absent, and a
sibling case's (cv06b's) number was cited here as if it were this
function's own measurement. See ``test_do_not_repeat_cites_the_recorded_
failure`` and ``test_analytic_target_is_the_tutorial_declared_quantity_
not_cv06bs_realized_one`` below.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
from types import ModuleType
from typing import Final

import pytest

CROSSVAL_DIR: Final = (
    pathlib.Path(__file__).resolve().parents[2] / "validation" / "crossval"
)
SCRIPT_PATH: Final = CROSSVAL_DIR / "07_sheen_lpf.py"
REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]


def _load_sheen_module() -> ModuleType:
    """Load 07_sheen_lpf.py as a throwaway module (not sys.modules-registered).

    Must succeed WITHOUT openEMS installed -- the script defers its openEMS
    import into functions (``_require_openems`` / ``_openems_common_setup``)
    specifically so this works.
    """
    assert SCRIPT_PATH.exists(), f"missing crossval script {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("_sheen_lpf_07", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_openems():
    """The module itself must not require openEMS at import time."""
    module = _load_sheen_module()
    assert hasattr(module, "REPRODUCE_GATE_RECORD")
    assert hasattr(module, "F_NOTCH_TUTORIAL_DECLARED_HZ")


def test_reproduce_gate_record_has_required_fields():
    module = _load_sheen_module()
    record = module.REPRODUCE_GATE_RECORD
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "expected_f_notch_tutorial_declared_hz", "gate", "status",
        "reproduced_f_notch_hz", "reproduced_f_notch_dev_pct", "log_path",
        "vessl_run_id", "verified_on",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"

    gate = record["gate"]
    assert gate["f_notch_lo_hz"] < gate["f_notch_hi_hz"]


def test_tutorial_citation_is_verifiable():
    """The reproduce-gate must cite a REAL, checkable tutorial path -- not
    an assertion that no tutorial exists. This test checks the citation has
    the shape of something a reviewer could independently verify (repo +
    path + how it was checked + when), not that any PARTICULAR tutorial is
    named -- the record's own content should be free to evolve without this
    test needing a rewrite."""
    module = _load_sheen_module()
    tutorial = module.REPRODUCE_GATE_RECORD["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".py"), (
        "tutorial citation should point at a real source file"
    )
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"


def test_do_not_repeat_cites_the_recorded_failure():
    """R1/R2 class: the rebuild must name the specific recorded failure it
    avoids repeating, not just assert a new approach in the abstract --
    issue #971's actual mistake (a sibling case's number cited as this
    function's own known-good)."""
    module = _load_sheen_module()
    do_not_repeat = module.REPRODUCE_GATE_RECORD["do_not_repeat"]
    assert "3.4286" in do_not_repeat
    assert "cv06b" in do_not_repeat or "06b" in do_not_repeat
    assert "Palace" in do_not_repeat


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path.

    This is the test that must go RED if someone later claims reproduced
    numbers without a log path pointing at the run that produced them --
    not a pinned number that rots after the first real VESSL run. Same
    tracked-path requirement ``test_coax_two_port_referee_header.py``'s
    twin test enforces: a FILLED (``status == "RUN"``) record's
    ``log_path`` must live under a git-TRACKED prefix, not ``.omx/`` or
    ``docs/research_notes/vessl_logs/`` (both gitignored -- unreadable by a
    reviewer outside the machine that ran the job).
    """
    module = _load_sheen_module()
    record = module.REPRODUCE_GATE_RECORD

    if record["status"] == "UNRUN":
        assert record["reproduced_f_notch_hz"] is None
        assert record["reproduced_f_notch_dev_pct"] is None
        assert record["log_path"] is None
        assert record["vessl_run_id"] is None
        assert record["verified_on"] is None
    else:
        assert record["status"] == "RUN", (
            f"unexpected status {record['status']!r}: only 'UNRUN' or 'RUN' "
            "are contract-valid"
        )
        assert record["reproduced_f_notch_hz"] is not None
        assert record["reproduced_f_notch_dev_pct"] is not None
        assert record["vessl_run_id"], "a filled record needs a vessl_run_id"
        assert record["verified_on"], "a filled record needs a verification date"
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"

        gitignored_prefixes = (".omx/", "docs/research_notes/vessl_logs/")
        tracked_prefixes = ("validation/crossval/_07_sheen_logs/",)
        assert not log_path_str.startswith(gitignored_prefixes), (
            f"log_path {log_path_str!r} lives under a GITIGNORED prefix -- "
            f"a FILLED (status == 'RUN') record needs a log a reviewer "
            f"OUTSIDE this machine can open; use a tracked prefix instead: "
            f"{tracked_prefixes!r} (same PR #548 lesson cv20/cv21 already "
            f"paid for)"
        )
        assert log_path_str.startswith(tracked_prefixes), (
            f"log_path {log_path_str!r} must live under a TRACKED prefix "
            f"{tracked_prefixes!r} once status is RUN"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but "
            f"its log_path {log_path} does not exist -- a claimed number "
            f"needs a real log, per external_solver_comparator.md step 2"
        )


def test_analytic_target_is_the_tutorial_declared_quantity_not_cv06bs_realized_one():
    """Naming-collision guard (team-lead ruling, 2026-09-10): two DIFFERENT
    physical quantities share the shape of an "F_NOTCH" constant across
    sibling crossval cases. ``06b_msl_notch_filter_uniform.py``'s
    ``F_NOTCH_AN`` uses its own as-built rfx board's LATTICE-REALIZED
    electrical trace width (635um, its own n_rows*DX convention, issue
    #723); this case's ``F_NOTCH_TUTORIAL_DECLARED_HZ`` uses the openEMS
    tutorial's DECLARED width (600um, the same value
    ``20_msl_phase_referee.py``'s Stage A independently recomputes as
    ``F_NOTCH_AN_HZ``). They must stay ~0.22% apart -- if a future edit
    collapses the eps_eff inputs and this test starts reading ~0%, someone
    silently made the two interchangeable, which is exactly the
    substitution issue #971 is about.
    """
    module = _load_sheen_module()
    f_declared = module.F_NOTCH_TUTORIAL_DECLARED_HZ

    # cv06b_msl_notch_filter_uniform.py's own docstring-pinned value
    # (validation/crossval/06b_msl_notch_filter_uniform.py:379-380,
    # u=2.500, eps_eff=2.882252): F_NOTCH_AN = 3.678954 GHz. Recomputed
    # here from the same closed form and the same realized-635um input
    # rather than importing cv06b, so this test does not couple to that
    # module's own import-time behaviour.
    f_cv06b_realized = 2.998e8 / (4.0 * 12e-3 * math.sqrt(2.882252))
    assert f_cv06b_realized == pytest.approx(3.678954e9, rel=1e-6)

    dev_pct = abs(f_declared - f_cv06b_realized) / f_cv06b_realized * 100.0
    assert 0.15 < dev_pct < 0.35, (
        f"F_NOTCH_TUTORIAL_DECLARED_HZ ({f_declared / 1e9:.4f} GHz) and "
        f"cv06b's realized-board F_NOTCH_AN ({f_cv06b_realized / 1e9:.4f} "
        f"GHz) should differ by ~0.22% (declared-vs-realized trace width, "
        f"2026-09-10 reconciliation) -- got {dev_pct:.3f}%, meaning either "
        f"constant's inputs changed without this test being updated to say "
        f"why."
    )
