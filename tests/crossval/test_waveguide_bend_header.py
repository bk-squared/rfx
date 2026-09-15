"""Header/record consistency checks for cv01's Meep tutorial reproduce-gate.

``validation/crossval/01_waveguide_bend.py`` has NO ``__main__`` guard --
importing it runs a real FDTD solve (and, if Meep is importable, a second
one) as bare module-level code. That is unlike ``07_sheen_lpf.py``,
``20_msl_phase_referee.py`` and ``21_coax_two_port_referee.py``, whose own
header tests load the module via ``importlib`` and just inspect
``REPRODUCE_GATE_RECORD`` -- doing the same here would trigger the real
solve on every test run. So this test reads ``REPRODUCE_GATE_RECORD``
STATICALLY, via ``ast.parse`` + ``ast.literal_eval`` on just that one
assignment's value, and never imports or execs the module at all.

This case's reproduce-gate is anchored differently from cv07's/cv20's/
cv21's: Meep's own docs page for the bent-waveguide transmittance example
(``doc/docs/Python_Tutorials/Basics.md``, "Transmittance Spectrum of a
Waveguide Bend") publishes a plot, not a number
(``doc/docs/images/Tut-bend-flux.png``) -- verified directly against the
raw upstream markdown, 2026-09-10. So the reproduce-gate here anchors on
our own recorded run of upstream's OWN, unmodified
``examples/bend-flux.py`` (vendored verbatim at
``validation/crossval/_01_waveguide_bend_upstream/bend-flux.py``, see that
directory's ``PROVENANCE.md`` and the producer,
``scripts/diagnostics/waveguide_bend_tutorial_meep.py``), not on a
published value -- a weaker anchor than cv20's/cv21's, and the record says
so in its own text.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
from typing import Final

CROSSVAL_DIR: Final = pathlib.Path(__file__).resolve().parents[2] / "validation" / "crossval"
SCRIPT_PATH: Final = CROSSVAL_DIR / "01_waveguide_bend.py"
REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]

VENDORED_PATH: Final = CROSSVAL_DIR / "_01_waveguide_bend_upstream" / "bend-flux.py"
PROVENANCE_PATH: Final = CROSSVAL_DIR / "_01_waveguide_bend_upstream" / "PROVENANCE.md"
PRODUCER_PATH: Final = REPO_ROOT / "scripts" / "diagnostics" / "waveguide_bend_tutorial_meep.py"

EXPECTED_GIT_BLOB_SHA: Final = "f56ab6492a3cc55ebc1fc0c682c4981508c51955"


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _read_reproduce_gate_record() -> dict:
    """Statically extract REPRODUCE_GATE_RECORD's literal value from
    01_waveguide_bend.py WITHOUT importing or exec'ing the module (see
    module docstring: that file runs a real FDTD solve at import time,
    unconditionally)."""
    assert SCRIPT_PATH.exists(), f"missing crossval script {SCRIPT_PATH}"
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))
    for node in tree.body:
        targets = None
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        if not targets:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "REPRODUCE_GATE_RECORD":
                return ast.literal_eval(node.value)
    raise AssertionError(
        f"REPRODUCE_GATE_RECORD not found as a module-level assignment in {SCRIPT_PATH}"
    )


def test_the_source_file_has_no_main_guard_so_this_test_must_stay_static():
    """Documents WHY this file does static AST extraction instead of the
    importlib pattern cv07/cv20/cv21's header tests use -- if a future edit
    adds a `__main__` guard to 01_waveguide_bend.py, that would be safe to
    import and this whole file could be simplified to match the others.
    This test is a tripwire for that, not a requirement that it stay this
    way forever."""
    assert "__main__" not in SCRIPT_PATH.read_text(encoding="utf-8"), (
        "01_waveguide_bend.py now has a __main__ guard -- this header test "
        "was written assuming bare module-level execution (no guard) and "
        "deliberately avoids importing the module for that reason; if a "
        "guard now exists, switch this test to the importlib pattern "
        "tests/crossval/test_coax_two_port_referee_header.py uses instead "
        "of leaving this static-AST workaround in place unnecessarily."
    )


def test_reproduce_gate_record_has_required_fields():
    record = _read_reproduce_gate_record()
    required_fields = {
        "stage", "tutorial", "do_not_repeat", "geometry", "documented_check",
        "status", "reproduced_passivity_ok", "reproduced_shape_ok",
        "reproduced_meep_version", "log_path", "vessl_run_id", "verified_on",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"REPRODUCE_GATE_RECORD missing fields: {missing}"

    tutorial = record["tutorial"]
    tutorial_fields = {"repo", "path", "verified_present_on", "verified_via", "submodule_pin_note"}
    missing_tutorial = tutorial_fields - set(tutorial.keys())
    assert not missing_tutorial, f"tutorial sub-record missing fields: {missing_tutorial}"


def test_tutorial_citation_is_verifiable():
    record = _read_reproduce_gate_record()
    tutorial = record["tutorial"]
    assert tutorial["repo"], "tutorial citation needs a repo"
    assert tutorial["path"].endswith(".py"), "tutorial citation should point at a real source file"
    assert tutorial["verified_present_on"], "tutorial citation needs a verification date"
    assert tutorial["verified_via"], "tutorial citation needs a verification method"


def test_record_states_plainly_that_no_number_is_published():
    """This is the one thing that must not get lost in a future edit: the
    record must say, in its own text, that there is no published number to
    check against -- otherwise a reader could mistake this for a cv20/cv21-
    strength anchor."""
    record = _read_reproduce_gate_record()
    combined = " ".join([record["documented_check"], record["geometry"]])
    assert "NONE PUBLISHED" in record["documented_check"] or "no published" in combined.lower() or "not published" in combined.lower() or "NONE" in record["documented_check"]


def test_do_not_repeat_names_the_existing_legs_divergence():
    """The do_not_repeat field exists specifically so a future reader does
    not mistake cv01's existing hand-ported Meep comparator leg (:187-200)
    for a reproduction of the upstream tutorial."""
    record = _read_reproduce_gate_record()
    do_not_repeat = record["do_not_repeat"]
    assert "hand-port" in do_not_repeat or "hand-ported" in do_not_repeat
    assert "187" in do_not_repeat or "flux-region" in do_not_repeat


def test_reproduce_gate_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant: UNRUN <=> no numbers, no log path. Same
    shape tests/crossval/test_coax_two_port_referee_header.py's twin test
    enforces for cv21."""
    record = _read_reproduce_gate_record()
    if record["status"] == "UNRUN":
        assert record["reproduced_passivity_ok"] is None
        assert record["reproduced_shape_ok"] is None
        assert record["reproduced_meep_version"] is None
        assert record["log_path"] is None
        assert record["vessl_run_id"] is None
        assert record["verified_on"] is None
    else:
        assert record["status"] == "RUN", (
            f"unexpected status {record['status']!r}: only 'UNRUN' or 'RUN' are contract-valid"
        )
        assert record["reproduced_passivity_ok"] is not None
        assert record["reproduced_shape_ok"] is not None
        assert record["vessl_run_id"], "a filled record needs a vessl_run_id"
        assert record["verified_on"], "a filled record needs a verification date"
        log_path_str = record["log_path"]
        assert log_path_str, "a filled-in reproduce_gate_record needs a log_path"

        gitignored_prefixes = (".omx/", "docs/research_notes/vessl_logs/")
        tracked_prefixes = ("validation/crossval/_01_waveguide_bend_logs/",)
        assert not log_path_str.startswith(gitignored_prefixes), (
            f"log_path {log_path_str!r} lives under a GITIGNORED prefix -- "
            "a FILLED (status == 'RUN') record needs a log a reviewer "
            f"OUTSIDE this machine can open; use a tracked prefix instead: "
            f"{tracked_prefixes!r}"
        )
        assert log_path_str.startswith(tracked_prefixes), (
            f"log_path {log_path_str!r} must live under a TRACKED prefix "
            f"{tracked_prefixes!r} once status is RUN"
        )
        log_path = REPO_ROOT / log_path_str
        assert log_path.exists(), (
            f"reproduce_gate_record claims status={record['status']!r} but "
            f"its log_path {log_path} does not exist -- a claimed result "
            f"needs a real log, per external_solver_comparator.md step 2"
        )


def test_vendored_upstream_file_is_byte_identical_to_its_recorded_provenance():
    """The whole point of vendoring instead of porting: this must be
    independently checkable, not just claimed. Recomputes the git blob sha
    the same way `git hash-object` would, without shelling out to git."""
    assert VENDORED_PATH.exists(), f"missing vendored file {VENDORED_PATH}"
    assert PROVENANCE_PATH.exists(), f"missing provenance note {PROVENANCE_PATH}"
    assert PRODUCER_PATH.exists(), f"missing producer script {PRODUCER_PATH}"
    data = VENDORED_PATH.read_bytes()
    got = _git_blob_sha(data)
    assert got == EXPECTED_GIT_BLOB_SHA, (
        f"validation/crossval/_01_waveguide_bend_upstream/bend-flux.py has "
        f"drifted: git blob sha {got} != recorded {EXPECTED_GIT_BLOB_SHA}. "
        "If Meep's own tutorial changed upstream, re-fetch and update the "
        "recorded sha deliberately (here, in PROVENANCE.md, and in the "
        "producer script) -- do not hand-edit the vendored copy."
    )
    producer_text = PRODUCER_PATH.read_text(encoding="utf-8")
    assert EXPECTED_GIT_BLOB_SHA in producer_text, (
        "the producer script's own recorded EXPECTED_GIT_BLOB_SHA has "
        "diverged from this test's -- both must cite the same upstream blob"
    )
