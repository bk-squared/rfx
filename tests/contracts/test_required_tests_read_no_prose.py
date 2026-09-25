"""A test that reads prose is a documentation check, and a required lane runs none.

Required checks compare numbers with a tolerance, never text (PI, 2026-09-24),
and a documentation mismatch never holds a merge (PI, 2026-09-22). Four of the
nine red ``main`` runs of 2026-09-22/24 were a notes table compared with a
script's output whose last printed digit followed the OpenBLAS kernel (#1262).
So a test that reads a note, a guide or a README is marked ``docs_consistency``;
the required gates deselect that marker and the non-required docs-consistency
workflow runs it.

``tests/_prose_reads.py`` enforces it at run time: it fails a test without the
marker that opened prose, unless the test itself carries
``reads_docs_for_gate(reason=...)`` because the file is part of a gate (a
pre-declaration's frozen sections, a known-limitations entry that must stay),
and it fails the session when prose is opened outside any test. These tests
show that the plugin is wired into every session this repository runs, that
the required gates deselect the marker, and -- by running planted tests
through the plugin in a separate interpreter, against a planted ``docs/`` --
that it catches each way a test reads prose, refuses a bad opt-out, and passes
what it should pass. They also pin, as lists a reviewer reads in the diff,
which tests opt out and which tests are documentation.
"""

from __future__ import annotations

import ast
import functools
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from tests import _prose_reads

REPO = Path(__file__).resolve().parents[2]
PR_TESTS = REPO / ".github" / "workflows" / "pr-tests.yml"
DOCS_WORKFLOW = REPO / ".github" / "workflows" / "docs-consistency.yml"

# --------------------------------------------------------------------------
# Planted tests, run through the plugin in a separate interpreter
# --------------------------------------------------------------------------

#: Written into the planted session's root; points the plugin's idea of the
#: repository's docs/ at the planted one, so nothing here needs a real page.
_ROOT_CONFTEST = '''
    import os
    import tests._prose_reads as prose_reads
    prose_reads._DOCS = {docs!r} + os.sep
'''

_PLANTED = {
    # The pre-#1262 shape: a note's text against what a script prints.
    "test_note_equals_tables.py": '''
        from pathlib import Path
        NOTE = Path({note!r})

        def test_note_equals_tables():
            assert NOTE.read_text() == NOTE.read_text()
    ''',
    "test_marked.py": '''
        import pytest
        from pathlib import Path

        @pytest.mark.docs_consistency
        def test_marked_note_check():
            assert "gate" in Path({note!r}).read_text()
    ''',
    "test_builtin_open.py": '''
        def test_reads_through_open():
            with open({note!r}, encoding="utf-8") as fh:
                assert fh.read()
    ''',
    "test_module_level.py": '''
        from pathlib import Path
        TEXT = Path({note!r}).read_text()

        def test_uses_nothing_from_the_text():
            assert True
    ''',
    "test_cached_fixture.py": '''
        import pytest
        from pathlib import Path

        @pytest.fixture(scope="module")
        def note():
            return Path({note!r}).read_text()

        @pytest.mark.docs_consistency
        def test_first_user_is_marked(note):
            assert note

        def test_second_user_is_not(note):
            assert note
    ''',
    # A fixture the test asks for itself is charged to the test, first use
    # and cached reuse alike.
    "test_dynamic_fixture.py": '''
        import pytest
        from pathlib import Path

        @pytest.fixture(scope="module")
        def note_on_request():
            return Path({note!r}).read_text()

        def test_asks_for_the_note_itself(request):
            assert request.getfixturevalue("note_on_request")

        def test_asks_again_for_the_cached_note(request):
            assert request.getfixturevalue("note_on_request")
    ''',
    "test_gate_opt_out.py": '''
        import pytest
        from pathlib import Path

        @pytest.mark.reads_docs_for_gate(reason="pre-declaration hash")
        def test_reads_for_a_named_gate():
            assert Path({note!r}).read_text()

        @pytest.mark.reads_docs_for_gate(reason="  ")
        def test_blank_reason():
            assert True

        @pytest.mark.reads_docs_for_gate
        def test_no_reason():
            assert True
    ''',
    # An opt-out that covers tests nobody chose is refused.
    "test_module_opt_out.py": '''
        import pytest
        from pathlib import Path
        pytestmark = pytest.mark.reads_docs_for_gate(reason="the whole module")

        def test_under_a_module_opt_out():
            assert Path({note!r}).read_text()
    ''',
    "test_class_opt_out.py": '''
        import pytest

        @pytest.mark.reads_docs_for_gate(reason="the whole class")
        class TestUnderAClassOptOut:
            def test_under_a_class_opt_out(self):
                assert True
    ''',
    # A note read while pytest parametrizes one function counts for that
    # function only, not for its neighbours in the module.
    "test_generated_params.py": '''
        import pytest
        from pathlib import Path

        def pytest_generate_tests(metafunc):
            if metafunc.function.__name__.startswith("test_params_"):
                metafunc.parametrize(
                    "line", Path({note!r}).read_text().splitlines()[:1], ids=["n"])

        @pytest.mark.docs_consistency
        def test_params_marked(line):
            assert line is not None

        def test_params_unmarked(line):
            assert line is not None

        def test_neighbour_reads_nothing():
            assert True
    ''',
    "test_not_prose.py": '''
        from pathlib import Path

        def test_reads_configuration_a_record_and_a_readme():
            assert Path({config!r}).read_text()
            assert Path({record!r}).read_text()
            assert Path({readme!r}).read_text()   # outside docs/: not seen
    ''',
}

#: What the plugin must say about each planted test: passed, or failed at
#: teardown with the message that names the rule it broke.
_READS = "reads prose"
_REASON = "needs a non-empty reason"
_PLACE = "must decorate the one test function"
_EXPECTED = {
    "test_note_equals_tables": ("error", _READS),
    "test_marked_note_check": "passed",
    "test_reads_through_open": ("error", _READS),
    "test_uses_nothing_from_the_text": ("error", _READS),
    "test_first_user_is_marked": "passed",
    "test_second_user_is_not": ("error", _READS),
    "test_asks_for_the_note_itself": ("error", _READS),
    "test_asks_again_for_the_cached_note": ("error", _READS),
    "test_reads_for_a_named_gate": "passed",
    "test_blank_reason": ("error", _REASON),
    "test_no_reason": ("error", _REASON),
    "test_under_a_module_opt_out": ("error", _PLACE),
    "test_under_a_class_opt_out": ("error", _PLACE),
    "test_params_marked[n]": "passed",
    "test_params_unmarked[n]": ("error", _READS),
    "test_neighbour_reads_nothing": "passed",
    "test_reads_configuration_a_record_and_a_readme": "passed",
}


def _planted_docs(tmp_path: Path) -> dict[str, str]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# A note\n\nThe gate reads this.\n", encoding="utf-8")
    (docs / "matrix.json").write_text('{"status": "ok"}\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    return dict(docs=str(docs), note=str(docs / "note.md"), record=str(docs / "matrix.json"),
                readme=str(tmp_path / "README.md"), config=str(tmp_path / "pyproject.toml"))


def _run_planted(root: Path, paths: dict[str, str]) -> subprocess.CompletedProcess:
    (root / "conftest.py").write_text(
        textwrap.dedent(_ROOT_CONFTEST).format(**paths), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests._prose_reads", f"--junitxml={root / 'report.xml'}",
         "--rootdir", str(root), str(root)],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )


def _outcomes(tmp_path: Path) -> dict[str, object]:
    paths = _planted_docs(tmp_path)
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, body in _PLANTED.items():
        (suite / name).write_text(textwrap.dedent(body).format(**paths), encoding="utf-8")
    proc = _run_planted(suite, paths)
    report = suite / "report.xml"
    assert report.is_file(), proc.stdout + proc.stderr
    outcomes: dict[str, object] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        kinds = {child.tag for child in case}
        if "error" in kinds:
            message = next(c for c in case if c.tag == "error").get("message", "")
            which = [m for m in (_READS, _REASON, _PLACE) if m in message]
            outcomes[case.get("name")] = ("error", which[0] if which else message)
        else:
            outcomes[case.get("name")] = ("failed" if "failure" in kinds else
                                          "skipped" if "skipped" in kinds else "passed")
    return outcomes


def test_the_plugin_fails_each_unmarked_way_of_reading_prose(tmp_path: Path) -> None:
    assert _outcomes(tmp_path) == _EXPECTED


def test_prose_opened_outside_any_test_fails_the_session(tmp_path: Path) -> None:
    """Fail closed: a sub-directory conftest that reads a note at import.

    Its one test passes; the session does not.
    """
    paths = _planted_docs(tmp_path)
    root = tmp_path / "outside"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "conftest.py").write_text(
        f"from pathlib import Path\nTEXT = Path({paths['note']!r}).read_text()\n",
        encoding="utf-8")
    (root / "sub" / "test_plain.py").write_text(
        "def test_plain():\n    assert True\n", encoding="utf-8")
    proc = _run_planted(root, paths)
    out = proc.stdout + proc.stderr
    assert "1 passed" in out, out
    assert proc.returncode == pytest.ExitCode.TESTS_FAILED, out
    assert "prose was opened outside any test" in out and "note.md" in out, out


def test_the_plugin_is_loaded_by_the_root_conftest(request) -> None:
    """Every session in this repository, required lanes included, runs it."""
    assert request.config.pluginmanager.has_plugin("tests._prose_reads")


def test_the_audit_hook_is_recording_in_this_session(request, tmp_path, monkeypatch) -> None:
    """Not only registered: this session's hook sees an open() of a prose file.

    Against a planted docs/, and the record is removed before teardown, so this
    unmarked test is not failed for the read it makes on purpose.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "page.md").write_text("text\n", encoding="utf-8")
    monkeypatch.setattr(_prose_reads, "_DOCS", str(docs) + os.sep)
    (docs / "page.md").read_text(encoding="utf-8")
    seen = _prose_reads._reads.pop(request.node.nodeid, [])
    assert any(p.endswith("page.md") for p in seen), seen


@pytest.mark.parametrize("path", [
    "docs/design_notes/x.md", "docs/public/guide/y.mdx", "docs/guides/z.rst",
])
def test_prose_is_what_it_says(path: str) -> None:
    assert _prose_reads.is_prose(REPO / path)


@pytest.mark.parametrize("path", [
    "docs/guides/sparameter_support_matrix.json", "CHANGELOG.md", "README.md",
    "changelog.d/1.fixed.md", "scripts/ci/DURATIONS.md", "pyproject.toml",
    "validation/crossval/README.md", "/tmp/elsewhere/docs/x.md",
])
def test_everything_else_is_not_prose(path: str) -> None:
    assert not _prose_reads.is_prose(path if path.startswith("/") else REPO / path)


def _pytest_runs(path: Path) -> list[str]:
    runs = []
    for job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].values():
        for step in job.get("steps", []):
            run = str(step.get("run", ""))
            if "pytest" in run and '-m "' in run:
                runs.append(run)
    return runs


def test_the_required_gates_deselect_the_marker_and_the_docs_lane_selects_it() -> None:
    """Both pr-tests gates override addopts, so each needs the marker in -m."""
    required = _pytest_runs(PR_TESTS)
    assert len(required) >= 2, required
    for run in required:
        assert "not docs_consistency" in run, run
    docs = _pytest_runs(DOCS_WORKFLOW)
    assert docs and all('-m "docs_consistency and not gpu"' in run for run in docs), docs
    addopts = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "and not docs_consistency'\"" in addopts


# --------------------------------------------------------------------------
# Who opts out, and what is documentation: pinned, so a change shows in the diff
# --------------------------------------------------------------------------

#: Every required test allowed to read prose, and why. Adding one is a gate
#: decision; it goes here, in the same change, where a reviewer reads it.
GATE_OPT_OUTS = {
    "tests/contracts/test_known_limitations_citations.py::"
    "test_every_pinned_issue_keeps_its_entry":
        "a known-limitations entry must stay while its defect is open: the one "
        "page a PR must keep right (PI, 2026-09-22)",
    "tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py::"
    "test_sections_0_to_7_are_the_pre_declaration_as_committed":
        "pre-declaration hash: sections 0-7 hold the windows and bars fixed "
        "before any arm ran; editing them after the result would loosen the gate",
}

#: Every test function with at least one docs_consistency case. Marking a test
#: as documentation takes it out of the required lanes, so a new entry here is
#: something a reviewer should see: a number check must not leave this way.
DOCS_CONSISTENCY_TESTS = frozenset({
    "tests/contracts/test_ci_workflows_contract.py::test_local_sh_and_the_runbook_list_the_same_steps_in_the_same_order",
    "tests/contracts/test_ci_workflows_contract.py::test_the_runbook_says_changes_must_be_a_required_check",
    "tests/contracts/test_ci_workflows_contract.py::test_the_runbook_tells_authors_to_run_it_before_every_push",
    "tests/contracts/test_empty_window_gradient_caveat_docpin.py::test_autodiff_guide_pins_fd_necessary_not_sufficient",
    "tests/contracts/test_empty_window_gradient_caveat_docpin.py::test_gradient_behavior_guide_pins_passing_witness_caveat",
    "tests/contracts/test_empty_window_gradient_caveat_docpin.py::test_inverse_design_guide_pins_split_window_precondition",
    "tests/contracts/test_empty_window_gradient_caveat_docpin.py::test_objective_docstring_pins_split_window_precondition",
    "tests/contracts/test_empty_window_gradient_caveat_docpin.py::test_sources_ports_guide_pins_lumped_rlc_not_a_port",
    "tests/contracts/test_evidence_numeric_provenance.py::test_each_classification_holds_mechanically",
    "tests/contracts/test_evidence_numeric_provenance.py::test_each_registered_site_still_carries_its_references",
    "tests/contracts/test_evidence_numeric_provenance.py::test_every_enumerated_document_is_classified",
    "tests/contracts/test_evidence_numeric_provenance.py::test_every_number_a_note_cites_matches_its_artifact",
    "tests/contracts/test_evidence_numeric_provenance.py::test_the_cited_population_is_still_present",
    "tests/contracts/test_guide_citations_resolve.py::test_every_relative_link_is_in_the_repository",
    "tests/contracts/test_known_limitations_citations.py::test_every_citation_link_points_at_the_issue_it_names",
    "tests/contracts/test_known_limitations_citations.py::test_no_entry_names_an_issue_it_does_not_cite",
    "tests/contracts/test_known_limitations_citations.py::test_the_page_cites_no_issue_outside_the_pinned_set",
    "tests/contracts/test_known_limitations_citations.py::test_the_page_is_reachable_from_the_readme_and_the_support_matrix",
    "tests/contracts/test_rcs_bistatic_caveat_docpin.py::test_compute_rcs_docstring_pins_bistatic_caveat",
    "tests/contracts/test_rcs_bistatic_caveat_docpin.py::test_public_guide_pins_bistatic_caveat",
    "tests/contracts/test_rcs_bistatic_caveat_docpin.py::test_rcsresult_docstring_pins_validation_scope",
    "tests/contracts/test_retired_msl_claims_are_gone.py::test_no_page_states_a_retired_msl_claim",
    "tests/contracts/test_support_matrix_parity.py::test_ad_traceable_no_is_not_overclaimed_in_markdown",
    "tests/contracts/test_support_matrix_parity.py::test_api_summary_status_cell_agrees_with_section_status",
    "tests/contracts/test_support_matrix_parity.py::test_json_numeric_claims_appear_in_markdown",
    "tests/contracts/test_support_matrix_parity.py::test_lane_map_headers_exist_in_markdown",
    "tests/contracts/test_support_matrix_parity.py::test_markdown_run_ids_appear_in_json",
    "tests/contracts/test_support_matrix_parity.py::test_status_token_polarity_agrees",
    "tests/contracts/test_test_durations_provenance.py::test_the_doc_reports_a_carried_count_that_fits_the_file",
    "tests/contracts/test_test_durations_provenance.py::test_the_doc_reports_the_entry_count_the_file_has",
    "tests/contracts/test_two_plane_gone_from_docs_and_scripts.py::test_no_allowlist_entry_is_stale",
    "tests/contracts/test_two_plane_gone_from_docs_and_scripts.py::test_no_live_two_plane_outside_the_allowlist",
    "tests/contracts/test_two_plane_gone_from_docs_and_scripts.py::test_no_pending_entry_is_stale",
    "tests/crossval/test_patch_msl_public_carriers.py::test_msl_current_replay_quotes_fixture_and_labels_historical_openems",
    "tests/crossval/test_patch_msl_public_carriers.py::test_msl_producer_current_geometry_quotes_committed_metadata",
    "tests/crossval/test_patch_msl_public_carriers.py::test_patch_demo_directivity_comment_uses_current_measurement",
    "tests/crossval/test_patch_msl_public_carriers.py::test_patch_farfield_beam_peak_prose_quotes_committed_cut_angles",
    "tests/unit/autodiff/test_ad_diagnostics.py::test_memory_reduction_docs_include_residual_inspection_boundary",
    "tests/unit/autodiff/test_estimate_ad_memory.py::test_memory_reduction_docs_separate_planning_from_certificate_evidence",
    "tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py::test_the_notes_results_are_what_tables_prints",
    "tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py::test_the_second_notes_results_are_still_what_its_tables_print",
    "tests/unit/nonuniform/test_msl_notch_fz_replay.py::test_the_notes_f6_is_the_instruments_f6_byte_for_byte",
    "tests/unit/nonuniform/test_results_git_sha_resolves.py::test_every_orphaned_sha_is_named_in_its_note",
    "tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_predeclaration_7_2_carries_an_explicit_supersession_note",
    "tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_predeclaration_7_2_describes_the_board_stage_2_actually_builds",
    "tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py::test_note_results_section_quotes_the_record",
})


def _mentions(node: ast.AST, defs: dict[str, ast.AST], marker: str) -> bool:
    """``pytest.mark.<marker>`` in the node, or in a module-level def or
    assignment the node names directly (so ``@_by_kind()`` counts the mark
    ``_by_kind`` builds). An attribute, not a word: a string that merely
    contains the marker's name (a ``-m`` expression) is not a mark."""
    named = [defs[n.id] for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in defs]
    return any(isinstance(a, ast.Attribute) and a.attr == marker
               and isinstance(a.value, ast.Attribute) and a.value.attr == "mark"
               for cur in [node, *named] for a in ast.walk(cur))


def _gate_reason(decorators: list[ast.expr]) -> str | None:
    for dec in decorators:
        for call in ast.walk(dec):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == _prose_reads.GATE_MARKER):
                reason = [k.value for k in call.keywords if k.arg == "reason"]
                try:
                    return ast.literal_eval(reason[0]) if reason else ""
                except ValueError:
                    return "<not a literal string>"
    return None


@functools.lru_cache(maxsize=None)
def _scan() -> tuple[dict[str, str], frozenset[str], tuple[str, ...]]:
    """Static: opt-outs with reasons, docs_consistency functions, and any
    opt-out placed on a module or class."""
    opt_outs: dict[str, str] = {}
    docs: set[str] = set()
    misplaced: list[str] = []
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if _prose_reads.MARKER not in text and _prose_reads.GATE_MARKER not in text:
            continue
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(text)
        defs: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defs[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defs[target.id] = node
        module_marks = [n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)]
        module_docs = any(_mentions(m, defs, _prose_reads.MARKER) for m in module_marks)
        if any(_mentions(m, {}, _prose_reads.GATE_MARKER) for m in module_marks):
            misplaced.append(rel)
        functions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append((node.name, node, []))
            elif isinstance(node, ast.ClassDef):
                if _gate_reason(node.decorator_list) is not None:
                    misplaced.append(f"{rel}::{node.name}")
                functions += [(f"{node.name}::{f.name}", f, node.decorator_list)
                              for f in node.body if isinstance(f, ast.FunctionDef)]
        for name, fn, class_decorators in functions:
            if not fn.name.startswith("test"):
                continue
            decorators = list(fn.decorator_list) + list(class_decorators)
            if module_docs or any(_mentions(d, defs, _prose_reads.MARKER) for d in decorators):
                docs.add(f"{rel}::{name}")
            reason = _gate_reason(fn.decorator_list)
            if reason is not None:
                opt_outs[f"{rel}::{name}"] = reason
    return opt_outs, frozenset(docs), tuple(misplaced)


def test_the_opt_outs_are_exactly_the_pinned_ones_with_their_reasons() -> None:
    opt_outs, _, _ = _scan()
    assert opt_outs == GATE_OPT_OUTS, (
        "the required tests allowed to read prose changed. Each one is a gate "
        "decision: edit GATE_OPT_OUTS in this file in the same change, with the "
        f"reason.\n  found:  {opt_outs}\n  pinned: {GATE_OPT_OUTS}")


def test_no_opt_out_sits_on_a_module_or_class() -> None:
    _, _, misplaced = _scan()
    assert not misplaced, (
        f"reads_docs_for_gate on a module or class opts out tests nobody chose: "
        f"{list(misplaced)}. Put it on the one test function that reads the file.")


def test_the_documentation_tests_are_exactly_the_pinned_ones() -> None:
    _, docs, _ = _scan()
    added, removed = sorted(docs - DOCS_CONSISTENCY_TESTS), sorted(DOCS_CONSISTENCY_TESTS - docs)
    assert not added and not removed, (
        "the set of docs_consistency tests changed. Marking a test as "
        "documentation takes it out of the required lanes, so it is listed in "
        "DOCS_CONSISTENCY_TESTS in the same change; a test that pins a number "
        f"does not belong there.\n  newly marked: {added}\n  no longer marked: {removed}")
