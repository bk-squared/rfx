"""Local selection contracts: AST-only decisions and recorded-cost accounting."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gate_test_selector", ROOT / "scripts/ci/select_gate_tests.py")
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


@pytest.fixture
def sources():
    return {
        "tests/unit/test_debye.py": "from rfx.materials.debye import Debye",
        "tests/unit/test_import.py": "import rfx.materials.debye as d",
        "tests/unit/test_parent.py": "from rfx.materials import debye as d",
        "tests/unit/test_top.py": "from rfx import Simulation",
        "tests/unit/test_bare.py": "import rfx",
        "tests/unit/test_other.py": 'text = "import rfx.materials.debye"',
        "tests/unit/test_child.py": "import rfx.materials.debye.child",
        "tests/unit/runners/test_light.py": "",
        "tests/unit/runners/test_heavy.py": "",
        "tests/unit/autodiff/test_boundary.py": "",
        "tests/unit/autodiff/test_unknown.py": "",
        "tests/contracts/test_contract.py": "from rfx.materials.debye import Debye",
    }


@pytest.fixture
def durations():
    return {
        "tests/unit/runners/test_light.py::test_x[a]": 100,
        "tests/unit/runners/test_heavy.py::test_x[a]": 200,
        "tests/unit/runners/test_heavy.py::test_x[b]": 201,
        "tests/unit/autodiff/test_boundary.py::test_x": 400,
        "tests/unit/test_top.py::test_x": 12.5,
        "tests/contracts/test_contract.py::test_x": 33,
    }


def test_no_rfx_or_test_changes_select_nothing(sources, durations):
    result = selector.select_tests(["docs/example.md", "scripts/tool.py"], sources, durations)
    assert result.files == ()


def test_debye_selects_only_direct_imports(sources, durations):
    result = selector.select_tests(["rfx/materials/debye.py"], sources, durations)
    assert set(result.files) == {
        "tests/unit/test_debye.py", "tests/unit/test_import.py", "tests/unit/test_parent.py",
    }


def test_central_change_adds_runners_and_autodiff_except_over_budget(sources, durations):
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations)
    assert set(result.files) == {
        "tests/unit/runners/test_light.py", "tests/unit/autodiff/test_boundary.py",
        "tests/unit/autodiff/test_unknown.py",
    }
    assert result.excluded == {"tests/unit/runners/test_heavy.py": 401}


def test_changed_test_overrides_budget(sources, durations):
    heavy = "tests/unit/runners/test_heavy.py"
    result = selector.select_tests(["rfx/api/_execute.py", heavy], sources, durations)
    assert heavy in result.files
    assert not result.excluded
    assert selector.select_tests([heavy], sources, durations).files == (heavy,)


def test_not_run_summary_matches_recorded_tests(sources, durations):
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations)
    assert result.not_run_count == 3
    assert result.not_run_seconds == 413.5
    assert result.selected_seconds == 500
    assert "3 recorded tests, 413.500000 s" in result.summary()
    assert "tests/unit/runners/test_heavy.py: 401.000000 s" in result.summary()
    empty = selector.select_tests([], sources, durations)
    assert empty.not_run_count == 5
    assert empty.not_run_seconds == 913.5


def test_execute_selects_real_regressions():
    paths = (
        "tests/unit/runners/test_component_h_materials.py",
        "tests/unit/autodiff/test_msl_forward_identity.py",
    )
    sources = {path: (ROOT / path).read_text() for path in paths}
    durations = json.loads((ROOT / ".test_durations").read_text())
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations)
    assert set(result.files) == set(paths)


@pytest.mark.parametrize("changed,expected", [
    ("rfx/__init__.py", {"tests/unit/test_top.py", "tests/unit/test_bare.py"}),
    ("rfx/materials/__init__.py", {"tests/unit/test_parent.py"}),
])
def test_package_init_imports(changed, expected, sources):
    assert set(selector.select_tests([changed], sources, {}).files) == expected


def test_expensive_direct_import_is_excluded_unless_edited(sources):
    path = "tests/unit/test_debye.py"
    durations = {f"{path}::test_x": 401}
    assert path not in selector.select_tests(["rfx/materials/debye.py"], sources, durations).files
    assert path in selector.select_tests(["rfx/materials/debye.py", path], sources, durations).files


def test_deleted_test_and_contract_are_not_selected(sources):
    result = selector.select_tests([
        "tests/unit/test_deleted.py", "tests/contracts/test_contract.py",
    ], sources, {})
    assert result.files == ()


@pytest.mark.parametrize("path", [
    "rfx/simulation.py", "rfx/core/example.py", "rfx/runners/example.py",
    "rfx/nonuniform.py", "rfx/measurement/example.py", "rfx/model/example.py",
])
def test_other_central_paths(path, sources, durations):
    assert "tests/unit/runners/test_light.py" in selector.select_tests([path], sources, durations).files


def test_git_diff_arguments_and_untracked_worktree(monkeypatch):
    calls = []

    def git(args, **kwargs):
        calls.append(args[1:])
        return {"merge-base": "abc\n", "diff": "rfx/materials/debye.py\0",
                "ls-files": "tests/unit/test_new.py\0"}[args[1]]

    monkeypatch.setattr(selector.subprocess, "check_output", git)
    assert selector.changed_paths("base", None, ROOT) == [
        "rfx/materials/debye.py", "tests/unit/test_new.py",
    ]
    assert calls == [["merge-base", "base", "HEAD"],
                     ["diff", "--name-only", "-z", "abc", "--"],
                     ["ls-files", "--others", "--exclude-standard", "-z"]]
    calls.clear()
    assert selector.changed_paths("base", "head", ROOT) == ["rfx/materials/debye.py"]
    assert calls == [["merge-base", "base", "head"],
                     ["diff", "--name-only", "-z", "abc", "head", "--"]]


def test_local_stage_numbers_and_separate_session():
    text = (ROOT / "scripts/ci/local.sh").read_text()
    names = re.search(r"^STEP_NAMES=\(([^)]*)\)", text, re.M).group(1).split()
    assert names[-2:] == ["contract-tests", "selected-tests"]
    assert [int(n) for n in re.findall(r"^begin (\d+)$", text, re.M)] == list(range(len(names)))
    stage = text.split("\nbegin 8\n")[1]
    assert '"$PYTHON" -m pytest "${selected_tests[@]}" -q' in stage
    assert ' -x' not in stage
    assert '-o addopts="" -m "not gpu and not docs_consistency" --strict-markers' in stage
    assert 'echo "nothing selected"' in stage
    assert stage.index('cat "$selection_dir/summary"') < stage.index('[ "$selected_status" -eq 0 ] || fail')


@pytest.mark.parametrize("source", ["import rfx", "from rfx import nonuniform"])
def test_top_level_imports_only_match_root_init(source):
    assert not selector.directly_imports(source, {"rfx.nonuniform"})
    assert selector.directly_imports(source, {"rfx"})


@pytest.mark.parametrize("source", [
    "import rfx.materials as materials",
    "from rfx.materials import Debye",
    "from rfx import materials",
])
def test_package_import_forms_respect_top_level_exception(source):
    assert selector.directly_imports(source, {"rfx.materials"}) == (source != "from rfx import materials")


@pytest.mark.parametrize("summary_file", [False, True])
def test_cli_stdout_and_summary(monkeypatch, tmp_path, capsys, summary_file):
    duration_path = tmp_path / "durations.json"
    duration_path.write_text(json.dumps({"tests/unit/test_new.py::test_x": 9}))
    summary_path = tmp_path / "summary.txt"
    args = ["select_gate_tests.py", "--base", "base", "--head", "head",
            "--durations", str(duration_path)]
    if summary_file:
        args += ["--summary", str(summary_path)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(selector.subprocess, "check_output", lambda *a, **k: str(ROOT))
    monkeypatch.setattr(selector, "changed_paths", lambda *a: ["tests/unit/test_new.py"])
    monkeypatch.setattr(selector, "read_test_sources", lambda *a: {"tests/unit/test_new.py": ""})
    assert selector.main() == 0
    output = capsys.readouterr()
    assert output.out == "tests/unit/test_new.py\n"
    summary = summary_path.read_text() if summary_file else output.err
    assert "1 files, recorded 9.000000 s" in summary
    assert "0 recorded tests, 0.000000 s" in summary
    if summary_file:
        assert output.err == ""
