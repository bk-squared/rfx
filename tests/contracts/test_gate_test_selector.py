"""Local selection contracts: AST-only decisions and recorded-cost accounting."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import tomllib
import sys
from pathlib import Path

import pytest

from scripts.ci.source_scan import STAGE2_EXCLUDED_MARKERS

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


def test_package_directory_selects_geometry_through_top_level_helper():
    sources = {
        "tests/unit/geometry/test_x.py": "from tests._fixture import build",
        "tests/_fixture.py": "from rfx import Simulation\ndef build(): return Simulation()",
        "tests/unit/rasterize_grid/test_wrong.py": "",
        "tests/unit/geometry_extra/test_other.py": "",
    }
    result = selector.select_tests(["rfx/geometry/rasterize_grid.py"], sources, {
        "tests/unit/geometry/test_x.py::test_x[a]": 3,
        "tests/unit/geometry/test_x.py::test_x[b]": 4,
    })
    assert result.files == ("tests/unit/geometry/test_x.py",)
    assert result.selected_seconds == 7
    assert "package directory tests added: 1 files, recorded 7.000000 s" in result.summary()


@pytest.mark.parametrize("changed", [
    "rfx/sources/ports/x.py", "rfx/sources/__init__.py", "rfx/sources/ports/data.json",
])
def test_package_directory_uses_first_package_at_any_depth(changed):
    result = selector.select_tests([changed], {
        "tests/unit/sources/test_root.py": "",
        "tests/unit/sources/ports/test_nested.py": "",
        "tests/unit/ports/test_other.py": "",
    }, {})
    assert result.files == (
        "tests/unit/sources/ports/test_nested.py", "tests/unit/sources/test_root.py",
    )
    assert "package directory tests added: 2 files, recorded 0.000000 s" in result.summary()


@pytest.mark.parametrize("changed", ["rfx/fidelity.py", "rfx/absent/x.py"])
def test_package_directory_missing_or_top_level_adds_nothing(changed):
    result = selector.select_tests([changed], {
        "tests/unit/fidelity/test_x.py": "", "tests/unit/geometry/test_x.py": "",
    }, {})
    assert result.files == ()
    assert "package directory tests added: 0 files, recorded 0.000000 s" in result.summary()


@pytest.mark.parametrize("edited", [False, True])
def test_package_directory_budget_excludes_heavy_unless_edited(edited):
    path = "tests/unit/geometry/test_heavy.py"
    changed = ["rfx/geometry/rasterize_grid.py"]
    if edited:
        changed.append(path)
    result = selector.select_tests(changed, {
        path: "", "tests/unit/geometry/test_boundary.py": "",
    }, {
        "tests/unit/geometry/test_heavy.py::test_x[a]": 200,
        "tests/unit/geometry/test_heavy.py::test_x[b]": 201,
        "tests/unit/geometry/test_boundary.py::test_x": 400,
    })
    if edited:
        assert result.files == (
            "tests/unit/geometry/test_boundary.py", "tests/unit/geometry/test_heavy.py",
        )
        assert result.excluded == {}
    else:
        assert result.files == ("tests/unit/geometry/test_boundary.py",)
        assert result.excluded == {"tests/unit/geometry/test_heavy.py": 401}
    assert "package directory tests added: 1 files, recorded 400.000000 s" in result.summary()


def test_package_directory_never_adds_contracts():
    result = selector.select_tests([
        "rfx/geometry/rasterize_grid.py", "rfx/contracts/x.py",
        "tests/contracts/geometry/test_x.py",
    ], {
        "tests/contracts/geometry/test_x.py": "import rfx.geometry.rasterize_grid",
        "tests/contracts/test_x.py": "from rfx import Simulation",
        "tests/unit/geometry/test_x.py": "",
    }, {})
    assert result.files == ("tests/unit/geometry/test_x.py",)


def test_package_directory_summary_counts_only_new_candidates():
    result = selector.select_tests([
        "rfx/runners/x.py", "rfx/runners/nested/y.py", "tests/unit/runners/test_edited.py",
    ], {
        "tests/unit/runners/test_direct.py": "import rfx.runners.x",
        "tests/unit/runners/test_edited.py": "",
        "tests/unit/runners/test_added.py": "",
    }, {"tests/unit/runners/test_added.py::test_x": 5}, central_paths=True)
    assert result.files == (
        "tests/unit/runners/test_added.py", "tests/unit/runners/test_direct.py",
        "tests/unit/runners/test_edited.py",
    )
    assert "package directory tests added: 0 files, recorded 0.000000 s" in result.summary()


def test_geometry_change_selects_oracle_that_imports_no_geometry():
    # #2005 shape: rfx/geometry changed, the oracle reaches it only through rfx.Simulation.
    sources = {
        "tests/oracle/test_leontovich.py": "from rfx import Simulation",
        "tests/oracle/test_other_oracle.py": "",
        "tests/locks/test_lock.py": "import numpy",
        "tests/unit/sparams/test_unrelated.py": "",
    }
    result = selector.select_tests(["rfx/geometry/rasterize_grid.py"], sources, {
        "tests/oracle/test_leontovich.py::test_x": 27.5,
        "tests/oracle/test_other_oracle.py::test_x[a]": 1,
        "tests/locks/test_lock.py::test_x": 2,
    })
    assert result.files == (
        "tests/locks/test_lock.py", "tests/oracle/test_leontovich.py",
        "tests/oracle/test_other_oracle.py",
    )
    assert result.selected_seconds == 30.5
    assert "oracle and lock tests added: 3 files, recorded 30.500000 s" in result.summary()


@pytest.mark.parametrize("changed", [
    "rfx/vmap_sweep.py", "rfx/__init__.py", "rfx/sources/ports/data.json",
])
def test_any_rfx_change_selects_oracle_and_locks(changed):
    result = selector.select_tests([changed], {
        "tests/oracle/test_o.py": "", "tests/locks/nested/test_l.py": "",
        "tests/oracles/test_not.py": "", "tests/unit/test_u.py": "",
    }, {})
    assert result.files == ("tests/locks/nested/test_l.py", "tests/oracle/test_o.py")


@pytest.mark.parametrize("changed", [
    ["docs/guide.md"], ["scripts/ci/local.sh"], ["tests/unit/test_u.py"], ["rfxsomething/x.py"],
])
def test_oracle_and_locks_not_added_without_rfx_change(changed):
    result = selector.select_tests(changed, {
        "tests/oracle/test_o.py": "", "tests/locks/test_l.py": "", "tests/unit/test_u.py": "",
    }, {})
    assert "tests/oracle/test_o.py" not in result.files
    assert "tests/locks/test_l.py" not in result.files
    assert "oracle and lock tests added: 0 files, recorded 0.000000 s" in result.summary()


@pytest.mark.parametrize("edited", [False, True])
def test_oracle_and_locks_budget_excludes_heavy_unless_edited(edited):
    heavy = "tests/locks/test_heavy.py"
    changed = ["rfx/vmap_sweep.py"] + ([heavy] if edited else [])
    result = selector.select_tests(changed, {heavy: "", "tests/oracle/test_light.py": ""}, {
        "tests/locks/test_heavy.py::test_x[a]": 300,
        "tests/locks/test_heavy.py::test_x[b]": 101,
        "tests/oracle/test_light.py::test_x": 400,
    })
    if edited:
        assert result.files == ("tests/locks/test_heavy.py", "tests/oracle/test_light.py")
        assert "oracle and lock tests added: 1 files, recorded 400.000000 s" in result.summary()
    else:
        assert result.files == ("tests/oracle/test_light.py",)
        assert result.excluded == {"tests/locks/test_heavy.py": 401}
        assert "oracle and lock tests added: 1 files, recorded 400.000000 s" in result.summary()


def test_oracle_summary_counts_only_files_no_other_rule_chose():
    result = selector.select_tests(["rfx/geometry/x.py", "tests/oracle/test_edited.py"], {
        "tests/oracle/test_edited.py": "",
        "tests/oracle/test_direct.py": "import rfx.geometry.x",
        "tests/oracle/test_added.py": "",
    }, {"tests/oracle/test_added.py::test_x": 5, "tests/oracle/test_direct.py::test_x": 7})
    assert result.files == (
        "tests/oracle/test_added.py", "tests/oracle/test_direct.py", "tests/oracle/test_edited.py",
    )
    assert "oracle and lock tests added: 1 files, recorded 5.000000 s" in result.summary()


def test_central_change_adds_runners_and_autodiff_except_over_budget(sources, durations):
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations, central_paths=True)
    assert set(result.files) == {
        "tests/unit/runners/test_light.py", "tests/unit/autodiff/test_boundary.py",
        "tests/unit/autodiff/test_unknown.py",
    }
    assert result.excluded == {"tests/unit/runners/test_heavy.py": 401}


def test_execute_defaults_to_only_direct_imports(sources, durations):
    direct = "tests/unit/runners/test_direct.py"
    sources[direct] = "from rfx.api._execute import execute"
    sources["tests/unit/test_execute.py"] = "import rfx.api._execute"
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations)
    assert set(result.files) == {direct, "tests/unit/test_execute.py"}
    assert not result.excluded


def test_disabled_central_summary_reports_full_set(sources, durations):
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations)
    line = (
        "central paths changed; full runners/autodiff suite not run (--central-paths disabled): "
        "4 files, recorded 901.000000 s"
    )
    assert line in result.summary().splitlines()
    for changed, enabled in [(["rfx/api/_execute.py"], True), ([], False),
                             (["rfx/materials/debye.py"], False)]:
        assert "central paths changed" not in selector.select_tests(
            changed, sources, durations, central_paths=enabled,
        ).summary()


@pytest.mark.parametrize("value,expected", [(None, []), ("", []), ("0", []),
                                           ("true", []), ("1", ["--central-paths"])])
def test_local_central_flag_is_opt_in(value, expected):
    stage = (ROOT / "scripts/ci/local.sh").read_text().split("\nbegin 9\n")[1]
    fragment = stage.split("central_path_args=()", 1)[1].split('"$PYTHON"', 1)[0]
    assert 'scripts/ci/select_gate_tests.py ${central_path_args[@]+"${central_path_args[@]}"}' in stage
    env = dict(os.environ)
    env.pop("RFX_GATE_CENTRAL_PATHS", None)
    if value is not None:
        env["RFX_GATE_CENTRAL_PATHS"] = value
    result = subprocess.run(
        ["bash", "-uc", "central_path_args=()" + fragment
         + 'for arg in ${central_path_args[@]+"${central_path_args[@]}"}; do echo "$arg"; done'],
        env=env, capture_output=True, text=True, check=True, timeout=10,
    )
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_central_paths_switch(monkeypatch, tmp_path, capsys, enabled):
    durations = tmp_path / "durations.json"
    durations.write_text("{}")
    args = ["select_gate_tests.py", "--base", "base", "--durations", str(durations)]
    if enabled:
        args.append("--central-paths")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(selector.subprocess, "check_output", lambda *a, **k: str(ROOT))
    monkeypatch.setattr(selector, "changed_paths", lambda *a: ["rfx/api/_execute.py"])
    monkeypatch.setattr(selector, "read_test_sources", lambda *a: {
        "tests/unit/runners/test_runner.py": "",
    })
    assert selector.main() == 0
    output = capsys.readouterr()
    assert output.out == ("tests/unit/runners/test_runner.py\n" if enabled else "")
    assert ("central paths changed" in output.err) is not enabled


def test_changed_test_overrides_budget(sources, durations):
    heavy = "tests/unit/runners/test_heavy.py"
    result = selector.select_tests(["rfx/api/_execute.py", heavy], sources, durations, central_paths=True)
    assert heavy in result.files
    assert not result.excluded
    assert selector.select_tests([heavy], sources, durations).files == (heavy,)


def test_not_run_summary_matches_recorded_tests(sources, durations):
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations, central_paths=True)
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
    result = selector.select_tests(["rfx/api/_execute.py"], sources, durations, central_paths=True)
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
    "rfx/stepping/example.py",
])
def test_other_central_paths(path, sources, durations):
    assert "tests/unit/runners/test_light.py" in selector.select_tests([path], sources, durations, central_paths=True).files


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
                     ["diff", "--name-only", "--no-renames", "-z", "abc", "--"],
                     ["ls-files", "--others", "--exclude-standard", "-z"]]
    calls.clear()
    assert selector.changed_paths("base", "head", ROOT) == ["rfx/materials/debye.py"]
    assert calls == [["merge-base", "base", "head"],
                     ["diff", "--name-only", "--no-renames", "-z", "abc", "head", "--"]]


def test_local_stage_numbers_and_separate_session():
    text = (ROOT / "scripts/ci/local.sh").read_text()
    names = re.search(r"^STEP_NAMES=\(([^)]*)\)", text, re.M).group(1).split()
    assert names[-2:] == ["contract-tests", "selected-tests"]
    assert [int(n) for n in re.findall(r"^begin (\d+)$", text, re.M)] == list(range(len(names)))
    stage = text.split("\nbegin 9\n")[1]
    assert 'stage2_pytest selected "${selected_tests[@]}"' in stage
    assert '"$PYTHON" -m pytest "${@:2}" -q' in stage
    assert ' -x' not in stage
    assert '-o addopts="" -m ' in stage
    assert "selection_dir=$(mktemp -d) || fail" in stage
    assert 'echo "nothing selected"' in stage
    assert stage.index('cat "$selection_dir/summary"') < stage.index('if [ "$outcome_status" -eq 5 ]; then')


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


@pytest.mark.parametrize("changed", ["rfx/core_x.py", "rfx/runners.py", "rfx/nonuniform_extra.py"])
def test_similar_names_are_not_central_paths(changed, sources, durations):
    assert selector.select_tests([changed], sources, durations, central_paths=True).files == ()


@pytest.mark.parametrize("changed", [[], ["rfx/materials/debye.py"]])
def test_syntax_error_is_selected_even_over_budget(changed, sources):
    broken = "tests/unit/test_broken.py"
    sources[broken] = "def test_broken(:\n    pass\n"
    result = selector.select_tests(changed, sources, {f"{broken}::test_broken": 401})
    assert broken in result.files
    assert broken not in result.excluded


def test_selected_stage_markers_match_pyproject():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    options = shlex.split(config["tool"]["pytest"]["ini_options"]["addopts"])
    default_expression = options[options.index("-m") + 1]
    stage = (ROOT / "scripts/ci/local.sh").read_text().split("\nbegin 9\n")[1]
    stage_expression = re.search(r'-m "([^"]+)"', stage).group(1)

    def excluded_markers(expression):
        terms = expression.split(" and ")
        assert all(re.fullmatch(r"not [a-z_]+", term) for term in terms)
        return {term.removeprefix("not ") for term in terms}

    assert excluded_markers(stage_expression) == excluded_markers(default_expression)
    assert excluded_markers(stage_expression) == STAGE2_EXCLUDED_MARKERS
    registered = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("source_scan:") for marker in registered)


@pytest.mark.parametrize("use_head", [False, True])
def test_rename_preserves_old_imports_and_central_expansion(tmp_path, use_head):
    repo = tmp_path / "repo"
    (repo / "rfx/core").mkdir(parents=True)
    (repo / "rfx/other").mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=True, timeout=60,
            env={
                "PATH": os.environ.get("PATH", ""), "HOME": str(repo),
                "GIT_CONFIG_GLOBAL": str(repo / "gitconfig"),
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
            },
        ).stdout.strip()

    (repo / "rfx/core/a.py").write_text("x = 1\n" * 40)
    git("init", "-q", "-b", "main")
    git("config", "diff.renames", "true")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("mv", "rfx/core/a.py", "rfx/other/a2.py")
    git("commit", "-qm", "rename")
    assert git("diff", "--name-status", base, "HEAD") == "R100\trfx/core/a.py\trfx/other/a2.py"
    sources = {
        "tests/unit/test_old.py": "import rfx.core.a",
        "tests/unit/test_new.py": "import rfx.other.a2",
        "tests/unit/runners/test_runner.py": "",
        "tests/unit/autodiff/test_ad.py": "",
    }
    paths = selector.changed_paths(base, "HEAD" if use_head else None, repo)
    assert set(selector.select_tests(paths, sources, {}, central_paths=True).files) == set(sources)
    assert set(paths) == {"rfx/core/a.py", "rfx/other/a2.py"}


def test_function_move_selects_both_real_source_scans_without_import_matches():
    paths = {
        "tests/unit/farfield/test_current_moment_monitor.py":
            "test_every_user_facing_return_carries_the_backstop",
        "tests/unit/materials/test_sheet_impedance.py":
            "test_every_fence_in_the_source_is_pinned",
    }
    sources = {path: (ROOT / path).read_text() for path in paths}
    changed = ["rfx/core/source_owner.py", "rfx/core/moved_owner.py"]
    modules = {selector.module_name(path) for path in changed}
    assert all(not selector.directly_imports(source, modules) for source in sources.values())
    result = selector.select_tests(changed, sources, {})
    for path, name in paths.items():
        assert f"{path}::{name}" in result.files
        assert path not in result.files


def test_source_scans_are_not_added_without_rfx_changes():
    path = "tests/unit/farfield/test_current_moment_monitor.py"
    sources = {path: (ROOT / path).read_text()}
    result = selector.select_tests(["docs/page.md", "scripts/helper.py"], sources, {})
    assert result.files == ()
    assert "source-scan tests added: 0 tests in 0 files, recorded 0.000000 s" in result.summary()


@pytest.mark.parametrize("edited", [False, True])
def test_whole_file_selection_wins_over_source_scan_node_ids(edited):
    path = "tests/unit/test_scan.py"
    source = ('import pytest\nimport rfx.core.changed\n'
              '@pytest.mark.source_scan\ndef test_scan():\n    return "rfx/api/a.py"\n')
    changed = ["rfx/core/changed.py", *([path] if edited else [])]
    result = selector.select_tests(changed, {path: source}, {f"{path}::test_scan": 5})
    assert result.files == (path,)
    assert result.selected_seconds == 5
    assert result.source_scan_count == 0
    assert result.source_scan_seconds == 0


def test_source_scan_node_ids_bypass_budget_and_account_for_parameters():
    path = "tests/unit/test_scan.py"
    source = ('import pytest\nimport rfx.core.changed\n'
              '@pytest.mark.source_scan\ndef test_scan():\n    return "rfx/api/a.py"\n'
              'def test_plain():\n    pass\n')
    durations = {f"{path}::test_scan[a]": 3, f"{path}::test_scan[b]": 4,
                 f"{path}::test_plain": 500}
    result = selector.select_tests(["rfx/core/changed.py"], {path: source}, durations)
    assert result.files == (f"{path}::test_scan",)
    assert not result.excluded
    assert result.partly_added == {path: 507}
    assert result.selected_seconds == result.source_scan_seconds == 7
    assert result.not_run_count == 1
    assert result.not_run_seconds == 500
    assert "source-scan tests added: 1 tests in 1 files, recorded 7.000000 s" in result.summary()
    assert "selected-tests: 1 files, recorded 7.000000 s" in result.summary()
    assert "excluded by >400 s rule: 0 files" in result.summary()
    assert "partly added by node id: 1 files" in result.summary()
    assert "source-scan tests that run: 1 tests, recorded 7.000000 s" in result.summary()


@pytest.mark.parametrize("imports_changed", [False, True])
def test_module_source_scan_mark_adds_whole_file_even_over_budget(imports_changed):
    path = "tests/unit/test_scan.py"
    source = ('import pytest\npytestmark = [pytest.mark.source_scan]\n'
              'def test_scan():\n    return "rfx/api/a.py"\n')
    if imports_changed:
        source = "import rfx.core.changed\n" + source
    result = selector.select_tests(["rfx/non_python_data", "rfx/core/changed.py"], {path: source},
                                   {f"{path}::test_scan": 401})
    assert result.files == (path,)
    assert result.source_scan_count == result.source_scan_files == 1
    assert result.source_scan_seconds == result.selected_seconds == 401
    assert result.not_run_count == 0
    assert not result.excluded


def test_class_source_scan_mark_selects_method_nodes_and_excludes_contracts():
    source = ('import pytest\n@pytest.mark.source_scan\nclass TestScanner:\n'
              '    def test_scan(self):\n        return "rfx/api/a.py"\n')
    path = "tests/unit/test_scan.py"
    result = selector.select_tests(["rfx/new.py"], {
        path: source, "tests/contracts/test_scan.py": source,
    }, {})
    assert result.files == (f"{path}::TestScanner::test_scan",)
    assert result.source_scan_count == result.source_scan_files == 1


def test_cli_emits_source_scan_node_ids(monkeypatch, tmp_path, capsys):
    durations = tmp_path / "durations.json"
    durations.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["select_gate_tests.py", "--base", "base",
                                     "--durations", str(durations)])
    monkeypatch.setattr(selector.subprocess, "check_output", lambda *a, **k: str(ROOT))
    monkeypatch.setattr(selector, "changed_paths", lambda *a: ["rfx/new.py"])
    monkeypatch.setattr(selector, "read_test_sources", lambda *a: {
        "tests/unit/test_scan.py": 'import pytest\n@pytest.mark.source_scan\n'
                                   'def test_scan():\n    return "rfx/api/a.py"\n',
    })
    assert selector.main() == 0
    output = capsys.readouterr()
    assert output.out == "tests/unit/test_scan.py::test_scan\n"
    assert "source-scan tests added: 1 tests in 1 files" in output.err


@pytest.mark.parametrize("whole_file", [False, True])
def test_source_scan_summary_splits_running_and_deselected_tests(whole_file):
    path = "tests/unit/test_scan.py"
    sources = {path: ('import pytest\n'
                     '@pytest.mark.source_scan\ndef test_run():\n    open("rfx/api/a.py")\n'
                     '@pytest.mark.slow\n@pytest.mark.source_scan\n'
                     'def test_slow():\n    open("rfx/api/a.py")\n')}
    if whole_file:
        sources[path] = "import rfx.core.changed\n" + sources[path]
    result = selector.select_tests(["rfx/core/changed.py"], sources, {
        f"{path}::test_run[a]": 3, f"{path}::test_run[b]": 4,
        f"{path}::test_slow": 9,
    })
    assert result.source_scan_run_count == result.source_scan_deselected_count == 1
    assert result.source_scan_run_seconds == 7
    assert result.source_scan_deselected_seconds == 9
    assert "source-scan tests that run: 1 tests, recorded 7.000000 s" in result.summary()
    assert "source-scan tests deselected by stage-2 markers: 1 tests, recorded 9.000000 s" in result.summary()
    assert result.files == ((path,) if whole_file else (f"{path}::test_run", f"{path}::test_slow"))


@pytest.mark.parametrize("scope", ["module", "class"])
def test_selector_summary_inherits_stage2_exclusion_marks(scope):
    path = "tests/unit/test_scan.py"
    if scope == "module":
        source = ('import pytest\npytestmark = [pytest.mark.source_scan, pytest.mark.gpu]\n'
                  'def test_scan():\n    open("rfx/api/a.py")\n')
        node = f"{path}::test_scan"
    else:
        source = ('import pytest\n@pytest.mark.source_scan\n@pytest.mark.docs_consistency\n'
                  'class TestScan:\n    def test_scan(self):\n        open("rfx/api/a.py")\n')
        node = f"{path}::TestScan::test_scan"
    result = selector.select_tests(["rfx/new.py"], {path: source}, {node: 5})
    assert result.source_scan_run_count == result.source_scan_run_seconds == 0
    assert result.source_scan_deselected_count == 1
    assert result.source_scan_deselected_seconds == 5


def test_summary_counts_marked_tests_in_an_edited_file_without_rfx_changes():
    path = "tests/unit/test_scan.py"
    source = ('import pytest\n@pytest.mark.source_scan\ndef test_run():\n'
              '    open("rfx/api/a.py")\n'
              '@pytest.mark.source_scan\n@pytest.mark.slow\ndef test_slow():\n'
              '    open("rfx/api/a.py")\n')
    result = selector.select_tests([path], {path: source}, {
        f"{path}::test_run": 3, f"{path}::test_slow": 4,
    })
    assert result.files == (path,)
    assert result.source_scan_count == 0
    assert result.source_scan_run_count == result.source_scan_deselected_count == 1
    assert result.source_scan_run_seconds == 3
    assert result.source_scan_deselected_seconds == 4


def test_local_shell_passes_selected_node_ids_unchanged(tmp_path):
    node = "tests/unit/test_scan.py::TestScanner::test_scan"
    (tmp_path / "files").write_text(node + "\n")
    stage = (ROOT / "scripts/ci/local.sh").read_text().split("selected_tests=()", 1)[1]
    read_array = "selected_tests=()" + stage.split("# Bash 3.2", 1)[0]
    result = subprocess.run(
        ["bash", "-uc", 'selection_dir="$1"\n' + read_array
         + 'stage2_pytest() { printf "%s\\n" "${@:2}"; }\n'
         + 'stage2_pytest selected "${selected_tests[@]}"', "bash", str(tmp_path)],
        capture_output=True, text=True, check=True, timeout=10,
    )
    assert result.stdout.splitlines() == [node]


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4, 5, 130])
def test_selected_status_handling_in_isolated_bash(tmp_path, status):
    # The stage now needs collection + JUnit evidence, even for exit zero.
    spec = importlib.util.spec_from_file_location(
        "stage2_contract", ROOT / "tests/contracts/test_gate_stage2_outcome.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_fragment(tmp_path, fake=True, first=status, retry=1)
    assert result.returncode == (0 if status in (0, 5) else 1)
    assert ("nothing to run" in result.stdout) == (status == 5)
