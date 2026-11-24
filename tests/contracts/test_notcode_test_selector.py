"""Literal reader selection and the required not-code workflow execution contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
SPEC = importlib.util.spec_from_file_location("select_notcode_tests", ROOT / "scripts/ci/select_notcode_tests.py")
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)

READERS = [
    ("docs/design_notes/schemas/rfx-design-ir-v2.schema.json", "tests/interop/test_interop_design_document.py"),
    ("docs/design_notes/schemas/rfx-design-ir-v2.schema.json", "tests/interop/test_interop_design_schema_contract.py"),
    ("docs/design_notes/schemas/rfx-design-ir-v2.schema.json", "tests/interop/test_interop_shape_codec.py"),
    ("docs/design_notes/patch_edgefed_s11_band_repin_results.json", "tests/locks/test_patch_edgefed_s11_passivity.py"),
    ("docs/design_notes/chain_closure_contract.md", "tests/oracle/test_waveguide_chain_battery_closure.py"),
    ("docs/public/run/memory-and-speed.mdx", "tests/unit/autodiff/test_ad_diagnostics.py"),
    ("docs/public/run/memory-and-speed.mdx", "tests/unit/autodiff/test_estimate_ad_memory.py"),
    ("docs/guides/sparameter_support_matrix.json", "tests/unit/autodiff/test_ad_surface_contract.py"),
    ("docs/guides/sparameter_support_matrix.json", "tests/unit/sparams/test_sparameter_support_contract.py"),
    ("docs/guides/known_limitations.md", "tests/unit/preflight/test_thin_conductor_thickness.py"),
]


@pytest.mark.parametrize("path,reader", READERS)
def test_real_readers(path, reader):
    assert reader in selector.select_tests([path], ROOT)


@pytest.mark.parametrize("name", ["test_public_docs_block_runner.py", "test_public_docs_bundle.py", "test_public_docs_code_quotes.py"])
def test_dynamic_readers_live_in_contracts(name):
    assert (ROOT / "tests/contracts" / name).is_file()
    assert not (ROOT / "tests/unit/docs" / name).exists()


def test_synthetic_literals_and_directories(tmp_path):
    """Docstrings, code constants and directory-prefix siblings cannot select tests."""
    for path in ("docs/guide.md", "docs/sibling.md", "rfx/core.py"):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    tests = tmp_path / "tests/unit"
    tests.mkdir(parents=True)
    sources = {
        "test_file.py": 'PATH = "./docs/guide.md"',
        "test_directory.py": 'PATH = "docs/"',
        "test_bare_directory.py": 'PATH = "docs"',
        "test_docstring.py": '\"\"\"docs/guide.md\"\"\"\nclass Reader:\n    \"\"\"docs/guide.md\"\"\"\n    def read(self):\n        \"\"\"docs/guide.md\"\"\"',
        "test_code.py": 'PATH = "rfx/core.py"',
        "test_parts.py": 'PARTS = ("docs", "guide.md")',
        "test_invalid.py": 'TEXT = "x" * 10000',
    }
    for name, source in sources.items():
        (tests / name).write_text(source)
    contracts = tmp_path / "tests/contracts"
    contracts.mkdir()
    (contracts / "test_contract.py").write_text('PATH = "docs/guide.md"')
    assert selector.select_tests(["docs/guide.md"], tmp_path) == [
        "tests/unit/test_bare_directory.py", "tests/unit/test_directory.py",
        "tests/unit/test_file.py", "tests/unit/test_parts.py",
    ]
    assert selector.select_tests(["docs-other/guide.md", "rfx/core.py"], tmp_path) == []
    assert selector.select_tests([], tmp_path) == []


@pytest.mark.parametrize("changed,expected", [
    ("docs/guide.md", ["tests/unit/test_file.py"]),
    ("docs/deleted/page.md", ["tests/unit/test_directory.py"]),
    ("docs/guidebook.md", []),
    ("etc/passwd", []),
    ("rfx/missing.py", []),
])
def test_absent_path_readers(tmp_path, changed, expected):
    """Deleted or renamed readers select by exact path or ancestor, excluding siblings and invalid paths."""
    tests = tmp_path / "tests/unit"
    tests.mkdir(parents=True)
    sources = {
        "test_file.py": 'PATH = "./docs/guide.md"',
        "test_directory.py": 'PATH = "docs/deleted/"',
        "test_sibling.py": 'PATH = "docs/gui"',
        "test_invalid.py": 'PATHS = (".", "..", "../docs/guide.md", "/docs/guide.md", "/etc/passwd", "rfx/missing.py")',
    }
    for name, source in sources.items():
        (tests / name).write_text(source)
    assert not (tmp_path / "docs").exists()
    assert selector.select_tests([changed], tmp_path) == expected


def notcode_step():
    """Read the single required shard step that executes the not-code branch."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-tests.yml").read_text())
    steps = workflow["jobs"]["fast-suite"]["steps"]
    return next(step for step in steps if step.get("if") == "needs.changes.outputs.code_changed == 'false'")


def test_workflow_and_classifier_wiring():
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-tests.yml").read_text())
    assert workflow["jobs"]["changes"]["outputs"]["notcode_tests"] == "${{ steps.classify.outputs.notcode_tests }}"
    step = notcode_step()
    assert step["env"]["NOTCODE_TESTS"] == "${{ needs.changes.outputs.notcode_tests }}"
    calls = [line.strip() for line in step["run"].splitlines() if "python -m pytest" in line and "--collect-only" not in line]
    assert len(calls) == 1
    assert '"${selected_files[@]}"' in calls[0]
    assert "--strict-markers" in calls[0] and "-p no:cacheprovider" in calls[0]
    assert "addopts" not in calls[0]
    script = (ROOT / "scripts/ci/classify_changes.sh").read_text()
    assert '"$here/select_notcode_tests.py" --stdin --null < "$changed_file"' in script
    assert 'echo "notcode_tests=$notcode_tests" >> "$GITHUB_OUTPUT"' in script
    assert selector.is_code_path("scripts/ci/select_notcode_tests.py")


@pytest.mark.parametrize("count,group,status,expected,runs", [
    (24, 2, 0, 0, True), (24, 20, 5, 0, True),
    (24, 1, 1, 1, True), (24, 1, 2, 2, True),
    (3, 1, 0, 0, True), (3, 2, 0, 0, False),
    (0, 1, 5, 0, True), (24, 1, 0, 0, False),
])
def test_notcode_branch_execution(tmp_path, count, group, status, expected, runs):
    """Execute the real shell branch with a recording pytest to check shards and exits."""
    fake = tmp_path / "pytest_stub.py"
    record = tmp_path / "calls.jsonl"
    fake.write_text(
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'args = sys.argv[1:]\n'
        'with Path(os.environ["CALLS"]).open("a") as stream:\n'
        '    stream.write(json.dumps(args) + "\\n")\n'
        'if "--collect-only" in args:\n'
        '    print(os.environ["COUNT"] + "/30 tests collected (some deselected) in 0.01s")\n'
        '    sys.exit(0)\n'
        'sys.exit(int(os.environ["STATUS"]))\n'
    )
    script = notcode_step()["run"]
    script = "\n".join(line for line in script.splitlines() if not line.strip().startswith("pytest tests/contracts "))
    script = script.replace("python -m pytest", f'"{sys.executable}" "{fake}"')
    result = subprocess.run(["bash", "-e", "-c", script], text=True, capture_output=True, env={
        **os.environ, "NOTCODE_TESTS": "tests/unit/test_reader.py" if (count != 24 or group != 1 or status != 0) else "",
        "SHARD_GROUP": str(group), "RUNNER_TEMP": str(tmp_path),
        "CALLS": str(record), "COUNT": str(count), "STATUS": str(status),
    })
    assert result.returncode == expected, result.stdout + result.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    executions = [call for call in calls if "--collect-only" not in call]
    assert bool(executions) is runs
    if executions:
        assert executions[0][-1] == "tests/unit/test_reader.py"
        assert "--strict-markers" in executions[0]
        assert ("--splits" in executions[0]) is (count >= 20)


def test_cli_reads_null_separated_paths():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/select_notcode_tests.py"), "--stdin", "--null"],
        input="./docs/guides/sparameter_support_matrix.json\0docs/missing page.md\0",
        text=True, capture_output=True, cwd=ROOT, check=True,
    )
    assert result.stdout.splitlines() == [
        "tests/unit/autodiff/test_ad_surface_contract.py",
        "tests/unit/sparams/test_sparameter_support_contract.py",
    ]


def test_classifier_emits_the_selection(tmp_path):
    """A successful docs diff emits its verdict and the same path list's readers."""
    git = tmp_path / "git"
    git.write_text('#!/bin/sh\nprintf "docs/guides/sparameter_support_matrix.json\\000"\n')
    git.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(ROOT / "scripts/ci/classify_changes.sh")],
        cwd=ROOT, text=True, capture_output=True, env={
            **os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PYTHON": sys.executable, "TMPDIR": str(tmp_path),
            "GITHUB_OUTPUT": str(output), "EVENT_NAME": "pull_request",
            "BASE_SHA": "base", "HEAD_SHA": "head",
        }, check=True)
    expected = "notcode_tests=tests/unit/autodiff/test_ad_surface_contract.py tests/unit/sparams/test_sparameter_support_contract.py"
    assert set(output.read_text().splitlines()) == {"code_changed=false", expected}
    assert expected in result.stdout
