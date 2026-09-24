"""The data-budget gate does what it claims (PI, 2026-09-24).

The repository holds code, tests and the frozen reference data a test reads;
measurement records go to ``bk-squared/rfx-archive``. The gate is mechanical
and so is everything that can go wrong with it: a budget off by one, a rename
counted as new data, a deletion counted against the PR that moves records out,
an allowlist glob that crosses directories, a fixture no test reads slipping
into ``tests/fixtures/``, an exception label read from a comma split. Each is
written down here against a git repository this file builds, through the same
``evaluate``/``failures``/``main`` the workflow step calls.

No network, no package import.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._git_tracked import git_available

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "check_data_budget.py"

_SPEC = importlib.util.spec_from_file_location("check_data_budget", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
budget = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = budget
_SPEC.loader.exec_module(budget)


# --------------------------------------------------------------------------
# What a data file is, and what the allowlist covers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "validation/x/results.json", "a/b.csv", "a/b.tsv", "run.log", "a/stdout.txt",
    "a/job.out", "a/job.rc", "a/job.exit", "a/job.started", "a/b.dat",
    "a/report.xml", "a/mesh.msh", "a/b.npz", "a/b.npy", "a/b.h5", "a/b.hdf5",
    "a/b.pkl", "a/logs.tar.gz", "a/b.JSON", "a/line.s2p", "a/line.s12p",
    ".test_durations",
])
def test_data_files_are_recognised(path: str) -> None:
    assert budget.is_data(path)


@pytest.mark.parametrize("path", [
    "rfx/core/yee.py", "docs/guide.md", "docs/agent/x.mdx", "a/fig.png",
    "scripts/job.yaml", "pyproject.toml", "a/b.json.py", "a/s2p", "a/b.sp",
    "tests/fixtures/mesh/box.step",
])
def test_code_prose_and_images_are_not_data(path: str) -> None:
    assert not budget.is_data(path)


@pytest.mark.parametrize("path", [
    ".test_durations",
    "docs/guides/api_symbol_inventory.json",
    "docs/guides/sparameter_support_matrix.json",
    "docs/guides/support_matrix.json",
    "validation/crossval/manifest.json",
    "docs/design_notes/schemas/rfx-design-ir-v2.schema.json",
    "docs/public/site_map.json",
    "docs/public/gallery/assets/patch_antenna/sparams.s1p",
    "scripts/ops/gpu_suite_shards.json",
    "tests/contracts/boundary_registry.json",
    "tests/fixtures/a.json",
    "tests/fixtures/deep/er/a.npz",
    "tests/data/preflight_split_snapshot/x.json",
    "tests/crossval/sheen_lpf/reference/openems_sheen.json",
])
def test_the_allowlist_covers_the_measured_paths(path: str) -> None:
    assert budget.allowlisted(path)


@pytest.mark.parametrize("path", [
    "validation/crossval/_26_oblique_results/rfx.json",
    "validation/crossval/sub/manifest.json",
    "docs/guides/sub/support_matrix.json",
    "docs/notes/x/support_matrix.json",
    "scripts/diagnostics/_artifacts/x.json",
    "tests/crossval/sheen_lpf/results/run.json",
    "tests/crossval/a/b/reference/x.json",
    "tests/unit/x.json",
    "xtests/fixtures/a.json",
    "sub/.test_durations",
])
def test_the_allowlist_does_not_leak(path: str) -> None:
    """`*` stays inside one directory, and every pattern is anchored at both ends."""
    assert not budget.allowlisted(path)


def test_every_frozen_data_home_is_on_the_allowlist() -> None:
    """Rule 3 guards exactly the homes the allowlist lets grow past the budget."""
    for path in ("tests/fixtures/a/b.json", "tests/data/b.json",
                 "tests/crossval/case/reference/b.json"):
        assert budget.FROZEN_HOME_RE.match(path), path
        assert budget.allowlisted(path), path


@pytest.mark.parametrize("path,tokens", [
    ("tests/fixtures/coax_chain_battery/fixture.json",
     ["fixture.json", "fixture", "coax_chain_battery"]),
    ("tests/fixtures/harminv_decimation/cv02/input-00.npz",
     ["input-00.npz", "input-00", "cv02"]),
    ("tests/data/v1.json", ["v1.json"]),  # "v1" is too short to name one file
    ("tests/fixtures/sweep/0.01_16.json", ["0.01_16.json", "sweep"]),  # stem has no letter
    ("tests/fixtures/a/logs.tar.gz", ["logs.tar.gz", "logs"]),
    # Only data suffixes come off: `split(".")[0]` would leave "rfx", which
    # every file under rfx/ names.
    ("tests/fixtures/rfx.golden_v2.json", ["rfx.golden_v2.json", "rfx.golden_v2"]),
    ("tests/crossval/sheen_lpf/reference/openems_sheen.json",
     ["openems_sheen.json", "openems_sheen"]),
])
def test_reference_tokens(path: str, tokens: list) -> None:
    """A file directly in its home gets no directory token: `fixtures` names everything."""
    assert budget.reference_tokens(path) == tokens


@pytest.mark.parametrize("token,text,named", [
    ("data.json", 'FIX / "data.json"', True),
    ("data.json", "tests/fixtures/x/data.json", True),
    ("data.json", "mydata.json", False),
    ("data.json", "my-data.json", False),
    ("data.json", "data.json5", False),
    ("data.json", "x.data.json", False),
    ("data", "CASES = ['data']", True),
    ("data", "old_data", False),
    ("data", "data-2", False),
    ("data", "data", True),
    ("data", "", False),
    ("data", "old_data, then data.", True),  # a later whole occurrence counts
])
def test_is_named_needs_a_whole_name(token: str, text: str, named: bool) -> None:
    assert budget.is_named(token, text) is named


# --------------------------------------------------------------------------
# The rules, against a repository this file builds
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    if not git_available():
        pytest.skip("git unavailable")
    root = tmp_path / "repo"
    for directory in ("rfx", "tests/fixtures", "tests/unit", "validation/study",
                      "docs/guides"):
        (root / directory).mkdir(parents=True)
    (root / "rfx" / "simulation.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "unit" / "test_x.py").write_text("def test_x(): pass\n",
                                                        encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def _commit(repo: Path, message: str = "change") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _lines(n: int) -> str:
    return "".join(f'{{"i": {i}}}\n' for i in range(n))


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _binary(repo: Path, rel: str, size: int) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0\1" * (size // 2) + b"\0" * (size % 2))


def _check(repo: Path, base: str, head: str) -> list:
    return budget.failures(budget.evaluate(base, head, repo))


def test_the_line_budget_is_500_inclusive(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/study/a.json", _lines(300))
    _write(repo, "validation/study/b.csv", _lines(200))
    head = _commit(repo)
    assert _check(repo, base, head) == []
    _write(repo, "validation/study/c.log", "one more line\n")
    over = _commit(repo)
    failures = _check(repo, base, over)
    assert len(failures) == 1 and "add 501 lines (budget 500)" in failures[0]
    assert "validation/study/a.json" in failures[0]


def test_modified_data_counts_only_the_lines_added(repo: Path) -> None:
    _write(repo, "validation/study/a.json", _lines(2000))
    base = _commit(repo, "a record already on main")
    _write(repo, "validation/study/a.json", _lines(2000) + _lines(10))
    head = _commit(repo)
    findings = budget.evaluate(base, head, repo)
    assert findings.line_total == 10
    assert _check(repo, base, head) == []


def test_deleting_records_is_free(repo: Path) -> None:
    """Moving records out to the archive is the point; it must never cost budget."""
    _write(repo, "validation/study/a.json", _lines(5000))
    base = _commit(repo, "records on main")
    (repo / "validation" / "study" / "a.json").unlink()
    head = _commit(repo, "move records to the archive")
    findings = budget.evaluate(base, head, repo)
    assert findings.changes == [] and _check(repo, base, head) == []


def test_a_pure_rename_adds_no_lines(repo: Path) -> None:
    _write(repo, "validation/study/a.json", _lines(5000))
    base = _commit(repo, "records on main")
    _git(repo, "mv", "validation/study/a.json", "validation/study/renamed.json")
    head = _commit(repo, "rename")
    findings = budget.evaluate(base, head, repo)
    assert [(c.path, c.status, c.added) for c in findings.changes] == [
        ("validation/study/renamed.json", "R", 0)
    ]
    assert _check(repo, base, head) == []


def test_code_is_not_data(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/study/huge.py", "x = 1\n" * 20000)
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_the_binary_budget_is_200_kB_inclusive(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _binary(repo, "validation/study/a.npz", 150_000)
    _binary(repo, "validation/study/b.npy", 50_000)
    head = _commit(repo)
    findings = budget.evaluate(base, head, repo)
    assert findings.binary_total == 200_000 and findings.line_total == 0
    assert _check(repo, base, head) == []
    _binary(repo, "validation/study/c.gz", 1)
    over = _commit(repo)
    failures = _check(repo, base, over)
    assert len(failures) == 1 and "200,001 bytes (budget 200,000)" in failures[0]


def test_allowlisted_data_does_not_spend_the_budget(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "docs/guides/api_symbol_inventory.json", _lines(3000))
    _write(repo, "tests/fixtures/case/fixture.json", _lines(3000))
    _write(repo, "tests/unit/test_case.py", 'FIX = "case"\n')
    head = _commit(repo)
    findings = budget.evaluate(base, head, repo)
    assert findings.line_total == 0
    assert _check(repo, base, head) == []


def test_the_size_cap_applies_on_the_allowlist_too(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/crossval/manifest.json", "x" * 1_000_001)
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 1 and "over their size cap" in failures[0]
    assert "(cap 1,000,000)  validation/crossval/manifest.json" in failures[0]


@pytest.mark.parametrize("path,named,cap", [
    ("tests/fixtures/case/fixture.json", {"tests/fixtures/case/fixture.json"}, 5_000_000),
    ("tests/data/x.json", {"tests/data/x.json"}, 5_000_000),
    ("tests/crossval/c/reference/r.json", {"tests/crossval/c/reference/r.json"}, 5_000_000),
    ("tests/fixtures/case/fixture.json", set(), 1_000_000),     # no reader names it
    ("validation/crossval/manifest.json", {"validation/crossval/manifest.json"}, 1_000_000),
    ("tests/contracts/boundary_registry.json", {"tests/contracts/boundary_registry.json"}, 1_000_000),
])
def test_only_a_named_file_in_a_frozen_home_gets_the_larger_cap(path, named, cap) -> None:
    assert budget.size_cap(path, frozenset(named)) == cap


def test_a_named_frozen_file_may_reach_5_MB(repo: Path) -> None:
    """Four fixtures tests read, added 2026-09-10..24, were 1.04-1.81 MB."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/case/fixture.json", "x" * 5_000_000)
    _write(repo, "tests/unit/test_case.py", 'FIX = "fixture.json"\n')
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_a_named_frozen_file_over_5_MB_fails(repo: Path) -> None:
    """Over the file cap, and so necessarily over the per-PR frozen budget too."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/case/fixture.json", "x" * 5_000_001)
    _write(repo, "tests/unit/test_case.py", 'FIX = "fixture.json"\n')
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 2
    assert "(cap 5,000,000)  tests/fixtures/case/fixture.json" in failures[0]
    assert "hold 5,000,001 bytes (budget 5,000,000 per PR)" in failures[1]


def test_an_edited_named_frozen_file_gets_the_larger_cap(repo: Path) -> None:
    """Rule 3 only asks about new files; the larger cap still needs a reader."""
    _write(repo, "tests/fixtures/case/fixture.json", _lines(5))
    _write(repo, "tests/unit/test_case.py", 'FIX = "fixture.json"\n')
    base = _commit(repo, "a small fixture on main")
    _write(repo, "tests/fixtures/case/fixture.json", "x" * 2_000_000)
    head = _commit(repo, "regenerate it larger")
    assert _check(repo, base, head) == []


@pytest.mark.parametrize("size,failing", [(1_000_000, 0), (1_000_001, 1)])
def test_an_unnamed_frozen_file_keeps_the_1_MB_cap(repo: Path, size: int, failing: int) -> None:
    """Edited, so rule 3 does not apply: only the cap can fail it."""
    _write(repo, "tests/fixtures/orphan_case/orphan.json", _lines(5))
    base = _commit(repo, "an unread fixture already on main")
    _write(repo, "tests/fixtures/orphan_case/orphan.json", "x" * size)
    head = _commit(repo, "grow it")
    failures = _check(repo, base, head)
    assert len(failures) == failing
    if failing:
        assert "(cap 1,000,000)  tests/fixtures/orphan_case/orphan.json" in failures[0]


def test_a_new_unnamed_frozen_file_over_1_MB_fails_both_rules(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/orphan_case/orphan.json", "x" * 2_000_000)
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 2
    assert "(cap 1,000,000)" in failures[0] and "no tracked file" in failures[1]


def test_the_size_cap_is_inclusive_and_test_durations_is_exempt(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/case/fixture.json", "x" * 1_000_000)
    _write(repo, "tests/unit/test_case.py", 'FIX = "fixture.json"\n')
    _write(repo, ".test_durations", "x" * 1_200_000)
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_a_new_fixture_no_test_names_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 1 and "no tracked file" in failures[0]
    assert "tests/fixtures/sweep/point_07.json" in failures[0]


@pytest.mark.parametrize("reader", [
    'PATH = FIX / "sweep" / "point_07.json"\n',   # by file name
    'CASES = ["point_07"]  # f"{case}.json"\n',   # by name without extension
    'for p in (FIX / "sweep").glob("*.json"):\n    pass\n',  # by its subdirectory
])
def test_a_new_fixture_a_test_names_passes(repo: Path, reader: str) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "tests/unit/test_sweep.py", reader)
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_a_reader_under_rfx_counts(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/data/table_v2.csv", "a,b\n1,2\n")
    _write(repo, "rfx/tables.py", 'NAME = "table_v2.csv"\n')
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_a_data_file_naming_the_fixture_is_not_a_reader(repo: Path) -> None:
    """A manifest.json listing the records would otherwise vouch for them."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "tests/fixtures/other/manifest.json", '["point_07.json"]\n')
    _write(repo, "tests/unit/test_other.py", 'X = "manifest.json"\n')
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 1 and "point_07.json" in failures[0]


def test_a_file_directly_in_a_home_is_not_named_by_the_home(repo: Path) -> None:
    """`fixtures` appears in every test; it cannot vouch for a file."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/orphan.json", _lines(5))
    _write(repo, "tests/unit/test_fix.py", 'FIX = ROOT / "tests" / "fixtures"\n')
    head = _commit(repo)
    assert len(_check(repo, base, head)) == 1


def test_digits_do_not_vouch_for_a_numeric_name(repo: Path) -> None:
    """A sweep point named `0.01_16.json` is not named by every `0` in the tree."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/data/0.01_16.json", _lines(5))
    _write(repo, "tests/unit/test_x.py", "X = [0, 0.01, 16]\n")
    head = _commit(repo)
    assert len(_check(repo, base, head)) == 1


@pytest.mark.parametrize("reader_path,reader", [
    ("tests/fixtures/sweep/README.md", "point_07.json is the 7th sweep point\n"),
    ("tests/unit/notes.yaml", "fixture: sweep/point_07.json\n"),
    ("tests/unit/notes.txt.md", "sweep/point_07.json\n"),
])
def test_only_a_py_file_is_a_reader(repo: Path, reader_path: str, reader: str) -> None:
    """A README beside #1198's relocated records vouched for all 205 of them."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, reader_path, reader)
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == 1 and "no tracked file" in failures[0]


@pytest.mark.parametrize("path,reader", [
    ("tests/unit/test_x.py", True),
    ("tests/conftest.py", True),
    ("tests/unit/sparams/conftest.py", True),
    ("tests/unit/_helpers.py", True),
    ("tests/__init__.py", True),
    ("rfx/io/tables.py", True),
    ("tests/unit/helpers.py", False),                       # neither a test nor a _helper
    ("tests/unit/capture_golden.py", False),
    ("tests/fixtures/option_b/_index.py", False),           # inside a frozen-data home
    ("tests/fixtures/case/test_case.py", False),
    ("tests/data/_index.py", False),
    ("tests/crossval/c/reference/make_reference.py", False),
    ("tests/crossval/c/test_c.py", True),                   # beside reference/, not in it
    ("scripts/diagnostics/_x.py", False),
    ("tests/unit/test_x.pyi", False),
])
def test_is_reader(path: str, reader: bool) -> None:
    assert budget.is_reader(path) is reader


@pytest.mark.parametrize("reader_path,counts", [
    ("tests/fixtures/sweep/_index.py", False),   # the stacked-relocation route
    ("tests/fixtures/_index.py", False),
    ("tests/unit/helpers.py", False),
    ("tests/unit/_helpers.py", True),
    ("tests/conftest.py", True),
])
def test_which_py_files_can_vouch(repo: Path, reader_path: str, counts: bool) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, reader_path, 'NAMES = ["sweep/point_07.json"]\n')
    head = _commit(repo)
    assert (_check(repo, base, head) == []) is counts


def test_a_test_of_this_gate_is_not_a_reader(repo: Path) -> None:
    """The gate's own tests name `sweep`, `point_07`, `orphan_case` for synthetic records."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "tests/contracts/test_gate.py",
           'SCRIPT = "scripts/ci/check_data_budget.py"\nX = FIX / "sweep" / "point_07.json"\n')
    head = _commit(repo)
    assert len(_check(repo, base, head)) == 1


@pytest.mark.parametrize("reader,named", [
    ('ROOT = FIX / "sweep"\n', True),
    ('ROOT = "tests/fixtures/sweep/"\n', True),
    ('ROOT = f"tests/fixtures/sweep/{n}"\n', True),
    ("# the sweep folder\nROOT = FIX\n", False),               # a comment
    ('"""Reads the sweep folder."""\nROOT = FIX\n', False),    # a docstring
    ('ROOT = FIX / "my_sweep_dir"\n', False),                   # part of a longer name
    ('ROOT = FIX / "sweep_v2"\n', False),
    ("sweep = 1\n", False),                                     # an identifier, not a literal
])
def test_the_folder_token_must_be_a_path_component_of_a_literal(
    repo: Path, reader: str, named: bool
) -> None:
    """A bare substring let `patch`, `msl`, `waveguide` vouch for relocated records."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "tests/unit/test_sweep.py", reader)
    head = _commit(repo)
    assert (_check(repo, base, head) == []) is named


def test_a_reader_that_does_not_parse_still_names_by_file_name(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "tests/unit/test_sweep.py", 'X = "point_07.json"\nif\n')
    head = _commit(repo)
    assert _check(repo, base, head) == []


def _named_fixture(repo: Path, name: str, size: int) -> None:
    _write(repo, f"tests/fixtures/case/{name}", "x" * size)
    _write(repo, "tests/unit/test_case.py", 'FIX = ROOT / "case"\n')


@pytest.mark.parametrize("second,failing", [(1_000_000, 0), (1_000_001, 1)])
def test_the_frozen_homes_take_5_MB_per_pr(repo: Path, second: int, failing: int) -> None:
    """Relocated #1198 was 10.3 MB; the largest real frozen-data PR was 2.34 MB."""
    base = _git(repo, "rev-parse", "HEAD")
    _named_fixture(repo, "a.json", 4_000_000)
    _write(repo, "tests/data/b.npz", "\0" * second)
    _write(repo, "tests/unit/test_data.py", 'B = "b.npz"\n')
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert len(failures) == failing
    if failing:
        assert "hold 5,000,001 bytes (budget 5,000,000 per PR)" in failures[0]


def test_an_edited_frozen_file_counts_its_whole_size(repo: Path) -> None:
    _named_fixture(repo, "a.json", 3_000_000)
    base = _commit(repo, "a large fixture on main")
    _named_fixture(repo, "a.json", 3_000_001)
    _named_fixture(repo, "b.json", 2_000_000)
    head = _commit(repo, "touch it and add another")
    failures = _check(repo, base, head)
    assert len(failures) == 1 and "hold 5,000,001 bytes" in failures[0]


@pytest.mark.parametrize("size,failing", [(200_000, 0), (200_001, 1)])
def test_the_text_byte_budget_is_200_kB_inclusive(repo: Path, size: int, failing: int) -> None:
    """Ten one-line JSON files of 949,820 bytes were `10 lines` and passed."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/study/a.json", "x" * (size - 100_000))
    _write(repo, "validation/study/b.json", "y" * 100_000)
    head = _commit(repo)
    findings = budget.evaluate(base, head, repo)
    assert findings.line_total == 2 and findings.text_byte_total == size
    failures = _check(repo, base, head)
    assert len(failures) == failing
    if failing:
        assert f"add {size:,} bytes (budget 200,000)" in failures[0]


@pytest.mark.parametrize("grown,failing", [(200_000, 0), (200_001, 1)])
def test_a_modified_text_file_counts_its_growth(repo: Path, grown: int, failing: int) -> None:
    _write(repo, "validation/study/a.json", "x" * 150_000)
    base = _commit(repo, "a record on main")
    _write(repo, "validation/study/a.json", "x" * (150_000 + grown))
    head = _commit(repo)
    assert budget.evaluate(base, head, repo).text_byte_total == grown
    assert len(_check(repo, base, head)) == failing


def test_a_shrinking_text_file_counts_nothing(repo: Path) -> None:
    _write(repo, "validation/study/a.json", "x" * 900_000)
    base = _commit(repo, "a record on main")
    _write(repo, "validation/study/a.json", "y" * 300_000)
    head = _commit(repo)
    assert budget.evaluate(base, head, repo).text_byte_total == 0


@pytest.mark.parametrize("path,counted", [
    ("validation/study/blob.qqq", True),     # no data suffix, binary: counted
    ("validation/study/model.bin", True),
    ("docs/figures/plot.png", False),        # an image
    ("docs/figures/paper.pdf", False),
])
def test_any_binary_that_is_not_an_image_is_counted(repo: Path, path: str, counted: bool) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _binary(repo, path, 300_000)
    head = _commit(repo)
    assert (len(_check(repo, base, head)) == 1) is counted


@pytest.mark.parametrize("path", ["docs/notes/big.md", "validation/study/job.yaml"])
def test_markdown_and_yaml_are_never_counted(repo: Path, path: str) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, path, "line\n" * 5000)
    head = _commit(repo)
    assert budget.evaluate(base, head, repo).changes == []


@pytest.mark.parametrize("path", [".gitattributes", "validation/.gitattributes"])
def test_adding_a_gitattributes_fails(repo: Path, path: str) -> None:
    """`validation/** -diff` turned a 9,000-line CSV into a passing 196 kB binary."""
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, path, "*.csv -diff\n")
    _write(repo, "validation/study/sweep.csv", _lines(9000))
    head = _commit(repo)
    failures = _check(repo, base, head)
    assert any(".gitattributes" in failure and path in failure for failure in failures)


def test_editing_a_gitattributes_fails_and_deleting_one_does_not(repo: Path) -> None:
    _write(repo, ".gitattributes", "*.sh text eol=lf\n")
    base = _commit(repo, "attributes on main")
    _write(repo, ".gitattributes", "*.sh text eol=lf\n*.json -diff\n")
    edited = _commit(repo, "edit")
    assert len(_check(repo, base, edited)) == 1
    (repo / ".gitattributes").unlink()
    deleted = _commit(repo, "delete")
    assert _check(repo, base, deleted) == []


def test_the_label_lets_a_gitattributes_through(repo: Path, monkeypatch) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, ".gitattributes", "*.npz filter=lfs\n")
    head = _commit(repo)
    assert _main(monkeypatch, repo, base, head) == 1
    assert _main(monkeypatch, repo, base, head, labels='["data-budget-exception"]') == 0


def test_a_reader_outside_tests_and_rfx_does_not_count(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    _write(repo, "docs/guides/notes.md", "see sweep/point_07.json\n")
    head = _commit(repo)
    assert len(_check(repo, base, head)) == 1


def test_an_edited_fixture_is_not_asked_for_a_reader(repo: Path) -> None:
    """Rule 3 is about files a PR brings in; existing ones are not re-litigated."""
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(5))
    base = _commit(repo, "an unread fixture already on main")
    _write(repo, "tests/fixtures/sweep/point_07.json", _lines(6))
    head = _commit(repo)
    assert _check(repo, base, head) == []


def test_a_fixture_renamed_into_a_home_needs_a_reader(repo: Path) -> None:
    _write(repo, "validation/study/point_07.json", _lines(5))
    base = _commit(repo, "a record on main")
    (repo / "tests" / "fixtures" / "sweep").mkdir(parents=True)
    _git(repo, "mv", "validation/study/point_07.json", "tests/fixtures/sweep/point_07.json")
    head = _commit(repo, "park it in fixtures")
    assert len(_check(repo, base, head)) == 1


def test_the_budget_is_measured_from_the_merge_base(repo: Path) -> None:
    """Records that landed on main after the branch point are not this PR's."""
    base_branch_point = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "rfx/simulation.py", "x = 2\n")
    head = _commit(repo, "the PR")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "validation/study/a.json", _lines(5000))
    main_tip = _commit(repo, "someone else's records on main")
    assert base_branch_point != main_tip
    assert _check(repo, main_tip, head) == []


# --------------------------------------------------------------------------
# The command line: exit codes, the label, the step summary
# --------------------------------------------------------------------------


def _violating_pr(repo: Path) -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/study/sweep.json", _lines(600))
    return base, _commit(repo, "a sweep")


def _main(monkeypatch, repo: Path, base: str, head: str, labels: str | None = None,
          event: str | None = None) -> int:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    if labels is None:
        monkeypatch.delenv("PR_LABELS_JSON", raising=False)
    else:
        monkeypatch.setenv("PR_LABELS_JSON", labels)
    if event is not None:
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    return budget.main(["--base", base, "--head", head, "--repo", str(repo)])


def test_a_violation_exits_1_and_says_where_the_data_goes(repo: Path, monkeypatch, capsys) -> None:
    base, head = _violating_pr(repo)
    assert _main(monkeypatch, repo, base, head) == 1
    err = capsys.readouterr().err
    assert "bk-squared/rfx-archive" in err
    assert "rfx/records/<YYYYMMDD>-<topic>/" in err
    assert "archive commit in the PR" in err
    assert "data-budget-exception" in err
    # The frozen-data homes are named only to say records do not belong there.
    assert "only frozen data that a test reads" in err
    assert "does not make\n  it one" in err


def test_a_clean_pr_exits_0_and_says_what_it_measured(repo: Path, monkeypatch, capsys) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "validation/study/small.json", _lines(10))
    head = _commit(repo)
    assert _main(monkeypatch, repo, base, head) == 0
    out = capsys.readouterr().out
    assert "outside the allowlist 10 lines (budget 500)" in out
    assert "bytes of text (budget 200,000)" in out


def test_the_exception_label_passes_loudly(repo: Path, monkeypatch, capsys, tmp_path) -> None:
    base, head = _violating_pr(repo)
    summary = tmp_path / "summary.md"
    summary.write_text("", encoding="utf-8")
    monkeypatch.setenv("PR_LABELS_JSON", '["lane:ci-infra", "data-budget-exception"]')
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    assert budget.main(["--base", base, "--head", head, "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("::warning ")
    assert "NOT enforced" in out and "add 600 lines" in out
    written = summary.read_text(encoding="utf-8")
    assert "Data budget NOT enforced" in written and "add 600 lines" in written


def test_a_label_containing_a_comma_is_not_the_exception(repo: Path, monkeypatch) -> None:
    base, head = _violating_pr(repo)
    assert _main(monkeypatch, repo, base, head, labels='["x,data-budget-exception"]') == 1


def test_other_labels_do_not_waive_anything(repo: Path, monkeypatch) -> None:
    base, head = _violating_pr(repo)
    assert _main(monkeypatch, repo, base, head, labels='["release", "lane:ci-infra"]') == 1


def test_labels_that_are_not_json_fail(repo: Path, monkeypatch) -> None:
    base, head = _violating_pr(repo)
    assert _main(monkeypatch, repo, base, head, labels="data-budget-exception") == 1


def test_a_failure_is_written_to_the_step_summary(repo: Path, monkeypatch, tmp_path) -> None:
    base, head = _violating_pr(repo)
    summary = tmp_path / "summary.md"
    summary.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("PR_LABELS_JSON", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    assert budget.main(["--base", base, "--head", head, "--repo", str(repo)]) == 1
    written = summary.read_text(encoding="utf-8")
    assert "Data budget failed" in written and "bk-squared/rfx-archive" in written


def test_a_push_has_nothing_to_measure(repo: Path, monkeypatch, capsys) -> None:
    assert _main(monkeypatch, repo, "", "", event="push") == 0
    assert "nothing to check" in capsys.readouterr().out


def test_a_pull_request_without_shas_fails(repo: Path, monkeypatch) -> None:
    """Only a non-PR event may skip; a PR event with no base is a broken wiring."""
    assert _main(monkeypatch, repo, "", "", event="pull_request") == 1
    assert _main(monkeypatch, repo, "", "") == 1


def test_an_unreachable_base_is_a_message_not_a_traceback(repo: Path, monkeypatch, capsys) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    assert _main(monkeypatch, repo, "0" * 40, head) == 1
    assert "fetch-depth: 0" in capsys.readouterr().err


def test_the_cli_runs_as_a_script(repo: Path, tmp_path) -> None:
    """The workflow calls the file, not the module: it must stand alone."""
    base, head = _violating_pr(repo)
    env = {"PATH": "/usr/bin:/bin", "PR_LABELS_JSON": json.dumps([])}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--base", base, "--head", head, "--repo", str(repo)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "add 600 lines (budget 500)" in result.stderr
