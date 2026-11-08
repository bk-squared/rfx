"""The public-docs block runner's --output-dir job: page order and per-block rows."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "docs" / "check_public_docs_blocks.py"

HEADER = "page\tblock\tdoc_line\tresult\tdetail"

PAGES = {
    "a_failing.md": (
        "# B\n\n"
        "```python\nprint('first')\n```\n\n"
        "```python\nraise ValueError('boom')\n```\n\n"
        "```python\nprint('never')\n```\n"
    ),
    "b_passing.md": (
        "```python\nvalue = 1\n```\n\n"
        "Text.\n\n"
        "```python\nassert value == 1\n```\n"
    ),
    "c_fragment.mdx": "```python\n...\n```\n",
    "d_no_python.md": "# Only prose\n\n```bash\nls\n```\n",
}

EXPECTED_ROWS = [
    HEADER,
    "a_failing.md\t1\t3\tran\t",
    "a_failing.md\t2\t7\tfailed\tValueError: boom",
    "a_failing.md\t3\t11\tnot reached\t",
    "b_passing.md\t1\t1\tran\t",
    "b_passing.md\t2\t7\tran\t",
    "c_fragment.mdx\t1\t1\tnot runnable\tplaceholder `...` statement",
]


def make_tree(root: Path, pages: dict[str, str]) -> None:
    for relative, text in pages.items():
        path = root / "docs" / "public" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_job(root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), "--output-dir", str(out), *extra],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},  # the runner decodes page output strictly
    )


def tsv_lines(out: Path) -> list[str]:
    return (out / "blocks.tsv").read_text(encoding="utf-8").splitlines()


def test_job_writes_one_row_per_block_in_page_order(tmp_path):
    make_tree(tmp_path, PAGES)
    out = tmp_path / "out"

    done = run_job(tmp_path, out)

    assert done.returncode == 0, done.stderr
    assert tsv_lines(out) == EXPECTED_ROWS


def test_job_writes_one_json_per_page_with_the_single_page_schema(tmp_path):
    make_tree(tmp_path, PAGES)
    out = tmp_path / "out"

    run_job(tmp_path, out)

    names = sorted(path.name for path in (out / "pages").iterdir())
    assert names == [
        "a_failing.md.json",
        "b_passing.md.json",
        "c_fragment.mdx.json",
        "d_no_python.md.json",
    ]
    failing = json.loads((out / "pages" / "a_failing.md.json").read_text(encoding="utf-8"))
    single = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(tmp_path), "--only", "a_failing.md", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = json.loads(single.stdout)
    assert set(failing) == set(expected)
    assert set(failing["pages"][0]) == set(expected["pages"][0])
    assert failing["has_failures"] is True
    assert failing["pages"][0]["page"] == "a_failing.md"
    assert failing["pages"][0]["status"] == "FAIL"
    assert failing["pages"][0]["first_failing_block"] == 2
    no_python = json.loads((out / "pages" / "d_no_python.md.json").read_text(encoding="utf-8"))
    assert no_python["pages"] == []
    assert no_python["has_failures"] is False


def test_job_exit_code_follows_strict(tmp_path):
    make_tree(tmp_path, PAGES)

    assert run_job(tmp_path, tmp_path / "plain").returncode == 0
    assert run_job(tmp_path, tmp_path / "strict", "--strict").returncode == 1


def test_only_restricts_the_job(tmp_path):
    make_tree(tmp_path, PAGES)
    out = tmp_path / "out"

    run_job(tmp_path, out, "--only", "b_passing")

    assert tsv_lines(out) == [
        HEADER,
        "b_passing.md\t1\t1\tran\t",
        "b_passing.md\t2\t7\tran\t",
    ]
    assert [path.name for path in (out / "pages").iterdir()] == ["b_passing.md.json"]


def test_pages_sort_by_relative_path_as_a_string(tmp_path):
    # '-' sorts before '/' in the relative path; comparing Path parts gives the reverse.
    make_tree(
        tmp_path,
        {
            "x/p.md": "```python\nvalue = 1\n```\n",
            "x-y.md": "```python\nvalue = 2\n```\n",
        },
    )
    out = tmp_path / "out"

    run_job(tmp_path, out)

    assert tsv_lines(out) == [
        HEADER,
        "x-y.md\t1\t1\tran\t",
        "x/p.md\t1\t1\tran\t",
    ]
    assert (out / "pages" / "x_p.md.json").is_file()


def test_timed_out_page_marks_runnable_blocks_not_reached_and_next_page_runs(tmp_path):
    make_tree(
        tmp_path,
        {
            "a_slow.md": (
                "```python\nprint('started', flush=True)\n```\n\n"
                "```python\nimport time\ntime.sleep(120)\n```\n"
            ),
            "b_after.md": "```python\nvalue = 1\n```\n",
        },
    )
    out = tmp_path / "out"

    done = run_job(tmp_path, out, "--timeout", "3")

    assert done.returncode == 0, done.stderr
    assert tsv_lines(out) == [
        HEADER,
        "a_slow.md\t1\t1\tnot reached\tpage timed out",
        "a_slow.md\t2\t5\tnot reached\t",
        "b_after.md\t1\t1\tran\t",
    ]


def test_blocks_run_with_the_interpreter_running_the_script(tmp_path):
    make_tree(
        tmp_path,
        {"p.md": f"```python\nimport sys\nassert sys.executable == {sys.executable!r}\n```\n"},
    )
    out = tmp_path / "out"

    run_job(tmp_path, out)

    assert tsv_lines(out) == [HEADER, "p.md\t1\t1\tran\t"]


def test_failure_inside_an_earlier_helper_names_the_calling_block(tmp_path):
    make_tree(
        tmp_path,
        {
            "p.md": (
                "```python\ndef h():\n    raise ValueError('inside')\n```\n\n"
                "```python\nx = 1\n```\n\n"
                "```python\nh()\n```\n\n"
                "```python\ny = 2\n```\n"
            ),
        },
    )
    out = tmp_path / "out"

    run_job(tmp_path, out)

    assert tsv_lines(out) == [
        HEADER,
        "p.md\t1\t1\tran\t",
        "p.md\t2\t6\tran\t",
        "p.md\t3\t10\tfailed\tValueError: inside",
        "p.md\t4\t14\tnot reached\t",
    ]
    page = json.loads((out / "pages" / "p.md.json").read_text(encoding="utf-8"))["pages"][0]
    assert page["first_failing_block"] == 1
    assert "raising_top_level_block" not in page


def test_undecodable_output_fails_that_page_and_the_next_page_runs(tmp_path):
    make_tree(
        tmp_path,
        {
            "a.md": '```python\nimport sys\nsys.stdout.buffer.write(b"\\xff\\xfe")\n```\n',
            "b.md": "```python\nvalue = 1\n```\n",
        },
    )
    out = tmp_path / "out"

    done = run_job(tmp_path, out, "--strict")

    assert done.returncode == 1, done.stderr
    assert tsv_lines(out) == [
        HEADER,
        "a.md\t1\t1\tnot reached\tpage failed, block unknown: UnicodeDecodeError: "
        "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
        "b.md\t1\t1\tran\t",
    ]
    page = json.loads((out / "pages" / "a.md.json").read_text(encoding="utf-8"))["pages"][0]
    assert page["status"] == "FAIL"
