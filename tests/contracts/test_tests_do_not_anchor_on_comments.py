"""Source-reading tests locate code without depending on comment wording."""

from __future__ import annotations

import ast
from bisect import bisect_left
import io
import inspect
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import tokenize

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path(__file__).relative_to(ROOT).as_posix()
TOOL_DIRECTIVES = ("noqa", "type:", "pragma", "fmt:", "isort:", "pylint:", "mypy:", "ruff:", "-*-")
EXEMPTIONS = (
    ("tests/crossval/rt5880_patch/reference/make_openems_reference.py",
     "patches a pinned upstream script by its comments (pin: A_TUTORIAL_SHA256); "
     "its copy proof compares a frozen slice of its own builder verbatim"),
)
pytestmark = pytest.mark.source_scan


def excluded_comment(text: str) -> bool:
    """Shebangs and tool directives do not locate source code."""
    body = text[1:]
    return body.startswith("!") or body.lstrip().startswith(TOOL_DIRECTIVES)


def scan_repository(root: Path) -> list[str]:
    """Return one diagnostic per literal/comment pair in tracked Python and shell."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root,
    ).decode("utf-8").split("\0")
    by_basename = {}
    for name in tracked:
        path = Path(name)
        if path.suffix in {".py", ".sh"} and name != CONTRACT:
            by_basename.setdefault(path.name, []).append(name)
    candidates = []
    scopes = {}
    exemptions = {path for path, reason in EXEMPTIONS if reason.strip()}
    for name in sorted(tracked):
        if not name.startswith("tests/") or not name.endswith(".py") or name in exemptions:
            continue
        with tokenize.open(root / name) as handle:
            source = handle.read()
        strings = [node for node in ast.walk(ast.parse(source, filename=name))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        literals = []
        for node in strings:
            for offset, line in enumerate(node.value.splitlines()):
                literal = line.strip()
                if (literal.startswith("#") and len(literal[1:].lstrip()) >= 6
                        and not excluded_comment(literal)):
                    literals.append((literal, name, node.lineno + offset))
        if literals:
            named_basenames = {node.value.rsplit("/", 1)[-1] for node in strings}
            scopes[name] = {source_name for basename, names in by_basename.items()
                            if basename in named_basenames for source_name in names}
            candidates.extend(literals)
    needed_literals = {}
    for literal, name, _ in candidates:
        for source_name in scopes[name]:
            needed_literals.setdefault(source_name, set()).add(literal)
    comments = {}
    for name in sorted(set().union(*scopes.values())):
        path = Path(name)
        with tokenize.open(root / path) as handle:
            source = handle.read()
        comments[name] = []
        if not any(literal in source for literal in needed_literals[name]):
            continue
        found = []
        if path.suffix == ".py":
            found = [(token.string.strip(), token.start[0])
                     for token in tokenize.generate_tokens(io.StringIO(source).readline)
                     if token.type == tokenize.COMMENT]
        elif path.suffix == ".sh":
            found = [(line.strip(), number) for number, line in enumerate(
                source.splitlines(), 1)
                if line.strip().startswith("#")]
        comments[name] = sorted((comment, number) for comment, number in found
                                if not excluded_comment(comment))
    hits = []
    for literal, name, number in sorted(set(candidates)):
        for source_name in sorted(scopes[name]):
            source_comments = comments[source_name]
            index = bisect_left(source_comments, (literal,))
            while index < len(source_comments) and source_comments[index][0].startswith(literal):
                _, source_number = source_comments[index]
                hits.append(f"{name}:{number}: names the comment at "
                            f"{source_name}:{source_number}; anchor on code")
                index += 1
    return sorted(hits)


def test_tracked_tests_do_not_anchor_on_comments():
    started = perf_counter()
    hits = scan_repository(ROOT)
    print(f"comment-anchor scan: {perf_counter() - started:.3f}s", file=sys.stderr)
    assert not hits, "\n".join(hits)


@pytest.fixture
def planted_tree(tmp_path):
    files = {
        "module.py": 'value = 1  # planted Python boundary\ntext = "# string boundary"\n#!/usr/bin/env python3\n# noqa: E501\n',
        "unmentioned.py": "# outside named scope\n",
        "worker.sh": "#!/bin/sh\n  # planted shell boundary\n# planted prefix boundary extended\n# tiny\n",
        "tests/test_sample.py": (
            'python = "# planted Python boundary"\n'
            'shell = "# planted shell boundary"\n'
            'prefix = "# planted prefix boundary"\n'
            'code = "value = 1"\n'
            'unmatched = "# absent boundary"\n'
            'short = "# tiny"\n'
            'string = "# string boundary"\n'
            'shebang = "#!/usr/bin/env python3"\n'
            'directive = "# noqa: E501"\n'
            'outside = "# outside named scope"\n'
            'paths = ("module.py", "scripts/worker.sh")\n'
        ),
        "untracked.py": "# absent boundary\n",
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "--template="], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "module.py", "worker.sh", "unmentioned.py", "tests/test_sample.py"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def assert_planted_hits(scan, root):
    """Expected locations come from the planted files, without scanner helpers."""
    assert scan(root) == [
        "tests/test_sample.py:1: names the comment at module.py:1; anchor on code",
        "tests/test_sample.py:2: names the comment at worker.sh:2; anchor on code",
        "tests/test_sample.py:3: names the comment at worker.sh:3; anchor on code",
    ], "planted Python, shell and prefix anchors must be refused; code, unmatched, short, string, shebang, directive and unnamed-file literals must pass"


def test_planted_comments_and_allowed_literals(planted_tree):
    assert_planted_hits(scan_repository, planted_tree)


@pytest.mark.parametrize(("original", "replacement"), [
    ('elif path.suffix == ".sh":', 'elif False:'),
    ('source_comments[index][0].startswith(literal)', 'source_comments[index][0] == literal'),
    ('if basename in named_basenames', 'if True'),
], ids=["M1-shell-comments-not-scanned", "M2-equality-instead-of-prefix", "M3-basename-scope-removed"])
def test_planted_assertion_catches_mutations(planted_tree, original, replacement):
    source = inspect.getsource(scan_repository)
    assert source.count(original) == 1
    namespace = dict(globals())
    exec(compile(source.replace(original, replacement), "<mutated-scanner>", "exec"), namespace)
    with pytest.raises(AssertionError, match="planted Python, shell and prefix anchors") as caught:
        assert_planted_hits(namespace["scan_repository"], planted_tree)
    print(f"mutation {original} -> {replacement}: {caught.value}", file=sys.stderr)
