"""Parse fenced Python in tracked public pages in the non-required docs lane."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from tests._git_tracked import tracked_set

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.docs_consistency

# Each exception names the documentation-batch issue; repaired or missing
# blocks must be removed from this list, never silently retained.
KNOWN_BROKEN: dict[tuple[str, int], str] = {}

PAGES = sorted(
    path for path in tracked_set(REPO)
    if path.startswith("docs/public/") and Path(path).suffix in {".md", ".mdx"}
)
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")


def python_blocks(text: str):
    """Yield (first code line, source), ignoring MDX comments outside fences."""
    fence = None
    in_comment = False
    lines = []
    first_line = 0
    is_python = False
    for number, line in enumerate(text.splitlines(), 1):
        if fence is None:
            visible = ""
            while line:
                if in_comment:
                    _, end, line = line.partition("*/}")
                    if not end:
                        break
                    in_comment = False
                else:
                    before, start, line = line.partition("{/*")
                    visible += before
                    if not start:
                        break
                    in_comment = True
            match = _FENCE.match(visible)
            if match:
                fence, info = match.groups()
                words = info.split()
                is_python = bool(words) and words[0].lower() in {"python", "py"}
                first_line = number + 1
                lines = []
        else:
            stripped = line.strip()
            if (len(stripped) >= len(fence)
                    and set(stripped) == {fence[0]}):
                if is_python:
                    yield first_line, python_source(lines)
                fence = None
            else:
                lines.append(line)
    if fence is not None and is_python:
        yield first_line, python_source(lines)


def python_source(lines: list[str]) -> str:
    source = textwrap.dedent("\n".join(lines))
    first = next((line for line in source.splitlines() if line.strip()), "")
    if first.startswith(">>> "):
        source = "\n".join(
            line[4:] if line.startswith((">>> ", "... ")) else ""
            for line in source.splitlines()
            if line.startswith((">>> ", "... ")) or line == "..."
        )
    return source


@pytest.mark.parametrize("page", sorted(set(PAGES) | {p for p, _ in KNOWN_BROKEN}) or [""], ids=str)
def test_public_python_blocks_parse(page: str) -> None:
    assert PAGES, "git ls-files found no public Markdown pages"
    expected = {line for path, line in KNOWN_BROKEN if path == page}
    errors = []
    seen = set()
    blocks = python_blocks((REPO / page).read_text(encoding="utf-8")) if page in PAGES else ()
    for first_line, source in blocks:
        seen.add(first_line)
        try:
            ast.parse(source)
        except SyntaxError as error:
            if first_line not in expected:
                errors.append(f"{page}:{first_line + (error.lineno or 1) - 1}: {error.msg}")
        else:
            if first_line in expected:
                errors.append(f"{page}:{first_line}: KNOWN_BROKEN block now parses; remove it")
    for first_line in sorted(expected - seen):
        errors.append(f"{page}:{first_line}: KNOWN_BROKEN block no longer exists; remove it")
    assert not errors, "\n".join(errors)
