"""Every public docs page opens with YAML frontmatter that parses to a mapping with a title.

PR 1704 was blocked by `docs/public/design/design-regions.mdx`, whose description held an unquoted
second colon. Only the non-required api-reference job parsed the frontmatter; nothing in tests/unit/docs
did. This test runs the `yaml.safe_load` that `scripts/build_public_docs_bundle.py` uses on every
docs/public/**/*.md(x) page. It lives in tests/contracts because pr-tests runs only tests/contracts on a
docs-only PR, which is exactly the PR that breaks frontmatter. The downstream Astro site parses with js-yaml,
which also rejects duplicate keys; PyYAML accepts them, so this test does not catch that case.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_PAGES = sorted(p for p in (_REPO / "docs" / "public").rglob("*") if p.suffix in {".md", ".mdx"})


def frontmatter_problem(text: str) -> str | None:
    """None if `text` opens with `---` frontmatter that parses to a mapping with a title, else why not."""
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        return "does not open with a '---' frontmatter line"
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.rstrip("\r") == "---")
    except StopIteration:
        return "frontmatter has no closing '---' line"
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        return "frontmatter is not valid YAML: " + " ".join(str(exc).split())
    if not isinstance(data, dict):
        return f"frontmatter parses to {type(data).__name__}, not a mapping"
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        return "frontmatter has no non-empty string 'title'"
    return None


def test_there_are_pages_to_check() -> None:
    assert len(_PAGES) >= 50, _PAGES  # an emptied or moved tree must not pass vacuously


@pytest.mark.reads_docs_for_gate(reason=(
    "Parses the YAML frontmatter the site build parses; a page that does not parse blocks the merge train "
    "(PR 1704). It checks syntax and the presence of a title, never the prose."))
@pytest.mark.parametrize("page", _PAGES, ids=lambda p: str(p.relative_to(_REPO)))
def test_page_frontmatter_parses(page: Path) -> None:
    problem = frontmatter_problem(page.read_text(encoding="utf-8"))
    assert problem is None, f"{page.relative_to(_REPO)}: {problem}"


@pytest.mark.parametrize("text, expected", [
    ("---\ntitle: A\ndescription: one: two\n---\nbody\n", "not valid YAML"),
    ("---\ntitle: A\ndescription: fine\nbody without a closing line\n", "no closing '---'"),
    ("title: A\n", "does not open"),
    ("---\n- a\n---\n", "not a mapping"),
    ("---\ndescription: x\n---\n", "no non-empty string 'title'"),
])
def test_the_checker_names_each_failure(text: str, expected: str) -> None:
    problem = frontmatter_problem(text)
    assert problem is not None and expected in problem, problem
