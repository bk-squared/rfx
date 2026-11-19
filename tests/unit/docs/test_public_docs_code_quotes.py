"""The site build converts quotes in JSX text inside <code>, so a copied command then fails."""

from pathlib import Path
import re

import pytest


FIX = ("write it as a markdown code span or a JS string expression ({'…'}); "
       "the site build converts quotes in JSX text")
QUOTES = re.compile('["\'“”‘’]')
CODE = re.compile(r'<code\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*>(.*?)</code\s*>', re.DOTALL)


def blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def offending_lines(page: str) -> list[str]:
    fence = None
    lines = []
    for line in page.splitlines(keepends=True):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if fence is not None:
            lines.append(blank(line))
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
            lines.append(blank(line))
        else:
            lines.append(line)
    text = "".join(lines)
    depth = 0
    quote = None
    escaped = False
    masked = []
    for char in text:
        if depth:
            masked.append("\n" if char == "\n" else " ")
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'`":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        elif char == "{":
            depth = 1
            masked.append(" ")
        else:
            masked.append(char)
    text = "".join(masked)
    text = re.sub(r"(`+)(?!`)(.*?)(?<!`)\1(?!`)", lambda match: blank(match[0]), text, flags=re.DOTALL)
    found = set()
    for element in CODE.finditer(text):
        body = re.sub(r'<[^>]*>', lambda match: blank(match[0]), element[1])
        for match in QUOTES.finditer(body):
            found.add(text.count("\n", 0, element.start(1) + match.start()) + 1)
    return [f"{number}: {page.splitlines()[number - 1]}" for number in sorted(found)]


@pytest.mark.parametrize("page, expected", [
    ('<p data-rfx="actions"><code>pip install "rfx-fdtd>=2.0"</code></p>',
     ['1: <p data-rfx="actions"><code>pip install "rfx-fdtd>=2.0"</code></p>']),
    ('<code>“x”</code>', ['1: <code>“x”</code>']),
    ('<code>\'x\'</code>', ["1: <code>'x'</code>"]),
    ('<code>‘x’</code>', ['1: <code>‘x’</code>']),
    ('<p data-rfx="actions">`pip install "rfx-fdtd>=2.0"`</p>', []),
    ('```mdx\n<code>"x"</code>\n```\n', []),
    ('~~~mdx\n<code>"x"</code>\n~~~\n', []),
    ('<code>{\'"x"\'}</code>', []),
    ('<code>9f1abcc</code>', []),
    ('``<code>"x"</code> ` example``', []),
    ('<code title="example > value">\n"x"\n</code>', ['2: "x"']),
    ('```\n<code>"ignored"</code>\n```\n<code>"x"</code>', ['4: <code>"x"</code>']),
    ('<code>{{value: "}"}} "x"</code>', ['1: <code>{{value: "}"}} "x"</code>']),
    ('<code><span title="example">plain</span></code>', []),
])
def test_offending_lines(page, expected):
    assert offending_lines(page) == expected


@pytest.mark.reads_docs_for_gate(reason=(
    "Checks JSX code syntax that lets the site build convert command quotes; "
    "quoted commands must use markdown code spans or JS string expressions."))
def test_public_docs_code_quotes():
    root = Path(__file__).resolve().parents[3]
    pages = sorted((root / "docs/public").rglob("*.mdx"))
    assert pages
    failures = [
        f"{page.relative_to(root)}:{line}: {FIX}"
        for page in pages
        for line in offending_lines(page.read_text(encoding="utf-8"))
    ]
    assert not failures, "\n".join(failures)
