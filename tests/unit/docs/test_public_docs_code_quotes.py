"""The site build converts quotes in JSX text inside <code>, so a copied command then fails."""

from pathlib import Path
import re
import string

import pytest


FIX = ("write it as a markdown code span or a JS string expression ({'…'}); "
       "the site build converts quotes in JSX text")
QUOTES = re.compile('["\'“”‘’]')
CODE = re.compile(r'<code\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*>(.*?)</code\s*>', re.DOTALL)


def blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def js_string_end(text: str, start: int) -> int:
    position = start + 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text[position] == text[start]:
            return position + 1
        else:
            position += 1
    return len(text)


def js_template_end(text: str, start: int) -> int:
    position = start + 1
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text[position] == "`":
            return position + 1
        elif text.startswith("${", position):
            position = expression_end(text, position + 1)
        else:
            position += 1
    return len(text)


def expression_end(text: str, start: int) -> int:
    position = start + 1
    depth = 1
    while position < len(text):
        char = text[position]
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            position = len(text) if end == -1 else end + 2
        elif text.startswith("//", position):
            end = text.find("\n", position + 2)
            position = len(text) if end == -1 else end
        elif char in "\"'":
            position = js_string_end(text, position)
        elif char == "`":
            position = js_template_end(text, position)
        else:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if not depth:
                    return position + 1
            position += 1
    return len(text)


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
    masked = []
    position = 0
    while position < len(text):
        char = text[position]
        if char == "\\" and position + 1 < len(text) and text[position + 1] in string.punctuation:
            masked.append(text[position:position + 2].replace("<", " "))
            position += 2
        elif char == "`":
            opener = re.match(r"`+", text[position:])[0]
            end = position + len(opener)
            boundary = re.search(r"\n[^\S\n]*\n", text[end:])
            limit = end + boundary.start() if boundary else len(text)
            closer = next((run for run in re.finditer(r"`+", text[end:limit])
                           if len(run[0]) == len(opener)), None)
            if closer is not None:
                end += closer.end()
                masked.append(blank(text[position:end]))
            else:
                masked.append(opener)
            position = end
        elif char == "{":
            end = expression_end(text, position)
            masked.append(blank(text[position:end]))
            position = end
        else:
            masked.append(char)
            position += 1
    text = "".join(masked)
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
    ('`{` and <code>"x"</code>', ['1: `{` and <code>"x"</code>']),
    ('\\{ <code>"x"</code>', ['1: \\{ <code>"x"</code>']),
    ("{/* it's */}\n<code>\"x\"</code>", ['2: <code>"x"</code>']),
    ('the ` key\n\n<code>"x"</code>\n\nuse `y`', ['3: <code>"x"</code>']),
    ('`<code>"x"</code>`', []),
    ('{ /* } " */ value }\n<code>"x"</code>', ['2: <code>"x"</code>']),
    ('{ // } "\nvalue }\n<code>"x"</code>', ['3: <code>"x"</code>']),
    ('{`outer ${`inner ${"}"}`} tail`}\n<code>"x"</code>', ['2: <code>"x"</code>']),
    ('<code>{"a\\\"}b"}</code>', []),
    ('\\<code>"x"</code>', []),
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
