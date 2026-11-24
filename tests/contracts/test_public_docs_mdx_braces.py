"""Refuse JavaScript expressions accidentally written as public MDX prose."""

from pathlib import Path
import re
import string

import pytest


FIX = (r"MDX reads {…} in prose as JavaScript: put the formula in a code span (`e^{+jωt}`), "
       r"escape the braces as \{ \}, or write it without braces (exp(+jωt)).")


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


def allowed_expression(expression: str) -> bool:
    """Only a whole comment or a single string literal without interpolation is prose-safe."""
    if not expression.endswith("}"):
        return False
    body = expression[1:-1].strip()
    if body.startswith("/*"):
        return body.find("*/", 2) == len(body) - 2
    if body.startswith(("'", '"', "`")):
        if body[0] == "`" and "${" in body:
            return False
        end = js_template_end(body, 0) if body[0] == "`" else js_string_end(body, 0)
        trailing_slashes = len(body[:-1]) - len(body[:-1].rstrip("\\"))
        return (len(body) >= 2 and body[-1] == body[0]
                and trailing_slashes % 2 == 0 and end == len(body))
    return False


def tag_end(text: str, start: int) -> int | None:
    """Skip JSX attributes, including quoted > and nested expression braces."""
    if not re.match(r"</?[A-Za-z][\w.:-]*(?=[\s/>])|</?>", text[start:]):
        return None
    position = start + 1
    while position < len(text):
        char = text[position]
        if char in "\"'":
            position = js_string_end(text, position)
        elif char == "{":
            position = expression_end(text, position)
        elif char == ">":
            return position + 1
        else:
            position += 1
    return None


def code_span_limit(text: str, start: int) -> int:
    """A code span cannot close in another block or table cell."""
    boundary = re.search(
        r"\n(?:[^\S\n]*\n| {0,3}(?:#|[-*+] |\d+[.)] |>+|\||<))", text[start:])
    limit = start + boundary.start() if boundary else len(text)
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    if re.match(r" {0,3}#", line):
        limit = min(limit, line_end)
    elif re.match(r" {0,3}\|", line):
        limit = min(limit, line_end)
        for token in re.finditer(r"\\.|\|", line):
            pipe = line_start + token.start()
            if token[0] == "|" and pipe >= start:
                limit = min(limit, pipe)
                break
    return limit


def offending_lines(page: str) -> list[str]:
    fence = None
    frontmatter = False
    lines = []
    for number, line in enumerate(page.splitlines(keepends=True)):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if number == 0 and line.strip() == "---":
            frontmatter = True
            lines.append(blank(line))
        elif frontmatter:
            lines.append(blank(line))
            if line.strip() == "---":
                frontmatter = False
        elif fence is not None:
            lines.append(blank(line))
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker and (marker[1][0] != "`" or "`" not in marker[2]):
            fence = marker[1]
            lines.append(blank(line))
        elif (not lines or not lines[-1].strip()) and re.match(r"^(?:import|export)\b", line):
            lines.append(blank(line))
        else:
            lines.append(line)
    text = "".join(lines)
    found = set()
    position = 0
    while position < len(text):
        char = text[position]
        if char == "\\" and position + 1 < len(text) and text[position + 1] in string.punctuation:
            position += 2
        elif char == "`":
            opener = re.match(r"`+", text[position:])[0]
            end = position + len(opener)
            limit = code_span_limit(text, end)
            closer = next((run for run in re.finditer(r"`+", text[end:limit])
                           if len(run[0]) == len(opener)), None)
            position = end + closer.end() if closer else end
        elif char == "<" and (end := tag_end(text, position)) is not None:
            position = end
        elif char == "{":
            end = expression_end(text, position)
            if allowed_expression(text[position:end]):
                position = end
            else:
                found.add(text.count("\n", 0, position) + 1)
                position += 1
        else:
            if char == "}":
                found.add(text.count("\n", 0, position) + 1)
            position += 1
    return [f"{number}: {page.splitlines()[number - 1]}" for number in sorted(found)]


@pytest.mark.parametrize("page, expected", [
    ("time convention e^{+jωt}", ["1: time convention e^{+jωt}"]),
    ("time convention `e^{+jωt}`", []),
    ("``e^{+jωt}``", []),
    ("```mdx\n{x}\n```", []),
    ("~~~mdx\n{x}\n~~~", []),
    (r"\{x\}", []),
    ("{/* note */}", []),
    ("{ /* note */ }", []),
    ("<code>{'\"x\"'}</code>", []),
    ('<code>{"x"}</code>', []),
    ("<code>{`plain`}</code>", []),
    ("<code>{`a ${b}`}</code>", ["1: <code>{`a ${b}`}</code>"]),
    ('{`outer ${`inner ${"}"}`} tail`}', ['1: {`outer ${`inner ${"}"}`} tail`}']),
    ("{`echo ${HOME}`}", ["1: {`echo ${HOME}`}"]),
    ("<Figure caption={x} />", []),
    ("<Figure\n prop={{a: 1}} />", []),
    ('<Figure title="{x} >" />', []),
    ('<Figure prop={{a: ">"}}>{x}</Figure>', ['1: <Figure prop={{a: ">"}}>{x}</Figure>']),
    ("a } b", ["1: a } b"]),
    ("a { b", ["1: a { b"]),
    ("| {a} |", ["1: | {a} |"]),
    ('---\ndescription: "{x}"\n---\nplain', []),
    ('import X from "{x}";\nexport const x = {a: 1};', []),
    ("{frontmatter.title}", ["1: {frontmatter.title}"]),
    ("{'x' + y}", ["1: {'x' + y}"]),
    ("{/* note */ value}", ["1: {/* note */ value}"]),
    ("{'unterminated}", ["1: {'unterminated}"]),
    (r"\<Figure prop={x} />", [r"1: \<Figure prop={x} />"]),
    ("the ` key\n\n{x}\n\nuse `y`", ["3: {x}"]),
    ("{x}\n{y}", ["1: {x}", "2: {y}"]),
    ("`{` and }", ["1: `{` and }"]),
    ("# Head ` tick\nText {x} and `code`", ["2: Text {x} and `code`"]),
    ("- a ` b\n- {x} `c`", ["2: - {x} `c`"]),
    ("| `x | {y} | z` |", ["1: | `x | {y} | z` |"]),
    (r"| `x \| {y} z` |", []),
    ("Some text\nimport the field {x}", ["2: import the field {x}"]),
    ("Some text\nexport the field {x}", ["2: export the field {x}"]),
    ("```js `x`\n{y}\n```", ["2: {y}"]),
    ("~~~js `x`\n{y}\n~~~", []),
    ("`wrapped\n{x}`", []),
])
def test_offending_lines(page, expected):
    assert offending_lines(page) == expected


@pytest.mark.parametrize("marker", ["#", "-", "*", "+", "1.", "2)", ">", "|", "<div>"])
@pytest.mark.parametrize("indent", ["", "   "])
def test_code_span_stops_at_block_start(marker, indent):
    line = f"{indent}{marker} {{x}} `code`"
    assert offending_lines(f"Text ` tick\n{line}") == [f"2: {line}"]


@pytest.mark.reads_docs_for_gate(reason=(
    "Checks MDX syntax the site build evaluates: a brace in prose is read as JavaScript "
    "and stops the build; formulas go in code spans or escaped braces."))
def test_public_docs_mdx_braces():
    root = Path(__file__).resolve().parents[2]
    pages = sorted((root / "docs/public").rglob("*.mdx"))
    assert pages
    failures = [
        f"{page.relative_to(root)}:{line}"
        for page in pages
        for line in offending_lines(page.read_text(encoding="utf-8"))
    ]
    assert not failures, "\n".join([*failures, FIX])
