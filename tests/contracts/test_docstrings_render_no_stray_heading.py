"""Markdown renderers treat a docstring line beginning with # and a digit as a heading.
Keep wrapped tracker references from creating extra headings in API pages.
"""

import ast
from itertools import dropwhile
from pathlib import Path
import re


def test_docstrings_render_no_stray_heading():
    root = Path(__file__).resolve().parents[2]
    failures = []
    for path in sorted((root / "rfx").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if doc is None:
                continue
            # cleandoc removes leading blank lines but preserves line order.
            raw = ast.get_docstring(node, clean=False).expandtabs().splitlines()
            leading = len(raw) - len(list(dropwhile(lambda line: not line.strip(), raw)))
            for offset, line in enumerate(doc.splitlines()):
                if re.match(r"^#[^\s#]", line.lstrip()):
                    lineno = node.body[0].lineno + leading + offset
                    failures.append(f"{path.relative_to(root)}:{lineno}: {line}")
    assert not failures, "Docstring lines render as headings:\n" + "\n".join(failures)
