#!/usr/bin/env python3
"""Render the showcase objective from its script into the public pages.

The excerpt is the inner ``J`` of ``make_objective`` in
``scripts/showcase/patch_sensitivity.py`` and the statement that calls
``jax.value_and_grad`` on it, copied line for line.
Run with --check to report documentation drift without modifying files.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "scripts/showcase/patch_sensitivity.py"
PAGES = ("docs/public/index.mdx", "docs/public/showcase/sensitivity-map.mdx")
START = "{/* rfx-showcase-objective:start */}"
END = "{/* rfx-showcase-objective:end */}"


def _function(body: list[ast.stmt], name: str) -> ast.FunctionDef:
    found = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise SystemExit(f"{SOURCE}: expected one function {name!r}, found {len(found)}")
    return found[0]


def _calls_value_and_grad_on_j(node: ast.stmt) -> bool:
    return any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
               and call.func.attr == "value_and_grad"
               and [a.id for a in call.args if isinstance(a, ast.Name)] == ["J"]
               for call in ast.walk(node))


def snippet(root: Path) -> str:
    source = (root / SOURCE).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    inner = _function(_function(tree.body, "make_objective").body, "J")
    objective = textwrap.dedent("\n".join(lines[inner.lineno - 1:inner.end_lineno]))
    stage = _function(tree.body, "stage_x64fd")
    calls = [node for node in stage.body if _calls_value_and_grad_on_j(node)]
    if len(calls) != 1:
        raise SystemExit(f"{SOURCE}: expected one value_and_grad(J) statement in stage_x64fd, "
                         f"found {len(calls)}")
    call = textwrap.dedent("\n".join(lines[calls[0].lineno - 1:calls[0].end_lineno]))
    return f"{objective}\n\n{call}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    code = snippet(args.repo_root)
    content = f"{START}\n\n```python\n{code}```\n\n{END}"
    changed = []
    for rel in PAGES:
        path = args.repo_root / rel
        original = path.read_text()
        if original.count(START) != 1 or original.count(END) != 1:
            raise SystemExit(f"Expected one generated objective excerpt in {rel}")
        updated = re.sub(re.escape(START) + r".*?" + re.escape(END),
                         lambda _: content, original, flags=re.S)
        if original != updated:
            changed.append(rel)
            if not args.check:
                path.write_text(updated)
    if args.check and changed:
        print("Regenerate with python scripts/sync_showcase_snippet.py: " + ", ".join(changed))
        return 1
    print("Showcase objective excerpt: " + ("updated " + ", ".join(changed) if changed else "in sync"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
