#!/usr/bin/env python3
"""Render the tested hello-world builder and run block into onboarding docs.

Run with --check to report documentation drift without modifying files.
"""
from __future__ import annotations

import argparse
import ast
import io
from pathlib import Path
import re
import textwrap
import tokenize

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- rfx-hello-world:start -->"
END = "<!-- rfx-hello-world:end -->"


def without_comments(source: str) -> str:
    tokens = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
              if token.type != tokenize.COMMENT]
    rendered = tokenize.untokenize(tokens)
    return "\n".join(line.rstrip() for line in rendered.splitlines() if line.strip())


def snippet(root: Path) -> str:
    source = (root / "examples/quickstart/hello_world.py").read_text()
    tree = ast.parse(source)
    builder = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "build_simulation")
    imports = [ast.get_source_segment(source, node) for node in tree.body
               if isinstance(node, ast.ImportFrom) and node.module == "rfx"]
    statements = [node for node in builder.body if not isinstance(node, ast.Return)
                  and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                           and isinstance(node.value.value, str))]
    lines = source.splitlines()
    body = textwrap.dedent("\n".join(lines[statements[0].lineno - 1:statements[-1].end_lineno]))
    run = source.split("    # docs-run-start\n", 1)[1].split("    # docs-run-end", 1)[0]
    return "\n\n".join(("\n".join(imports), without_comments(body),
                           without_comments(textwrap.dedent(run)))) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    code = snippet(args.repo_root)
    changed = []
    for rel in ("README.md", "docs/public/guide/first-run.mdx"):
        path = args.repo_root / rel
        start, end = (("{/* rfx-hello-world:start */}", "{/* rfx-hello-world:end */}")
                      if path.suffix == ".mdx" else (START, END))
        content = f"{start}\n\n```python\n{code}```\n\n{end}"
        original = path.read_text()
        if original.count(start) != 1 or original.count(end) != 1:
            raise SystemExit(f"Expected one generated snippet in {rel}")
        updated = re.sub(re.escape(start) + r".*?" + re.escape(end),
                         lambda _: content, original, flags=re.S)
        if original != updated:
            changed.append(rel)
            if not args.check:
                path.write_text(updated)
    if args.check and changed:
        print("Regenerate with python scripts/sync_intro_snippet.py: " + ", ".join(changed))
        return 1
    print("Hello-world snippet: " + ("updated " + ", ".join(changed) if changed else "in sync"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
