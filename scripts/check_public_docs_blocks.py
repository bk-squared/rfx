#!/usr/bin/env python3
"""Execute every python code block in docs/public and report what happens.

Each page's runnable blocks are concatenated in document order into one script
and executed once, because blocks on a page normally depend on earlier blocks
(``sim = rfx.Simulation(...)`` in block 1, ``sim.run()`` in block 3). A page is
therefore checked the way a reader consumes it: top to bottom, in one process.

What a PASS proves
------------------
The page's python blocks, run in order on CPU, import cleanly and complete
without raising. That is all.

What a PASS does NOT prove
--------------------------
Nothing about physics. A page can pass while printing a wrong resonance, an
unconverged result, a directivity that disagrees with the reference solver, or
an energy sum above one. It also does not check prose, numbers quoted in prose,
narrative claims, or whether the printed output matches what the page says the
output will be. Those need a human or a separate oracle. This script answers one
question only: does the code the docs tell people to run still run?

A FAIL is likewise not always a docs defect -- a block may reference a data file
the reader is expected to supply, or a third-party package that is absent here.
Read the reported exception before concluding anything.

Warnings are left at Python's default filter so the output matches what a reader
running the snippet actually sees; ``--grep-deprecations`` surfaces any
DeprecationWarning a page emits, which is how an API scheduled for removal
becomes visible before it breaks the docs.

Usage::

    python3 scripts/check_public_docs_blocks.py
    python3 scripts/check_public_docs_blocks.py --only guide/probes-sparams
    python3 scripts/check_public_docs_blocks.py --format json --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DOC_EXTS = {".md", ".mdx"}
PYTHON_INFO_PREFIXES = ("python", "py")
BLOCK_MARKER = "# ===== rfx-doc-block {number} (doc line {line}) ====="
MARKER_RE = re.compile(r"^# ===== rfx-doc-block (\d+) \(doc line (\d+)\) =====$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})[ \t]*(\S*)")
PLACEHOLDER_RE = re.compile(r"<[a-z_][a-z0-9_]*>", re.IGNORECASE)
EXC_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)\b")


@dataclass
class BlockReport:
    number: int
    doc_line: int
    info: str
    n_lines: int
    fragment_reason: str | None = None

    @property
    def is_fragment(self) -> bool:
        return self.fragment_reason is not None


@dataclass
class PageReport:
    page: str
    blocks: list[BlockReport]
    status: str = "pending"
    returncode: int | None = None
    wall_seconds: float = 0.0
    first_failing_block: int | None = None
    first_failing_doc_line: int | None = None
    exception_head: list[str] = field(default_factory=list)
    deprecations: list[str] = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def fragments(self) -> list[BlockReport]:
        return [b for b in self.blocks if b.is_fragment]

    @property
    def runnable(self) -> list[BlockReport]:
        return [b for b in self.blocks if not b.is_fragment]


@dataclass
class RunReport:
    repo_root: str
    timeout: int
    pages: list[PageReport]

    @property
    def executed(self) -> list[PageReport]:
        return [p for p in self.pages if p.status in {"PASS", "FAIL"}]

    @property
    def n_pass(self) -> int:
        return sum(1 for p in self.pages if p.status == "PASS")

    @property
    def n_fail(self) -> int:
        return sum(1 for p in self.pages if p.status == "FAIL")

    @property
    def n_fragment_only(self) -> int:
        return sum(1 for p in self.pages if p.status == "fragment-only")

    @property
    def has_failures(self) -> bool:
        return self.n_fail > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to research/rfx.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="PAGE",
        help="Restrict to these pages, by path relative to docs/public, with or "
        "without extension (e.g. guide/probes-sparams). A directory prefix "
        "selects everything under it.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-page wall-clock limit in seconds (default: 600).",
    )
    parser.add_argument(
        "--grep-deprecations",
        action="store_true",
        help="Report each DeprecationWarning a page's run emitted.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def is_doc_file(path: Path) -> bool:
    return path.suffix in DOC_EXTS


def iter_doc_pages(public_root: Path) -> list[Path]:
    return sorted(p for p in public_root.rglob("*") if p.is_file() and is_doc_file(p))


def selects(rel_page: str, selectors: list[str]) -> bool:
    stem = rel_page.rsplit(".", 1)[0]
    for raw in selectors:
        sel = raw.strip().strip("/")
        if sel in {rel_page, stem}:
            return True
        if rel_page.startswith(f"{sel}/") or stem.startswith(f"{sel}/"):
            return True
    return False


def is_python_info(info: str) -> bool:
    lowered = info.lower()
    if not lowered:
        return False
    head = re.split(r"[^a-z0-9+]", lowered, maxsplit=1)[0]
    return head in PYTHON_INFO_PREFIXES


def extract_blocks(text: str) -> list[tuple[str, int, str]]:
    """Return (info, doc_line_of_fence, code) for each python fenced block."""
    lines = text.splitlines()
    blocks: list[tuple[str, int, str]] = []
    i = 0
    while i < len(lines):
        match = FENCE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        fence, info = match.group(2), match.group(3)
        closer = re.compile(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not closer.match(lines[j]):
            body.append(lines[j])
            j += 1
        if is_python_info(info):
            blocks.append((info, i + 1, "\n".join(body)))
        i = j + 1
    return blocks


def _is_ellipsis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def classify_fragment(code: str) -> str | None:
    """Return a reason if the block is a non-runnable doc fragment, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        if PLACEHOLDER_RE.search(code):
            return f"placeholder `<...>` (SyntaxError: {exc.msg})"
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    # A bare `...` statement and an Ellipsis passed as an argument
    # (`sim.add_port(...)`) both parse fine but are placeholders, not code.
    # Ellipsis inside a subscript is NOT a placeholder -- `S[..., bins]` is
    # ordinary numpy indexing, and `Callable[..., int]` is a type annotation.
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and _is_ellipsis(node.value):
            return "placeholder `...` statement"
        if isinstance(node, ast.Call):
            if any(_is_ellipsis(arg) for arg in node.args):
                return "placeholder `...` argument"
            if any(_is_ellipsis(kw.value) for kw in node.keywords):
                return "placeholder `...` argument"
    if not tree.body:
        return "no executable statements"
    return None


def build_page_report(page: Path, public_root: Path) -> PageReport:
    rel = page.relative_to(public_root).as_posix()
    blocks = [
        BlockReport(
            number=n,
            doc_line=doc_line,
            info=info,
            n_lines=len(code.splitlines()),
            fragment_reason=classify_fragment(code),
        )
        for n, (info, doc_line, code) in enumerate(extract_blocks(page.read_text(encoding="utf-8")), 1)
    ]
    return PageReport(page=rel, blocks=blocks)


def concatenate(page: Path, public_root: Path, report: PageReport) -> str:
    codes = {
        n: code
        for n, (_, _, code) in enumerate(extract_blocks(page.read_text(encoding="utf-8")), 1)
    }
    parts = [
        BLOCK_MARKER.format(number=b.number, line=b.doc_line) + "\n" + codes[b.number]
        for b in report.runnable
    ]
    return "\n".join(parts) + "\n"


def block_for_line(script: str, lineno: int) -> tuple[int | None, int | None]:
    """Map a line in the concatenated script back to (block number, doc line)."""
    found: tuple[int | None, int | None] = (None, None)
    for index, line in enumerate(script.splitlines(), 1):
        if index > lineno:
            break
        match = MARKER_RE.match(line)
        if match:
            found = (int(match.group(1)), int(match.group(2)))
    return found


def exception_head(output: str, limit: int = 3) -> list[str]:
    """The exception type/message lines at the end of a traceback."""
    lines = [line for line in output.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        if EXC_RE.match(lines[index]):
            return lines[index : index + limit]
    return lines[-limit:] if lines else []


def find_deprecations(output: str) -> list[str]:
    seen: list[str] = []
    for line in output.splitlines():
        if "DeprecationWarning" not in line:
            continue
        text = line.split("DeprecationWarning:", 1)[-1].strip() or line.strip()
        if text not in seen:
            seen.append(text)
    return seen


def run_page(page: Path, public_root: Path, repo_root: Path, timeout: int) -> PageReport:
    report = build_page_report(page, public_root)
    if not report.runnable:
        report.status = "fragment-only" if report.blocks else "no-python-blocks"
        return report

    script = concatenate(page, public_root, report)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["JAX_PLATFORMS"] = "cpu"
    env["MPLBACKEND"] = "Agg"

    with tempfile.TemporaryDirectory(prefix="rfx-doc-blocks-") as workdir:
        script_path = Path(workdir) / "page_blocks.py"
        script_path.write_text(script, encoding="utf-8")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                ["python3", str(script_path)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = completed.stdout + completed.stderr
            report.returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", "replace")
            report.returncode = 124
            report.exception_head = [f"TimeoutExpired: exceeded --timeout {timeout}s"]
        report.wall_seconds = round(time.monotonic() - started, 1)

    report.deprecations = find_deprecations(output)
    if report.returncode == 0:
        report.status = "PASS"
        return report

    report.status = "FAIL"
    if not report.exception_head:
        report.exception_head = exception_head(output)
    frames = re.findall(rf'File "{re.escape(str(script_path))}", line (\d+)', output)
    if frames:
        report.first_failing_block, report.first_failing_doc_line = block_for_line(
            script, int(frames[-1])
        )
    return report


def make_report(repo_root: Path, args: argparse.Namespace) -> RunReport:
    public_root = repo_root / "docs" / "public"
    pages: list[PageReport] = []
    for page in iter_doc_pages(public_root):
        rel = page.relative_to(public_root).as_posix()
        if args.only and not selects(rel, args.only):
            continue
        report = run_page(page, public_root, repo_root, args.timeout)
        if report.status == "no-python-blocks":
            continue
        pages.append(report)
    return RunReport(repo_root=str(repo_root), timeout=args.timeout, pages=pages)


def format_text(report: RunReport, *, show_deprecations: bool) -> str:
    header = (
        f"{'page':40s} {'#blk':>4s} {'status':13s} {'fail@blk (doc line)':>20s}  exception"
    )
    lines = [f"repo_root: {report.repo_root}", f"timeout:   {report.timeout}s", "", header, "-" * 110]
    for page in report.pages:
        if page.first_failing_block is not None:
            where = f"{page.first_failing_block} (line {page.first_failing_doc_line})"
        else:
            where = "-"
        head = page.exception_head[0] if page.exception_head else "-"
        lines.append(f"{page.page:40s} {page.n_blocks:4d} {page.status:13s} {where:>20s}  {head[:60]}")
        for extra in page.exception_head[1:]:
            lines.append(f"{'':61s}  {extra[:60]}")
        for fragment in page.fragments:
            lines.append(
                f"{'':46s}fragment: blk{fragment.number} "
                f"(doc line {fragment.doc_line}) - {fragment.fragment_reason}"
            )
        if show_deprecations:
            for text in page.deprecations:
                lines.append(f"{'':46s}DeprecationWarning: {text[:80]}")
    lines.extend(
        [
            "-" * 110,
            f"pages executed: {len(report.executed)}   PASS: {report.n_pass}   "
            f"FAIL: {report.n_fail}   fragment-only: {report.n_fragment_only}",
            f"total blocks: {sum(p.n_blocks for p in report.pages)}   "
            f"fragments: {sum(len(p.fragments) for p in report.pages)}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    public_root = repo_root / "docs" / "public"
    if not public_root.is_dir():
        raise SystemExit(f"docs/public not found under {repo_root}")

    started = time.monotonic()
    report = make_report(repo_root, args)
    elapsed = round(time.monotonic() - started, 1)

    if args.format == "json":
        payload = asdict(report)
        payload["has_failures"] = report.has_failures
        payload["wall_seconds"] = elapsed
        print(json.dumps(payload, indent=2))
    else:
        print(format_text(report, show_deprecations=args.grep_deprecations))
        print(f"wall time: {elapsed}s")
    return 1 if args.strict and report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
