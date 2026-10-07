"""Classify selected-tests JUnit output; exit 10 requests one serial retry."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def crash_nodeid(problem: ET.Element) -> str | None:
    """xdist 3.8 crash reports are pytest setup errors in JUnit XML."""
    text = problem.text or ""
    match = re.fullmatch(r"worker 'gw[0-9]+' crashed while running '(.+)'", text)
    if (problem.tag == "error" and match
            and problem.get("message") == f'failed on setup with "{text}"'):
        return match[1]
    return None


def classify(status: int, report: Path, selected: list[str]) -> tuple[str, list[str], list[str]]:
    if status in (0, 5):
        return "pass", [], []
    if status != 1:
        return "fail", [], []
    try:
        root = ET.parse(report).getroot()
        suites = list(root.iter("testsuite"))
        problems = list(root.iter("failure")) + list(root.iter("error"))
        declared = sum(int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites)
    except (OSError, ET.ParseError, ValueError):
        return "fail", [], []
    if not problems or declared != len(problems):
        return "fail", [], []
    nodes = []
    files = []
    for problem in problems:
        node = crash_nodeid(problem)
        if node is None:
            return "fail", [], []
        path, separator, _ = node.partition("::")
        if not separator or path not in selected or "\n" in node or "\r" in node:
            return "fail", [], []
        if node not in nodes:
            nodes.append(node)
        if path not in files:
            files.append(path)
    return "retry", files, nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=int)
    parser.add_argument("report", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("retry_files", type=Path)
    parser.add_argument("crashed_nodes", type=Path)
    args = parser.parse_args()
    outcome, files, nodes = classify(args.status, args.report, args.selected.read_text().splitlines())
    args.retry_files.write_text("".join(f"{path}\n" for path in files))
    args.crashed_nodes.write_text("".join(f"{node}\n" for node in nodes))
    return {"pass": 0, "fail": 1, "retry": 10}[outcome]


if __name__ == "__main__":
    sys.exit(main())
