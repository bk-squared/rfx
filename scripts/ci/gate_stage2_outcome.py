"""Fail-closed selected-tests classification using collection and JUnit evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def crash_nodeid(problem: ET.Element) -> str | None:
    """xdist crash reports are pytest setup errors, not arbitrary failure text."""
    text = problem.text or ""
    match = re.fullmatch(r"worker 'gw[0-9]+' crashed while running '(.+)'", text)
    if (problem.tag == "error" and match
            and problem.get("message") == f'failed on setup with "{text}"'):
        return match[1]
    return None


def junit_key(node: str) -> tuple[str, str]:
    # Match pytest's JUnit address encoding, preserving dots and :: in parameters.
    path, bracket, params = node.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def classify(status: int, report: Path, collection: Path) -> dict:
    result = dict(outcome="FAIL", files=[], nodes=[], collected=0, reported=0, parallel=False)
    if status == 5:
        result["outcome"] = "NOTHING"
        return result
    try:
        records = [json.loads(p.read_text()) for p in sorted(collection.glob("*.json"))]
        if not records or not records[0]:
            return result
        if any(not isinstance(r, list) or r != records[0] for r in records):
            return result
        nodes = records[0]
        if any(not isinstance(n, str) or "::" not in n or "\n" in n or "\r" in n for n in nodes):
            return result
        keys = {junit_key(n): n for n in nodes}
        if len(keys) != len(nodes):
            return result  # Ambiguous JUnit names cannot prove execution coverage.
        result["collected"] = len(nodes)
        root = ET.parse(report).getroot()
        suites = list(root.iter("testsuite"))
        if not suites:
            return result
        problems = list(root.iter("failure")) + list(root.iter("error"))
        declared = sum(int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites)
        if declared != len(problems):
            return result
        seen = set()
        successful = set()
        for case in root.iter("testcase"):
            node = keys.get((case.get("classname"), case.get("name")))
            if node is None:
                return result
            seen.add(node)
            if not list(case.iter("error")) and not list(case.iter("failure")):
                successful.add(node)
        result["reported"] = len(seen)
        if status == 0:
            if not problems and successful == set(nodes):
                result["outcome"] = "PASS"
            return result
        if status != 1:
            return result
        if not problems:
            return result
        crashed = []
        for problem in problems:
            node = crash_nodeid(problem)
            if node not in nodes:
                return result
            if node not in crashed:
                crashed.append(node)
        crash_files = {n.split("::", 1)[0] for n in crashed}
        missing_files = {n.split("::", 1)[0] for n in nodes if n not in seen}
        result.update(outcome="RETRY", nodes=crashed,
                      files=sorted(crash_files | missing_files),
                      parallel=bool(missing_files - crash_files))
    except (OSError, ET.ParseError, ValueError, TypeError):
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=int)
    parser.add_argument("report", type=Path)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = classify(args.status, args.report, args.collection)
    args.output.with_suffix(".json").write_text(json.dumps(result))
    args.output.with_suffix(".files").write_text("".join(f"{p}\n" for p in result["files"]))
    args.output.with_suffix(".mode").write_text("parallel" if result["parallel"] else "serial")
    print(f'collected {result["collected"]}; results {result["reported"]}; {result["outcome"]}')
    for node in result["nodes"]:
        print(f"worker crash: {node}")
    return {"PASS": 0, "NOTHING": 5, "FAIL": 1, "RETRY": 10}[result["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
