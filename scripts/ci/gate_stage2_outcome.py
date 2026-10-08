"""Fail-closed selected-tests classification using collection and JUnit evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def crash_nodeid(problem: ET.Element) -> str | None:
    """xdist 3.8 / pytest 9.1 encode all crash phases as setup errors in XML."""
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


def classify(status: int, report: Path, collection: Path, must_run: list[str] | None = None) -> dict:
    result = dict(outcome="FAIL", files=[], nodes=[], must_run=[], never_executed=[],
                  collected=0, reported=0, parallel=False, collection_skips=0, skipped_collectors=[],
                  diagnostics=[])
    required = set(must_run or [])
    result["never_executed"] = sorted(required)
    if status == 5:
        result["outcome"] = "FAIL" if required else "NOTHING"
        return result
    try:
        paths = sorted(collection.glob("*.json"))
        records = [json.loads(p.read_text()) for p in paths]
        if not records or not records[0]:
            return result
        if any(not isinstance(r, list) or r != records[0] for r in records):
            return result
        nodes = records[0]
        if any(not isinstance(n, str) or "::" not in n or "\n" in n or "\r" in n for n in nodes):
            return result
        keys = {junit_key(n): n for n in nodes}
        if len(keys) != len(nodes):
            result["diagnostics"].append(f"ambiguous collected nodeids: {nodes!r}")
            return result  # Ambiguous JUnit names cannot prove execution coverage.
        result["collected"] = len(nodes)
        skip_records = [json.loads(p.with_suffix(".skips").read_text())
                        if p.with_suffix(".skips").exists() else [] for p in paths]
        if any(not isinstance(r, list) or r != skip_records[0] for r in skip_records):
            result["diagnostics"].append("inconsistent collection skip records")
            return result
        skips = {}
        skip_names = {}
        for record in skip_records[0]:
            if (not isinstance(record, list) or len(record) != 2
                    or any(not isinstance(value, str) for value in record)):
                result["diagnostics"].append(f"invalid collection skip: {record!r}")
                return result
            key = junit_key(record[0])
            if key in keys or key in skips:
                result["diagnostics"].append(f"ambiguous collection skip: {record[0]}")
                return result
            skips[key] = record[1]
            skip_names[key] = record[0]
        root = ET.parse(report).getroot()
        suites = list(root.iter("testsuite"))
        if not suites:
            return result
        problems = list(root.iter("failure")) + list(root.iter("error"))
        declared = sum(int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites)
        if declared != len(problems):
            return result
        seen = set()
        seen_skips = set()
        successful = set()
        problem_nodes = {}
        for case in root.iter("testcase"):
            key = (case.get("classname"), case.get("name"))
            node = keys.get(key)
            if node is None:
                skipped = case.find("skipped")
                if (key in skips and len(case) == 1 and skipped is not None
                        and skipped.get("message") == "collection skipped"
                        and skipped.text == skips[key]):
                    seen_skips.add(key)
                    result["collection_skips"] = len(seen_skips)
                    result["skipped_collectors"] = sorted(skip_names[k] for k in seen_skips)
                    continue
                result["diagnostics"].append(f"unaccounted testcase: classname={key[0]!r}, name={key[1]!r}")
                continue
            seen.add(node)
            for problem in list(case.iter("error")) + list(case.iter("failure")):
                problem_nodes[problem] = node
                crash = crash_nodeid(problem)
                if crash is not None and crash != node:
                    result["diagnostics"].append(f"crash nodeid {crash!r} disagrees with testcase {node!r}")
            if not list(case.iter("error")) and not list(case.iter("failure")):
                successful.add(node)
        result["reported"] = len(seen)
        result["never_executed"] = sorted(required - successful)
        for key in skips.keys() - seen_skips:
            result["diagnostics"].append(f"unreported collection skip: {key!r}")
        if result["diagnostics"]:
            return result
        if status == 0:
            if not problems and successful == set(nodes) and not result["never_executed"]:
                result["outcome"] = "PASS"
            for node in sorted(set(nodes) - seen):
                result["diagnostics"].append(f"unreported collected nodeid: {node}")
            return result
        if status != 1:
            return result
        if not problems:
            return result
        crashed = []
        for problem in problems:
            node = crash_nodeid(problem)
            if node not in nodes or problem_nodes.get(problem) != node:
                result["diagnostics"].append(
                    f"unrecognised problem: {problem.tag} {problem.get('message')!r}; "
                    f"testcase={problem_nodes.get(problem)!r}, crash nodeid={node!r}")
                return result
            if node not in crashed:
                crashed.append(node)
        crash_files = {n.split("::", 1)[0] for n in crashed}
        missing_files = {n.split("::", 1)[0] for n in nodes if n not in seen}
        result.update(outcome="RETRY", nodes=crashed,
                      must_run=sorted(set(crashed) | (set(nodes) - seen)),
                      files=sorted(crash_files | missing_files),
                      parallel=bool(missing_files - crash_files))
    except (OSError, ET.ParseError, ValueError, TypeError) as error:
        result["diagnostics"].append(f"invalid evidence: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=int)
    parser.add_argument("report", type=Path)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--must-run", type=Path, help="First execution outcome JSON")
    args = parser.parse_args()
    must_run = json.loads(args.must_run.read_text())["must_run"] if args.must_run else None
    result = classify(args.status, args.report, args.collection, must_run)
    args.output.with_suffix(".json").write_text(json.dumps(result))
    args.output.with_suffix(".files").write_text("".join(f"{p}\n" for p in result["files"]))
    args.output.with_suffix(".mode").write_text("parallel" if result["parallel"] else "serial")
    print(f'collected {result["collected"]}; results {result["reported"]}; {result["outcome"]}')
    for diagnostic in result["diagnostics"]:
        print(diagnostic)
    for node in result["skipped_collectors"]:
        # A selected module skipped wholesale at collection ran nothing: say so on every outcome.
        print(f"collection skip accepted (no test of it ran): {node}")
    for node in result["nodes"]:
        print(f"worker crash: {node}")
    for node in result["never_executed"]:
        print(f"never executed successfully in retry: {node}")
    return {"PASS": 0, "NOTHING": 5, "FAIL": 1, "RETRY": 10}[result["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
