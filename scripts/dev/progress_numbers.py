"""Print the rfx 2.0 progress numbers of one commit, read from that commit only.

The layer numbers come from the layer checker committed at the commit
(``scripts/ci/check_layer_structure.py``), run on the commit's own sources; the
run()/forward() numbers are the literal exception tables of the contract
committed there (``tests/contracts/test_run_forward_one_pipeline.py``), each
entry a known difference or refusal still to remove before 2.0. Nothing is read
from the working tree, and nothing under ``rfx/`` is imported.

    python scripts/dev/progress_numbers.py <sha> [--repo PATH]
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[2]
CHECKER = "scripts/ci/check_layer_structure.py"
CONTRACT = "tests/contracts/test_run_forward_one_pipeline.py"
TABLES = ("KNOWN_ROWS", "TWO_GUARD_REFUSALS", "KNOWN_DIRECTIONS")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def committed(repo: Path, sha: str, path: str) -> str | None:
    if not git(repo, "ls-tree", sha, "--", path):
        return None
    return git(repo, "show", f"{sha}:{path}")


def checker_at(repo: Path, sha: str) -> types.ModuleType:
    """The layer checker as committed at ``sha``; its REPO points at ``repo``."""
    source = committed(repo, sha, CHECKER)
    if source is None:
        raise SystemExit(f"{sha} has no {CHECKER}")
    module = types.ModuleType("layer_checker_at_commit")
    module.__file__ = str(repo / CHECKER)
    sys.modules[module.__name__] = module
    exec(compile(source, CHECKER, "exec"), module.__dict__)
    return module


def layer_numbers(repo: Path, sha: str) -> dict:
    gate = checker_at(repo, sha)
    result = gate.count(gate.Sources(repo, sha))
    # An older checker counts fewer rules; a count it does not make is None ("not counted").
    def size(rules, *names):
        return None if any(r not in rules for r in names) else sum(len(rules[r]) for r in names)

    files = {}
    for path in gate.FIVE_FILES:
        rules = result["baseline"]["files"][path]
        files[path] = {
            "units": len(set().union(*(rules.get(r, ()) for r in "SMAG"))),
            "of": result.get("unit_counts", {}).get(path),
            "imports": size(rules, "I"),
            "declarations": size(rules, "D1", "D2"),
        }
    outside = result["baseline"]["outside"]
    reverse = result["baseline"].get("reverse")
    defined = result.get("caller_definitions")
    five = (None if defined is None else
            sum(e.partition(":")[0] in gate.FIVE_MODULES for entries in defined.values() for e in entries))
    return {
        "files": files,
        "outside": {path: {r: len(rules.get(r, ())) for r in ("S", "M")} for path, rules in outside.items()},
        "reverse": None if reverse is None else sum(len(rules["R"]) for rules in reverse.values()),
        "defined_five": five,
        "defined_other": None if defined is None else sum(len(e) for e in defined.values()) - five,
    }


def contract_numbers(repo: Path, sha: str) -> dict:
    """Entry count of each literal exception table; None when the commit has no such table."""
    source = committed(repo, sha, CONTRACT)
    found = dict.fromkeys(TABLES)
    if source is None:
        return found
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in TABLES):
            value = node.value
            if not isinstance(value, (ast.Tuple, ast.List, ast.Dict)):
                raise SystemExit(f"{CONTRACT}: {node.targets[0].id} is not a literal table")
            found[node.targets[0].id] = len(value.keys if isinstance(value, ast.Dict) else value.elts)
    return found


def shown(value) -> str:
    return "not counted at this commit" if value is None else str(value)


def report(sha: str, header: str, layers: dict, contract: dict) -> None:
    print(f"commit {sha} {header}")
    print("five files (2.0 condition: every number 0)")
    totals = []
    for path, n in layers["files"].items():
        total = n["units"] + (n["imports"] or 0) + (n["declarations"] or 0)
        totals.append(total)
        print(f"  {path}: {total} (layer units {n['units']} of {shown(n['of'])}, imports {shown(n['imports'])}, "
              f"declarations {shown(n['declarations'])})")
    print(f"  five files total: {sum(totals)}")
    print(f"  layer imports of a five-file definition: {shown(layers['defined_five'])}")
    s = sum(v["S"] for v in layers["outside"].values())
    m = sum(v["M"] for v in layers["outside"].values())
    print(f"outside the layer packages: time step S={s}, measurement M={m}")
    for path, v in sorted(layers["outside"].items()):
        print(f"  {path}: S={v['S']} M={v['M']}")
    print(f"reverse imports from layers to callers: R={shown(layers['reverse'])} "
          f"(of them names another caller defines: {shown(layers['defined_other'])})")
    print(f"run()/forward() contract literal exceptions ({CONTRACT}):")
    for table, count in contract.items():
        print(f"  {table}: {'absent' if count is None else count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sha")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    sha = git(args.repo, "rev-parse", "--verify", f"{args.sha}^{{commit}}").strip()
    header = git(args.repo, "show", "-s", "--format=%cI %s", sha).strip()
    report(sha, header, layer_numbers(args.repo, sha), contract_numbers(args.repo, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
