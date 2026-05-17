#!/usr/bin/env python3
"""GATE-B2 mechanical check for the subgrid runner decomposition.

This script enforces the structural gates agreed for Branch B2 (Phase-3b
decomposition).  It is a pure static analysis — it does not import or run
the FDTD code — so it is safe to run in CI without JAX.

Gates
-----
G1  No function in ``rfx/subgridding/jit_runner.py`` exceeds 600 source lines.
G2  No function in ``rfx/subgridding/jit_runner.py`` exceeds 15 parameters.
G3  ``rfx/subgridding/jit_runner.py`` contains 0 occurrences of the dead
    search-loop knob prefixes ``material_sat_e_h_trace_`` / ``box_shadow_sync_``.
G4  ``rfx/subgridding/disjoint_3d.py`` defines at most 1 ``step_disjoint_z_slab*``
    stepper.

Each gate prints PASS/FAIL with the offending detail.  The script exits with
status 0 only when every gate passes; otherwise it exits non-zero.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JIT_RUNNER = REPO_ROOT / "rfx" / "subgridding" / "jit_runner.py"
DISJOINT_3D = REPO_ROOT / "rfx" / "subgridding" / "disjoint_3d.py"

MAX_FUNCTION_LINES = 600
MAX_FUNCTION_PARAMS = 15
DEAD_KNOB_RE = re.compile(r"material_sat_e_h_trace_|box_shadow_sync_")
Z_SLAB_DEF_RE = re.compile(r"^def step_disjoint_z_slab", re.MULTILINE)


def _iter_functions(tree: ast.AST):
    """Yield every function/async-function node, including nested ones."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _param_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Total declared parameters: pos-only, normal, kw-only, *args, **kwargs."""
    args = node.args
    count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg is not None:
        count += 1
    if args.kwarg is not None:
        count += 1
    return count


def check_g1(source: str) -> tuple[bool, list[str]]:
    """G1: no function in jit_runner.py exceeds MAX_FUNCTION_LINES lines."""
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in _iter_functions(tree):
        nlines = node.end_lineno - node.lineno + 1
        if nlines > MAX_FUNCTION_LINES:
            offenders.append(
                f"{node.name} = {nlines} lines "
                f"(@{node.lineno}-{node.end_lineno}, limit {MAX_FUNCTION_LINES})"
            )
    return (not offenders), offenders


def check_g2(source: str) -> tuple[bool, list[str]]:
    """G2: no function in jit_runner.py exceeds MAX_FUNCTION_PARAMS params."""
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in _iter_functions(tree):
        nparams = _param_count(node)
        if nparams > MAX_FUNCTION_PARAMS:
            offenders.append(
                f"{node.name} = {nparams} params "
                f"(@{node.lineno}, limit {MAX_FUNCTION_PARAMS})"
            )
    return (not offenders), offenders


def check_g3(source: str) -> tuple[bool, list[str]]:
    """G3: 0 dead-knob prefix occurrences in jit_runner.py."""
    matches = DEAD_KNOB_RE.findall(source)
    if matches:
        lines = [
            f"line {i}: {line.strip()}"
            for i, line in enumerate(source.splitlines(), start=1)
            if DEAD_KNOB_RE.search(line)
        ]
        return False, lines
    return True, []


def check_g4(source: str) -> tuple[bool, list[str]]:
    """G4: at most 1 step_disjoint_z_slab* def in disjoint_3d.py."""
    defs = [
        line.split("(")[0].replace("def ", "").strip()
        for line in source.splitlines()
        if line.startswith("def step_disjoint_z_slab")
    ]
    return (len(defs) <= 1), [f"{len(defs)} z_slab steppers: {defs}"]


def main() -> int:
    if not JIT_RUNNER.is_file():
        print(f"FAIL  cannot find {JIT_RUNNER}", file=sys.stderr)
        return 2
    if not DISJOINT_3D.is_file():
        print(f"FAIL  cannot find {DISJOINT_3D}", file=sys.stderr)
        return 2

    jit_src = JIT_RUNNER.read_text()
    disjoint_src = DISJOINT_3D.read_text()

    gates = [
        ("G1", "no jit_runner function > 600 source lines", check_g1(jit_src)),
        ("G2", "no jit_runner function > 15 parameters", check_g2(jit_src)),
        ("G3", "0 dead-knob occurrences in jit_runner", check_g3(jit_src)),
        ("G4", "<= 1 z_slab stepper in disjoint_3d", check_g4(disjoint_src)),
    ]

    all_ok = True
    print("GATE-B2 mechanical check")
    print("=" * 60)
    for name, desc, (ok, details) in gates:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {desc}")
        if not ok:
            all_ok = False
            for detail in details:
                print(f"         - {detail}")
    print("=" * 60)
    print("RESULT:", "ALL GATES PASS" if all_ok else "GATE FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
