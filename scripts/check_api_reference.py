#!/usr/bin/env python
"""API-reference regeneration gate (roadmap W4.7).

The pdoc-rendered reference (``docs/api``, gitignored snapshot) drifts
silently when the public API changes. This script pins the public surface in
a committed inventory and fails CI when the package and the inventory
disagree, so an API change cannot ship without regenerating the reference.

Inventory = top-level ``rfx`` exports (kind + parameter names for callables)
plus public ``Simulation`` methods (parameter names). Parameter NAMES only —
default-value reprs vary across environments and would make the gate flaky.

Usage:
    python scripts/check_api_reference.py            # verify (CI mode)
    python scripts/check_api_reference.py --write    # regenerate inventory
    python scripts/check_api_reference.py --html-dir docs/api
        # additionally assert the rendered HTML contains the required symbols

Exit codes: 0 ok, 1 drift/gate failure.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INVENTORY_PATH = REPO_ROOT / "docs/guides/api_symbol_inventory.json"

# W4.7 acceptance gate: the regenerated reference must contain these.
REQUIRED_SIMULATION_METHODS = ("add_msl_port", "forward", "compute_waveguide_s_matrix")
REQUIRED_FORWARD_PARAMS = ("distributed", "port_s11_freqs")
REQUIRED_HTML_SYMBOLS = ("add_msl_port", "port_s11_freqs", "distributed")


def _params(obj) -> list[str] | None:
    try:
        return [p for p in inspect.signature(obj).parameters if p != "self"]
    except (TypeError, ValueError):
        return None


def build_inventory() -> dict:
    import rfx
    from rfx import Simulation

    exports: dict[str, dict] = {}
    for name in sorted(dir(rfx)):
        if name.startswith("_"):
            continue
        obj = getattr(rfx, name)
        if inspect.ismodule(obj):
            continue
        if inspect.isclass(obj):
            kind = "class"
        elif callable(obj):
            kind = "function"
        else:
            kind = "object"
        entry: dict = {"kind": kind}
        if kind == "function":
            params = _params(obj)
            if params is not None:
                entry["params"] = params
        exports[name] = entry

    methods = {
        name: _params(fn) or []
        for name, fn in inspect.getmembers(Simulation, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    return {
        "_comment": (
            "Public API surface pinned by scripts/check_api_reference.py. "
            "Regenerate with: python scripts/check_api_reference.py --write "
            "(and rebuild docs/api with pdoc: "
            "python -m pdoc -o docs/api rfx '!rfx.dashboard')."
        ),
        "rfx_exports": exports,
        "simulation_methods": dict(sorted(methods.items())),
    }


def check_required_gates(inv: dict) -> list[str]:
    errors = []
    methods = inv["simulation_methods"]
    for m in REQUIRED_SIMULATION_METHODS:
        if m not in methods:
            errors.append(f"required Simulation method missing from surface: {m}")
    fwd = methods.get("forward", [])
    for p in REQUIRED_FORWARD_PARAMS:
        if p not in fwd:
            errors.append(f"Simulation.forward lost required parameter: {p}")
    return errors


def check_html(html_dir: Path) -> list[str]:
    errors = []
    if not html_dir.is_dir():
        return [f"html dir not found: {html_dir}"]
    blob = "".join(
        p.read_text(errors="ignore") for p in sorted(html_dir.rglob("*.html"))
    )
    if not blob:
        return [f"no rendered .html under {html_dir}"]
    for sym in REQUIRED_HTML_SYMBOLS:
        if sym not in blob:
            errors.append(f"rendered reference is missing required symbol: {sym}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate the inventory")
    ap.add_argument("--html-dir", type=Path, default=None,
                    help="also gate a rendered pdoc tree (e.g. docs/api)")
    args = ap.parse_args()

    live = build_inventory()

    errors = check_required_gates(live)

    if args.write:
        if errors:
            print("\n".join(f"GATE FAIL: {e}" for e in errors))
            return 1
        INVENTORY_PATH.write_text(json.dumps(live, indent=2) + "\n")
        print(f"wrote {INVENTORY_PATH}")
        return 0

    if not INVENTORY_PATH.exists():
        print(f"DRIFT FAIL: {INVENTORY_PATH} missing — run with --write")
        return 1
    pinned = json.loads(INVENTORY_PATH.read_text())
    for key in ("rfx_exports", "simulation_methods"):
        if pinned.get(key) != live.get(key):
            pin_d, live_d = pinned.get(key, {}), live.get(key, {})
            added = sorted(set(live_d) - set(pin_d))
            removed = sorted(set(pin_d) - set(live_d))
            changed = sorted(
                k for k in set(pin_d) & set(live_d) if pin_d[k] != live_d[k]
            )
            errors.append(
                f"DRIFT in {key}: added={added} removed={removed} changed={changed} "
                "— the public API moved without regenerating the reference. Run "
                "'python scripts/check_api_reference.py --write' and rebuild "
                "docs/api with pdoc."
            )

    if args.html_dir is not None:
        errors.extend(check_html(args.html_dir))

    if errors:
        print("\n".join(f"FAIL: {e}" for e in errors))
        return 1
    print("api reference surface: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
