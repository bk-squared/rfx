#!/usr/bin/env python3
"""Report the committed vs regenerated WR-90 NU flux broad-E5 envelope (#574).

A regeneration that silently replaces a committed gate's artifact is
unreviewable: the diff is 16 float changes with no statement of what moved or
whether the gate still binds anything. This prints that statement.

It deliberately does NOT write a fixture. Promoting the new envelope is a
separate, reviewed step, because the gate must be RE-DERIVED from the new
measurement (tests/_gate_policy.gate_from_envelope) rather than inherited — and
a gate re-derived from an artifact that got worse would loosen itself, which is
the dependency-closure trap #576 put an absolute ceiling in front of.

    python scripts/diagnostics/i574_compare_e5_envelope.py --new A.json --committed B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _cases(env: dict) -> dict:
    return {c["tag"]: c for c in env.get("cases", [])}


def _summary_max(env: dict) -> float:
    # Read the key the GATE reads (`envelope_summary.max_mag_abs_diff_across_cases`)
    # first. An earlier revision looked only in a `summary` dict these envelopes
    # do not have, fell through to the max over cases, and happened to agree —
    # because the gate test asserts those are equal. Agreeing by luck is not the
    # same as reporting the gated quantity.
    es = env.get("envelope_summary") or {}
    if "max_mag_abs_diff_across_cases" in es:
        return float(es["max_mag_abs_diff_across_cases"])
    s = env.get("summary") or {}
    for key in ("max_mag_abs_diff", "max_abs_diff", "envelope_max"):
        if key in s:
            return float(s[key])
    cases = _cases(env)
    if not cases:
        raise SystemExit("no summary and no cases — is this an envelope file?")
    return max(float(c["max_mag_abs_diff"]) for c in cases.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--committed", required=True)
    args = ap.parse_args(argv)

    new = json.loads(Path(args.new).read_text())
    old = json.loads(Path(args.committed).read_text())
    cn, co = _cases(new), _cases(old)

    print(f"{'case':28} {'committed':>10} {'regenerated':>12} {'delta':>10} "
          f"{'factor':>8}")
    common = [t for t in co if t in cn]
    for tag in sorted(common):
        a = float(co[tag]["max_mag_abs_diff"])
        b = float(cn[tag]["max_mag_abs_diff"])
        fac = (a / b) if b > 0 else float("inf")
        print(f"{tag:28} {a:>10.6f} {b:>12.6f} {b - a:>+10.6f} {fac:>7.2f}x")
    for tag in sorted(set(co) - set(cn)):
        print(f"{tag:28} {'MISSING in the regeneration':>44}")
    for tag in sorted(set(cn) - set(co)):
        print(f"{tag:28} {'NEW case, absent from the committed set':>44}")

    mo, mn = _summary_max(old), _summary_max(new)
    if mn == mo:
        verdict = "IDENTICAL — is this actually a regenerated file?"
    elif mn < mo:
        verdict = f"{mo / mn:.2f}x better"
    else:
        verdict = f"{mn / mo:.2f}x WORSE"
    print(f"\nenvelope max: committed {mo:.6f} -> regenerated {mn:.6f}  ({verdict})")
    print(f"cases: committed {len(co)}, regenerated {len(cn)}, common {len(common)}")

    # What the gate WOULD become, reported rather than applied.
    try:
        from tests._gate_policy import gate_from_envelope
        for quantum, label in ((100, "1/100"), (1000, "1/1000")):
            print(f"  gate re-derived from the new envelope at quantum {label}: "
                  f"{gate_from_envelope(mn, quantum=quantum)}")
    except Exception as exc:                       # pragma: no cover
        print(f"  (gate policy unavailable here: {exc})")
    print("  committed tolerance in the fixture: "
          f"{old.get('max_mag_abs_tol', old.get('tolerance', 'n/a'))}")

    if mn > mo:
        print("\nREGRESSION: the regenerated envelope is WORSE than the committed "
              "one. Do NOT promote it and do NOT re-derive a looser gate from it "
              "— find out what changed first (#576 dependency-closure rule).")
        return 1
    print("\nNot promoting anything: this script only reports. Promotion is a "
          "reviewed step that re-derives the gate from the new envelope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
