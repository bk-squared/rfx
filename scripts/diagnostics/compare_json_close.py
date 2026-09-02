"""Compare two JSON artifacts field-by-field with a float tolerance.

Usage: ``python scripts/diagnostics/compare_json_close.py A.json B.json [--rel 1e-9]``

Exit 0 and print one line when every leaf agrees (floats to ``--rel``
relative, ``1e-12`` absolute; everything else exactly); exit 1 and list the
first differences otherwise.  Written for the VESSL cv06b falsifier job
(#812 P3): try-1 showed that a byte ``diff`` of the same artifact regenerated
on two hosts fails on the last float digit (-32.389876972878234 vs ...23),
which is not staleness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _walk(x, y, path, rel, bad):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in x and k in y:
                _walk(x[k], y[k], f"{path}.{k}", rel, bad)
            else:
                bad.append(f"{path}.{k}: present on one side only")
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(f"{path}: length {len(x)} vs {len(y)}")
            return
        for i, (u, v) in enumerate(zip(x, y)):
            _walk(u, v, f"{path}[{i}]", rel, bad)
    elif isinstance(x, float) or isinstance(y, float):
        if isinstance(x, bool) or isinstance(y, bool) or not math.isclose(x, y, rel_tol=rel, abs_tol=1e-12):
            bad.append(f"{path}: {x!r} vs {y!r}")
    elif x != y:
        bad.append(f"{path}: {x!r} vs {y!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rel", type=float, default=1e-9)
    args = ap.parse_args(argv)
    with open(args.a) as fa, open(args.b) as fb:
        a, b = json.load(fa), json.load(fb)
    bad: list[str] = []
    _walk(a, b, "", args.rel, bad)
    if bad:
        print(f"DIFFER ({len(bad)} leaves beyond rel {args.rel:g}): " + "; ".join(bad[:8]))
        return 1
    print(f"artifacts agree to rel {args.rel:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
