#!/usr/bin/env python
"""Fail the two-solver crossval lane BEFORE pytest if either solver is absent.

Why this is a repo script and not inline YAML: `_configs/.claude/rules/vessl-jobs.md`
forbids heredocs in a VESSL `run:` block -- VESSL re-indents the block inside its own
wrapper, which moves a `<<'EOF'` terminator off column 0, and the job dies at parse
time before any work runs (verified 2026-07-31, amc run 369367250654). `sh -n` on the
extracted block does not catch it. A repo script is also unit-testable locally.

Why the probe exists at all. The 13 external legs in the four gpu-marked crossval
files are guarded by `pytest.importorskip`, so a lane WITHOUT the solvers reports 13
clean SKIPs and exits 0. That green-with-skips outcome is precisely the dishonest
coverage claim issue #717 was filed about, and it is the outcome every lane in this
repo has produced so far. Probing here converts it into a loud failure.

The same probe catches the other failure mode from
`docs/design_notes/717_crossval_lane_decision.md` Sec. 1: under pytest 9,
`importorskip` does NOT absorb a *broken* import. A meep built against the wrong numpy
raises ImportError, which propagates as a test FAILURE -- an environment defect wearing
the shape of a cross-solver physics disagreement.

Prints one machine-readable JSON line to `--out` (if given) and a human summary to
stdout. Exits nonzero on the first thing that is not true.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write the provenance block as JSON here")
    args = ap.parse_args()

    info: dict[str, object] = {"python": sys.version.split()[0]}
    failures: list[str] = []

    import numpy

    info["numpy"] = numpy.__version__
    if not numpy.__version__.startswith("1."):
        # Not a style preference. Every conda-forge pymeep build through 1.34.0
        # declares numpy >=1.11.3,<2.0a0, and the openEMS bindings in the parent
        # image were compiled against numpy<2. numpy 2 here means one of them is
        # already broken and pytest would report it as a physics failure.
        failures.append(
            f"numpy {numpy.__version__} is not the numpy-1 ABI that pymeep and the "
            "openEMS bindings are built against"
        )

    try:
        import meep

        info["meep"] = meep.__version__
    except Exception as exc:  # noqa: BLE001 -- the point is to report ANY import failure
        info["meep"] = None
        failures.append(f"meep did not import: {type(exc).__name__}: {exc}")

    try:
        from CSXCAD.CSXCAD import ContinuousStructure  # noqa: F401
        from openEMS.openEMS import openEMS  # noqa: F401

        info["openems"] = "import-ok"
    except Exception as exc:  # noqa: BLE001
        info["openems"] = None
        failures.append(f"openEMS/CSXCAD bindings did not import: {type(exc).__name__}: {exc}")

    try:
        import jax

        import rfx

        info["jax"] = jax.__version__
        info["jax_devices"] = [str(d) for d in jax.devices()]
        info["rfx"] = getattr(rfx, "__version__", "?")
        info["rfx_file"] = rfx.__file__
    except Exception as exc:  # noqa: BLE001
        failures.append(f"rfx/jax did not import: {type(exc).__name__}: {exc}")

    for key in ("RFX_OPENEMS_COMMIT", "RFX_PYMEEP_VERSION", "RFX_SOLVER_PARENT_IMAGE"):
        import os

        info[key.lower()] = os.environ.get(key)

    for key, value in info.items():
        print(f"{key}={value}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if failures:
        print("", file=sys.stderr)
        for line in failures:
            print(f"FATAL: {line}", file=sys.stderr)
        print(
            "\nThis lane exists to EXECUTE the 13 external legs. Without both solvers "
            "they importorskip away and the lane would be green and empty (#717).",
            file=sys.stderr,
        )
        return 1

    print("both solvers present; the 13 external legs will execute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
