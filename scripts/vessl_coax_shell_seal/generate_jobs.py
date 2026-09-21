#!/usr/bin/env python3
"""Job specifications for the coax shell-seal diagnostic — one per arm.

Six jobs, submitted together: the three rung-4 arms that vary the shell against
the as-shipped control, and the declared-radius arm at all three rungs so the
question "do the rungs then realize one line" has an answer.

Regenerate with::

    python scripts/vessl_coax_shell_seal/generate_jobs.py \
        --sha <pushed HEAD> --src <worktree path> --out <directory>
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vessl_coax_chain_battery"))
from generate_jobs import CLUSTER, IMAGE, RUNS, TEMPLATE  # noqa: E402

DRIVER = "scripts/diagnostics/coax_shell_seal_diagnostic.py"

# (arm, rung, dut). Arm 0 is the control and must reproduce the battery's rung-4
# thru. The two one-port loads exist because the two-port result carries no Z0.
CASES = [(0, 4, "thru"), (1, 4, "thru"), (2, 4, "thru"),
         (3, 4, "thru"), (3, 6, "thru"), (3, 9, "thru")]
EDGE_CASES = [(4, 4, "thru"), (4, 6, "thru"), (4, 9, "thru"), (5, 4, "thru"),
              (4, 6, "r25"), (4, 6, "r100")]

DESCRIPTIONS = {
    0: "as shipped (control)",
    4: "arm 3's geometry, conductors realized as PEC edge masks",
    5: "the as-shipped geometry, conductors realized as PEC edge masks",
    1: "shell 3 cells thick, inner radius as shipped, extending outward",
    2: "every cell outside the shell's inner radius is conductor",
    3: "shell inner radius at the declared outer radius b, extending outward 3 cells",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", choices=("sigma", "edge"), default="sigma",
                    help="sigma: arms 0-3; edge: arms 4-5, the PEC-edge realization")
    args = ap.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, rung, dut in (CASES if args.cases == "sigma" else EDGE_CASES):
        # The DUT goes BEFORE the rung: "arm4-r6-r25" would extend "arm4-r6",
        # and vessl_submit.sh finds a run directory by `-name "<prefix>*"`.
        key = (f"shell-arm{arm}-r{rung}" if dut == "thru"
               else f"shell-arm{arm}-{dut}-r{rung}")
        prefix = f"coax-shell-{key}"
        text = TEMPLATE.format(
            name=f"rfx-{prefix}",
            description=(f"Coax shell-seal diagnostic arm {arm} at {rung} annulus "
                         f"cells, {dut}: {DESCRIPTIONS[arm]}."),
            tag=key, cluster=CLUSTER, preset="gpu-rtx4090", image=IMAGE,
            sha=args.sha, src=args.src, runs=RUNS, prefix=prefix, driver=DRIVER,
            cli=f"--arm {arm} --rung {rung} --dut {dut}")
        path = dest / f"{key}.yaml"
        path.write_text(text)
        written.append((path, prefix))

    prefixes = [p for _, p in written]
    bad = [(x, y) for x, y in itertools.permutations(prefixes, 2) if y.startswith(x)]
    if bad:
        raise SystemExit(f"run prefixes collide: {bad}")

    for path, prefix in written:
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    print(f"\n{len(written)} job specifications in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
