#!/usr/bin/env python3
"""Write one VESSL job specification per (rung, board) of the coax open-end
check sweep (``scripts/diagnostics/coax_open_check_config.py``).

Same job body as the closed-can campaign (``generate_closed_can_jobs.py``:
pinned commit, clean-tree guard, node-local copy, EXIT-trap collection, JAX
0.6.2 installed before the first import, a GPU-backend guard), on a VRAM-band
preset. The sweep runs twice: at a commit whose ``rfx/`` carries the shipped
lane, and at the fix's commit; ``--label`` keeps the two sets' run
directories apart.

    python scripts/vessl_coax_chain_battery/generate_open_check_jobs.py \\
        --sha <commit> --src <run tree> --out <directory> --label old|new \\
        [--preset gpu-8gb]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_jobs import CLUSTER, IMAGE, RUNS  # noqa: E402
from generate_closed_can_jobs import COMMAND, TEMPLATE, TIMEOUT_S  # noqa: E402

DRIVER = "scripts/diagnostics/coax_open_check_config.py"
# (rung, board length in mm). The lane's twelve probe planes need about 66
# cells of line, so 25 mm is the shortest board that holds them at 4 annulus
# cells (0.355 mm cells); 40 mm is the battery's own one-port board.
SWEEP = ((4, 40.0), (4, 25.0), (6, 40.0), (6, 25.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, choices=("old", "new"))
    ap.add_argument("--preset", default="gpu-8gb", choices=("gpu-8gb", "gpu-24gb"))
    args = ap.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for rung, z_mm in SWEEP:
        key = f"open-check-{args.label}-r{rung}-z{z_mm:g}"
        prefix = f"coax-{key}"
        text = TEMPLATE.format(
            name=f"rfx-{prefix}",
            description=(f"Coax open-end check sweep ({args.label} lane): {rung} annulus "
                         f"cells on an 8 x 8 x {z_mm:g} mm board, 12 and 24 record units."),
            tag=key, cluster=CLUSTER, preset=args.preset, image=IMAGE, sha=args.sha,
            src=args.src, runs=RUNS, prefix=prefix,
            commands=COMMAND.format(timeout=TIMEOUT_S, driver=DRIVER,
                                    cli=f"--rung {rung} --z-mm {z_mm:g}", prefix=prefix))
        path = dest / f"{key}.yaml"
        path.write_text(text)
        print(f"sh scripts/vessl_submit.sh {path} {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
