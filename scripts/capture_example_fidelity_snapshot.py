#!/usr/bin/env python3
"""Regenerate ``tests/data/example_fidelity_snapshot.json`` (#737 P4).

EMISSION-DRIFT SNAPSHOT, not a zero-advisory bar and not a physics check:
this pins the TEXT every audited example EMITS at build time from
``preflight()``/``fidelity_report()``, so drift fails CI. Nothing here
time-steps, so a green gate says nothing about whether an example's OUTPUT
numbers are right (that is tests/contracts/test_tutorial_examples.py for
six tutorials, tests/unit/api/test_diagnostics.py for hello_world, and the
weekly crossval-external job for the eight scheduled crossval cases), and
it is not an endorsement that today's advisories are correct or complete.

Regenerate after a DELIBERATE change to a committed example's declared
geometry, materials, or preflight-relevant config -- never to silence a
drift the gate correctly caught. If ``test_example_fidelity_contract.py``
fails and the diff is NOT an intentional change, that is the gate working:
fix the script (or investigate why realized != declared), do not re-pin.

No solves: every number below comes from ``sim.preflight()`` and
``sim.fidelity_report()``, neither of which time-steps. The snapshot today
holds 51 (script, builder, variant) triples across the 34 auditable
scripts of the 137 discovered under examples/ + validation/ (measured
2026-09-16 at 6d721a56; the 2026-08-28 capture was 33 triples over 23
scripts; since then cv07, cv15, cv24 and eight validation/research scripts
joined the audited set -- 18 new variants, none removed).
CPU-only; no GPU, and JAX warmup dominates.

Every optional dependency in ``_example_fidelity_lib.OPTIONAL_DEPENDENCIES``
(today: optax) must be installed to regenerate. Without them this script
STOPS on the import rather than writing a snapshot with those variants
missing -- a partial snapshot would fail the gate on every machine that does
have them.

The #729 site-1 defect (domain extents were NODE-count sums, one cell too
long) was FIXED by PR #734 (merged a5a72280) and this snapshot was
re-captured after it: cv11's WR-90 guide now reads 23000/11000 um, which
is what #722 measures. Domain extents in this file are usable again; the
snapshot's own ``_comment`` records that re-capture and the later
2026-09-02 one (#833 item 2).

Run from the repo root::

    JAX_ENABLE_X64=0 python scripts/capture_example_fidelity_snapshot.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _example_fidelity_lib as lib  # noqa: E402


def main() -> int:
    t0 = time.time()
    snapshot: dict[str, dict] = {}
    n = 0
    for relpath, fn, label, sim in lib.iter_audited_variants():
        key = f"{relpath}::{fn}::{label}"
        snapshot[key] = lib.digest_variant(sim)
        n += 1
        print(f"  captured [{n:2d}] {key}")
    wall = time.time() - t0

    out = dict(
        _comment=(
            "Regenerate with: JAX_ENABLE_X64=0 python "
            "scripts/capture_example_fidelity_snapshot.py -- SNAPSHOT of "
            "today's advisories, not a zero-advisory endorsement (see this "
            "script's module docstring and test_example_fidelity_contract.py). "
            "Nothing here time-steps: this pins build-time EMISSION "
            "only, so it is not evidence about any example's output "
            "numbers. The #729-site-1 defect (domain "
            "realized_extent_um/n_cells read NODE counts, one cell too "
            "long) was FIXED by PR #734 (a5a72280) and this snapshot was "
            "re-captured after that fix, so domain extents here are "
            "usable again (cv11 reads 23000/11000 um, matching #722). "
            "Re-captured 2026-09-02 "
            "after the report started reading the rasterizer's exact "
            "float64 node line (#833 item 2): realized_um / cell_um / "
            "face_residual_um on non-uniform lanes moved by the float32 "
            "cell-size widening (<= 0.011 um), 108 on-node residuals "
            "went to 0.0, 12 one-cell bodies stopped reading sub_cell, "
            "and 6 'inside-absorber' rows on domain-filling bodies "
            "(convergence_floor fixture; hi face read 27000.0002 um "
            "against a 27000.0 um domain) went away; no n_cells or "
            "preflight row moved."
        ),
        variants=snapshot,
    )
    lib.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lib.SNAPSHOT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {lib.SNAPSHOT_PATH.relative_to(REPO_ROOT)}: "
          f"{n} variants in {wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
