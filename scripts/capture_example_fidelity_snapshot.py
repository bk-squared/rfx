#!/usr/bin/env python3
"""Regenerate ``tests/data/example_fidelity_snapshot.json`` (#737 P4).

SNAPSHOT, not a zero-advisory bar: this pins what every audited example's
``preflight()``/``fidelity_report()`` say TODAY, so drift fails CI. It is
NOT an endorsement that today's advisories are correct or complete. This
gate's job is drift detection, not correctness certification -- and one
pinned quantity is known wrong: every ``domain`` row's realized extent and
n_cells is one cell too large (``rfx/fidelity.py`` sums the NODE-count
slice; #729 site 1, fixed by the open PR #734, not here). When #734 lands
this snapshot will drift, and re-capturing it belongs to that PR.

Regenerate after a DELIBERATE change to a committed example's declared
geometry, materials, or preflight-relevant config -- never to silence a
drift the gate correctly caught. If ``test_example_fidelity_contract.py``
fails and the diff is NOT an intentional change, that is the gate working:
fix the script (or investigate why realized != declared), do not re-pin.

No solves: every number below comes from ``sim.preflight()`` and
``sim.fidelity_report()``, neither of which time-steps. Wall clock for all
33 (script, builder, variant) triples across the 23 auditable scripts:
``wrote tests/data/example_fidelity_snapshot.json: 33 variants in 52.2s``
(measured 2026-08-28; CPU-only, no GPU/JAX warmup dominates).

Every optional dependency in ``_example_fidelity_lib.OPTIONAL_DEPENDENCIES``
(today: optax) must be installed to regenerate. Without them this script
STOPS on the import rather than writing a snapshot with those variants
missing -- a partial snapshot would fail the gate on every machine that does
have them.

Domain extents in this file are NODE-count sums, i.e. one cell too long:
cv11's WR-90 guide is pinned at 24000/12000 um where #722 measures
23000/11000. See the note above -- the fix is PR #734's, and this file
follows it rather than forking it.

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
            "The #729-site-1 defect (domain realized_extent_um/n_cells "
            "read NODE counts, one cell too long) was FIXED by PR #734, "
            "and this snapshot was re-captured after that fix. Do not "
            "quote a domain extent from this file."
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
