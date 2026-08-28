#!/usr/bin/env python3
"""Regenerate ``tests/data/example_fidelity_snapshot.json`` (#737 P4).

SNAPSHOT, not a zero-advisory bar: this pins what every audited example's
``preflight()``/``fidelity_report()`` say TODAY, so drift fails CI. It is
NOT an endorsement that today's advisories are correct or complete -- #742
is open (~45% of advisory codes never fire in this corpus; the
``absorber_budget_exceeds_axis`` check false-fires on PEC-closed axes, e.g.
``validation/tmtt_paper/waveguide_dielectric_taper.py``, which declares
z=Boundary(lo='pec', hi='pec') and still gets an absorber-budget warning).
Tighten the bar to zero-advisory only when #742 lands; until then this
gate's job is drift detection, not correctness certification.

Regenerate after a DELIBERATE change to a committed example's declared
geometry, materials, or preflight-relevant config -- never to silence a
drift the gate correctly caught. If ``test_example_fidelity_contract.py``
fails and the diff is NOT an intentional change, that is the gate working:
fix the script (or investigate why realized != declared), do not re-pin.

No solves: every number below comes from ``sim.preflight()`` and
``sim.fidelity_report()``, neither of which time-steps. Wall clock ~30s for
all 33 (script, builder, variant) triples across the 23 auditable scripts
(measured 2026-08-27; CPU-only, no GPU/JAX warmup dominates).

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
            "script's module docstring and test_example_fidelity_contract.py)."
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
