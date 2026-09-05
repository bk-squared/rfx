"""Backfill the auxiliary-echo record invariant (#888) into a committed
``lattice_witness.json``, without re-running any physics.

WHY THIS EXISTS RATHER THAN A RE-RUN. ``lattice_witness.evaluate`` now emits an
``aux_echo`` block and a ``precond_aux_echo_record`` gate for every rung, so
every FUTURE run of cv04 / cv22 / cv23 carries the witness. The three
``lattice_witness.json`` files already in the repo were produced on the run
machines of their lanes, and regenerating them here moves numbers this lane is
forbidden to move: rebuilding cv22 / cv23 from their committed ``rfx.json``
reproduces every scalar only to ~1e-12 relative (platform float), and re-running
cv04's FDTD moves ``mean_dR_lattice_gated`` in the fourth digit. So the witness
is INSERTED into the committed documents and nothing else in them is touched.

WHAT IT WRITES. Into each ``rungs.<name>``:

  * ``aux_echo``   -- the block ``cv22_dispersive_gates.slab_aux_echo`` computes
                      from the rung's declared geometry (``nx_interior``,
                      ``dx_div``) and its own ``dt_s`` / ``n_steps``;
  * ``gates.precond_aux_echo_record`` -- ``aux_echo.ok``, placed directly after
    ``precond_tail_witness`` so the key order matches ``evaluate``'s.

Nothing else in the document changes; the script asserts that.

Usage (from the repo root)::

    python validation/crossval/comparators/emit_aux_echo_witness.py --check
    python validation/crossval/comparators/emit_aux_echo_witness.py --write
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv22_dispersive_gates as G  # noqa: E402

# (results dir, the case's declared interior width in cells at dx_div = 1).
# cv04 runs G.NX_INTERIOR (600, its own pre-declared rig); cv22 and cv23 run
# G.NX_INTERIOR_R3 (1000, the round-3 box). Both are declared constants of the
# family, not numbers read back out of a run.
CASES = (
    ("validation/crossval/_04_fresnel_results", G.NX_INTERIOR),
    ("validation/crossval/_22_dispersive_results", G.NX_INTERIOR_R3),
    ("validation/crossval/_23_lossy_results", G.NX_INTERIOR_R3),
)


def _rfx_records(results_dir: str) -> dict:
    """``rungs.<name> -> run.record`` from the case's committed ``rfx*.json``,
    for the cross-check. cv04 has no ``rfx.json``; its rungs cross-check
    against the declared rig alone."""
    import lattice_witness as LW
    if not os.path.isfile(os.path.join(results_dir, "rfx.json")):
        return {}
    return {name: (ad.get("run") or {}).get("record") or {}
            for name, ad in LW.rungs_from_results(results_dir).items()}


def aux_echo_for_rung(rung: dict, nx_interior_default: int, record: dict | None = None) -> dict:
    """The ``aux_echo`` block for one committed rung, from geometry only."""
    K = int(rung.get("dx_div", 1))
    rec = record or {}
    nxi = rec.get("nx_interior", nx_interior_default)
    echo = G.slab_aux_echo(int(nxi) // K, float(rung["dt_s"]), dx_div=K,
                           n_steps=int(rung["n_steps"]))
    cells = G.rig_cells(int(nxi) // K, K)
    for key in ("nx", "x_lo", "probe_refl", "probe_trans", "n_cpml"):
        if key in rec and int(rec[key]) != int(cells[key]):
            raise ValueError(f"rig bookkeeping drift on {key}: record {rec[key]} vs geometry {cells[key]}")
    return echo


def backfill(doc: dict, nx_interior_default: int, records: dict) -> dict:
    """A copy of ``doc`` with the witness inserted into every rung."""
    out = copy.deepcopy(doc)
    for name, rung in out["rungs"].items():
        echo = aux_echo_for_rung(rung, nx_interior_default, records.get(name))
        rung["aux_echo"] = echo
        gates = rung["gates"]
        rebuilt = {}
        for k, v in gates.items():
            rebuilt[k] = v
            if k == "precond_tail_witness":
                rebuilt["precond_aux_echo_record"] = bool(echo["ok"])
        if "precond_aux_echo_record" not in rebuilt:
            rebuilt["precond_aux_echo_record"] = bool(echo["ok"])
        rung["gates"] = rebuilt
    return out


def _stripped(doc: dict) -> dict:
    """``doc`` with exactly what this script adds removed, for the no-drift check."""
    out = copy.deepcopy(doc)
    for rung in out["rungs"].values():
        rung.pop("aux_echo", None)
        rung["gates"] = {k: v for k, v in rung["gates"].items() if k != "precond_aux_echo_record"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite the artifacts in place")
    ap.add_argument("--check", action="store_true", help="report the ratios; write nothing")
    a = ap.parse_args(argv)
    rc = 0
    for rel, nxi in CASES:
        path = os.path.join(_REPO_ROOT, rel, "lattice_witness.json")
        if not os.path.isfile(path):
            print(f"[skip] {rel}/lattice_witness.json absent")
            continue
        with open(path) as fh:
            doc = json.load(fh)
        new = backfill(doc, nxi, _rfx_records(os.path.join(_REPO_ROOT, rel)))
        assert _stripped(new) == _stripped(doc), f"{rel}: the backfill changed something else"
        for name, rung in new["rungs"].items():
            e = rung["aux_echo"]
            flag = "ok" if e["ok"] else "FAIL"
            print(f"{rel.split('/')[-1]:28s} {name:16s} record {e['record_steps']:6d} / arrival "
                  f"{e['echo_arrival_steps']:6d} = {e['record_over_echo_arrival']:.3f}  [{flag}]")
            rc = rc or (0 if e["ok"] else 1)
        if a.write:
            with open(path, "w") as fh:
                json.dump(new, fh, indent=1)
            print(f"  wrote {rel}/lattice_witness.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
