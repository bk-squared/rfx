"""Replay base-branch profiles exactly, including all four returned fields.

The fixture was generated on the cluster at the generator commit named below,
before any change to the helper; its own header repeats the run provenance.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

from rfx.mesh_edges import SheetEdge, edge_aware_profile


LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/edge_aware_profiles_today/fixture.json",
    "generator": "scripts/dev/freeze_edge_aware_today.py",
    "commit": "e0237d6890b6d518d0593f332b0fae8f1aec88d5",
    "date": "2026-11-26",
    "run_id": "lr-nu-261126072858-z5km4",
    "host": "remi cluster cpu, rfx-gate image, Linux x86_64 glibc 2.35, numpy 2.4.6, jax 0.10.2",
    "pinned_until": "2027-03-26",
}


def test_edge_aware_profiles_match_today_bit_for_bit():
    path = Path(__file__).resolve().parents[2] / LOCK_PROVENANCE["fixture"]
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == 1
    assert frozen["provenance"]["pytest_exitstatus"] == 0
    entries = frozen.get("entries")
    if entries is None:
        entries = json.loads(gzip.decompress(base64.b64decode(frozen["entries_gzip_base64"])))
    assert entries, "the freeze must contain returning calls"
    scan_callers = {
        caller for entry in entries for caller in entry["callers"]
        if caller.startswith("scan:")
    }
    assert scan_callers == {
        "scan:c=2.30", "scan:c=2.40", "scan:c=2.45", "scan:c=2.50",
        "scan:c=2.85", "scan:c=2.90", "scan:c=2.95", "scan:c=3.00",
    }
    for index, entry in enumerate(entries):
        args = dict(entry["args"])
        for name in ("lo", "hi", "dx", "max_ratio", "edge_offset"):
            args[name] = float.fromhex(args[name])
        if args["boundary_cell"] is not None:
            args["boundary_cell"] = float.fromhex(args["boundary_cell"])
        args["faces"] = [float.fromhex(v) for v in args["faces"]]
        args["sheet_edges"] = [
            SheetEdge(float.fromhex(position), metal_side)
            for position, metal_side in args["sheet_edges"]
        ]
        actual = edge_aware_profile(**args)
        for name in ("cells", "nodes", "metal_end_nodes", "edge_cells"):
            expected_hex = entry["outputs"][name]
            expected = np.asarray([float.fromhex(v) for v in expected_hex], dtype=np.float64)
            values = np.asarray(getattr(actual, name), dtype=np.float64)
            label = f"entry {index}, {name}, callers={entry['callers']}"
            assert np.array_equal(values, expected), label
            actual_hex = np.asarray([float(v).hex() for v in values], dtype=str)
            assert np.array_equal(actual_hex, np.asarray(expected_hex, dtype=str)), label
