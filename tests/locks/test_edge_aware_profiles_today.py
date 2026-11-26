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

import pytest

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
    mismatches = _profile_mismatches(entries)
    assert not mismatches, "\n".join(mismatches)


def _profile_mismatches(entries):
    mismatches = []
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
        try:
            actual = edge_aware_profile(**args)
        except Exception as exc:
            mismatches.append(f"entry {index}, callers={entry['callers']}: {type(exc).__name__}: {exc}")
            continue
        for name in ("cells", "nodes", "metal_end_nodes", "edge_cells"):
            expected_hex = entry["outputs"][name]
            expected = np.asarray([float.fromhex(v) for v in expected_hex], dtype=np.float64)
            values = np.asarray(getattr(actual, name), dtype=np.float64)
            label = f"entry {index}, {name}, callers={entry['callers']}"
            actual_hex = np.asarray([float(v).hex() for v in values], dtype=str)
            if not (np.array_equal(values, expected)
                    and np.array_equal(actual_hex, np.asarray(expected_hex, dtype=str))):
                mismatches.append(f"{label}: expected={expected_hex!r}, actual={actual_hex.tolist()!r}")
    return mismatches


def test_lock_reports_every_differing_entry(monkeypatch):
    from types import SimpleNamespace
    import sys

    args = {name: float(value).hex() for name, value in
            dict(lo=0, hi=1, dx=1, max_ratio=1.3, edge_offset=0.35).items()}
    args.update(boundary_cell=None, faces=[], sheet_edges=[])
    outputs = {name: [float(1).hex()] for name in
               ("cells", "nodes", "metal_end_nodes", "edge_cells")}
    entries = [dict(args=args, outputs=outputs, callers=[f"caller:{index}"]) for index in range(2)]
    monkeypatch.setattr(sys.modules[__name__], "edge_aware_profile",
                        lambda **kwargs: SimpleNamespace(**{name: [2.0] for name in outputs}))
    mismatches = _profile_mismatches(entries)
    assert len(mismatches) == 8
    for index in range(2):
        assert sum(f"entry {index}," in message for message in mismatches) == 4
    with pytest.raises(AssertionError, match="caller:1"):
        assert not mismatches, "\n".join(mismatches)
