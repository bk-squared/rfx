"""Retain a concrete paper observation and its same-run settling witness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rfx.api._sparams import settling_verdict


def retain_observation(directory, label: str, result, observable) -> dict:
    """Host-only reporting, deliberately outside every differentiated closure."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = directory / f"{label}.npz"
    np.savez_compressed(arrays, time_series=np.asarray(result.time_series),
                        observable=np.asarray(observable),
                        settling_probe_info=np.asarray(result.settling_probe_info, dtype=int))
    db = result.settling_db
    record = {
        "label": label,
        "settling_db": db,
        "settling_verdict": settling_verdict(db),
        "settling_witness": result.settling_witness,
        "arrays": arrays.name,
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "scope": "same-run point-probe ring-down diagnostic; passing does not establish RF accuracy",
    }
    (directory / f"{label}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"[settling] {label}: {record['settling_verdict']} ({db} dB); {arrays}")
    return record
