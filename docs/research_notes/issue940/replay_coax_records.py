"""Evaluate the retained real point records in long-double precision; no solve."""
from pathlib import Path
import argparse
import json

import numpy as np


def replay(path):
    data = np.load(path)
    rows = []
    for drive in range(2):
        record = data[f"time_series_{drive}"].astype(np.longdouble)
        assert record.shape == (3000, 2) and np.all(np.isfinite(record))
        power = record * record
        tail_count = len(record) // 10
        ratio = power[-tail_count:].mean(axis=0) / power.max(axis=0)
        db = 10 * np.log10(ratio)
        assert np.all(np.isfinite(db))
        reported = float(data["settling_db"][drive])
        assert abs(float(db.max()) - reported) < 1e-12
        rows.append({
            "drive": drive,
            "longdouble_component_db": [float(value) for value in db],
            "worst_db": float(db.max()), "reported_db": reported,
            "worst_to_minus40_margin_db": float(-40 - db.max()),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", type=Path,
                        default=Path(__file__).with_name("coax_baseline.npz"))
    print(json.dumps(replay(parser.parse_args().record), indent=2, allow_nan=False))
