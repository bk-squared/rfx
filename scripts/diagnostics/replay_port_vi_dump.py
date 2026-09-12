#!/usr/bin/env python3
"""Replay an rfx raw V/I port dump into an S-matrix.

This script is intentionally independent of the production port extractors: it
loads raw V/I phasors, follows the recorded wave/assembly convention, and compares
that replay against a production S-matrix stored in the same dump file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from rfx.validation import (
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
)


def _complex_to_jsonable(arr: np.ndarray) -> list:
    arr = np.asarray(arr)
    return [[[[float(z.real), float(z.imag)] for z in row] for row in plane] for plane in arr]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="generic V/I or explicit MSL v3/v4 .npz dump")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--write-json", type=Path, help="optional JSON output path")
    args = parser.parse_args(argv)

    dump = load_port_vi_dump_npz(args.dump)
    replayed = replay_smatrix_from_port_vi_dump(dump)
    payload = {
        "source": replayed.source,
        "freqs_hz": replayed.freqs.tolist(),
        "port_names": list(replayed.port_names),
        "s_params_layout": "S[receiver_port, driven_port, frequency_index]",
        "s_params_complex": _complex_to_jsonable(replayed.s_params),
        "metadata": dict(dump.metadata),
    }

    exit_code = 0
    if dump.production_smatrix is not None:
        production = type(
            "ProductionSMatrix",
            (),
            {"s_params": dump.production_smatrix, "freqs": dump.freqs},
        )()
        comparison = compare_replayed_smatrix(
            replayed,
            production,
            atol=args.atol,
            rtol=args.rtol,
        )
        payload["production_comparison"] = {
            "ok": comparison.ok,
            "max_abs_diff": comparison.max_abs_diff,
            "max_allowed": comparison.max_allowed,
            "atol": comparison.atol,
            "rtol": comparison.rtol,
        }
        payload["status"] = "passed" if comparison.ok else "failed"
        print(comparison.summary())
        if not comparison.ok:
            exit_code = 1
    else:
        payload["status"] = "replay_only"
        print(
            "PASS replay-only: no production_smatrix stored; "
            f"ports={replayed.s_params.shape[0]}, freqs={replayed.s_params.shape[2]}"
        )

    if args.write_json:
        args.write_json.write_text(json.dumps(payload, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
