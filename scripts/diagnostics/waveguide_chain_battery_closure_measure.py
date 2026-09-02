#!/usr/bin/env python3
"""WR-90 chain battery — interior power-closure witness driver (v1.8 WP3).

Runs the pre-declared measurement of
``tests/test_waveguide_chain_battery_closure.py`` (route A: the flux lane's
``1 − |S11|² − |S21|²``; route B: the same power balance read off two
``add_flux_monitor`` planes inside the guide) and writes
``tests/fixtures/waveguide_chain_battery/closure_witness.json``. The
declaration, the gate and the branch table live in that test module's
docstring; this file only builds, runs, records and persists.

The measurement is ONE attempt (R2, rfx-tightened). Re-running it to obtain a
different number is a new attempt and needs a written falsifier first.

Usage (from a clean checkout; the rfx import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/waveguide_chain_battery_closure_measure.py \
        --out tests/fixtures/waveguide_chain_battery/closure_witness.json \
        --run-id <vessl run id or "local"> --run-lane <vessl|local>

Lanes ``normalize="flux"`` only (``normalize=True`` never enters). Nothing from
``rfx/probes/refplane.py`` is imported. ``fixture.json`` is never written here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402

import rfx  # noqa: E402

from tests import test_waveguide_chain_battery_closure as C  # noqa: E402

DRIVER = "scripts/diagnostics/waveguide_chain_battery_closure_measure.py"
DEFAULT_OUT = "tests/fixtures/waveguide_chain_battery/closure_witness.json"


def git_sha(override: str | None) -> str:
    if override:
        return override
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — provenance only, never fatal
        return "unknown"


def provenance(args) -> dict:
    return {
        "commit": git_sha(args.git_sha),
        "run_id": args.run_id,
        "run_lane": args.run_lane,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "precision": "float32",
        "python": sys.version.split()[0],
        "hostname": platform.node(),
        "rfx_version": getattr(rfx, "__version__", "?"),
        "recapture_entry_point": DRIVER,
        "recapture_command": (
            f"PYTHONPATH=. python {DRIVER} --out {DEFAULT_OUT} "
            "--run-id <run id> --run-lane <vessl|local>"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="artifact path, relative to the repo root")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--run-lane", default="local")
    ap.add_argument("--git-sha", default=None)
    args = ap.parse_args()

    record = C.measure_closure_witness()
    record["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    record["provenance"] = provenance(args)
    record["driver"] = DRIVER

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out)

    print(f"wrote {args.out}")
    print(f"  verdict           : {record['verdict']}")
    print(f"  max|closure_S - closure_M| = {record['max_abs_diff']:.6e} "
          f"(gate {record['declaration']['gate_abs_diff']})")
    print(f"  worst bin         : {record['worst_bin_hz'] / 1e9:.2f} GHz  "
          f"port {record['closure_s_at_worst']:.6e}  "
          f"interior {record['closure_m_at_worst']:.6e}")
    print(f"  band centre bin   : {record['freqs_hz'][record['band_centre_bin']] / 1e9:.2f} GHz  "
          f"port {record['closure_s_at_centre']:.6e}  "
          f"interior {record['closure_m_at_centre']:.6e}")
    print(f"  settling_db       : flux lane {record['flux_lane']['settling_db']}, "
          f"device {record['device_run']['settling_db']:.2f} dB, "
          f"reference {record['reference_run']['settling_db']:.2f} dB")
    print(f"  wall time         : {record['wall_time_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
