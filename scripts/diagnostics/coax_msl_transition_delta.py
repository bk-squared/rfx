#!/usr/bin/env python3
"""Store the coax-to-MSL transition's own S on its committed fixture.

The transition lane is cross-family and frozen for new work, but it shares
``stamp_coaxial_line`` with the coaxial lanes, so a change to how the coax
conductors are realized reaches it whether or not anyone intended that. This
records the lane's S on its own attempt-2 fixture so the SAME command can be
run on two trees and the difference read off, rather than argued about.

Run it on each tree, then diff::

    PYTHONPATH=. python scripts/diagnostics/coax_msl_transition_delta.py \\
        --out <run-dir> --run-id before
    python scripts/diagnostics/coax_msl_transition_delta.py \\
        --compare <before.json> <after.json>

The fixture, the step count, the band and every probe setting come from the
test module itself (``_build_coax_msl_transition_sim_attempt2`` and
``_attempt2_kwargs``), so this measures the lane and not a board of my own.
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


def _load_fixture_module():
    sys.path.insert(0, str(REPO / "tests" / "unit" / "sparams"))
    import test_coax_msl_transition as T
    return T


def measure() -> dict:
    import rfx
    import jax
    T = _load_fixture_module()
    sim = T._build_coax_msl_transition_sim_attempt2()
    kwargs = T._attempt2_kwargs(T.N_STEPS_2)
    res = sim.compute_coax_msl_transition(**kwargs)
    S = np.asarray(res.s_params)
    return {
        "rfx_file": str(Path(rfx.__file__).resolve()),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "fixture": "_build_coax_msl_transition_sim_attempt2",
        "n_steps": int(T.N_STEPS_2),
        "freqs_hz": np.asarray(T.FREQS_2, dtype=float).tolist(),
        "port_names": list(res.port_names),
        "s_params_real": np.real(S).astype(float).tolist(),
        "s_params_imag": np.imag(S).astype(float).tolist(),
        "abs_s": np.abs(S).astype(float).tolist(),
        "status": str(getattr(res, "status", "")),
        "settling_db": np.asarray(getattr(res, "settling_db", []),
                                  dtype=float).tolist(),
    }


def compare(before_path: Path, after_path: Path) -> int:
    a = json.loads(before_path.read_text())
    b = json.loads(after_path.read_text())
    Sa = (np.asarray(a["measured"]["s_params_real"])
          + 1j * np.asarray(a["measured"]["s_params_imag"]))
    Sb = (np.asarray(b["measured"]["s_params_real"])
          + 1j * np.asarray(b["measured"]["s_params_imag"]))
    if Sa.shape != Sb.shape:
        raise SystemExit(f"shape {Sa.shape} vs {Sb.shape}; not the same lane")
    d = np.abs(Sb - Sa)
    names = a["measured"].get("port_names", ["0", "1"])
    freqs = np.asarray(a["measured"]["freqs_hz"], dtype=float)
    print(f"[delta] before {a['commit'][:8]}  after {b['commit'][:8]}")
    print(f"[delta] fixture {a['measured']['fixture']}, "
          f"{a['measured']['n_steps']} steps, {freqs.size} bins "
          f"{freqs.min()/1e9:.3g}-{freqs.max()/1e9:.3g} GHz")
    print(f"[delta] max |dS| over every entry and bin: {d.max():.6e}")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            print(f"[delta]   S{names[i]}{names[j]}: max |dS| {d[i, j].max():.6e}, "
                  f"|S| before {np.abs(Sa[i, j]).min():.5f}-{np.abs(Sa[i, j]).max():.5f}, "
                  f"after {np.abs(Sb[i, j]).min():.5f}-{np.abs(Sb[i, j]).max():.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))
    if not args.out:
        raise SystemExit("--out is required unless --compare")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                  text=True).strip()
    rec = {
        "schema": "rfx.coax_msl_transition_delta", "schema_version": 1,
        "commit": sha, "run_id": args.run_id,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    path = out_dir / "coax_msl_transition_delta.json"
    path.write_text(json.dumps(rec, indent=1))      # persisted BEFORE the solve
    rec["measured"] = m = measure()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)
    S = np.abs(np.asarray(m["abs_s"]))
    print(f"[delta] {sha[:8]} status {m['status']} settling {m['settling_db']}")
    print(f"[delta] |S| range {S.min():.5f}-{S.max():.5f}")
    print(f"[delta] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
