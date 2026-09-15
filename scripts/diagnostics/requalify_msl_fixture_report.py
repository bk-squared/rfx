#!/usr/bin/env python3
"""Re-evaluate a saved aligned-coupon report without running FDTD or writing goldens.

Run from the intended checkout with PYTHONPATH=$PWD JAX_PLATFORMS=cpu.
The complete original capture is retained, with its original provenance and
qualification untouched. A separate analysis records the current helper and
analyzer hashes. This does not turn an older capture into a final-source run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from tests import _msl_fixture_qualification as qualification_helper

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def decode_complex(value, expected_shape, name):
    pairs = np.asarray(value, dtype=float)
    if pairs.shape != (*expected_shape, 2):
        raise ValueError(f"{name} has shape {pairs.shape}, expected {(*expected_shape, 2)}")
    return pairs[..., 0] + 1j * pairs[..., 1]


def requalify(original):
    """Interpret only the known aligned drawing and its actual reference planes."""
    if original["fixture_id"] != "931-msl-254um-600um-aligned-v1":
        raise ValueError("unsupported fixture: this comparator is for the aligned MSL coupon")
    geometry = original["geometry"]
    for name, expected in (("substrate_height_m", 254e-6),
                           ("trace_width_m", 600e-6),
                           ("eps_r", 3.66)):
        np.testing.assert_allclose(geometry[name], expected, rtol=0, atol=1e-12)
    probe_zero = [planes[0] for planes in geometry["probe_x_m"]]
    np.testing.assert_allclose(probe_zero, qualification_helper.REFERENCE_PLANES_M,
                               rtol=0, atol=2e-9)
    freqs = np.asarray(original["freqs_hz"], dtype=float)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("frequency vector must be nonempty and one-dimensional")
    n_freqs = freqs.size
    projected = decode_complex(original["S_projected"], (2, 2, n_freqs), "S_projected")
    raw = decode_complex(original["S_raw"], (2, 2, n_freqs), "S_raw")
    result = SimpleNamespace(
        freqs=freqs, S=projected, S_raw=raw,
        Z0=decode_complex(original["Z0"], (2, n_freqs), "Z0"),
        beta=decode_complex(original["beta"], (n_freqs,), "beta"),
        assembly=original["assembly"],
    )
    for name, shape in (("reliable", (2, n_freqs)), ("settling_db", (2,)),
                        ("cond_a", (n_freqs,)), ("beta_railed", (2, n_freqs))):
        value = original.get(name)
        if value is not None:
            value = np.asarray(value)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        setattr(result, name, value)
    return qualification_helper.qualify_result(result), projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True, help="name of this offline analysis")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path; original and prior evidence remain immutable")
    source_bytes = args.input.read_bytes()
    original = json.loads(source_bytes)
    corrected, projected = requalify(original)
    golden_path = REPO_ROOT / "tests/fixtures/msl_s_matrix_golden.npy"
    golden_bytes = golden_path.read_bytes()
    golden = np.load(golden_path)
    if golden.shape != projected.shape:
        raise ValueError("golden and capture frequency shapes differ")
    delta = np.abs(projected - golden)
    helper_path = Path(qualification_helper.__file__).resolve()
    analyzer_path = Path(__file__).resolve()
    report = {
        "kind": "offline_requalification",
        "run_name": args.run_name,
        "fresh_fdtd_run": False,
        "golden_written": False,
        "reason": "Compare probe-0 S at x=5 and 9 mm against its 4 mm electrical length; "
                  "the former comparator used the 10 mm feed separation.",
        "input_report": {"path": str(args.input.resolve()), "sha256": sha256(source_bytes)},
        "original_capture": original,
        "analysis": {
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   cwd=REPO_ROOT, text=True).strip(),
            "source_hashes": {
                str(path.relative_to(REPO_ROOT)): sha256(path.read_bytes())
                for path in (analyzer_path, helper_path)
            },
            "numpy_version": np.__version__,
            "reference_planes_m": list(qualification_helper.REFERENCE_PLANES_M),
            "original_failures": original["qualification"]["failures"],
            "corrected_failures": corrected["failures"],
            "corrected_qualification": corrected,
            "unchanged_golden": {
                "path": str(golden_path.relative_to(REPO_ROOT)),
                "sha256": sha256(golden_bytes),
                "max_abs_delta": float(np.max(delta)),
                "mismatching_complex_entries": int(np.count_nonzero(
                    ~np.isfinite(delta) | (delta > .002 + .005 * np.abs(golden)))),
                "total_complex_entries": int(projected.size),
                "rtol": .005, "atol": .002,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "analysis": report["analysis"]}, indent=2))
    return int(bool(corrected["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
