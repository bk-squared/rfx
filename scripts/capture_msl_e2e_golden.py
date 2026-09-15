#!/usr/bin/env python3
"""Qualify the aligned live coupon before any explicit golden write.

Always run from this checkout with PYTHONPATH=$PWD JAX_PLATFORMS=cpu.
Default operation writes a named report and full raw/projected arrays, NEVER
changes the golden. --write-golden also requires a passing 2x-refinement
report and an independently named same-mesh confirmation report. Existing
rtol=.005/atol=.002 drift tolerance is unchanged. Frozen replay binaries
and their original 80 um board are deliberately outside this operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import rfx
import jax
import jax.numpy as jnp
import numpy as np

from tests.unit.autodiff import test_msl_sparam_ad as t
from tests._msl_fixture_qualification import qualify_result

REPO_ROOT = Path(__file__).resolve().parents[1]


def complex_pairs(a):
    a = np.asarray(a)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def complex_array(a):
    a = np.asarray(a)
    return a[..., 0] + 1j * a[..., 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-periods", type=float, default=t.E2E_PERIODS)
    parser.add_argument("--refinement-report", type=Path)
    parser.add_argument("--confirmation-report", type=Path)
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        parser.error("this repair campaign requires JAX_PLATFORMS=cpu")
    if args.output.exists():
        parser.error("choose a new report path; named evidence must not be overwritten")
    source_files = (Path(t.__file__), REPO_ROOT / "tests/_msl_fixture_qualification.py",
                    Path(__file__).resolve())
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=REPO_ROOT, text=True).strip()
    source_hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files}
    print(f"rfx.__file__={rfx.__file__}", flush=True)
    sim = t._build_aligned_e2e_sim(refinement=args.refinement)
    rz = t._assert_trace_sheet_realized(sim)
    preflight = t._assert_e2e_preflight(sim)
    print(json.dumps([issue.to_dict() for issue in preflight], indent=2), flush=True)
    freqs = jnp.linspace(t.F_MAX / 10, t.F_MAX, 10, dtype=jnp.float32)
    result = sim.compute_msl_s_matrix(freqs=freqs, num_periods=args.num_periods)
    qualification = qualify_result(result)
    projected = np.asarray(result.S, dtype=np.complex128)
    raw = projected if result.S_raw is None else np.asarray(result.S_raw)
    report = {
        "run_name": args.run_name,
        "fixture_id": t.E2E_REPAIR_ID,
        "source_sha": source_sha,
        "source_hashes": source_hashes,
        "rfx_file": rfx.__file__,
        "preflight": [issue.to_dict() for issue in preflight],
        "jax_version": jax.__version__, "backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "dtype": str(np.asarray(result.S).dtype),
        "refinement": args.refinement, "num_periods": args.num_periods,
        "geometry": {"substrate_height_m": t.H_SUB, "trace_width_m": t.W_TRACE,
                     "launch_plane_separation_m": t.L_LINE, "eps_r": t.EPS_R,
                     "domain_m": t.E2E_DOMAIN,
                     "boundary_spacing_m": [d / args.refinement for d in t.E2E_SPACING],
                     "dz_profile_m": np.repeat(t.E2E_DZ_PROFILE / args.refinement, args.refinement).tolist(),
                     "grid_shape": [rz.grid.nx, rz.grid.ny, rz.grid.nz],
                     "trace_model": "PEC sheet at substrate interface; nominal 35um copper",
                     "drive_waveform": {"kind": "differentiated Gaussian",
                                        "f0_hz": t.E2E_WAVEFORM.f0,
                                        "bandwidth": t.E2E_WAVEFORM.bandwidth,
                                        "cutoff": t.E2E_WAVEFORM.cutoff,
                                        "amplitude": t.E2E_WAVEFORM.amplitude},
                     "launch_x_m": [t.PORT_MARGIN, t.PORT_MARGIN + t.L_LINE],
                     "s_reference_x_m": qualification["reference_planes_m"],
                     "probe_x_m": [[.005,.006,.007,.008,.009], [.009,.008,.007,.006,.005]]},
        "freqs_hz": np.asarray(result.freqs).tolist(),
        "S_raw": complex_pairs(raw), "S_projected": complex_pairs(projected),
        "Z0": complex_pairs(result.Z0), "beta": complex_pairs(result.beta),
        "assembly": result.assembly, "qualification": qualification,
    }
    for name in ("reliable", "settling_db", "cond_a", "beta_railed", "passivity_correction"):
        value = getattr(result, name, None)
        report[name] = None if value is None else np.asarray(value).tolist()
    old = np.load(t.E2E_GOLDEN_PATH)
    report["old_golden_max_abs_delta"] = float(np.max(np.abs(projected - old)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if qualification["failures"]:
        raise SystemExit("UNQUALIFIED: report preserved; golden unchanged")
    if args.write_golden:
        if args.refinement != 1 or args.num_periods != t.E2E_PERIODS:
            parser.error("capture must use the exact base test mesh and integration horizon")
        if not args.refinement_report or not args.confirmation_report:
            parser.error("capture needs both refinement and independent confirmation evidence")
        for path, expected_refinement in ((args.refinement_report, 2),
                                           (args.confirmation_report, 1)):
            other = json.loads(path.read_text())
            assert other["fixture_id"] == report["fixture_id"]
            assert other["source_sha"] == report["source_sha"], "production revisions differ"
            assert other["backend"] == report["backend"] == "cpu"
            assert other["dtype"] == report["dtype"]
            assert other["x64_enabled"] == report["x64_enabled"]
            assert other["source_hashes"] == report["source_hashes"], "fixture sources changed"
            assert other["run_name"] != report["run_name"], "independent named run required"
            assert other["refinement"] == expected_refinement
            assert other["num_periods"] == report["num_periods"]
            assert not other["qualification"]["failures"], "other run is unqualified"
            np.testing.assert_array_equal(other["freqs_hz"], report["freqs_hz"])
            if expected_refinement == 1:
                np.testing.assert_allclose(projected, complex_array(other["S_projected"]),
                                           rtol=.005, atol=.002)
            else:
                # Predeclared 2% raw-complex mesh budget, all ten bins.
                np.testing.assert_allclose(raw, complex_array(other["S_raw"]),
                                           rtol=0, atol=.02)
        np.save(t.E2E_GOLDEN_PATH, projected)
        manifest = t.E2E_GOLDEN_PATH.with_suffix(".json")
        report["qualification_reports"] = [str(args.refinement_report), str(args.confirmation_report)]
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Qualified golden written: {t.E2E_GOLDEN_PATH}; manifest: {manifest}")
        print("Record the measured re-pin decision and named reports in the E2E test docstring before committing.")


if __name__ == "__main__":
    main()
