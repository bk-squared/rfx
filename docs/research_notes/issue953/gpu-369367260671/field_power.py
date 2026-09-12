"""Run the three approved cv06b arms sequentially, retaining raw MSL evidence.

Adds ordinary raw-dump requests and post-run result copies. The production
producer retains its frequencies, duration, sources, gates and returned S.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(os.environ.get("RFX_WORK", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))

import numpy as np
import rfx
from rfx.api._sparams import settling_verdict

assert Path(rfx.__file__).resolve().is_relative_to(ROOT)
LABELS = ("baseline", "stub_1cell", "stub_narrow")


def main(out, build_only=False, *, audit_runner=True):
    out.mkdir(parents=True, exist_ok=False)
    producer = ROOT / "scripts/diagnostics/cv06b_build_falsifiers.py"
    spec = importlib.util.spec_from_file_location("cv06b_falsifiers_recorded", producer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = module.prepare_inputs()
    assert tuple(item[0] for item in inputs) == LABELS
    plan = {
        "scope": "Three approved geometries; source/field/extraction algorithms unchanged; G1 baseline only",
        "source_sha": os.environ.get("RFX_SHA"),
        "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(module.CV06B.read_bytes()).hexdigest(),
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_freqs": 100, "num_periods": 20.0,
        "consumed_plan_check_requested": audit_runner,
        "inputs": {},
        "limits": ["No general MSL power calibration claim", "No unique historical frequency-shift attribution"],
    }
    for label, cv, sim, geometry in inputs:
        grid = sim._build_grid()
        plan["inputs"][label] = dict(
            realized_metal=geometry,
            frequency_reference=module.frequency_reference(cv, label, geometry),
            grid_shape=list(grid.shape), dx_m=float(grid.dx), dt_s=float(grid.dt),
            n_steps=int(grid.num_timesteps(num_periods=20.0)),
        )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if build_only:
        print(json.dumps(plan, indent=2))
        return 0

    original = rfx.Simulation.compute_msl_s_matrix
    observations = []
    consumed = []
    import rfx.simulation as engine
    original_run = engine.run
    if audit_runner:
        checker_path = ROOT / "docs/research_notes/issue953/consumed_plan.py"
        checker_spec = importlib.util.spec_from_file_location("_cv06b_consumed_plan", checker_path)
        checker = importlib.util.module_from_spec(checker_spec)
        checker_spec.loader.exec_module(checker)
        allowed_stub_box = checker.stub_box_from_geometry(inputs[0][3])

        def checked_run(*args, **kwargs):
            arm = len(observations)
            drive = len(consumed) - 2 * arm
            assert arm < 3 and drive in (0, 1)
            signature = checker.fingerprint_run_call(
                original_run, args, kwargs, allowed_stub_box=allowed_stub_box,
            )
            if arm:
                checker.compare_run_plans(consumed[drive]["signature"], signature)
            consumed.append(dict(label=inputs[arm][0], drive=drive, signature=signature))
            (out / "consumed-run-plans.json").write_text(json.dumps(consumed, indent=2) + "\n")
            return original_run(*args, **kwargs)

        engine.run = checked_run

    def observed(self, *args, **kwargs):
        index = len(observations)
        assert index < len(inputs) and self is inputs[index][2]
        assert not args and kwargs == dict(n_freqs=100, num_periods=20.0)
        label = inputs[index][0]
        result = original(self, **kwargs, raw_3probe_dump_path=str(out / f"{label}-raw-vi.npz"))
        arrays = {"freqs_hz": np.asarray(result.freqs), "S": np.asarray(result.S)}
        for name in ("S_raw", "Z0", "beta", "settling_db", "reliable",
                     "reference_impedances", "cond_a", "passivity_correction",
                     "beta_railed", "assembly"):
            value = getattr(result, name, None)
            if value is not None:
                arrays[name] = np.asarray(value)
        np.savez_compressed(out / f"{label}-result.npz", **arrays)
        raw = arrays.get("S_raw", arrays["S"])
        finite = np.isfinite(raw).all(axis=(0, 1))
        gains = (np.linalg.svd(np.moveaxis(raw, -1, 0), compute_uv=False)[:, 0]**2
                 if finite.all() else None)
        settling = arrays.get("settling_db")
        settling_status = ([settling_verdict(v) for v in settling.ravel()]
                           if settling is not None and settling.size else ["absent"])
        observations.append(dict(
            label=label,
            raw_max_coherent_power_gain=None if gains is None else float(np.max(gains)),
            nonfinite_frequency_count=int(np.count_nonzero(~finite)),
            settling_db=None if settling is None else settling.tolist(),
            settling_status_by_drive=settling_status,
            settling_screen_pass=all(v == "pass" for v in settling_status),
            raw_vi=f"{label}-raw-vi.npz", result=f"{label}-result.npz",
        ))
        (out / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
        return result

    # Reuse exactly the already validated three instances, as the producer
    # would after its ordinary prepare_inputs call. No arm runs in parallel.
    module.prepare_inputs = lambda: inputs
    rfx.Simulation.compute_msl_s_matrix = observed
    saved_argv = sys.argv
    outcome = {"complete": False, "field_matrix_calls": 0}
    try:
        sys.argv = [str(producer), "--out-dir", str(out)]
        rc = module.main()
        assert len(observations) == 3
        if audit_runner:
            assert len(consumed) == 6, "all six field runs must pass the consumed-plan check"
        outcome.update(complete=True, producer_return_code=rc,
                       runner_plan_verified=bool(audit_runner and len(consumed) == 6),
                       all_settling_screens_pass=all(o["settling_screen_pass"] is True for o in observations))
        return rc
    except BaseException:
        outcome["error"] = traceback.format_exc()
        raise
    finally:
        rfx.Simulation.compute_msl_s_matrix = original
        engine.run = original_run
        sys.argv = saved_argv
        outcome["field_matrix_calls"] = len(observations)
        (out / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out.resolve(), args.build_only))
