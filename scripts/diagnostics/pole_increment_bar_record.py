"""Record the Drude B2/B2' pilot and rate cases from a committed checkout."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import jax
import numpy as np

from rfx import ringdown
from scripts.diagnostics.drude_plasma_resonance_record import period_record, pole_record, source_record
from tests.oracle._drude_plasma_scene import FREQ_MAX, references, scene, waveform


def measure(nu, smoke_steps=0):
    if jax.config.x64_enabled:
        raise ValueError("B2 requires x64 off")
    pulse = waveform()
    sim, grid, n_steps = scene(nu, "float32", 16, pulse)
    dt = float(grid.dt)
    ref = references(nu, dt)
    source = source_record(pulse, n_steps, dt)
    actual_steps = smoke_steps or n_steps
    start = time.monotonic()
    result = sim.run(n_steps=actual_steps, compute_s_params=False)
    series = np.asarray(jax.block_until_ready(result.time_series))
    wall = time.monotonic() - start
    record = dict(nu=nu, delta=nu / 2, dt=dt, **ref, n_steps=actual_steps,
                  run_wall_s=wall, steps_per_s=actual_steps / wall,
                  projected_3p4m_step_hours=3.4e6 * wall / actual_steps / 3600,
                  waveform=source, cases=[], smoke_only=bool(smoke_steps))
    if smoke_steps:
        return record
    for periods in (8, 12, 16):
        stop = int(np.ceil(periods / (ref["f_cf"] * dt)))
        prefix = series[:stop]
        off = source["source_off_step"]
        model = ringdown.identify(prefix, dt, off, stop, freq_max=FREQ_MAX)
        identified = pole_record(model, ref["f_cf"], ref["f_cf"], ref["f_main"])
        envelope = period_record(prefix, off, dt, ref["f_cf"])
        nearest = identified["nearest"]
        if nearest is None or not .5 * ref["f_cf"] <= nearest["freq_hz"] <= 1.5 * ref["f_cf"]:
            raise ValueError("cannot judge: no pole within [0.5,1.5] f_cf")
        row = dict(periods=periods, n_steps=stop, identification=identified,
                   period_maxima=envelope, b2_pass=nearest["relative_error_cf"] <= 1e-3)
        if periods in (8, 16):
            row["decay_prediction"] = float(np.exp(-nu / 2 * (periods - 2.5) / ref["f_cf"]))
        if periods == 8:
            row["b2prime_start"] = {1e7: .97287, 1.063e7: .97119, 1.126e7: .96951}[nu]
            row["b2prime_pass"] = bool(np.all(np.array(envelope["last_first_ratio"]) <= 1.01))
        record["cases"].append(row)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, choices=(1e7, 1.063e7, 1.126e7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("smoke steps must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    if not args.smoke_steps and args.output.resolve().is_relative_to(root):
        parser.error("measurement records must be outside the repository")
    record = dict(source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    start = time.monotonic()
    rc = 0
    try:
        record.update(measure(args.nu, args.smoke_steps))
        if any(not row["b2_pass"] or not row.get("b2prime_pass", True) for row in record["cases"]):
            rc = 1
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        rc = 1
    record.update(wall_s=time.monotonic() - start, exit_code=rc,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                      1 if sys.platform == "darwin" else 1024))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
