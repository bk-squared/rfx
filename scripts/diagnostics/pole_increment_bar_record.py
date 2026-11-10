"""Record the Drude and Lorentz B2/B2' cases from a committed checkout."""
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

from rfx import Box, GaussianPulse, Simulation, lorentz_pole, ringdown
from scripts.diagnostics.drude_plasma_resonance_record import period_record, pole_record, source_record
from tests.oracle._drude_plasma_scene import (
    DOMAIN, DX, EPS0, FREQ_MAX, PROBES, SOURCE, map_frequency, references, scene, waveform,
)


def lorentz_reference(nu, dt):
    """Hand-built longitudinal map and residual, without coefficient helpers."""
    w0 = 2 * np.pi * 3e9
    ddt = np.float64(nu) / 2 * np.float64(dt)
    restoring = w0**2 * dt**2 / (1 + ddt)
    exact = ((2 - w0**2 * dt**2) / (1 + ddt),
             -(1 - ddt) / (1 + ddt), EPS0 * 2 * w0**2 * dt**2 / (1 + ddt))
    rounded = tuple(float(np.float32(v)) for v in exact)
    return dict(a=float(exact[0]), b=float(exact[1]), c=float(exact[2]),
                a32=rounded[0], b32=rounded[1], c32=rounded[2], Cc=1 / EPS0,
                r=rounded[0] + rounded[1] - 1 + restoring,
                f_cf=map_frequency(*exact, 1 / EPS0, dt),
                f_main=map_frequency(*rounded, 1 / EPS0, dt))


def lorentz_scene(nu):
    pulse = GaussianPulse(f0=5.5e9, bandwidth=1.0)
    sim = Simulation(freq_max=16e9, domain=DOMAIN, dx=DX,
                     boundary="pec", precision="float32")
    sim.add_material("lorentz", eps_r=1., sigma=0.,
                     lorentz_poles=[lorentz_pole(2., 2 * np.pi * 3e9, nu / 2)])
    sim.add(Box((0., 0., 0.), DOMAIN), material="lorentz")
    sim.add_source(SOURCE, component="ez", waveform=pulse, amplitude_kind="current")
    for position in PROBES:
        sim.add_probe(position, component="ez")
    grid = sim._build_grid()
    selected = next((1e7 + k * 1e4 for k in range(100001)
                     if abs(lorentz_reference(1e7 + k * 1e4, float(grid.dt))["r"]) >= 2**-24), None)
    if selected != nu:
        raise ValueError(f"Lorentz selection rule gives {selected}, requested {nu}")
    ref = lorentz_reference(nu, float(grid.dt))
    return sim, grid, int(np.ceil(16 / (ref["f_cf"] * grid.dt))), pulse, ref


def measure(nu, smoke_steps=0):
    if jax.config.x64_enabled:
        raise ValueError("B2 requires x64 off")
    is_lorentz = nu == 1.183e7
    if is_lorentz:
        sim, grid, n_steps, pulse, ref = lorentz_scene(nu)
    else:
        pulse = waveform()
        sim, grid, n_steps = scene(nu, "float32", 16, pulse)
        ref = references(nu, float(grid.dt))
    dt = float(grid.dt)
    freq_max = 16e9 if is_lorentz else FREQ_MAX
    source = source_record(pulse, n_steps, dt, (4.5e9, 5.5e9) if is_lorentz else (.5e9, 1.4e9))
    actual_steps = smoke_steps or n_steps
    start = time.monotonic()
    result = sim.run(n_steps=actual_steps, compute_s_params=False)
    series = np.asarray(jax.block_until_ready(result.time_series))
    wall = time.monotonic() - start
    record = dict(medium="lorentz" if is_lorentz else "drude",
                  nu=nu, delta=nu / 2, dt=dt, **ref, n_steps=actual_steps,
                  run_wall_s=wall, steps_per_s=actual_steps / wall,
                  projected_3p4m_step_hours=3.4e6 * wall / actual_steps / 3600,
                  waveform=source, cases=[], smoke_only=bool(smoke_steps))
    if smoke_steps:
        return record
    for periods in (8, 12, 16):
        stop = int(np.ceil(periods / (ref["f_cf"] * dt)))
        prefix = series[:stop]
        off = source["source_off_step"]
        model = ringdown.identify(prefix, dt, off, stop, freq_max=freq_max)
        identified = pole_record(model, ref["f_cf"], ref["f_cf"], ref["f_main"], freq_max=freq_max)
        envelope = period_record(prefix, off, dt, ref["f_cf"])
        nearest = identified["nearest"]
        if nearest is None or not .5 * ref["f_cf"] <= nearest["freq_hz"] <= 1.5 * ref["f_cf"]:
            raise ValueError("cannot judge: no pole within [0.5,1.5] f_cf")
        row = dict(periods=periods, n_steps=stop, identification=identified,
                   period_maxima=envelope, b2_pass=nearest["relative_error_cf"] <= 1e-3)
        if periods in (8, 16):
            row["decay_prediction"] = float(np.exp(-nu / 2 * (periods - 2.5) / ref["f_cf"]))
        if is_lorentz:
            first = envelope["first_full_period"]
            last = envelope["last_full_period"]
            separation = (last["n_start"] + last["n_stop"] - first["n_start"] - first["n_stop"]) * dt / 2
            row["decay_prediction"] = float(np.exp(-nu / 2 * separation))
        if periods in (8, 16) and not is_lorentz:  # A11c: B2' is judged at N = 16 too
            row["b2prime_start"] = ({1e7: .97287, 1.063e7: .97119, 1.126e7: .96951}[nu] if periods == 8
                                    else row["decay_prediction"])
            row["b2prime_pass"] = bool(np.all(np.array(envelope["last_first_ratio"]) <= 1.01))
        record["cases"].append(row)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, choices=(1e7, 1.063e7, 1.126e7, 1.183e7), required=True)
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
