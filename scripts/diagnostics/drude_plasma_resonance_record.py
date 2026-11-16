"""Record one P0 case on unchanged main; output belongs in the job's NFS folder.

Example (repository on PYTHONPATH):
  python scripts/diagnostics/drude_plasma_resonance_record.py \
      --nu 10000000 --precision float64 --periods 8 --output /runs/pole/pilot/result.json

No frequency/growth bar is judged here. Refusals and identification failures
are recorded, then returned as nonzero exits for the design's stop rules.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from rfx import ringdown
from rfx.measurement.record_end import source_end_step
from tests._x64_compat import enable_x64
from tests.oracle._drude_plasma_scene import (
    DOMAIN, FREQ_MAX, PROBES, SOURCE, references, scene, selected_rates, waveform,
)


def source_record(pulse, n_steps, dt, band_hz=(.5e9, 1.4e9)):
    """Use ringdown's source-off sampler and tolerance, without transit delay."""
    off, detail = source_end_step([(pulse, 0.0)], n_steps, dt,
                                  tolerance=1e-6, return_detail=True)
    if off is None or off >= n_steps:
        raise ValueError("cannot judge: source has not turned off")
    magnitude, peak, _ = detail[0]
    ratio = float(np.max(magnitude[off:]) / peak)
    if ratio > 1e-6:
        raise ValueError("cannot judge: sources-off check failed")
    samples = np.asarray(jax.vmap(pulse)(jnp.arange(n_steps, dtype=jnp.float32) * dt))
    # Only the input spectrum uses FFT. Pole identification is always identify().
    nfft = 1 << int(np.ceil(np.log2(8 * n_steps)))
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64), n=nfft))
    frequencies = np.fft.rfftfreq(nfft, float(dt))
    band = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    db_min = float(20 * np.log10(spectrum[band].min() / spectrum.max()))
    if db_min < -20:
        raise ValueError(f"waveform band minimum {db_min} dB is below -20 dB")
    return dict(f0=pulse.f0, bandwidth=pulse.bandwidth, cutoff=pulse.cutoff,
                amplitude=pulse.amplitude, amplitude_kind="current",
                tau_s=pulse.tau, source_off_step=int(off), source_off_s=off * float(dt),
                source_off_ratio=ratio, source_off_tol=1e-6,
                sampled_band_min_db=db_min, spectrum_bin_hz=float(frequencies[1]),
                sampled_spectrum_peak_hz=float(frequencies[np.argmax(spectrum)]))


def period_record(series, off, dt, f_cf):
    """Bins from source-off, plus exact first/last full-period bar windows.

N denotes the absolute record end, so the final off-anchored bin can be
partial. Its width is explicit; the bar uses the full period ending at N.
"""
    period_steps = 1.0 / (f_cf * dt)
    stop = len(series)

    def window(lo, hi):
        return dict(n_start=int(lo), n_stop=int(hi),
                    duration_periods=(hi - lo) / period_steps,
                    max_abs_e=np.max(np.abs(series[lo:hi]), axis=0).tolist())

    bins = []
    k = 0
    while int(np.ceil(off + k * period_steps)) < stop:
        lo = int(np.ceil(off + k * period_steps))
        hi = min(stop, int(np.ceil(off + (k + 1) * period_steps)))
        bins.append(window(lo, hi))
        k += 1
    first = window(off, min(stop, int(np.ceil(off + period_steps))))
    last = window(max(off, int(np.ceil(stop - period_steps))), stop)
    previous = window(max(off, int(np.ceil(stop - 2 * period_steps))), last["n_start"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.asarray(last["max_abs_e"]) / first["max_abs_e"]
        growth = np.log(np.asarray(last["max_abs_e"]) / previous["max_abs_e"]) * f_cf
    return dict(from_source_off=bins, first_full_period=first,
                last_full_period=last, penultimate_full_period=previous,
                last_first_ratio=ratios.tolist(),
                last_two_periods_log_slope_per_s=growth.tolist())


def pole_record(model, f_ref, f_cf, f_main, freq_max=FREQ_MAX):
    rows = []
    for s, amplitude, residues in zip(model.s, model.amplitude, model.c):
        frequency = float(s.imag / (2 * np.pi))
        if 0 <= frequency <= .9 * freq_max:
            rows.append(dict(freq_hz=frequency, decay_per_s=float(-s.real),
                             amplitude=float(amplitude),
                             residue_abs_per_probe=np.abs(residues).tolist()))
    nearest = min(rows, key=lambda p: abs(p["freq_hz"] - f_ref)) if rows else None
    if nearest is not None:
        nearest = dict(nearest,
                       relative_error_cf=abs(nearest["freq_hz"] / f_cf - 1),
                       relative_error_main=abs(nearest["freq_hz"] / f_main - 1))
    growing = [dict(real_per_s=float(s.real), imag_per_s=float(s.imag),
                    freq_hz=float(s.imag / (2 * np.pi)), amplitude=float(a))
               for s, a in zip(model.s_growing, model.growing_amplitude)]
    return dict(poles_in_band=rows, nearest=nearest, f_ref=f_ref,
                nearest_0815ghz=min(rows, key=lambda p: abs(p["freq_hz"] - .815e9)) if rows else None,
                s_growing=growing, rank=model.rank, n_decimated=model.n_decimated,
                decimation_factors=list(model.factors), n_window=model.n_window,
                freq_max=freq_max, guard=.9, sv_rel=1e-6, unit_tol=1e-6)


def measure(args, record):
    pulse = waveform()
    sim, grid, n_steps = scene(args.nu, args.precision, args.periods, pulse)
    dt = float(grid.dt)
    ref = references(args.nu, dt)
    rates = selected_rates(dt)
    if args.nu not in [row["nu"] for row in rates.values()]:
        raise ValueError("requested nu does not match the predeclared selection rule")
    record.update(nu=args.nu, delta=args.nu / 2, precision=args.precision,
                  periods=args.periods, n_steps=n_steps, dt=dt,
                  grid_shape=list(map(int, grid.shape)), nodes=int(np.prod(grid.shape)),
                  declared_domain_m=DOMAIN,
                  realized_domain_m=[grid.node_of(a, n-1) - grid.node_of(a, 0)
                                     for a, n in zip("xyz", grid.shape)],
                  source_m=SOURCE, probes_m=PROBES, selected_rates=rates, **ref)
    record["waveform"] = source_record(pulse, n_steps, grid.dt)
    actual_steps = args.smoke_steps or n_steps
    record["actual_steps"] = actual_steps
    print(json.dumps({"before_run": record}), flush=True)
    start = time.monotonic()
    result = sim.run(n_steps=actual_steps, compute_s_params=False)
    series = np.asarray(jax.block_until_ready(result.time_series))
    record["run_wall_s"] = time.monotonic() - start
    record["steps_per_s"] = actual_steps / record["run_wall_s"]
    record["series_shape"] = list(series.shape)
    record["series_dtype"] = str(series.dtype)
    record["finite"] = bool(np.isfinite(series).all())
    if args.smoke_steps:
        record["smoke_only"] = True
        return
    np.savez_compressed(args.output.parent / "probes.npz", series=series, dt=dt)
    off = record["waveform"]["source_off_step"]
    record["period_maxima"] = period_record(series, off, dt, ref["f_cf"])
    if not record["finite"]:
        record["identification_unavailable"] = "nonfinite record"
        return
    f_ref = ref["f_cf"] if args.precision == "float64" else ref["f_main"]
    model = ringdown.identify(series, dt, off, n_steps, freq_max=FREQ_MAX)
    record["identification"] = pole_record(model, f_ref, ref["f_cf"], ref["f_main"])
    if not any(.5 * f_ref <= p["freq_hz"] <= 1.5 * f_ref
               for p in record["identification"]["poles_in_band"]):
        raise ValueError("cannot judge: no pole in [0.5, 1.5] f_ref")


def json_safe(value):
    """JSON null for nonfinite scalars; the finite flag preserves the distinction."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.periods not in (6, 8) or args.smoke_steps < 0:
        parser.error("P0 periods must be 6 or 8; smoke steps must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    local_smoke = args.smoke_steps and args.output.resolve().is_relative_to(root / "output" / "pole")
    if args.output.resolve().is_relative_to(root) and not local_smoke:
        parser.error("measurement records must be outside the repository")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    record = dict(job=os.environ.get("REMI_JOB"), jax_version=jax.__version__,
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    rc = 0
    try:
        if args.precision == "float32" and jax.config.x64_enabled:
            raise ValueError("main float32 requires x64 disabled")
        with enable_x64() if args.precision == "float64" else nullcontext():
            measure(args, record)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        rc = 1
    finally:
        record["wall_s"] = time.monotonic() - start
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
        record["exit_code"] = rc
        args.output.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n")
        print(json.dumps(json_safe(record), allow_nan=False), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
