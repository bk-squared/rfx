#!/usr/bin/env python3
# The v1 pmap runner (rfx.runners.distributed.run_distributed) was removed in #1296; this script ran on the commit its records name.
"""Measure public sim.run exchange-interval traces, spectra, and CPU timings.

Run from the worktree root (no pytest):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    PYTHONPATH=$PWD /Users/byungkwankim/Documents/rfx/.venv/bin/python \
    scripts/diagnostics/measure_exchange_interval_error.py

Only this diagnostic and its output directory are written. Solver code and
tests are neither patched nor monkeypatched. --analyze-only rebuilds artifacts
from the saved float32 traces without running a simulation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from unittest import mock


@contextmanager
def _validation_bypassed(k):
    """No-op the exchange_interval entry validation while k > 1 runs."""
    if k == 1:
        yield
        return
    import rfx.api._execute as _execute
    import rfx.runners.distributed as _v1
    import rfx.runners.distributed_v2 as _v2
    with mock.patch.object(_execute, "validate_exchange_interval", lambda v: None), \
         mock.patch.object(_v2, "validate_exchange_interval", lambda v: None), \
         mock.patch.object(_v1, "validate_exchange_interval", lambda v: None):
        yield

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CASES = {"M1": (400, [12, 24, 26, 36], [2, 4]),
         "M2": (800, [12, 24, 26, 36], [2]),
         "M3": (2000, [36], [2, 4])}
KS = (1, 2, 4)
REPEATS = 5
DOMAIN_M = (0.048, 0.016, 0.016)
SOURCE_M = (0.010, 0.008, 0.008)
DX_M = 0.001
C0_M_S = 299792458.0
BAND_HZ = (8e9, 16e9)
ANALYTIC_SOURCE = (
    "https://engineering.purdue.edu/wcchew/ece604f20/Lecture%20Notes/Lect21.pdf"
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def command(args):
    env = "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=$PWD"
    return env + " " + shlex.join([sys.executable, str(Path(__file__).relative_to(ROOT)), *args])


def clean_json(value):
    """Use JSON null for undefined numbers, strings for +/- infinity."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def tag(case, ndev, k):
    return f"{case}_single" if ndev == 1 else f"{case}_d{ndev}_K{k}"


def configs(case):
    return [(1, 1)] + [(d, k) for d in CASES[case][2] for k in KS]


def db20(value):
    return float(20 * np.log10(value)) if value > 0 else -math.inf


def trace_metrics(trace, reference, single, dt):
    # Subtract and square in float64; the stored solver samples stay float32.
    a, b, s = (np.asarray(x, dtype=np.float64) for x in (trace, reference, single))
    diff = a - b
    peak = float(np.max(np.abs(s)))
    rms = float(np.sqrt(np.mean(s * s)))
    finite = bool(np.isfinite(a).all())
    diff_finite = bool(np.isfinite(diff).all())
    hits = np.flatnonzero(np.abs(diff) > 1e-3 * peak)
    nonfinite = np.flatnonzero(~np.isfinite(a))
    max_diff = float(np.max(np.abs(diff))) if diff_finite else None
    rms_diff = float(np.sqrt(np.mean(diff * diff))) if diff_finite else None
    ratio = max_diff / peak if max_diff is not None and peak else None
    return {
        "single_peak_V_per_m": peak, "single_rms_V_per_m": rms,
        "max_abs_diff_V_per_m": max_diff,
        "max_abs_diff_over_single_peak": ratio,
        "max_abs_diff_over_single_peak_dB": db20(ratio) if ratio is not None else None,
        "rms_diff_V_per_m": rms_diff,
        "rms_diff_over_single_rms": rms_diff / rms if rms_diff is not None and rms else None,
        "threshold_V_per_m": 1e-3 * peak,
        "first_exceeds_step_index_0based": int(hits[0]) if hits.size else None,
        "first_exceeds_completed_step_1based": int(hits[0] + 1) if hits.size else None,
        "first_exceeds_time_s": float(hits[0] * dt) if hits.size else None,
        "trace_finite": finite,
        "first_nonfinite_step_index_0based": int(nonfinite[0]) if nonfinite.size else None,
    }


def analytic_modes():
    """TE/TM relative to z; m,n,p refer to x,y,z respectively."""
    limits = [math.ceil(2 * BAND_HZ[1] * length / C0_M_S) for length in DOMAIN_M]
    modes = []
    for m in range(limits[0] + 1):
        for n in range(limits[1] + 1):
            for p in range(limits[2] + 1):
                f = C0_M_S / 2 * math.sqrt(sum(
                    (v / length) ** 2 for v, length in zip((m, n, p), DOMAIN_M)))
                if not BAND_HZ[0] <= f <= BAND_HZ[1]:
                    continue
                families = []
                if p >= 1 and (m > 0 or n > 0):
                    families.append("TE")
                if m >= 1 and n >= 1:
                    families.append("TM")
                for family in families:
                    coupling = 0.0
                    if family == "TM":
                        def ez(pos):
                            x, y, z = pos
                            a, b, c = DOMAIN_M
                            return (math.sin(m * math.pi * x / a)
                                    * math.sin(n * math.pi * y / b)
                                    * math.cos(p * math.pi * z / c))
                        coupling = ez(SOURCE_M) * ez((0.036, 0.008, 0.008))
                    modes.append({"family_to_z": family, "m": m, "n": n, "p": p,
                                  "label": f"{family}({m},{n},{p})", "frequency_Hz": f,
                                  "ideal_Ez_source_probe_product_dimensionless": coupling,
                                  "ideal_Ez_source_probe_nonzero": abs(coupling) > 1e-12})
    return sorted(modes, key=lambda row: (row["frequency_Hz"], row["label"]))


def spectrum(trace, dt):
    """Unwindowed, unpadded |rFFT|, with parabolic interpolation of |FFT|."""
    from scipy.signal import find_peaks

    values = np.asarray(trace, dtype=np.float64)
    freq = np.fft.rfftfreq(len(values), dt)
    if not np.isfinite(values).all():
        return freq, np.full(freq.shape, np.nan), {
            "status": "nonfinite_trace", "lowest_three_peaks": [], "all_local_maxima_in_band": []}
    mag = np.abs(np.fft.rfft(values))
    in_band = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
    # No amplitude/prominence cutoff: retain every local maximum in the band.
    # The report exposes every candidate and the raw spectrum, including any
    # ambiguity between spectral maxima and physical cavity eigenmodes.
    bins, props = find_peaks(mag, prominence=0)
    candidates = []
    for j, i in enumerate(bins):
        if not in_band[i]:
            continue
        left, center, right = mag[i - 1:i + 2]
        denom = left - 2 * center + right
        offset = 0.5 * (left - right) / denom if denom else 0.0
        interpolated_frequency = float((i + offset) / (len(values) * dt))
        if not BAND_HZ[0] <= interpolated_frequency <= BAND_HZ[1]:
            continue
        candidates.append({
            "fft_bin": int(i), "bin_offset": float(offset),
            "frequency_Hz": interpolated_frequency,
            "magnitude_V_per_m": float(center - 0.25 * (left - right) * offset),
            "bin_magnitude_V_per_m": float(center),
            "prominence_V_per_m": float(props["prominences"][j]),
        })
    return freq, mag, {"status": "ok" if len(candidates) >= 3 else "fewer_than_three_peaks",
                       "lowest_three_peaks": candidates[:3],
                       "all_local_maxima_in_band": candidates}


def measure(case, traces, manifest, out, device_count=None):
    import jax
    import rfx
    from rfx import GaussianPulse, Simulation

    jax.config.update("jax_enable_x64", False)
    devices = jax.devices()
    if len(devices) != 4 or any(d.platform != "cpu" for d in devices):
        raise RuntimeError(f"Expected four CPU devices, got {devices}")
    if Path(rfx.__file__).resolve().parent != ROOT / "rfx":
        raise RuntimeError(f"Imported rfx outside worktree: {rfx.__file__}")
    n_steps, xs, _ = CASES[case]
    sim = Simulation(freq_max=BAND_HZ[1], domain=DOMAIN_M, dx=DX_M,
                     boundary="pec", cpml_layers=0, precision="float32")
    sim.add_source(SOURCE_M, "ez", waveform=GaussianPulse(f0=8e9, bandwidth=0.5),
                   amplitude_kind="field")
    for x in xs:
        sim.add_probe((x * 1e-3, 0.008, 0.008), "ez")
    manifest["environment"].update({
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_enable_x64": jax.config.x64_enabled,
        "jax_devices": [str(d) for d in devices], "rfx_file": rfx.__file__,
    })
    configurations = configs(case)
    if device_count is not None:
        configurations = [(d, k) for d, k in configurations if d == device_count]
    for ndev, k in configurations:
        name = tag(case, ndev, k)
        manifest["runs"][name] = {"case": case, "device_count": ndev,
                                  "exchange_interval": k, "n_steps": n_steps,
                                  "probe_x_mm": xs, "timings_s": [], "warnings": [],
                                  "status": "pending"}
    # One full-length warmup per configuration; five timed repeat rounds.
    # Rotate the order by one configuration each round to expose ordering.
    for repeat in range(-1, REPEATS):
        ordered = configurations if repeat == -1 else (
            configurations[repeat % len(configurations):]
            + configurations[:repeat % len(configurations)])
        for ndev, k in ordered:
            name = tag(case, ndev, k)
            row = manifest["runs"][name]
            if row["status"] == "failed":
                continue
            phase = "warmup" if repeat == -1 else f"repeat{repeat + 1}"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    start = time.perf_counter()
                    # K > 1 is refused at every public entry since this
                    # measurement was taken (the refusal is what it supports).
                    # Bypass only the entry validation here, as the always-on
                    # growth test does; the production scan and its one-cell
                    # ghost exchange are untouched.
                    with _validation_bypassed(k):
                        result = sim.run(n_steps=n_steps, skip_preflight=True,
                                         devices=devices[:ndev] if ndev > 1 else None,
                                         exchange_interval=k)
                    jax.block_until_ready((result.state, result.time_series))
                    values = np.asarray(result.time_series).copy()
                    elapsed = time.perf_counter() - start
                if values.shape != (n_steps, len(xs)) or values.dtype != np.float32:
                    raise RuntimeError(f"Unexpected output: {values.shape}, {values.dtype}")
                traces[f"{name}_{phase}"] = values
                if repeat == 0:
                    traces[name] = values
                row["warnings"] = sorted(set(row["warnings"]) | {str(w.message) for w in caught})
                row.update({"dt_s": float(result.dt), "grid_shape_nodes": list(result.grid.shape),
                            "field_dtype": str(result.state.ez.dtype),
                            "trace_dtype": str(values.dtype),
                            "probe_indices": [list(result.grid.position_to_index((x * 1e-3, .008, .008)))
                                              for x in xs],
                            "source_index": list(result.grid.position_to_index(SOURCE_M)),
                            "status": "complete" if repeat == REPEATS - 1 else "running"})
                nx_per = math.ceil(result.grid.shape[0] / ndev)
                row["x_padding_nodes"] = nx_per * ndev - result.grid.shape[0]
                row["seam_first_owned_x_mm"] = [d * nx_per * DX_M * 1e3 for d in range(1, ndev)]
                if repeat == -1:
                    row["warmup_s"] = elapsed
                else:
                    row["timings_s"].append(elapsed)
                manifest["execution_order"].append({"run": name, "phase": phase, "wall_s": elapsed})
                print(f"{name} {phase}: {elapsed:.6f} s; finite={np.isfinite(values).all()}", flush=True)
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                print(f"{name} {phase}: {row['error']}", flush=True)
    for ndev, k in configurations:
        name = tag(case, ndev, k)
        row = manifest["runs"][name]
        if row["timings_s"]:
            row["median_wall_s"] = float(np.median(row["timings_s"]))
        if name in traces:
            canonical = traces[name]
            row["repeat_max_abs_diff_from_repeat1_V_per_m"] = [
                float(np.max(np.abs(traces[f"{name}_repeat{i}"].astype(np.float64)
                                    - canonical.astype(np.float64))))
                for i in range(1, len(row["timings_s"]) + 1)]
    np.savez_compressed(out / "traces.npz", **traces)
    write_json(out / "run_manifest.json", manifest)


def analyze(traces, manifest, out):
    results = {"metadata": manifest, "measurements": {}, "summary": [],
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "analytic_modes": analytic_modes(), "did_not_run": [], "unavailable_spectral_results": [],
               "spectrum_method": {
                   "band_Hz": list(BAND_HZ), "fft": "abs(numpy.fft.rfft(trace.astype(float64)))",
                   "window": "rectangular (all samples)", "zero_padding": False,
                   "detrend": False, "amplitude_normalization": "none",
                   "peak_rule": "lowest three local maxima in 8-16 GHz; no prominence cutoff",
                   "interpolation": "parabola on three linear-magnitude bins",
                   "comparison_pairing": "ascending frequency rank, not asserted mode identity",
                   "analytic_source_url": ANALYTIC_SOURCE,
               }}
    for case, (steps, xs, _) in CASES.items():
        refname = tag(case, 1, 1)
        if refname not in traces:
            results["did_not_run"].append(f"{case}: no single-device trace")
            continue
        single = traces[refname]
        cm = {"n_steps": steps, "probe_x_mm": xs, "runs": {}}
        results["measurements"][case] = cm
        dt = manifest["runs"][refname]["dt_s"]
        cm.update({"dt_s": dt, "record_length_s": steps * dt,
                   "last_sample_time_s": (steps - 1) * dt})
        traces[f"{case}_time_s"] = np.arange(steps, dtype=np.float64) * dt
        for ndev, k in configs(case):
            name = tag(case, ndev, k)
            if name not in traces:
                results["did_not_run"].append(f"{name}: no trace")
                continue
            run = dict(manifest["runs"][name])
            cm["runs"][name] = run
            if run["status"] != "complete":
                results["did_not_run"].append(f"{name}: {run['status']}")
            if ndev > 1:
                references = {"vs_single": refname, "vs_K1": tag(case, ndev, 1)}
                run["comparisons"] = {}
                for comp, reference_name in references.items():
                    if reference_name not in traces:
                        continue
                    run["comparisons"][comp] = [
                        {"probe_x_mm": x, **trace_metrics(traces[name][:, j], traces[reference_name][:, j],
                                                         single[:, j], dt)}
                        for j, x in enumerate(xs)]
                run["timing_ratio_to_single"] = run["median_wall_s"] / manifest["runs"][refname]["median_wall_s"]
                run["timing_ratio_to_K1"] = run["median_wall_s"] / manifest["runs"][tag(case, ndev, 1)]["median_wall_s"]
            if case == "M3":
                freq, mag, spec = spectrum(traces[name][:, 0], dt)
                traces["M3_frequency_Hz"] = freq
                traces[f"{name}_fft_magnitude_V_per_m"] = mag
                run["spectrum"] = spec
                if len(spec["lowest_three_peaks"]) < 3:
                    results["unavailable_spectral_results"].append(
                        f"{name}: {len(spec['lowest_three_peaks'])} local maxima in 8-16 GHz; "
                        "missing peak frequencies and their shifts/magnitude changes are undefined")
                run["spectrum"]["fft_bin_spacing_Hz"] = 1 / (steps * dt)
                for peak in spec["lowest_three_peaks"]:
                    nearby = sorted(results["analytic_modes"], key=lambda m: abs(m["frequency_Hz"] - peak["frequency_Hz"]))
                    closest_f = nearby[0]["frequency_Hz"]
                    peak["nearest_analytic_modes"] = [m["label"] for m in nearby if abs(m["frequency_Hz"] - closest_f) < 1.0]
                    peak["nearest_analytic_frequency_Hz"] = closest_f
                    peak["offset_from_nearest_analytic_percent"] = 100 * (peak["frequency_Hz"] / closest_f - 1)
        if case == "M3":
            for name, run in cm["runs"].items():
                if run["device_count"] == 1:
                    continue
                run["spectral_comparisons"] = {}
                for comp, ref in (("vs_single", refname), ("vs_K1", tag(case, run["device_count"], 1))):
                    peaks = run["spectrum"]["lowest_three_peaks"]
                    reference = cm["runs"][ref]["spectrum"]["lowest_three_peaks"]
                    run["spectral_comparisons"][comp] = [
                        {"peak_rank": i + 1,
                         "frequency_shift_percent": 100 * (p["frequency_Hz"] / q["frequency_Hz"] - 1),
                         "peak_magnitude_change_dB": db20(p["magnitude_V_per_m"] / q["magnitude_V_per_m"])}
                        for i, (p, q) in enumerate(zip(peaks, reference))]
        for name, run in cm["runs"].items():
            if run["device_count"] == 1:
                continue
            for comp, rows in run["comparisons"].items():
                metrics = [r["max_abs_diff_over_single_peak_dB"] for r in rows]
                spectral = run.get("spectral_comparisons", {}).get(comp, [])
                results["summary"].append({
                    "case": case, "device_count": run["device_count"],
                    "exchange_interval": run["exchange_interval"], "comparison": comp,
                    "worst_max_abs_diff_over_single_peak_dB": max(metrics) if None not in metrics else None,
                    "worst_abs_frequency_shift_percent": max((abs(r["frequency_shift_percent"]) for r in spectral), default=None),
                    "worst_abs_peak_magnitude_change_dB": max((abs(r["peak_magnitude_change_dB"]) for r in spectral), default=None),
                    "median_wall_s": run["median_wall_s"],
                    "timing_ratio_to_single": run["timing_ratio_to_single"],
                    "timing_ratio_to_K1": run["timing_ratio_to_K1"],
                })
    np.savez_compressed(out / "traces.npz", **traces)
    write_json(out / "results.json", results)
    make_figures(results, traces, out)
    write_report(results, out)
    return results


def fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, str):
        return {"-Infinity": "−∞", "Infinity": "+∞"}.get(value, value)
    if isinstance(value, bool):
        return str(value).lower()
    if value == -math.inf:
        return "−∞"
    return f"{value:.{digits}g}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def write_report(results, out):
    meta = results["metadata"]
    parts = ["# Exchange interval: PEC CPU measurements", "",
             f"UTC: {meta['created_utc']}. Branch `{meta['branch']}`; HEAD `{meta['head']}`; origin/main `{meta['origin_main']}`.",
             "", "## Reproduction", "", "```sh", f"cd {shlex.quote(str(ROOT))}",
             *meta["commands"], "```", "",
             "These are the recorded measurement invocations. The final script's first command alone "
             "runs all 18 configurations; the second command appended the 4-device M3 batch to the "
             "initial measurements.", "",
             "Analysis-only command:", "", "```sh", command(["--analyze-only", "--output", str(out)]), "```", "",
             "## Configuration and definitions", "",
             "PEC vacuum box: 48 × 16 × 16 mm; dx=1 mm; float32 Yee fields; "
             "`GaussianPulse(f0=8e9, bandwidth=0.5, amplitude=1.0, cutoff=3.0)`; "
             "soft Ez source at (10,8,8) mm, `amplitude_kind='field'` (the PEC legacy field-increment convention). "
             "`freq_max=16e9`, `cpml_layers=0`, `skip_preflight=True`. No solver or test changes.", "",
             "M1: 400 steps, x=12,24,26,36 mm probes, 2 and 4 devices. "
             "M2: 800 steps, the same four probes, 2 devices. "
             "M3: 2000 steps, x=36 mm probe, 2 and 4 devices. All probes have y=z=8 mm. "
             "Each measurement batch reuses one Simulation object; all configurations use the same fixture. "
             "The 4-device M3 batch was run separately after the initial M1/M2/2-device-M3 batch.", "",
             "The realized grid is 49 × 17 × 17 nodes (48 × 16 × 16 intervals). "
             "Two devices use 25 x-nodes per padded slab, one padding node, seam at x=25 mm. "
             "Four devices use 13 x-nodes per padded slab, three padding nodes, seams at x=13,26,39 mm. "
             "Ghost width is 1 node. The requested x=24 mm probe is retained; its realized index is 24, "
             "adjacent to the 2-device seam. These coordinates were not moved.", "",
             "Trace comparisons use the first timed repeat. For either reference, "
             "max ratio = max(abs(distributed−reference))/max(abs(single)); dB = 20 log10(max ratio). "
             "RMS ratio = RMS(distributed−reference)/RMS(single). Both denominators use the single trace at that probe. "
             "Subtractions and reductions use float64; stored samples are float32. "
             "The crossing is the first zero-based source/scan step index with abs(diff)>0.001×single peak; "
             "time=index×dt. Samples are recorded after that step's field update. "
             "JSON also records the one-based completed step. ‘—’ means no crossing/not applicable/undefined; "
             "−∞ dB denotes an exactly zero ratio. JSON uses strings for infinities and null for undefined values.", "",
             "Timings are synchronized public `sim.run` wall time plus transfer/copy of the probe trace, "
             "including setup and any tracing/compilation performed by that call. "
             "All returned field arrays and traces are blocked before stopping the timer. "
             "One full-length warmup per configuration precedes five timed repeats; configuration order rotates "
             "by one each round. No concurrent simulation processes are used. "
             "Ratios are t(K)/t(K=1) and t(distributed)/t(single); these are elapsed-time ratios. "
             "CPU timings do not transfer to GPU.", "",
             f"Environment: `{json.dumps(meta['environment'], sort_keys=True)}`", "",
             "## Summary rows", "",
             table(["M", "devices", "K", "reference", "worst max ratio (dB)", "max |peak shift| (%)",
                    "max |peak change| (dB)", "median (s)", "t/tK1", "t/tsingle"],
                   [[r['case'], r['device_count'], r['exchange_interval'], r['comparison'],
                     fmt(r['worst_max_abs_diff_over_single_peak_dB']), fmt(r['worst_abs_frequency_shift_percent']),
                     fmt(r['worst_abs_peak_magnitude_change_dB']), fmt(r['median_wall_s']),
                     fmt(r['timing_ratio_to_K1']), fmt(r['timing_ratio_to_single'])] for r in results['summary']]), ""]
    for case, cm in results["measurements"].items():
        parts += [f"## {case}: traces and timings", "",
                  f"dt={cm['dt_s']:.12g} s; record length={cm['record_length_s'] * 1e9:.9g} ns; "
                  f"last sample time={cm['last_sample_time_s'] * 1e9:.9g} ns.", "",
                  table(["run", "warmup (s)", "five repeats (s)", "median (s)", "max repeat difference (V/m)"],
                        [[name, fmt(r.get('warmup_s')), ', '.join(fmt(t) for t in r['timings_s']),
                          fmt(r.get('median_wall_s')), fmt(max(r.get('repeat_max_abs_diff_from_repeat1_V_per_m', [math.nan])))]
                         for name, r in cm['runs'].items()]), ""]
        rows = []
        for name, run in cm["runs"].items():
            for comp, probes in run.get("comparisons", {}).items():
                for p in probes:
                    rows.append([name, comp, p['probe_x_mm'], fmt(p['single_peak_V_per_m']),
                                 fmt(p['max_abs_diff_over_single_peak']), fmt(p['max_abs_diff_over_single_peak_dB']),
                                 fmt(p['rms_diff_over_single_rms']), fmt(p['first_exceeds_step_index_0based']),
                                 fmt(p['first_exceeds_time_s'] * 1e9 if p['first_exceeds_time_s'] is not None else None),
                                 fmt(p['trace_finite'])])
        parts += [table(["run", "reference", "probe x (mm)", "single peak (V/m)", "max ratio (1)",
                         "max ratio (dB)", "RMS ratio (1)", "first step (0-based)", "first time (ns)", "finite"], rows),
                  "", f"![{case} traces and differences]({case}.png)", ""]
    if "M3" in results["measurements"]:
        parts += ["## M3: FFT peaks and analytic frequencies", "",
                  "Spectrum: abs(rFFT(Ez)), all 2000 samples, no detrending, no taper, no zero-padding, "
                  "no amplitude normalization. FFT magnitude units are V/m (the unnormalized sum of field samples). "
                  "Search band is explicitly 8–16 GHz. All local maxima in the band are retained in JSON; "
                  "the lowest three are tabulated. No prominence threshold is imposed. "
                  "At bin i, δ=(A[i−1]−A[i+1])/[2(A[i−1]−2A[i]+A[i+1])]; "
                  "f=(i+δ)/(N dt), Apeak=A[i]−(A[i−1]−A[i+1])δ/4. "
                  "Comparisons pair peaks by ascending frequency rank; this does not establish physical mode identity. "
                  "The nearest analytic frequency is a numeric nearest-neighbor annotation, not a mode assignment.", "",
                  "Analytic PEC frequencies: f=(c/2)√[(m/Lx)²+(n/Ly)²+(p/Lz)²], c=299792458 m/s. "
                  "TE/TM are defined relative to z: TE has m,n≥0, m+n>0, p≥1; TM has m,n≥1, p≥0. "
                  "TE has Ez=0. For TM the Ez spatial factor is sin(mπx/Lx)sin(nπy/Ly)cos(pπz/Lz). "
                  f"See [Purdue, Cavity Resonators, §21.2]({ANALYTIC_SOURCE}).", ""]
        peak_rows, comparison_rows = [], []
        for name, r in results["measurements"]["M3"]["runs"].items():
            for i, p in enumerate(r["spectrum"]["lowest_three_peaks"], 1):
                peak_rows.append([name, i, fmt(p['frequency_Hz'] / 1e9, 9), fmt(p['magnitude_V_per_m'], 9),
                                  ', '.join(p['nearest_analytic_modes']), fmt(p['nearest_analytic_frequency_Hz'] / 1e9, 9),
                                  fmt(p['offset_from_nearest_analytic_percent'])])
            for comp, rows in r.get("spectral_comparisons", {}).items():
                comparison_rows += [[name, comp, p['peak_rank'], fmt(p['frequency_shift_percent']),
                                     fmt(p['peak_magnitude_change_dB'])] for p in rows]
        parts += [table(["run", "peak rank", "measured (GHz)", "|FFT| (V/m)", "nearest analytic modes",
                         "analytic (GHz)", "offset from analytic (%)"], peak_rows), "",
                  table(["run", "reference", "peak rank", "frequency shift (%)", "peak magnitude change (dB)"], comparison_rows), "",
                  "All analytic TE/TM modes in 8–16 GHz:", "",
                  table(["mode (x,y,z indices)", "frequency (GHz)", "ideal Ez source×probe factor (1)"],
                        [[m['label'], fmt(m['frequency_Hz'] / 1e9, 9),
                          fmt(m['ideal_Ez_source_probe_product_dimensionless'])] for m in results['analytic_modes']]), ""]
    parts += ["## Files and completion", "",
              f"- Script: `{Path(__file__).resolve()}` (uncommitted).",
              f"- All trace samples and FFT magnitudes: `{out / 'traces.npz'}`.",
              "  Keys M1/M2/M3_single and M1/M2/M3_dN_KK hold first timed repeats; "
              "suffixes _warmup and _repeat1 through _repeat5 hold every run. "
              "_time_s arrays use index×dt; M3_frequency_Hz and _fft_magnitude_V_per_m hold spectra.",
              f"- Full metrics: `{out / 'results.json'}`; run provenance: `{out / 'run_manifest.json'}`.",
              *[f"- Figure: [{m}.png]({m}.png)." for m in results['measurements']],
              f"- Report: `{out / 'REPORT.md'}`.",
              "- Did not run: " + ('; '.join(results['did_not_run']) or 'none.'),
              "- Unavailable spectral results: " + ('; '.join(results['unavailable_spectral_results']) or 'none.'),
              f"- Longest individual warmup/timed call: {max((x['wall_s'] for x in meta['execution_order']), default=0):.6f} s.",
              "- Runner warning text, if emitted, is preserved verbatim in JSON as provenance.", ""]
    (out / "REPORT.md").write_text("\n".join(parts))


def make_figures(results, traces, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def thin_zero_ticks(ax):
        # A large symmetric-log range can place +/-linthresh labels on top
        # of the zero label. Remove those labels, keeping all plotted samples.
        transform = ax.yaxis.get_transform().transform
        low, high = ax.get_ylim()
        span = float(transform(np.array([high]))[0] - transform(np.array([low]))[0])
        zero = float(transform(np.array([0.0]))[0])
        ticks = [v for v in ax.get_yticks() if low <= v <= high and
                 (v == 0 or abs(float(transform(np.array([v]))[0]) - zero) / span >= 0.07)]
        ax.set_yticks(ticks)

    colors = {"single": "black", 1: "#0072B2", 2: "#D55E00", 4: "#009E73"}
    for case, cm in results["measurements"].items():
        devices = CASES[case][2]
        nrows = len(cm['probe_x_mm']) * len(devices)
        if case == "M3":
            nrows += 1
        fig, axes = plt.subplots(nrows, 3, figsize=(17, 2.6 * nrows), squeeze=False, constrained_layout=True)
        t = traces[f"{case}_time_s"] * 1e9
        single = traces[tag(case, 1, 1)]
        for dindex, ndev in enumerate(devices):
            k1 = traces.get(tag(case, ndev, 1))
            if k1 is None:
                continue
            for j, x in enumerate(cm["probe_x_mm"]):
                axrow = axes[dindex * len(cm['probe_x_mm']) + j]
                peak = np.max(np.abs(single[:, j]))
                axrow[0].plot(t, single[:, j], color=colors['single'], lw=1, label='single')
                for k in KS:
                    name = tag(case, ndev, k)
                    if name not in traces:
                        continue
                    values = traces[name][:, j].astype(np.float64)
                    axrow[0].plot(t, values, lw=.85, color=colors[k], alpha=.8, label=f'K={k}')
                    axrow[1].plot(t, (values - single[:, j]) / peak, lw=.85, color=colors[k], label=f'K={k}')
                    axrow[2].plot(t, (values - k1[:, j]) / peak, lw=.85, color=colors[k], label=f'K={k}')
                for col, ax in enumerate(axrow):
                    ax.set_title(f'{ndev} devices; x={x} mm; ' + ['Ez traces', 'difference vs single', 'difference vs K=1'][col])
                    ax.set_xlabel('Time (ns)')
                    ax.set_ylabel('Ez (V/m)' if col == 0 else 'ΔEz / single peak (1)')
                    ax.set_yscale('symlog', linthresh=max(float(peak) * 1e-4, 1e-30) if col == 0 else 1e-4)
                    ax.yaxis.get_major_locator().set_params(numticks=7)
                    thin_zero_ticks(ax)
                    ax.grid(True, alpha=.25)
                    ax.legend(fontsize=8, loc='best', ncol=2)
        if case == 'M3':
            freq = traces['M3_frequency_Hz'] / 1e9
            reference = traces['M3_single_fft_magnitude_V_per_m']
            for name, r in cm['runs'].items():
                key = f'{name}_fft_magnitude_V_per_m'
                mag = traces[key]
                label = 'single' if r['device_count'] == 1 else f"{r['device_count']} dev K={r['exchange_interval']}"
                color = colors['single' if r['device_count'] == 1 else r['exchange_interval']]
                linestyle = '--' if r['device_count'] == 4 else '-'
                axes[-1, 0].plot(freq, mag, label=label, color=color, lw=1, ls=linestyle)
                for p in r['spectrum']['lowest_three_peaks']:
                    axes[-1, 0].plot(p['frequency_Hz'] / 1e9, p['magnitude_V_per_m'], 'o', color=color, ms=4)
                if r['device_count'] > 1:
                    axes[-1, 1].plot(freq, (mag - reference) / np.max(reference), label=label, color=color, ls=linestyle)
                    k1mag = traces[f"M3_d{r['device_count']}_K1_fft_magnitude_V_per_m"]
                    axes[-1, 2].plot(freq, (mag - k1mag) / np.max(reference), label=label, color=color, ls=linestyle)
            for col, ax in enumerate(axes[-1]):
                ax.set_xlim(*(f / 1e9 for f in BAND_HZ))
                ax.set_xlabel('Frequency (GHz)')
                ax.set_ylabel('|FFT| (V/m)' if col == 0 else 'Normalized Δ|FFT| (1)')
                ax.set_title(['Raw FFT and selected peaks', 'FFT magnitude difference vs single', 'FFT magnitude difference vs K=1'][col])
                ax.set_yscale('log' if col == 0 else 'symlog', **({} if col == 0 else {'linthresh': 1e-4}))
                ax.yaxis.get_major_locator().set_params(numticks=7)
                if col > 0:
                    thin_zero_ticks(ax)
                ax.grid(True, alpha=.25)
                ax.legend(fontsize=8)
            for f in sorted({m['frequency_Hz'] for m in results['analytic_modes']}):
                axes[-1, 0].axvline(f / 1e9, color='gray', alpha=.25, lw=.7)
        fig.suptitle(f'{case}: 48 × 16 × 16 mm PEC; dx=1 mm; {cm["n_steps"]} steps; float32; virtual CPU devices\n'
                     'Time panels use symmetric log scales; difference denominators are single-device peaks', fontsize=13)
        fig.savefig(out / f'{case}.png', dpi=145)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/rfx-K'))
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--device-count', type=int, choices=[1, 2, 4],
                        help='Measure only this device count; preserve other saved configurations.')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    global np
    import numpy as np

    if args.analyze_only or args.case != 'all' or args.device_count is not None:
        manifest = json.loads((out / 'run_manifest.json').read_text()) if (out / 'run_manifest.json').exists() else None
        traces = {}
        if (out / 'traces.npz').exists():
            with np.load(out / 'traces.npz') as data:
                traces = {key: data[key] for key in data.files}
    else:
        manifest, traces = None, {}
    if manifest is None:
        try:
            chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            chip = platform.processor()
        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'branch': git('branch', '--show-current'), 'head': git('rev-parse', 'HEAD'),
            'origin_main': git('rev-parse', 'origin/main'),
            'initial_git_status': git('status', '--short'),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment': {'python': sys.version, 'executable': sys.executable,
                            'platform': platform.platform(), 'cpu': chip, 'logical_cpu_count': os.cpu_count(),
                            'JAX_PLATFORMS': os.environ.get('JAX_PLATFORMS'), 'XLA_FLAGS': os.environ.get('XLA_FLAGS'),
                            'PYTHONPATH': os.environ.get('PYTHONPATH')},
            'commands': [], 'runs': {}, 'execution_order': [],
        }
    if not args.analyze_only:
        manifest['commands'].append(command(sys.argv[1:]))
        for case in CASES if args.case == 'all' else [args.case]:
            measure(case, traces, manifest, out, args.device_count)
    results = analyze(traces, manifest, out)
    print(f"Saved {out / 'REPORT.md'}; {out / 'results.json'}; {out / 'traces.npz'}", flush=True)
    print('Did not run: ' + ('; '.join(results['did_not_run']) or 'none'), flush=True)


if __name__ == '__main__':
    main()
