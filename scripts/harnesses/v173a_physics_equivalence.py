"""V173A aligned patch-shaped composition sentinel (#931 fixture repair).

This is a synthetic source/material/sheet/CPML/Harminv regression, not an
antenna matching or accuracy test. The strongest probe mode has no spatial
mode label. The Hann-windowed FFT minimum is an unnormalised probe-spectrum
statistic, not S11. Physical patch mode coverage lives in
``tests/locks/test_patch_edgefed_resonance_harminv.py`` and mode-labelled far-field/reference regression
coverage in ``tests/crossval/test_patch_canonical_farfield_e4.py``.

The review's finite-board antenna proposal is deliberately not adopted here:
retain the 50 x 50 x 20 mm truncated uniform-laminate/infinite-ground model,
but represent nominal 35 um copper by a sheet at the 1.5 mm laminate face.
The 28 x 36 mm foil and substrate are fixed physical dimensions. dx=0.5 mm
places ALL faces on nodes and resolves the substrate with three cells. This
is a buildable etched footprint with explicitly idealized ground and feed;
it makes no finite-board, connector, loss or mesh-convergence claim.

Run from this checkout with PYTHONPATH=$PWD JAX_PLATFORMS=cpu. A capture is
written only to --output; this harness never updates the committed baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

_DX = 0.5e-3
_F_MAX = 4e9
_DOMAIN = (0.05, 0.05, 0.02)
_FR4_EPS = 4.3
_SUB_T = 0.0015
_PATCH_X = 0.028
_PATCH_Y = 0.036
_SRC_POS = (0.025, 0.018, 0.0005)
_PROBE_POS = (0.025, 0.018, 0.001)
_WAVEFORM = GaussianPulse(f0=_F_MAX / 2, bandwidth=0.8)
_N_STEPS = 6000  # same physical duration as the retired 1 mm / 3000 record


def _build_sim(*, dx=_DX) -> Simulation:
    sim = Simulation(
        freq_max=_F_MAX, domain=_DOMAIN, dx=dx,
        boundary="cpml", cpml_layers=round(0.004 / dx), pec_faces={"z_lo"},
    )
    sim.add_material("fr4", eps_r=_FR4_EPS)
    sim.add(Box((0., 0., 0.), (_DOMAIN[0], _DOMAIN[1], _SUB_T)), material="fr4")
    lo = ((_DOMAIN[0] - _PATCH_X) / 2, (_DOMAIN[1] - _PATCH_Y) / 2, _SUB_T)
    hi = (lo[0] + _PATCH_X, lo[1] + _PATCH_Y, _SUB_T)
    sim.add_thin_conductor(Box(lo, hi), sigma_bulk=5.8e7, thickness=35e-6)
    # Both Ez observations are live vertical gap edges below the foil.
    sim.add_source(_SRC_POS, "ez", waveform=_WAVEFORM, amplitude_kind="current")
    sim.add_probe(_PROBE_POS, "ez")
    return sim


def _probe_spectrum_min(ts, dt):
    """Return an unnormalised FFT log-amplitude minimum and its frequency."""
    freqs = np.fft.rfftfreq(len(ts), dt)
    spectrum = np.abs(np.fft.rfft(ts * np.hanning(len(ts))))
    band = (freqs >= 1.5e9) & (freqs <= 3.5e9)
    f_band, s_band = freqs[band], spectrum[band]
    if not np.all(np.isfinite(s_band)) or not np.max(s_band) > 0:
        raise ValueError("probe spectrum must be finite and nonzero")
    db = 20 * np.log10(s_band + float(np.max(s_band)) * 1e-6)
    idx = int(np.argmin(db))
    return float(db[idx]), float(f_band[idx])


def _measure(ts, dt):
    if not np.all(np.isfinite(ts)) or not np.max(np.abs(ts)) > 0:
        raise ValueError("probe trace must be finite and nonzero")
    # A pole fit measures free ring-down, not the driven pulse. Start six
    # Gaussian widths past its centre: |I| <= 12 exp(-36) = 2.8e-15 A,
    # independently of the observed resonance. Remove the static DC remnant.
    start_time = _WAVEFORM.t0 + 6 * _WAVEFORM.tau
    start = int(np.ceil(start_time / dt))
    if len(ts) - start < 20:
        raise ValueError("record ends before the source-free ring-down window")
    ringdown = ts[start:] - np.mean(ts[start:])
    modes = harminv(ringdown, dt, f_min=1.5e9, f_max=3.5e9, min_Q=5., decimate=False)
    if not modes:
        raise ValueError("no observable probe mode in the declared band")
    strongest = max(modes, key=lambda m: m.amplitude)
    db, freq = _probe_spectrum_min(ts, dt)
    return {
        "dominant_probe_mode_hz": float(strongest.freq),
        "probe_spectrum_min_db": db,
        "probe_spectrum_min_f_hz": freq,
        "modes": [m._asdict() for m in modes],
        "ringdown_start_s": start_time, "ringdown_start_sample": start,
        "source_tail_bound_A": float(12 * np.exp(-36)),
        "tail_to_peak": float(np.max(np.abs(ts[-len(ts)//10:])) / np.max(np.abs(ts))),
        "ringdown_tail_to_peak": float(np.max(np.abs(ringdown[-len(ringdown)//10:])) / np.max(np.abs(ringdown))),
    }


def run_capture(*, n_steps=_N_STEPS, dx=_DX, trace_output=None, replay_trace=None):
    root = Path(__file__).resolve().parents[2]
    rfx_import = str(Path(__import__("rfx").__file__).resolve())
    assert Path(rfx_import).is_relative_to(root / "rfx"), rfx_import
    if replay_trace is None:
        sim = _build_sim(dx=dx)
        result = sim.run(n_steps=n_steps)
        ts, dt = np.asarray(result.time_series)[:, 0], float(result.dt)
    else:
        # Explicit offline instrument analysis, not an independent solver run.
        record = np.load(replay_trace)
        ts, dt = record["trace"], float(record["dt"])
        if len(ts) != n_steps:
            raise ValueError("replayed trace length differs from declared n_steps")
    if trace_output is not None:
        Path(trace_output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_output, trace=ts, dt=dt)
    payload = _measure(ts, dt)
    payload.update({
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rfx_import": rfx_import,
        "jax_version": jax.__version__, "backend": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled), "dt": dt, "n_steps": n_steps,
        "fixture": {"dx_m": dx, "domain_m": list(_DOMAIN), "substrate_height_m": _SUB_T,
                    "eps_r": _FR4_EPS, "patch_size_m": [_PATCH_X, _PATCH_Y],
                    "sheet_z_m": _SUB_T, "source_m": list(_SRC_POS), "probe_m": list(_PROBE_POS),
                    "cpml_thickness_m": 0.004, "ground": "z_lo PEC",
                    "source_amplitude_kind": "current",
                    "waveform": {"f0_hz": _WAVEFORM.f0, "bandwidth": _WAVEFORM.bandwidth,
                                 "cutoff": _WAVEFORM.cutoff, "amplitude_A": _WAVEFORM.amplitude}},
    })
    if replay_trace is not None:
        payload["replayed_trace"] = str(replay_trace)
        payload["replayed_trace_sha256"] = hashlib.sha256(Path(replay_trace).read_bytes()).hexdigest()
    assert Path(payload["rfx_import"]).is_relative_to(root / "rfx"), payload["rfx_import"]
    return payload, ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--replay-trace", type=Path, help="offline extraction only; NOT an independent FDTD capture")
    parser.add_argument("--dx", type=float, default=_DX)
    parser.add_argument("--n-steps", type=int, default=_N_STEPS)
    args = parser.parse_args()
    payload, trace = run_capture(n_steps=args.n_steps, dx=args.dx,
                                 trace_output=args.output.with_suffix(".npz"),
                                 replay_trace=args.replay_trace)
    payload["run_name"] = args.run_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
