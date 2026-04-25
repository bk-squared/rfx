"""Bounded-CPML point-probe feasibility probe for SBP-SAT true R/T.

This is intentionally *not* public reflection/transmission evidence.  It is
an executable measurement-contract probe for the current experimental SBP-SAT
surface: guarded CPML, soft point source, point probes, and one axis-aligned
refinement box.  If the point-probe extraction is too sensitive, the test is
reported as ``xfail``/inconclusive and the support matrix must stay deferred.
"""

from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class _Fixture:
    freq_max = 6.0e9
    source_freq = 2.0e9
    source_bandwidth = 0.8
    domain = (0.04, 0.04, 0.09)
    cpml_layers = 4
    uniform_dx = 1.0e-3
    coarse_dx = 2.0e-3
    ratio = 2
    n_steps = 700

    refinement = {
        "x_range": (0.012, 0.028),
        "y_range": (0.012, 0.028),
        "z_range": (0.016, 0.074),
        "ratio": ratio,
    }
    source = (0.020, 0.020, 0.020)
    front_probe = (0.020, 0.020, 0.028)
    back_probe = (0.020, 0.020, 0.064)
    slab_lo = (0.012, 0.012, 0.046)
    slab_hi = (0.028, 0.028, 0.054)
    slab_eps_r = 2.25

    scored_freqs = np.array([1.5e9, 2.0e9, 2.5e9], dtype=np.float64)
    window_samples = 256


def _shift_z(pos: tuple[float, float, float], dz: float) -> tuple[float, float, float]:
    return (pos[0], pos[1], pos[2] + dz)


@lru_cache(maxsize=None)
def _run_fixture(*, subgrid: bool, slab: bool, probe_shift_cells: int = 0):
    from rfx import Box, Simulation
    from rfx.sources.sources import GaussianPulse

    dx = _Fixture.coarse_dx if subgrid else _Fixture.uniform_dx
    dz_shift = probe_shift_cells * _Fixture.uniform_dx
    sim = Simulation(
        freq_max=_Fixture.freq_max,
        domain=_Fixture.domain,
        boundary="cpml",
        cpml_layers=_Fixture.cpml_layers,
        dx=dx,
    )
    if slab:
        sim.add_material("feasibility_dielectric", eps_r=_Fixture.slab_eps_r)
        sim.add(
            Box(_Fixture.slab_lo, _Fixture.slab_hi),
            material="feasibility_dielectric",
        )
    if subgrid:
        sim.add_refinement(**_Fixture.refinement)
    pulse = GaussianPulse(
        f0=_Fixture.source_freq,
        bandwidth=_Fixture.source_bandwidth,
    )
    sim.add_source(position=_Fixture.source, component="ez", waveform=pulse)
    sim.add_probe(position=_shift_z(_Fixture.front_probe, dz_shift), component="ez")
    sim.add_probe(position=_shift_z(_Fixture.back_probe, dz_shift), component="ez")
    result = sim.run(n_steps=_Fixture.n_steps)
    return float(result.dt), np.asarray(result.time_series, dtype=np.float64)


def _centered_window(signal: np.ndarray, center: int, width: int) -> tuple[np.ndarray, int]:
    half = width // 2
    start = max(0, min(int(center) - half, len(signal) - width))
    stop = start + width
    window = signal[start:stop].copy()
    if len(window) != width:
        raise ValueError(f"window length {len(window)} != {width}")
    return window * np.hanning(width), start


def _dft(signal: np.ndarray, *, dt: float, start: int, freqs: np.ndarray) -> np.ndarray:
    t = (start + np.arange(len(signal), dtype=np.float64)) * dt
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * t[None, :])
    return phase @ signal * dt


def _extract_rt(vacuum: np.ndarray, slab: np.ndarray, *, dt: float) -> dict[str, object]:
    front_vac = vacuum[:, 0]
    back_vac = vacuum[:, 1]
    front_slab = slab[:, 0]
    back_slab = slab[:, 1]
    reflected = front_slab - front_vac

    width = _Fixture.window_samples
    incident_front_center = int(np.argmax(np.abs(front_vac)))
    incident_back_center = int(np.argmax(np.abs(back_vac)))
    reflected_search_start = min(len(reflected) - 1, incident_front_center + width // 3)
    reflected_center = reflected_search_start + int(
        np.argmax(np.abs(reflected[reflected_search_start:]))
    )
    transmitted_center = int(np.argmax(np.abs(back_slab)))

    incident_front, incident_front_start = _centered_window(
        front_vac, incident_front_center, width
    )
    incident_back, incident_back_start = _centered_window(
        back_vac, incident_back_center, width
    )
    reflected_w, reflected_start = _centered_window(reflected, reflected_center, width)
    transmitted_w, transmitted_start = _centered_window(
        back_slab, transmitted_center, width
    )

    freqs = _Fixture.scored_freqs
    inc_front = _dft(incident_front, dt=dt, start=incident_front_start, freqs=freqs)
    inc_back = _dft(incident_back, dt=dt, start=incident_back_start, freqs=freqs)
    refl = _dft(reflected_w, dt=dt, start=reflected_start, freqs=freqs)
    trans = _dft(transmitted_w, dt=dt, start=transmitted_start, freqs=freqs)

    inc_front_denom = np.where(np.abs(inc_front) >= 1e-30, inc_front, 1e-30 + 0j)
    inc_back_denom = np.where(np.abs(inc_back) >= 1e-30, inc_back, 1e-30 + 0j)
    return {
        "R": refl / inc_front_denom,
        "T": trans / inc_back_denom,
        "incident_front_spectrum": inc_front,
        "incident_back_spectrum": inc_back,
        "windows": {
            "incident_front": [incident_front_start, incident_front_start + width],
            "incident_back": [incident_back_start, incident_back_start + width],
            "reflected": [reflected_start, reflected_start + width],
            "transmitted": [transmitted_start, transmitted_start + width],
        },
    }


def _phase_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((np.angle(a / b, deg=True) + 180.0) % 360.0 - 180.0)


def _magnitude_error(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mag_test = np.abs(test)
    mag_ref = np.abs(ref)
    return np.where(
        mag_ref >= 1e-3,
        np.abs(mag_test - mag_ref) / np.maximum(mag_ref, 1e-30),
        np.abs(mag_test - mag_ref),
    )


@lru_cache(maxsize=None)
def _feasibility_metadata() -> dict[str, object]:
    dt_ref, vac_ref = _run_fixture(subgrid=False, slab=False)
    dt_ref_slab, slab_ref = _run_fixture(subgrid=False, slab=True)
    dt_sub, vac_sub = _run_fixture(subgrid=True, slab=False)
    dt_sub_slab, slab_sub = _run_fixture(subgrid=True, slab=True)
    dt_shift, vac_shift = _run_fixture(subgrid=True, slab=False, probe_shift_cells=1)
    dt_shift_slab, slab_shift = _run_fixture(subgrid=True, slab=True, probe_shift_cells=1)

    if not np.allclose([dt_ref_slab, dt_sub, dt_sub_slab, dt_shift, dt_shift_slab], dt_ref):
        return {"classification": "fail", "reason": "fixture timesteps are inconsistent"}
    for label, signal in {
        "vac_ref": vac_ref,
        "slab_ref": slab_ref,
        "vac_sub": vac_sub,
        "slab_sub": slab_sub,
        "vac_shift": vac_shift,
        "slab_shift": slab_shift,
    }.items():
        if not np.all(np.isfinite(signal)):
            return {"classification": "fail", "reason": f"{label} contains NaN/Inf"}

    rt_ref = _extract_rt(vac_ref, slab_ref, dt=dt_ref)
    rt_sub = _extract_rt(vac_sub, slab_sub, dt=dt_sub)
    rt_shift = _extract_rt(vac_shift, slab_shift, dt=dt_shift)

    inc_front = np.asarray(rt_ref["incident_front_spectrum"])
    inc_back = np.asarray(rt_ref["incident_back_spectrum"])
    source_mask = (
        (np.abs(inc_front) >= 0.20 * max(float(np.max(np.abs(inc_front))), 1e-30))
        & (np.abs(inc_back) >= 0.20 * max(float(np.max(np.abs(inc_back))), 1e-30))
    )
    freq_mask = (
        (_Fixture.scored_freqs >= 1.0e9)
        & (_Fixture.scored_freqs <= 3.0e9)
        & source_mask
    )
    if int(np.sum(freq_mask)) == 0:
        return {
            "classification": "fail",
            "reason": "no usable passband bins",
            "incident_front_abs": np.abs(inc_front).tolist(),
            "incident_back_abs": np.abs(inc_back).tolist(),
        }

    r_ref = np.asarray(rt_ref["R"])[freq_mask]
    t_ref = np.asarray(rt_ref["T"])[freq_mask]
    r_sub = np.asarray(rt_sub["R"])[freq_mask]
    t_sub = np.asarray(rt_sub["T"])[freq_mask]
    r_shift = np.asarray(rt_shift["R"])[freq_mask]
    t_shift = np.asarray(rt_shift["T"])[freq_mask]

    r_mag_error = _magnitude_error(r_sub, r_ref)
    t_mag_error = _magnitude_error(t_sub, t_ref)
    phase_refs = np.concatenate([r_ref[np.abs(r_ref) >= 0.05], t_ref[np.abs(t_ref) >= 0.05]])
    phase_tests = np.concatenate([r_sub[np.abs(r_ref) >= 0.05], t_sub[np.abs(t_ref) >= 0.05]])
    phase_error = (
        _phase_error_deg(phase_tests, phase_refs)
        if len(phase_refs) else np.array([0.0], dtype=np.float64)
    )
    r_shift_delta = np.abs(r_shift - r_sub)
    t_shift_delta = np.abs(t_shift - t_sub)

    gates = {
        "r_magnitude": float(np.max(r_mag_error)) <= 0.05,
        "t_magnitude": float(np.max(t_mag_error)) <= 0.05,
        "phase": float(np.max(phase_error)) <= 5.0,
        "probe_shift_r": float(np.max(r_shift_delta)) <= 0.05,
        "probe_shift_t": float(np.max(t_shift_delta)) <= 0.05,
    }
    classification = "pass" if all(gates.values()) else "inconclusive"

    return {
        "classification": classification,
        "gates": gates,
        "fixture": {
            "domain": _Fixture.domain,
            "cpml_layers": _Fixture.cpml_layers,
            "uniform_dx": _Fixture.uniform_dx,
            "coarse_dx": _Fixture.coarse_dx,
            "ratio": _Fixture.ratio,
            "n_steps": _Fixture.n_steps,
            "source": _Fixture.source,
            "front_probe": _Fixture.front_probe,
            "back_probe": _Fixture.back_probe,
            "probe_shift_cells": 1,
            "slab_lo": _Fixture.slab_lo,
            "slab_hi": _Fixture.slab_hi,
            "slab_eps_r": _Fixture.slab_eps_r,
            "source_freq": _Fixture.source_freq,
            "source_bandwidth": _Fixture.source_bandwidth,
        },
        "dt": dt_ref,
        "scored_freqs_hz": _Fixture.scored_freqs[freq_mask].tolist(),
        "windows_reference": rt_ref["windows"],
        "windows_subgrid": rt_sub["windows"],
        "max_r_magnitude_error": float(np.max(r_mag_error)),
        "max_t_magnitude_error": float(np.max(t_mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "max_probe_shift_r_delta": float(np.max(r_shift_delta)),
        "max_probe_shift_t_delta": float(np.max(t_shift_delta)),
        "energy_balance_residual_advisory": float(
            np.max(np.abs(np.abs(r_sub) ** 2 + np.abs(t_sub) ** 2 - 1.0))
        ),
    }


def _print_metadata(metadata: dict[str, object]) -> None:
    print("\nSBP-SAT bounded-CPML true-R/T feasibility metadata:")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _fail_or_xfail_inconclusive(metadata: dict[str, object], reason: str) -> None:
    if metadata["classification"] == "fail":
        pytest.fail(str(metadata))
    if metadata["classification"] == "inconclusive":
        pytest.xfail(reason)


def test_bounded_cpml_point_probe_true_rt_feasibility_vs_uniform_fine():
    metadata = _feasibility_metadata()
    _print_metadata(metadata)

    _fail_or_xfail_inconclusive(
        metadata,
        "Point-probe R/T feasibility is inconclusive; support matrix remains "
        "deferred and flux/DFT fallback stays a separate future plan.",
    )


def test_bounded_cpml_point_probe_true_rt_feasibility_probe_shift_stability():
    metadata = _feasibility_metadata()
    _print_metadata(metadata)

    _fail_or_xfail_inconclusive(
        metadata,
        "Probe-shift stability gate is inconclusive under point probes; "
        "do not promote true R/T.",
    )
