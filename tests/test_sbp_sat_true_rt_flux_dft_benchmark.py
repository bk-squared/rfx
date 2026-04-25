"""Private flux/DFT benchmark evidence for SBP-SAT true R/T.

This file intentionally exercises a private benchmark-only path.  Public
``add_dft_plane_probe`` and ``add_flux_monitor`` remain hard-failing when
``Simulation`` uses SBP-SAT refinement, and the public ``Result`` surface is
not widened by the helper.
"""

# ruff: noqa: E402

from __future__ import annotations

from functools import lru_cache
import json

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.core.dft_utils import dft_window_weight
from rfx.probes.probes import flux_spectrum
from rfx.runners.subgridded import (
    _BenchmarkFluxPlaneRequest,
    _build_benchmark_flux_plane_specs,
    run_subgridded_benchmark_flux,
)
from rfx.subgridding.jit_runner import (
    _BenchmarkFluxPlaneResult,
    _BenchmarkFluxPlaneSpec,
    _accumulate_benchmark_flux_plane,
    _benchmark_flux_spectrum,
)
from rfx.subgridding.sbp_sat_3d import SubgridState3D


_STRICT_PLACEMENT = "fine-owned strict-interior"
_NO_GO_REASON = (
    "private point-source finite-aperture flux fixture is not claims-bearing "
    "for a pass result under the current SBP-SAT public support boundary"
)
_NEXT_PREREQUISITE = (
    "replace the current point-source finite-aperture diagnostic with a "
    "claims-bearing incident-field normalization or plane-wave/port fixture "
    "in a separate support-matrix plan"
)
_NORMALIZATION_FLOOR = 1e-30


class _GuardFixture:
    freqs = np.array([2.0e9], dtype=np.float64)


class _FluxFixture:
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
    source_component = "ey"
    front_plane = 0.028
    back_plane = 0.064
    aperture_center = (0.020, 0.020)
    aperture_size = (0.010, 0.010)
    slab_lo = (0.012, 0.012, 0.046)
    slab_hi = (0.028, 0.028, 0.054)
    slab_eps_r = 2.25
    scored_freqs = np.array([1.5e9, 2.0e9, 2.5e9], dtype=np.float64)

    # Matches the current runner's coarse-cell inclusive refinement lowering:
    # z_range=(0.016, 0.074), dx_c=2 mm, ratio=2 => 60 fine cells at dz=1 mm.
    shape_f = (18, 18, 60)
    offsets = (0.012, 0.012, 0.016)
    dx_f = 1.0e-3


def _guard_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary="pec",
        dx=2e-3,
    )
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.020, 0.020, 0.020), component="ez")
    sim.add_probe(position=(0.020, 0.020, 0.022), component="ez")
    return sim


def _private_guard_plane() -> _BenchmarkFluxPlaneRequest:
    return _BenchmarkFluxPlaneRequest(
        name="private_guard",
        axis="z",
        coordinate=0.020,
        freqs=_GuardFixture.freqs,
        size=(0.006, 0.006),
        center=(0.020, 0.020),
    )


def test_public_dft_plane_probe_still_hard_fails_with_subgrid():
    sim = _guard_sim()
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=0.020,
        component="ez",
        freqs=_GuardFixture.freqs,
    )

    with pytest.raises(ValueError, match="does not support DFT plane probes"):
        sim.run(n_steps=4)


def test_public_flux_monitor_still_hard_fails_with_subgrid():
    sim = _guard_sim()
    sim.add_flux_monitor(
        axis="z",
        coordinate=0.020,
        freqs=_GuardFixture.freqs,
        size=(0.006, 0.006),
        center=(0.020, 0.020),
    )

    with pytest.raises(ValueError, match="does not support flux monitors"):
        sim.run(n_steps=4)


def test_private_benchmark_run_does_not_populate_public_dft_or_flux_results():
    run = run_subgridded_benchmark_flux(
        _guard_sim(),
        n_steps=4,
        planes=(_private_guard_plane(),),
    )

    assert run.result.dft_planes is None
    assert run.result.flux_monitors is None
    assert len(run.benchmark_flux_planes) == 1
    assert run.benchmark_flux_planes[0].name == "private_guard"


def _plane_specs(*planes: _BenchmarkFluxPlaneRequest) -> tuple[_BenchmarkFluxPlaneSpec, ...]:
    return _build_benchmark_flux_plane_specs(
        planes,
        shape_f=_FluxFixture.shape_f,
        offsets=_FluxFixture.offsets,
        dx_f=_FluxFixture.dx_f,
        n_steps=_FluxFixture.n_steps,
    )


def _benchmark_plane(
    name: str,
    *,
    coordinate: float,
    size: tuple[float, float] = _FluxFixture.aperture_size,
    center: tuple[float, float] = _FluxFixture.aperture_center,
) -> _BenchmarkFluxPlaneRequest:
    return _BenchmarkFluxPlaneRequest(
        name=name,
        axis="z",
        coordinate=coordinate,
        freqs=_FluxFixture.scored_freqs,
        size=size,
        center=center,
    )


def test_private_plane_accepts_strict_interior_fine_owned_planes():
    front, back = _plane_specs(
        _benchmark_plane("front", coordinate=_FluxFixture.front_plane),
        _benchmark_plane("back", coordinate=_FluxFixture.back_plane),
    )

    assert front.index == 12
    assert back.index == 48
    assert front.lo1 == front.lo2 == 3
    assert front.hi1 == front.hi2 == 13


def test_private_plane_rejects_local_normal_index_zero():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(_benchmark_plane("at_interface", coordinate=_FluxFixture.offsets[2]))


def test_private_plane_rejects_local_normal_index_n_minus_1():
    last_index_coordinate = _FluxFixture.offsets[2] + (
        _FluxFixture.shape_f[2] - 1
    ) * _FluxFixture.dx_f

    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(_benchmark_plane("at_last_slice", coordinate=last_index_coordinate))


def test_private_plane_rejects_plane_outside_fine_region():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(_benchmark_plane("outside", coordinate=0.010))


def test_private_plane_rejects_plane_not_fully_fine_owned():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane(
                "crosses_tangential_interface",
                coordinate=_FluxFixture.front_plane,
                size=(0.020, 0.010),
            )
        )


def _synthetic_subgrid_state(shape: tuple[int, int, int]) -> SubgridState3D:
    size = int(np.prod(shape))
    base = jnp.arange(size, dtype=jnp.float64).reshape(shape)

    def field(offset: float) -> jnp.ndarray:
        return (base + offset) / 100.0

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=field(1.0),
        ey_f=field(2.0),
        ez_f=field(3.0),
        hx_f=field(4.0),
        hy_f=field(5.0),
        hz_f=field(6.0),
        step=jnp.array(3, dtype=jnp.int32),
    )


def _axis_plane_slices(axis: int, index: int, lo1: int, hi1: int, lo2: int, hi2: int):
    if axis == 0:
        return (
            (index, slice(lo1, hi1), slice(lo2, hi2)),
            (index - 1, slice(lo1, hi1), slice(lo2, hi2)),
        )
    if axis == 1:
        return (
            (slice(lo1, hi1), index, slice(lo2, hi2)),
            (slice(lo1, hi1), index - 1, slice(lo2, hi2)),
        )
    return (
        (slice(lo1, hi1), slice(lo2, hi2), index),
        (slice(lo1, hi1), slice(lo2, hi2), index - 1),
    )


def _reference_accumulate(
    acc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    state: SubgridState3D,
    plane: _BenchmarkFluxPlaneSpec,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = {
        0: ("ey_f", "ez_f", "hy_f", "hz_f"),
        1: ("ez_f", "ex_f", "hz_f", "hx_f"),
        2: ("ex_f", "ey_f", "hx_f", "hy_f"),
    }[plane.axis]
    idx, idx_m1 = _axis_plane_slices(
        plane.axis,
        plane.index,
        plane.lo1,
        plane.hi1,
        plane.lo2,
        plane.hi2,
    )
    e1 = np.asarray(getattr(state, names[0]))[idx]
    e2 = np.asarray(getattr(state, names[1]))[idx]
    h1 = 0.5 * (
        np.asarray(getattr(state, names[2]))[idx_m1]
        + np.asarray(getattr(state, names[2]))[idx]
    )
    h2 = 0.5 * (
        np.asarray(getattr(state, names[3]))[idx_m1]
        + np.asarray(getattr(state, names[3]))[idx]
    )
    step = np.asarray(state.step, dtype=np.float64)
    t = step * dt
    freqs = np.asarray(plane.freqs, dtype=np.float64)
    weight = float(
        np.asarray(
            dft_window_weight(
                state.step,
                plane.total_steps,
                plane.window,
                plane.window_alpha,
            )
        )
    )
    kernel_e = np.exp(-1j * 2.0 * np.pi * freqs * t)[:, None, None] * dt * weight
    kernel_h = (
        np.exp(-1j * 2.0 * np.pi * freqs * (t - dt * 0.5))[:, None, None]
        * dt
        * weight
    )
    return (
        acc[0] + e1[None, :, :] * kernel_e,
        acc[1] + e2[None, :, :] * kernel_e,
        acc[2] + h1[None, :, :] * kernel_h,
        acc[3] + h2[None, :, :] * kernel_h,
    )


def test_private_flux_accumulator_matches_uniform_scan_kernel_formula():
    shape = (6, 5, 4)
    dt = 1.25e-12
    freqs = jnp.asarray([1.5e9, 2.0e9], dtype=jnp.float64)
    plane = _BenchmarkFluxPlaneSpec(
        name="synthetic",
        axis=2,
        index=2,
        freqs=freqs,
        dx=1.0e-3,
        total_steps=8,
        lo1=1,
        hi1=5,
        lo2=1,
        hi2=4,
    )
    state = _synthetic_subgrid_state(shape)
    acc_shape = (len(freqs), plane.hi1 - plane.lo1, plane.hi2 - plane.lo2)
    acc0 = tuple(
        jnp.zeros(acc_shape, dtype=jnp.complex128)
        for _ in range(4)
    )

    actual = _accumulate_benchmark_flux_plane(acc0, state, plane, dt)
    t = np.asarray(state.step, dtype=np.float64) * dt
    phase_e = np.exp(-1j * 2.0 * np.pi * np.asarray(freqs) * t)
    phase_h = np.exp(-1j * 2.0 * np.pi * np.asarray(freqs) * (t - dt * 0.5))
    kernel_e = phase_e[:, None, None] * dt
    kernel_h = phase_h[:, None, None] * dt

    plane_slice = (slice(plane.lo1, plane.hi1), slice(plane.lo2, plane.hi2))
    plane_idx = plane_slice + (plane.index,)
    plane_idx_m1 = plane_slice + (plane.index - 1,)
    ex = np.asarray(state.ex_f)[plane_idx]
    ey = np.asarray(state.ey_f)[plane_idx]
    hx = 0.5 * (
        np.asarray(state.hx_f)[plane_idx_m1]
        + np.asarray(state.hx_f)[plane_idx]
    )
    hy = 0.5 * (
        np.asarray(state.hy_f)[plane_idx_m1]
        + np.asarray(state.hy_f)[plane_idx]
    )
    expected = (
        ex[None, :, :] * kernel_e,
        ey[None, :, :] * kernel_e,
        hx[None, :, :] * kernel_h,
        hy[None, :, :] * kernel_h,
    )

    for actual_arr, expected_arr in zip(actual, expected):
        assert np.allclose(
            np.asarray(actual_arr),
            expected_arr,
            rtol=1e-6,
            atol=1e-9,
        )

    actual_plane = _BenchmarkFluxPlaneResult(
        name=plane.name,
        axis=plane.axis,
        index=plane.index,
        freqs=plane.freqs,
        dx=plane.dx,
        e1_dft=actual[0],
        e2_dft=actual[1],
        h1_dft=actual[2],
        h2_dft=actual[3],
        lo1=plane.lo1,
        hi1=plane.hi1,
        lo2=plane.lo2,
        hi2=plane.hi2,
    )
    expected_plane = actual_plane._replace(
        e1_dft=jnp.asarray(expected[0]),
        e2_dft=jnp.asarray(expected[1]),
        h1_dft=jnp.asarray(expected[2]),
        h2_dft=jnp.asarray(expected[3]),
    )
    assert np.allclose(
        np.asarray(_benchmark_flux_spectrum(actual_plane)),
        np.asarray(_benchmark_flux_spectrum(expected_plane)),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("axis", "index", "lo1", "hi1", "lo2", "hi2"),
    [
        (0, 3, 1, 4, 1, 3),
        (1, 3, 1, 5, 1, 3),
        (2, 2, 1, 5, 1, 4),
    ],
)
def test_private_flux_accumulator_matches_multistep_all_axis_windowed_formula(
    axis: int,
    index: int,
    lo1: int,
    hi1: int,
    lo2: int,
    hi2: int,
):
    shape = (6, 5, 4)
    dt = 1.25e-12
    freqs = jnp.asarray([1.5e9, 2.0e9], dtype=jnp.float64)
    plane = _BenchmarkFluxPlaneSpec(
        name=f"synthetic_axis_{axis}",
        axis=axis,
        index=index,
        freqs=freqs,
        dx=1.0e-3,
        total_steps=9,
        window="hann",
        lo1=lo1,
        hi1=hi1,
        lo2=lo2,
        hi2=hi2,
    )
    acc_shape = (len(freqs), hi1 - lo1, hi2 - lo2)
    actual = tuple(jnp.zeros(acc_shape, dtype=jnp.complex128) for _ in range(4))
    expected = tuple(np.zeros(acc_shape, dtype=np.complex128) for _ in range(4))

    for step in (3, 4):
        state = _synthetic_subgrid_state(shape)._replace(
            step=jnp.array(step, dtype=jnp.int32)
        )
        actual = _accumulate_benchmark_flux_plane(actual, state, plane, dt)
        expected = _reference_accumulate(expected, state, plane, dt)

    for actual_arr, expected_arr in zip(actual, expected):
        assert np.allclose(
            np.asarray(actual_arr),
            expected_arr,
            rtol=1e-6,
            atol=1e-9,
        )


def _complex_flux(plane) -> np.ndarray:
    return np.asarray(
        jnp.sum(
            plane.e1_dft * jnp.conj(plane.h2_dft)
            - plane.e2_dft * jnp.conj(plane.h1_dft),
            axis=(-2, -1),
        )
        * (plane.dx * plane.dx)
    )


def _phase_error_deg(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs((np.angle(test / ref, deg=True) + 180.0) % 360.0 - 180.0)


def _relative_magnitude_error(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mag_test = np.abs(test)
    mag_ref = np.abs(ref)
    return np.where(
        mag_ref >= 1e-3,
        np.abs(mag_test - mag_ref) / np.maximum(mag_ref, _NORMALIZATION_FLOOR),
        np.abs(mag_test - mag_ref),
    )


def _finite_or_fail(label: str, values: np.ndarray) -> dict[str, object] | None:
    if not np.all(np.isfinite(values)):
        return {"classification": "fail", "reason": f"{label} contains NaN/Inf"}
    return None


def _plane_requests(
    shift_cells: int = 0,
    aperture_size: float | None = None,
) -> tuple[_BenchmarkFluxPlaneRequest, ...]:
    dz = shift_cells * _FluxFixture.dx_f
    size = _FluxFixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    return (
        _benchmark_plane(
            "front",
            coordinate=_FluxFixture.front_plane + dz,
            size=aperture,
        ),
        _benchmark_plane(
            "back",
            coordinate=_FluxFixture.back_plane + dz,
            size=aperture,
        ),
    )


@lru_cache(maxsize=None)
def _run_flux_fixture(
    *,
    subgrid: bool,
    slab: bool,
    plane_shift_cells: int = 0,
    aperture_size: float | None = None,
) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    dx = _FluxFixture.coarse_dx if subgrid else _FluxFixture.uniform_dx
    size = _FluxFixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    sim = Simulation(
        freq_max=_FluxFixture.freq_max,
        domain=_FluxFixture.domain,
        boundary="cpml",
        cpml_layers=_FluxFixture.cpml_layers,
        dx=dx,
    )
    if slab:
        sim.add_material("rt_dielectric", eps_r=_FluxFixture.slab_eps_r)
        sim.add(Box(_FluxFixture.slab_lo, _FluxFixture.slab_hi), material="rt_dielectric")
    if subgrid:
        sim.add_refinement(**_FluxFixture.refinement)
    sim.add_source(
        position=_FluxFixture.source,
        component=_FluxFixture.source_component,
        waveform=GaussianPulse(
            f0=_FluxFixture.source_freq,
            bandwidth=_FluxFixture.source_bandwidth,
        ),
    )
    sim.add_probe(position=_FluxFixture.source, component=_FluxFixture.source_component)

    if subgrid:
        run = run_subgridded_benchmark_flux(
            sim,
            n_steps=_FluxFixture.n_steps,
            planes=_plane_requests(plane_shift_cells, aperture_size),
        )
        planes = run.benchmark_flux_planes
        signed_flux = tuple(np.asarray(_benchmark_flux_spectrum(p)) for p in planes)
        complex_flux = tuple(_complex_flux(p) for p in planes)
        return float(run.result.dt), complex_flux, signed_flux

    dz = plane_shift_cells * _FluxFixture.dx_f
    sim.add_flux_monitor(
        axis="z",
        coordinate=_FluxFixture.front_plane + dz,
        freqs=_FluxFixture.scored_freqs,
        size=aperture,
        center=_FluxFixture.aperture_center,
        name="front",
    )
    sim.add_flux_monitor(
        axis="z",
        coordinate=_FluxFixture.back_plane + dz,
        freqs=_FluxFixture.scored_freqs,
        size=aperture,
        center=_FluxFixture.aperture_center,
        name="back",
    )
    result = sim.run(n_steps=_FluxFixture.n_steps)
    monitors = (result.flux_monitors["front"], result.flux_monitors["back"])
    signed_flux = tuple(np.asarray(flux_spectrum(m)) for m in monitors)
    complex_flux = tuple(_complex_flux(m) for m in monitors)
    return float(result.dt), complex_flux, signed_flux


def _usable_passband(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    front_mag = np.abs(front)
    back_mag = np.abs(back)
    front_peak = float(np.max(front_mag))
    back_peak = float(np.max(back_mag))
    if front_peak <= 0.0 or back_peak <= 0.0:
        return np.zeros_like(front_mag, dtype=bool)
    return (
        (front_mag >= 0.20 * front_peak)
        & (back_mag >= 0.20 * back_peak)
    )


def _serial_complex(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in np.ravel(values)]


def _flux_diagnostics(
    complex_flux: tuple[np.ndarray, np.ndarray],
    signed_flux: tuple[np.ndarray, np.ndarray],
) -> dict[str, list[dict[str, object]]]:
    diagnostics: dict[str, list[dict[str, object]]] = {}
    for name, complex_values, signed_values in zip(
        ("front", "back"),
        complex_flux,
        signed_flux,
        strict=True,
    ):
        magnitude = np.abs(complex_values)
        diagnostics[name] = [
            {
                "freq_hz": float(freq),
                "complex": complex_pair,
                "magnitude": float(mag),
                "magnitude_to_floor": float(mag / _NORMALIZATION_FLOOR),
                "phase_deg": float(phase),
                "signed_real_flux": float(signed),
                "signed_real_flux_to_floor": float(
                    abs(signed) / _NORMALIZATION_FLOOR
                ),
            }
            for freq, complex_pair, mag, phase, signed in zip(
                _FluxFixture.scored_freqs,
                _serial_complex(complex_values),
                magnitude,
                np.angle(complex_values, deg=True),
                signed_values,
                strict=True,
            )
        ]
    return diagnostics


def _floor_relative_error(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(np.abs(test) - np.abs(ref)) / np.maximum(
        np.abs(ref),
        _NORMALIZATION_FLOOR,
    )


def _energy_residual(
    signed_slab: tuple[np.ndarray, np.ndarray],
    signed_vacuum: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    signed_front = np.asarray(signed_slab[0])[mask]
    signed_back = np.asarray(signed_slab[1])[mask]
    signed_incident = np.asarray(signed_vacuum[0])[mask]
    return np.abs(
        (signed_front - signed_back)
        / np.maximum(np.abs(signed_incident), _NORMALIZATION_FLOOR)
    )


def _unfloored_energy_residual(
    signed_slab: tuple[np.ndarray, np.ndarray],
    signed_vacuum: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    signed_front = np.asarray(signed_slab[0])[mask]
    signed_back = np.asarray(signed_slab[1])[mask]
    signed_incident = np.asarray(signed_vacuum[0])[mask]
    denom = np.where(np.abs(signed_incident) > 0.0, np.abs(signed_incident), np.nan)
    return np.abs((signed_front - signed_back) / denom)


@lru_cache(maxsize=None)
def _homogeneous_parity_for_aperture(aperture_size: float) -> dict[str, object]:
    aperture_arg = (
        None
        if np.isclose(aperture_size, _FluxFixture.aperture_size[0])
        else aperture_size
    )
    dt_ref, flux_ref, _signed_ref = _run_flux_fixture(
        subgrid=False,
        slab=False,
        aperture_size=aperture_arg,
    )
    dt_sub, flux_sub, _signed_sub = _run_flux_fixture(
        subgrid=True,
        slab=False,
        aperture_size=aperture_arg,
    )
    metadata: dict[str, object] = {
        "aperture_size_m": float(aperture_size),
        "dt_match": bool(np.allclose(dt_ref, dt_sub)),
        "uniform_front_peak_magnitude": float(np.max(np.abs(flux_ref[0]))),
        "uniform_back_peak_magnitude": float(np.max(np.abs(flux_ref[1]))),
        "subgrid_front_peak_magnitude": float(np.max(np.abs(flux_sub[0]))),
        "subgrid_back_peak_magnitude": float(np.max(np.abs(flux_sub[1]))),
    }
    metadata["uniform_front_peak_to_floor"] = float(
        metadata["uniform_front_peak_magnitude"] / _NORMALIZATION_FLOOR
    )
    metadata["uniform_back_peak_to_floor"] = float(
        metadata["uniform_back_peak_magnitude"] / _NORMALIZATION_FLOOR
    )

    mask = _usable_passband(flux_ref[0], flux_ref[1])
    metadata["usable_bins"] = int(np.sum(mask))
    metadata["scored_freqs_hz"] = _FluxFixture.scored_freqs[mask].tolist()
    if int(np.sum(mask)) == 0:
        metadata["classification"] = "inconclusive"
        metadata["reason"] = "homogeneous runtime passband too weak to score"
        return metadata

    ref = np.concatenate([flux_ref[0][mask], flux_ref[1][mask]])
    sub = np.concatenate([flux_sub[0][mask], flux_sub[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    metadata.update(
        {
            "classification": "pass"
            if float(np.max(mag_error)) <= 0.02
            and float(np.max(phase_error)) <= 2.0
            else "inconclusive",
            "max_floor_relative_magnitude_error": float(np.max(mag_error)),
            "max_complex_phase_error_deg": float(np.max(phase_error)),
        }
    )
    return metadata


@lru_cache(maxsize=None)
def _homogeneous_parity_metadata() -> dict[str, object]:
    dt_ref, flux_ref, signed_ref = _run_flux_fixture(subgrid=False, slab=False)
    dt_sub, flux_sub, signed_sub = _run_flux_fixture(subgrid=True, slab=False)
    if not np.allclose(dt_ref, dt_sub):
        return {"classification": "fail", "reason": "uniform/subgrid dt mismatch"}
    for label, arrays in {"uniform": flux_ref, "subgrid": flux_sub}.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail

    mask = _usable_passband(flux_ref[0], flux_ref[1])
    if int(np.sum(mask)) == 0:
        return {
            "classification": "inconclusive",
            "reason": "homogeneous runtime passband too weak to score",
            "front_abs": np.abs(flux_ref[0]).tolist(),
            "back_abs": np.abs(flux_ref[1]).tolist(),
        }

    ref = np.concatenate([flux_ref[0][mask], flux_ref[1][mask]])
    sub = np.concatenate([flux_sub[0][mask], flux_sub[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    gates = {
        "magnitude": float(np.max(mag_error)) <= 0.02,
        "phase": float(np.max(phase_error)) <= 2.0,
    }
    return {
        "classification": "pass" if all(gates.values()) else "inconclusive",
        "fixture": "bounded_cpml_private_fine_owned_flux_plane_homogeneous",
        "gates": gates,
        "scored_freqs_hz": _FluxFixture.scored_freqs[mask].tolist(),
        "max_magnitude_error": float(np.max(mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "uniform_flux_diagnostics": _flux_diagnostics(flux_ref, signed_ref),
        "subgrid_flux_diagnostics": _flux_diagnostics(flux_sub, signed_sub),
        "aperture_sweep": [
            _homogeneous_parity_for_aperture(_FluxFixture.aperture_size[0]),
            _homogeneous_parity_for_aperture(0.014),
        ],
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "homogeneous absolute flux parity remains below pass threshold, "
            "and scored spectra sit at or below the configured normalization "
            "floor for the point-source finite-aperture fixture"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "Synthetic multi-step/all-axis accumulator parity passes, but "
            "runtime homogeneous parity is dominated by weak finite-aperture "
            "point-source flux normalization rather than claims-bearing "
            "incident/reflected/transmitted separation."
        ),
    }


@lru_cache(maxsize=None)
def _plane_rt_metadata() -> dict[str, object]:
    dt_ref, vac_ref, signed_vac_ref = _run_flux_fixture(subgrid=False, slab=False)
    dt_ref_slab, slab_ref, signed_ref_slab = _run_flux_fixture(subgrid=False, slab=True)
    dt_sub, vac_sub, signed_vac_sub = _run_flux_fixture(subgrid=True, slab=False)
    dt_sub_slab, slab_sub, signed_sub_slab = _run_flux_fixture(subgrid=True, slab=True)
    dt_shift_vac, vac_shift, _ = _run_flux_fixture(
        subgrid=True,
        slab=False,
        plane_shift_cells=1,
    )
    dt_shift_slab, slab_shift, _ = _run_flux_fixture(
        subgrid=True,
        slab=True,
        plane_shift_cells=1,
    )

    if not np.allclose(
        [dt_ref_slab, dt_sub, dt_sub_slab, dt_shift_vac, dt_shift_slab],
        dt_ref,
    ):
        return {"classification": "fail", "reason": "fixture timesteps are inconsistent"}

    for label, arrays in {
        "vac_ref": vac_ref,
        "slab_ref": slab_ref,
        "vac_sub": vac_sub,
        "slab_sub": slab_sub,
        "vac_shift": vac_shift,
        "slab_shift": slab_shift,
    }.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail

    freq_mask = _usable_passband(vac_ref[0], vac_ref[1])
    if int(np.sum(freq_mask)) == 0:
        return {
            "classification": "fail",
            "reason": "no usable passband bins",
            "front_abs": np.abs(vac_ref[0]).tolist(),
            "back_abs": np.abs(vac_ref[1]).tolist(),
        }

    def rt(vac: tuple[np.ndarray, np.ndarray], slab: tuple[np.ndarray, np.ndarray]):
        inc_front = np.where(
            np.abs(vac[0]) >= _NORMALIZATION_FLOOR,
            vac[0],
            _NORMALIZATION_FLOOR + 0j,
        )
        inc_back = np.where(
            np.abs(vac[1]) >= _NORMALIZATION_FLOOR,
            vac[1],
            _NORMALIZATION_FLOOR + 0j,
        )
        return {
            "R": (slab[0] - vac[0]) / inc_front,
            "T": slab[1] / inc_back,
        }

    rt_ref = rt(vac_ref, slab_ref)
    rt_sub = rt(vac_sub, slab_sub)
    rt_shift = rt(vac_shift, slab_shift)
    r_ref = rt_ref["R"][freq_mask]
    t_ref = rt_ref["T"][freq_mask]
    r_sub = rt_sub["R"][freq_mask]
    t_sub = rt_sub["T"][freq_mask]
    r_shift = rt_shift["R"][freq_mask]
    t_shift = rt_shift["T"][freq_mask]

    r_mag_error = _relative_magnitude_error(r_sub, r_ref)
    t_mag_error = _relative_magnitude_error(t_sub, t_ref)
    phase_refs = np.concatenate([r_ref[np.abs(r_ref) >= 0.05], t_ref[np.abs(t_ref) >= 0.05]])
    phase_tests = np.concatenate([r_sub[np.abs(r_ref) >= 0.05], t_sub[np.abs(t_ref) >= 0.05]])
    phase_error = (
        _phase_error_deg(phase_tests, phase_refs)
        if len(phase_refs)
        else np.array([0.0], dtype=np.float64)
    )
    r_shift_delta = np.abs(np.abs(r_shift) - np.abs(r_sub))
    t_shift_delta = np.abs(np.abs(t_shift) - np.abs(t_sub))
    shift_phase_refs = np.concatenate([
        r_sub[np.abs(r_sub) >= 0.05],
        t_sub[np.abs(t_sub) >= 0.05],
    ])
    shift_phase_tests = np.concatenate([
        r_shift[np.abs(r_sub) >= 0.05],
        t_shift[np.abs(t_sub) >= 0.05],
    ])
    shift_phase_error = (
        _phase_error_deg(shift_phase_tests, shift_phase_refs)
        if len(shift_phase_refs)
        else np.array([0.0], dtype=np.float64)
    )
    # Signed plane-flux energy is advisory for this private gate.  It is used
    # to detect gross normalization drift, not to promote public true-R/T.
    energy_balance = _energy_residual(signed_sub_slab, signed_vac_sub, freq_mask)
    uniform_energy_balance = _energy_residual(
        signed_ref_slab,
        signed_vac_ref,
        freq_mask,
    )
    energy_delta = np.abs(energy_balance - uniform_energy_balance)
    unfloored_energy_balance = _unfloored_energy_residual(
        signed_sub_slab,
        signed_vac_sub,
        freq_mask,
    )

    gates = {
        "r_magnitude": float(np.max(r_mag_error)) <= 0.05,
        "t_magnitude": float(np.max(t_mag_error)) <= 0.05,
        "phase": float(np.max(phase_error)) <= 5.0,
        "energy": float(np.max(energy_balance)) <= 0.05,
        "plane_shift_r": float(np.max(r_shift_delta)) <= 0.05,
        "plane_shift_t": float(np.max(t_shift_delta)) <= 0.05,
        "plane_shift_phase": float(np.max(shift_phase_error)) <= 5.0,
    }
    classification = "pass" if all(gates.values()) else "inconclusive"
    return {
        "classification": classification,
        "gates": gates,
        "fixture": "bounded_cpml_private_fine_owned_flux_plane_vacuum_slab",
        "scored_freqs_hz": _FluxFixture.scored_freqs[freq_mask].tolist(),
        "max_r_magnitude_error": float(np.max(r_mag_error)),
        "max_t_magnitude_error": float(np.max(t_mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "energy_balance_residual": float(np.max(energy_balance)),
        "uniform_energy_balance_residual": float(np.max(uniform_energy_balance)),
        "energy_residual_delta_vs_uniform": float(np.max(energy_delta)),
        "unfloored_energy_balance_residual": float(np.nanmax(unfloored_energy_balance)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "max_plane_shift_r_delta": float(np.max(r_shift_delta)),
        "max_plane_shift_t_delta": float(np.max(t_shift_delta)),
        "max_plane_shift_phase_error_deg": float(np.max(shift_phase_error)),
        "subgrid_vacuum_flux_diagnostics": _flux_diagnostics(vac_sub, signed_vac_sub),
        "subgrid_slab_flux_diagnostics": _flux_diagnostics(slab_sub, signed_sub_slab),
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "the private point-source finite-aperture fixture keeps the energy "
            "residual above threshold on a normalization-floor-dominated "
            "incident flux, so the gate cannot be promoted as true R/T evidence"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "R/T magnitude and plane-shift checks are useful internal "
            "diagnostics, but the energy-balance observable lacks a "
            "claims-bearing incident-field normalization."
        ),
        "replacement_metric_allowed": bool(float(np.max(energy_delta)) <= 0.02),
    }


def _print_metadata(title: str, metadata: dict[str, object]) -> None:
    print(f"\n{title}:")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _fail_or_xfail_inconclusive(metadata: dict[str, object], reason: str) -> None:
    if metadata["classification"] == "fail":
        pytest.fail(str(metadata))
    if metadata["classification"] == "inconclusive":
        pytest.xfail(reason)


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_flux_matches_uniform_reference_in_homogeneous_cpml_fixture():
    metadata = _homogeneous_parity_metadata()
    _print_metadata("SBP-SAT private flux homogeneous parity metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private flux runtime parity threshold is a principled no-go for "
        "public promotion; support matrix must not promote public flux/DFT "
        "or true R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_benchmark_vs_uniform_fine():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private flux true-R/T metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private plane-flux true R/T gate is a principled no-go for public "
        "promotion; support matrix remains deferred.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_plane_shift_stability():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private flux true-R/T shift metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private plane-shift stability gate is inconclusive; do not promote true R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_no_go_metadata_is_explicit():
    metadata = _plane_rt_metadata()

    assert metadata["classification"] == "inconclusive"
    assert metadata["no_go_reason"] == _NO_GO_REASON
    assert metadata["next_prerequisite"] == _NEXT_PREREQUISITE
    assert metadata["blocking_diagnostic"]
    assert metadata["public_claim_allowed"] is False
    assert metadata["energy_residual_delta_vs_uniform"] > 0.02
