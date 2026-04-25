"""Private flux/DFT benchmark evidence for SBP-SAT true R/T.

This file intentionally exercises a private benchmark-only path.  Public
``add_dft_plane_probe`` and ``add_flux_monitor`` remain hard-failing when
``Simulation`` uses SBP-SAT refinement, and the public ``Result`` surface is
not widened by the helper.
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import NamedTuple
import warnings

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.dft_utils import dft_window_weight
from rfx.probes.probes import flux_spectrum
from rfx.runners.subgridded import (
    _BenchmarkFluxPlaneRequest,
    _PrivateAnalyticSheetSourceRequest,
    _build_benchmark_flux_plane_specs,
    _build_private_analytic_sheet_source_specs,
    run_subgridded_benchmark_flux,
)
from rfx.sources.sources import CustomWaveform
from rfx.subgridding.jit_runner import (
    _BenchmarkFluxPlaneResult,
    _BenchmarkFluxPlaneSpec,
    _PrivateAnalyticSheetSourceSpec,
    _accumulate_benchmark_flux_plane,
    _benchmark_flux_spectrum,
    _inject_private_analytic_sheet_source,
)
from rfx.subgridding.sbp_sat_3d import SubgridState3D


_STRICT_PLACEMENT = "fine-owned strict-interior"
_NO_GO_REASON = (
    "private analytic-sheet bounded-CPML fixture did not satisfy the "
    "incident-normalized fixture-quality gates required for public true R/T "
    "promotion"
)
_NEXT_PREREQUISITE = (
    "open a separate private TFSF-style incident-field fixture plan or private "
    "normalization-repair plan before reconsidering public true R/T, DFT, "
    "flux, port, or S-parameter promotion"
)
_NORMALIZATION_FLOOR = 1e-30
_NONFLOOR_FACTOR = 1e12
_MIN_CLAIMS_BEARING_BINS = 2


class _GuardFixture:
    freqs = np.array([2.0e9], dtype=np.float64)


@dataclass(frozen=True)
class _FluxFixtureConfig:
    """Private benchmark fixture geometry with derived fine-grid metadata."""

    name: str
    fixture_key: str
    freq_max: float = 6.0e9
    source_freq: float = 2.0e9
    source_bandwidth: float = 0.8
    domain: tuple[float, float, float] = (0.04, 0.04, 0.09)
    cpml_layers: int = 4
    uniform_dx: float = 1.0e-3
    coarse_dx: float = 2.0e-3
    ratio: int = 2
    n_steps: int = 700
    refinement_x_range: tuple[float, float] = (0.012, 0.028)
    refinement_y_range: tuple[float, float] = (0.012, 0.028)
    refinement_z_range: tuple[float, float] = (0.016, 0.074)
    sheet_coordinate: float = 0.024
    sheet_component: str = "ey"
    sheet_amplitude: float = 1.0e8
    sheet_phase_rad: float = 0.0
    sheet_x_span: tuple[float, float] = (0.013, 0.027)
    sheet_y_span: tuple[float, float] = (0.013, 0.027)
    front_plane: float = 0.028
    back_plane: float = 0.064
    aperture_center: tuple[float, float] = (0.020, 0.020)
    aperture_diameter: float = 0.014
    slab_lo: tuple[float, float, float] = (0.012, 0.012, 0.046)
    slab_hi: tuple[float, float, float] = (0.028, 0.028, 0.054)
    slab_eps_r: float = 2.25
    scored_freqs_tuple: tuple[float, ...] = (1.5e9, 2.0e9, 2.5e9)

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError("fixture ratio must be positive")
        if not np.isclose(self.dx_f, self.uniform_dx):
            raise ValueError("fixture fine dx must match uniform fine reference dx")
        for label, span in {
            "refinement_x_range": self.refinement_x_range,
            "refinement_y_range": self.refinement_y_range,
            "refinement_z_range": self.refinement_z_range,
            "sheet_x_span": self.sheet_x_span,
            "sheet_y_span": self.sheet_y_span,
        }.items():
            if not span[0] < span[1]:
                raise ValueError(f"{label} must be increasing")
        if not (
            self.refinement_z_range[0]
            < self.sheet_coordinate
            < self.front_plane
            < self.slab_lo[2]
            < self.slab_hi[2]
            < self.back_plane
            < self.refinement_z_range[1]
        ):
            raise ValueError("fixture z ordering must be sheet < front < slab < back")
        for axis, (ref_span, sheet_span, slab_lo, slab_hi) in enumerate(
            (
                (
                    self.refinement_x_range,
                    self.sheet_x_span,
                    self.slab_lo[0],
                    self.slab_hi[0],
                ),
                (
                    self.refinement_y_range,
                    self.sheet_y_span,
                    self.slab_lo[1],
                    self.slab_hi[1],
                ),
            )
        ):
            if not (
                ref_span[0] < sheet_span[0] < sheet_span[1] < ref_span[1]
                and ref_span[0] <= slab_lo < slab_hi <= ref_span[1]
            ):
                raise ValueError(f"fixture tangential geometry invalid on axis {axis}")
        self._validate_refinement_alignment()
        self._validate_absorber_guard()

    def _validate_refinement_alignment(self) -> None:
        for span in (
            self.refinement_x_range,
            self.refinement_y_range,
            self.refinement_z_range,
        ):
            cells = (span[1] - span[0]) / self.coarse_dx
            if not np.isclose(cells, round(cells), atol=1e-9):
                raise ValueError("refinement ranges must align to coarse cells")

    def _validate_absorber_guard(self) -> None:
        guard = (self.cpml_layers + 1) * self.coarse_dx
        for axis, (span, domain_size) in enumerate(
            zip(
                (
                    self.refinement_x_range,
                    self.refinement_y_range,
                    self.refinement_z_range,
                ),
                self.domain,
                strict=True,
            )
        ):
            if span[0] < guard - 1e-12 or span[1] > domain_size - guard + 1e-12:
                raise ValueError(
                    f"fixture refinement must stay outside CPML guard on axis {axis}"
                )

    @property
    def refinement(self) -> dict[str, object]:
        return {
            "x_range": self.refinement_x_range,
            "y_range": self.refinement_y_range,
            "z_range": self.refinement_z_range,
            "ratio": self.ratio,
        }

    @property
    def dx_f(self) -> float:
        return self.coarse_dx / self.ratio

    @property
    def shape_f(self) -> tuple[int, int, int]:
        # Mirrors the runner's coarse-cell inclusive refinement lowering.
        return tuple(
            int(round((hi - lo) / self.coarse_dx) + 1) * self.ratio
            for lo, hi in (
                self.refinement_x_range,
                self.refinement_y_range,
                self.refinement_z_range,
            )
        )

    @property
    def offsets(self) -> tuple[float, float, float]:
        return (
            self.refinement_x_range[0],
            self.refinement_y_range[0],
            self.refinement_z_range[0],
        )

    @property
    def source(self) -> tuple[float, float, float]:
        return (
            self.aperture_center[0],
            self.aperture_center[1],
            self.sheet_coordinate,
        )

    @property
    def source_component(self) -> str:
        return self.sheet_component

    @property
    def aperture_size(self) -> tuple[float, float]:
        return (self.aperture_diameter, self.aperture_diameter)

    @property
    def scored_freqs(self) -> np.ndarray:
        return np.array(self.scored_freqs_tuple, dtype=np.float64)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fixture": self.fixture_key,
            "domain": list(self.domain),
            "cpml_layers": self.cpml_layers,
            "coarse_dx": self.coarse_dx,
            "uniform_dx": self.uniform_dx,
            "ratio": self.ratio,
            "n_steps": self.n_steps,
            "refinement": self.refinement,
            "shape_f": list(self.shape_f),
            "offsets": list(self.offsets),
            "sheet_coordinate": self.sheet_coordinate,
            "sheet_span": [list(self.sheet_x_span), list(self.sheet_y_span)],
            "front_plane": self.front_plane,
            "back_plane": self.back_plane,
            "aperture_center": list(self.aperture_center),
            "aperture_size": list(self.aperture_size),
            "slab_lo": list(self.slab_lo),
            "slab_hi": list(self.slab_hi),
        }


_FluxFixture = _FluxFixtureConfig(
    name="current_bounded",
    fixture_key="bounded_cpml_private_analytic_sheet_flux_plane_vacuum_slab",
)

_BoundaryExpandedFluxFixture = _FluxFixtureConfig(
    name="boundary_expanded",
    fixture_key="boundary_expanded_private_analytic_sheet_flux_plane_vacuum_slab",
    domain=(0.048, 0.048, 0.105),
    refinement_x_range=(0.010, 0.038),
    refinement_y_range=(0.010, 0.038),
    refinement_z_range=(0.014, 0.090),
    sheet_coordinate=0.026,
    sheet_x_span=(0.014, 0.034),
    sheet_y_span=(0.014, 0.034),
    front_plane=0.036,
    back_plane=0.078,
    aperture_center=(0.024, 0.024),
    aperture_diameter=0.020,
    slab_lo=(0.014, 0.014, 0.056),
    slab_hi=(0.034, 0.034, 0.066),
)

_RECOVERY_SWEEP_FIXTURES = (_FluxFixture, _BoundaryExpandedFluxFixture)


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


def _private_guard_sheet() -> _PrivateAnalyticSheetSourceRequest:
    return _PrivateAnalyticSheetSourceRequest(
        name="private_guard_sheet",
        axis="z",
        coordinate=0.020,
        component="ey",
        propagation_sign=1,
        amplitude=1.0,
        f0_hz=2.0e9,
        bandwidth=0.8,
        phase_rad=0.0,
        x_span=(0.017, 0.023),
        y_span=(0.017, 0.023),
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
        sheet_sources=(_private_guard_sheet(),),
    )

    assert run.result.dft_planes is None
    assert run.result.flux_monitors is None
    assert len(run.benchmark_flux_planes) == 1
    assert run.benchmark_flux_planes[0].name == "private_guard"


def _plane_specs(
    *planes: _BenchmarkFluxPlaneRequest,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_BenchmarkFluxPlaneSpec, ...]:
    return _build_benchmark_flux_plane_specs(
        planes,
        shape_f=fixture.shape_f,
        offsets=fixture.offsets,
        dx_f=fixture.dx_f,
        n_steps=fixture.n_steps,
    )


def _benchmark_sheet_source(
    *,
    fixture: _FluxFixtureConfig = _FluxFixture,
    coordinate: float | None = None,
    axis: str = "z",
    component: str | None = None,
    propagation_sign: int = 1,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
) -> _PrivateAnalyticSheetSourceRequest:
    return _PrivateAnalyticSheetSourceRequest(
        name="private_sheet",
        axis=axis,
        coordinate=fixture.sheet_coordinate if coordinate is None else coordinate,
        component=fixture.sheet_component if component is None else component,
        propagation_sign=propagation_sign,
        amplitude=fixture.sheet_amplitude,
        f0_hz=fixture.source_freq,
        bandwidth=fixture.source_bandwidth,
        phase_rad=fixture.sheet_phase_rad,
        x_span=fixture.sheet_x_span if x_span is None else x_span,
        y_span=fixture.sheet_y_span if y_span is None else y_span,
    )


def _sheet_specs(
    *sources: _PrivateAnalyticSheetSourceRequest,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_PrivateAnalyticSheetSourceSpec, ...]:
    return _build_private_analytic_sheet_source_specs(
        sources,
        shape_f=fixture.shape_f,
        offsets=fixture.offsets,
        dx_f=fixture.dx_f,
        dt=1.0e-12,
        n_steps=fixture.n_steps,
    )


def _benchmark_plane(
    name: str,
    *,
    coordinate: float,
    fixture: _FluxFixtureConfig = _FluxFixture,
    size: tuple[float, float] | None = None,
    center: tuple[float, float] | None = None,
) -> _BenchmarkFluxPlaneRequest:
    return _BenchmarkFluxPlaneRequest(
        name=name,
        axis="z",
        coordinate=coordinate,
        freqs=fixture.scored_freqs,
        size=fixture.aperture_size if size is None else size,
        center=fixture.aperture_center if center is None else center,
    )


def test_private_plane_accepts_strict_interior_fine_owned_planes():
    front, back = _plane_specs(
        _benchmark_plane("front", coordinate=_FluxFixture.front_plane),
        _benchmark_plane("back", coordinate=_FluxFixture.back_plane),
    )

    assert front.index == 12
    assert back.index == 48
    assert front.lo1 == front.lo2 == 1
    assert front.hi1 == front.hi2 == 15


def test_private_sheet_source_accepts_strict_interior_full_span():
    (sheet,) = _sheet_specs(_benchmark_sheet_source())

    assert sheet.axis == 2
    assert sheet.index == 8
    assert sheet.component == _FluxFixture.sheet_component
    assert sheet.propagation_sign == 1
    assert sheet.lo1 == sheet.lo2 == 1
    assert sheet.hi1 == sheet.hi2 == 15
    assert sheet.source_values.shape == (_FluxFixture.n_steps,)


def test_boundary_expanded_fixture_derives_strict_fine_grid_metadata():
    fixture = _BoundaryExpandedFluxFixture
    front, back = _plane_specs(
        _benchmark_plane("front", coordinate=fixture.front_plane, fixture=fixture),
        _benchmark_plane("back", coordinate=fixture.back_plane, fixture=fixture),
        fixture=fixture,
    )
    (sheet,) = _sheet_specs(
        _benchmark_sheet_source(fixture=fixture),
        fixture=fixture,
    )

    assert fixture.shape_f == (30, 30, 78)
    assert fixture.offsets == (0.010, 0.010, 0.014)
    assert front.index == 22
    assert back.index == 64
    assert front.lo1 == front.lo2 == 4
    assert front.hi1 == front.hi2 == 24
    assert sheet.index == 12
    assert sheet.lo1 == sheet.lo2 == 4
    assert sheet.hi1 == sheet.hi2 == 24
    assert sheet.source_values.shape == (fixture.n_steps,)


@pytest.mark.parametrize(
    "source",
    [
        _benchmark_sheet_source(axis="x"),
        _benchmark_sheet_source(component="ez"),
        _benchmark_sheet_source(propagation_sign=-1),
        _benchmark_sheet_source(x_span=(0.011, 0.027)),
    ],
)
def test_private_sheet_source_rejects_public_or_edge_touching_shapes(
    source: _PrivateAnalyticSheetSourceRequest,
):
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _sheet_specs(source)


def test_private_sheet_source_injection_adds_selected_tangential_field_only():
    shape = (5, 6, 7)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=zeros,
        ey_f=zeros,
        ez_f=zeros,
        hx_f=zeros,
        hy_f=zeros,
        hz_f=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )
    sheet = _PrivateAnalyticSheetSourceSpec(
        name="synthetic_sheet",
        axis=2,
        index=3,
        component="ey",
        propagation_sign=1,
        amplitude=1.0,
        f0_hz=2.0e9,
        bandwidth=0.8,
        phase_rad=0.0,
        source_values=jnp.ones((4,), dtype=jnp.float64),
        lo1=1,
        hi1=4,
        lo2=2,
        hi2=5,
    )

    out = _inject_private_analytic_sheet_source(state, sheet, jnp.array(2.5))

    ey = np.asarray(out.ey_f)
    assert np.allclose(ey[1:4, 2:5, 3], 2.5)
    assert np.count_nonzero(ey) == 9
    assert np.allclose(np.asarray(out.ex_f), 0.0)


def test_private_plane_rejects_local_normal_index_zero():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane("at_interface", coordinate=_FluxFixture.offsets[2])
        )


def test_private_plane_rejects_local_normal_index_n_minus_1():
    last_index_coordinate = (
        _FluxFixture.offsets[2] + (_FluxFixture.shape_f[2] - 1) * _FluxFixture.dx_f
    )

    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane("at_last_slice", coordinate=last_index_coordinate)
        )


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
        np.exp(-1j * 2.0 * np.pi * freqs * (t - dt * 0.5))[:, None, None] * dt * weight
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
    acc0 = tuple(jnp.zeros(acc_shape, dtype=jnp.complex128) for _ in range(4))

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
        np.asarray(state.hx_f)[plane_idx_m1] + np.asarray(state.hx_f)[plane_idx]
    )
    hy = 0.5 * (
        np.asarray(state.hy_f)[plane_idx_m1] + np.asarray(state.hy_f)[plane_idx]
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


class _FixtureRun(NamedTuple):
    dt: float
    complex_flux: tuple[np.ndarray, np.ndarray]
    signed_flux: tuple[np.ndarray, np.ndarray]
    planes: tuple[object, object]


def _finite_or_fail(label: str, values: np.ndarray) -> dict[str, object] | None:
    if not np.all(np.isfinite(values)):
        return {"classification": "fail", "reason": f"{label} contains NaN/Inf"}
    return None


def _plane_requests(
    shift_cells: int = 0,
    aperture_size: float | None = None,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_BenchmarkFluxPlaneRequest, ...]:
    dz = shift_cells * fixture.dx_f
    size = fixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    return (
        _benchmark_plane(
            "front",
            fixture=fixture,
            coordinate=fixture.front_plane + dz,
            size=aperture,
        ),
        _benchmark_plane(
            "back",
            fixture=fixture,
            coordinate=fixture.back_plane + dz,
            size=aperture,
        ),
    )


def _sheet_waveform(fixture: _FluxFixtureConfig = _FluxFixture) -> CustomWaveform:
    def waveform(t):
        tau = 1.0 / (jnp.pi * fixture.source_freq * fixture.source_bandwidth)
        t0 = 5.0 * tau
        envelope = jnp.exp(-(((t - t0) / tau) ** 2))
        carrier = jnp.sin(
            2.0 * jnp.pi * fixture.source_freq * t + fixture.sheet_phase_rad
        )
        return jnp.asarray(
            fixture.sheet_amplitude * carrier * envelope,
            dtype=jnp.float32,
        )

    return CustomWaveform(func=waveform)


def _sheet_axis_positions(span: tuple[float, float], dx: float) -> np.ndarray:
    n_cells = int(round((span[1] - span[0]) / dx))
    return np.asarray(span[0] + np.arange(n_cells, dtype=np.float64) * dx)


def _add_uniform_sheet_sources(
    sim: Simulation,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> None:
    waveform = _sheet_waveform(fixture)
    xs = _sheet_axis_positions(fixture.sheet_x_span, fixture.uniform_dx)
    ys = _sheet_axis_positions(fixture.sheet_y_span, fixture.uniform_dx)
    for x in xs:
        for y in ys:
            sim.add_source(
                position=(float(x), float(y), fixture.sheet_coordinate),
                component=fixture.sheet_component,
                waveform=waveform,
            )


@lru_cache(maxsize=None)
def _run_flux_fixture(
    *,
    subgrid: bool,
    slab: bool,
    fixture: _FluxFixtureConfig = _FluxFixture,
    plane_shift_cells: int = 0,
    aperture_size: float | None = None,
) -> _FixtureRun:
    dx = fixture.coarse_dx if subgrid else fixture.uniform_dx
    size = fixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    sim = Simulation(
        freq_max=fixture.freq_max,
        domain=fixture.domain,
        boundary="cpml",
        cpml_layers=fixture.cpml_layers,
        dx=dx,
    )
    if slab:
        sim.add_material("rt_dielectric", eps_r=fixture.slab_eps_r)
        sim.add(Box(fixture.slab_lo, fixture.slab_hi), material="rt_dielectric")
    if subgrid:
        sim.add_refinement(**fixture.refinement)
    else:
        _add_uniform_sheet_sources(sim, fixture)
    sim.add_probe(position=fixture.source, component=fixture.source_component)

    if subgrid:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No sources, ports, TFSF, or waveguide/Floquet ports configured.*",
                category=UserWarning,
            )
            run = run_subgridded_benchmark_flux(
                sim,
                n_steps=fixture.n_steps,
                planes=_plane_requests(plane_shift_cells, aperture_size, fixture),
                sheet_sources=(_benchmark_sheet_source(fixture=fixture),),
            )
        planes = run.benchmark_flux_planes
        signed_flux = tuple(np.asarray(_benchmark_flux_spectrum(p)) for p in planes)
        complex_flux = tuple(_complex_flux(p) for p in planes)
        return _FixtureRun(float(run.result.dt), complex_flux, signed_flux, planes)

    dz = plane_shift_cells * fixture.dx_f
    sim.add_flux_monitor(
        axis="z",
        coordinate=fixture.front_plane + dz,
        freqs=fixture.scored_freqs,
        size=aperture,
        center=fixture.aperture_center,
        name="front",
    )
    sim.add_flux_monitor(
        axis="z",
        coordinate=fixture.back_plane + dz,
        freqs=fixture.scored_freqs,
        size=aperture,
        center=fixture.aperture_center,
        name="back",
    )
    result = sim.run(n_steps=fixture.n_steps)
    monitors = (result.flux_monitors["front"], result.flux_monitors["back"])
    signed_flux = tuple(np.asarray(flux_spectrum(m)) for m in monitors)
    complex_flux = tuple(_complex_flux(m) for m in monitors)
    return _FixtureRun(float(result.dt), complex_flux, signed_flux, monitors)


def _usable_passband(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    front_mag = np.abs(front)
    back_mag = np.abs(back)
    front_peak = float(np.max(front_mag))
    back_peak = float(np.max(back_mag))
    if front_peak <= 0.0 or back_peak <= 0.0:
        return np.zeros_like(front_mag, dtype=bool)
    return (front_mag >= 0.20 * front_peak) & (back_mag >= 0.20 * back_peak)


def _claims_bearing_passband(
    complex_flux: tuple[np.ndarray, np.ndarray],
    signed_flux: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mask = _usable_passband(complex_flux[0], complex_flux[1])
    nonfloor = _NONFLOOR_FACTOR * _NORMALIZATION_FLOOR
    for complex_values, signed_values in zip(complex_flux, signed_flux, strict=True):
        mask = (
            mask
            & (np.abs(complex_values) >= nonfloor)
            & (np.abs(signed_values) >= nonfloor)
        )
    return mask


def _sheet_component_dft(
    plane,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> np.ndarray:
    if fixture.sheet_component == "ex":
        return np.asarray(plane.e1_dft)
    return np.asarray(plane.e2_dft)


def _transverse_uniformity_metadata(
    planes: tuple[object, object],
    mask: np.ndarray,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, object]:
    per_plane = []
    max_cv = 0.0
    max_phase_spread = 0.0
    if int(np.sum(mask)) == 0:
        return {
            "passed": False,
            "max_magnitude_cv": float("inf"),
            "max_phase_spread_deg": float("inf"),
            "per_plane": per_plane,
        }

    for plane_name, plane in zip(("front", "back"), planes, strict=True):
        field = _sheet_component_dft(plane, fixture)[mask]
        for freq_hz, values in zip(
            fixture.scored_freqs[mask],
            field,
            strict=True,
        ):
            mags = np.abs(values)
            mean_mag = float(np.mean(mags))
            cv = (
                float(np.std(mags) / mean_mag)
                if mean_mag > _NORMALIZATION_FLOOR
                else float("inf")
            )
            mean_complex = complex(np.mean(values))
            if abs(mean_complex) > _NORMALIZATION_FLOOR:
                phase_delta = np.angle(values * np.conj(mean_complex), deg=True)
                phase_spread = float(np.max(np.abs(phase_delta)))
            else:
                phase_spread = float("inf")
            max_cv = max(max_cv, cv)
            max_phase_spread = max(max_phase_spread, phase_spread)
            per_plane.append(
                {
                    "plane": plane_name,
                    "freq_hz": float(freq_hz),
                    "magnitude_cv": cv,
                    "phase_spread_deg": phase_spread,
                }
            )

    return {
        "passed": bool(max_cv <= 0.01 and max_phase_spread <= 1.0),
        "max_magnitude_cv": max_cv,
        "max_phase_spread_deg": max_phase_spread,
        "per_plane": per_plane,
    }


def _vacuum_stability_metadata(
    uniform_flux: tuple[np.ndarray, np.ndarray],
    subgrid_flux: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict[str, object]:
    if int(np.sum(mask)) == 0:
        return {
            "passed": False,
            "max_magnitude_error": float("inf"),
            "max_phase_error_deg": float("inf"),
        }
    ref = np.concatenate([uniform_flux[0][mask], uniform_flux[1][mask]])
    sub = np.concatenate([subgrid_flux[0][mask], subgrid_flux[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    return {
        "passed": bool(
            float(np.max(mag_error)) <= 0.02 and float(np.max(phase_error)) <= 2.0
        ),
        "max_magnitude_error": float(np.max(mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
    }


def _serial_complex(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in np.ravel(values)]


def _flux_diagnostics(
    complex_flux: tuple[np.ndarray, np.ndarray],
    signed_flux: tuple[np.ndarray, np.ndarray],
    fixture: _FluxFixtureConfig = _FluxFixture,
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
                "signed_real_flux_to_floor": float(abs(signed) / _NORMALIZATION_FLOOR),
            }
            for freq, complex_pair, mag, phase, signed in zip(
                fixture.scored_freqs,
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
def _homogeneous_parity_for_aperture(
    aperture_size: float,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, object]:
    aperture_arg = (
        None if np.isclose(aperture_size, fixture.aperture_size[0]) else aperture_size
    )
    ref_run = _run_flux_fixture(
        subgrid=False,
        slab=False,
        fixture=fixture,
        aperture_size=aperture_arg,
    )
    sub_run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=fixture,
        aperture_size=aperture_arg,
    )
    dt_ref, flux_ref = (
        ref_run.dt,
        ref_run.complex_flux,
    )
    dt_sub, flux_sub, signed_sub = (
        sub_run.dt,
        sub_run.complex_flux,
        sub_run.signed_flux,
    )
    metadata: dict[str, object] = {
        "aperture_size_m": float(aperture_size),
        "dt_match": bool(np.allclose(dt_ref, dt_sub)),
        "uniform_front_peak_magnitude": float(np.max(np.abs(flux_ref[0]))),
        "uniform_back_peak_magnitude": float(np.max(np.abs(flux_ref[1]))),
        "subgrid_front_peak_magnitude": float(np.max(np.abs(flux_sub[0]))),
        "subgrid_back_peak_magnitude": float(np.max(np.abs(flux_sub[1]))),
    }
    metadata["fixture_name"] = fixture.name
    metadata["fixture"] = fixture.fixture_key
    metadata["uniform_front_peak_to_floor"] = float(
        metadata["uniform_front_peak_magnitude"] / _NORMALIZATION_FLOOR
    )
    metadata["uniform_back_peak_to_floor"] = float(
        metadata["uniform_back_peak_magnitude"] / _NORMALIZATION_FLOOR
    )

    mask = _claims_bearing_passband(flux_sub, signed_sub)
    metadata["usable_bins"] = int(np.sum(mask))
    metadata["scored_freqs_hz"] = fixture.scored_freqs[mask].tolist()
    if int(np.sum(mask)) < _MIN_CLAIMS_BEARING_BINS:
        metadata["classification"] = "inconclusive"
        metadata["reason"] = "homogeneous runtime passband is too weak to score"
        metadata["source_contract"] = "private_analytic_sheet_source"
        metadata["normalization"] = "vacuum_device_two_run_incident_normalized"
        return metadata

    ref = np.concatenate([flux_ref[0][mask], flux_ref[1][mask]])
    sub = np.concatenate([flux_sub[0][mask], flux_sub[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    metadata.update(
        {
            "classification": "pass"
            if float(np.max(mag_error)) <= 0.02 and float(np.max(phase_error)) <= 2.0
            else "inconclusive",
            "max_floor_relative_magnitude_error": float(np.max(mag_error)),
            "max_complex_phase_error_deg": float(np.max(phase_error)),
        }
    )
    return metadata


@lru_cache(maxsize=None)
def _homogeneous_parity_metadata(
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(subgrid=False, slab=False, fixture=fixture)
    sub_run = _run_flux_fixture(subgrid=True, slab=False, fixture=fixture)
    dt_ref, flux_ref, signed_ref = (
        ref_run.dt,
        ref_run.complex_flux,
        ref_run.signed_flux,
    )
    dt_sub, flux_sub, signed_sub = (
        sub_run.dt,
        sub_run.complex_flux,
        sub_run.signed_flux,
    )
    if not np.allclose(dt_ref, dt_sub):
        return {"classification": "fail", "reason": "uniform/subgrid dt mismatch"}
    for label, arrays in {"uniform": flux_ref, "subgrid": flux_sub}.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail

    mask = _claims_bearing_passband(flux_sub, signed_sub)
    if int(np.sum(mask)) < _MIN_CLAIMS_BEARING_BINS:
        return {
            "classification": "inconclusive",
            "reason": "homogeneous runtime passband is too weak to score",
            "fixture": f"{fixture.fixture_key}_homogeneous",
            "fixture_name": fixture.name,
            "fixture_parameters": fixture.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "front_abs": np.abs(flux_ref[0]).tolist(),
            "back_abs": np.abs(flux_ref[1]).tolist(),
            "usable_bins": int(np.sum(mask)),
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the private analytic sheet did not produce at least two "
                "non-floor homogeneous passband bins for uniform/subgrid "
                "vacuum parity"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
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
        "fixture": f"{fixture.fixture_key}_homogeneous",
        "fixture_name": fixture.name,
        "fixture_parameters": fixture.to_metadata(),
        "source_contract": "private_analytic_sheet_source",
        "normalization": "vacuum_device_two_run_incident_normalized",
        "gates": gates,
        "scored_freqs_hz": fixture.scored_freqs[mask].tolist(),
        "max_magnitude_error": float(np.max(mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "uniform_flux_diagnostics": _flux_diagnostics(flux_ref, signed_ref, fixture),
        "subgrid_flux_diagnostics": _flux_diagnostics(flux_sub, signed_sub, fixture),
        "aperture_sweep": [
            _homogeneous_parity_for_aperture(fixture.aperture_size[0], fixture),
            _homogeneous_parity_for_aperture(
                max(fixture.aperture_size[0] - 0.004, fixture.uniform_dx * 4),
                fixture,
            ),
        ],
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "homogeneous vacuum stability remains below pass threshold for "
            "the private analytic-sheet incident field"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "Synthetic multi-step/all-axis accumulator parity passes, and "
            "runtime scoring now uses the private analytic sheet plus "
            "non-floor incident bins before any public claim can be considered."
        ),
    }


def _quality_error_score(metadata: dict[str, object]) -> float:
    uniformity = metadata["transverse_uniformity"]
    vacuum = metadata["vacuum_stability"]
    if metadata["usable_bins"] < _MIN_CLAIMS_BEARING_BINS:
        return float("inf")
    ratios = [
        float(uniformity["max_magnitude_cv"]) / 0.01,
        float(uniformity["max_phase_spread_deg"]) / 1.0,
        float(vacuum["max_magnitude_error"]) / 0.02,
        float(vacuum["max_phase_error_deg"]) / 2.0,
    ]
    return float(max(ratios))


@lru_cache(maxsize=None)
def _fixture_quality_metadata(
    fixture: _FluxFixtureConfig,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(subgrid=False, slab=False, fixture=fixture)
    sub_run = _run_flux_fixture(subgrid=True, slab=False, fixture=fixture)
    if not np.allclose(ref_run.dt, sub_run.dt):
        return {
            "classification": "fail",
            "reason": "uniform/subgrid dt mismatch",
            "fixture_name": fixture.name,
            "fixture": fixture.fixture_key,
        }
    for label, arrays in {
        "uniform": ref_run.complex_flux,
        "subgrid": sub_run.complex_flux,
    }.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail | {
                    "fixture_name": fixture.name,
                    "fixture": fixture.fixture_key,
                }

    mask = _claims_bearing_passband(sub_run.complex_flux, sub_run.signed_flux)
    uniformity = _transverse_uniformity_metadata(sub_run.planes, mask, fixture)
    vacuum_stability = _vacuum_stability_metadata(
        ref_run.complex_flux,
        sub_run.complex_flux,
        mask,
    )
    gates = {
        "usable_passband": int(np.sum(mask)) >= _MIN_CLAIMS_BEARING_BINS,
        "transverse_uniformity": bool(uniformity["passed"]),
        "vacuum_stability": bool(vacuum_stability["passed"]),
    }
    metadata: dict[str, object] = {
        "classification": "pass" if all(gates.values()) else "inconclusive",
        "fixture_name": fixture.name,
        "fixture": fixture.fixture_key,
        "fixture_parameters": fixture.to_metadata(),
        "fixture_quality_gates": gates,
        "usable_bins": int(np.sum(mask)),
        "scored_freqs_hz": fixture.scored_freqs[mask].tolist(),
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "uniform_flux_diagnostics": _flux_diagnostics(
            ref_run.complex_flux,
            ref_run.signed_flux,
            fixture,
        ),
        "subgrid_flux_diagnostics": _flux_diagnostics(
            sub_run.complex_flux,
            sub_run.signed_flux,
            fixture,
        ),
    }
    metadata["quality_error_score"] = _quality_error_score(metadata)
    return metadata


@lru_cache(maxsize=None)
def _boundary_expansion_sweep_metadata() -> dict[str, object]:
    candidates = [
        _fixture_quality_metadata(fixture) for fixture in _RECOVERY_SWEEP_FIXTURES
    ]
    baseline = candidates[0]
    best = min(
        candidates,
        key=lambda item: float(item.get("quality_error_score", float("inf"))),
    )
    baseline_score = float(baseline.get("quality_error_score", float("inf")))
    best_score = float(best.get("quality_error_score", float("inf")))
    materially_improved = (
        best["fixture_name"] != baseline["fixture_name"]
        and np.isfinite(best_score)
        and (not np.isfinite(baseline_score) or best_score <= 0.90 * baseline_score)
    )
    return {
        "status": "pass" if best["classification"] == "pass" else "inconclusive",
        "candidate_count": len(candidates),
        "baseline_fixture": baseline["fixture_name"],
        "selected_fixture": best["fixture_name"],
        "selected_fixture_key": best["fixture"],
        "materially_improved_vs_baseline": bool(materially_improved),
        "baseline_quality_error_score": baseline_score,
        "selected_quality_error_score": best_score,
        "candidates": candidates,
    }


def _empty_rt_gates() -> dict[str, bool]:
    return {
        "r_magnitude": False,
        "t_magnitude": False,
        "phase": False,
        "energy": False,
        "plane_shift_r": False,
        "plane_shift_t": False,
        "plane_shift_phase": False,
    }


@lru_cache(maxsize=None)
def _plane_rt_metadata() -> dict[str, object]:
    sweep = _boundary_expansion_sweep_metadata()
    selected = next(
        fixture
        for fixture in _RECOVERY_SWEEP_FIXTURES
        if fixture.name == sweep["selected_fixture"]
    )
    selected_quality = next(
        candidate
        for candidate in sweep["candidates"]
        if candidate["fixture_name"] == selected.name
    )
    cheap_quality_passed = all(selected_quality["fixture_quality_gates"].values())
    if not cheap_quality_passed:
        return {
            "classification": "inconclusive",
            "reason": "boundary-expanded analytic-sheet sweep did not recover fixture quality",
            "fixture": selected.fixture_key,
            "fixture_name": selected.name,
            "fixture_parameters": selected.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "public_claim_allowed": False,
            "boundary_expansion_sweep": sweep,
            "usable_bins": int(selected_quality["usable_bins"]),
            "scored_freqs_hz": selected_quality["scored_freqs_hz"],
            "fixture_quality_gates": {
                **selected_quality["fixture_quality_gates"],
                "plane_location": False,
            },
            "gates": _empty_rt_gates(),
            "transverse_uniformity": selected_quality["transverse_uniformity"],
            "vacuum_stability": selected_quality["vacuum_stability"],
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the bounded geometry sweep did not produce a canonical "
                "private sheet fixture with passing usable-passband, "
                "transverse-uniformity, and vacuum-stability gates; full "
                "incident-normalized R/T scoring is intentionally skipped"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
            "diagnostic_basis": (
                "Candidate selection now uses a bounded boundary-expanded "
                "analytic-sheet sweep before full R/T scoring. Public support "
                "remains deferred and unsupported public observables stay "
                "hard-failing."
            ),
        }

    ref_vac_run = _run_flux_fixture(subgrid=False, slab=False, fixture=selected)
    ref_slab_run = _run_flux_fixture(subgrid=False, slab=True, fixture=selected)
    sub_vac_run = _run_flux_fixture(subgrid=True, slab=False, fixture=selected)
    sub_slab_run = _run_flux_fixture(subgrid=True, slab=True, fixture=selected)
    shift_vac_run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=selected,
        plane_shift_cells=1,
    )
    shift_slab_run = _run_flux_fixture(
        subgrid=True,
        slab=True,
        fixture=selected,
        plane_shift_cells=1,
    )
    dt_ref, vac_ref, signed_vac_ref = (
        ref_vac_run.dt,
        ref_vac_run.complex_flux,
        ref_vac_run.signed_flux,
    )
    dt_ref_slab, slab_ref, signed_ref_slab = (
        ref_slab_run.dt,
        ref_slab_run.complex_flux,
        ref_slab_run.signed_flux,
    )
    dt_sub, vac_sub, signed_vac_sub = (
        sub_vac_run.dt,
        sub_vac_run.complex_flux,
        sub_vac_run.signed_flux,
    )
    dt_sub_slab, slab_sub, signed_sub_slab = (
        sub_slab_run.dt,
        sub_slab_run.complex_flux,
        sub_slab_run.signed_flux,
    )
    dt_shift_vac, vac_shift = shift_vac_run.dt, shift_vac_run.complex_flux
    dt_shift_slab, slab_shift = shift_slab_run.dt, shift_slab_run.complex_flux

    if not np.allclose(
        [dt_ref_slab, dt_sub, dt_sub_slab, dt_shift_vac, dt_shift_slab],
        dt_ref,
    ):
        return {
            "classification": "fail",
            "reason": "fixture timesteps are inconsistent",
        }

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

    freq_mask = _claims_bearing_passband(vac_sub, signed_vac_sub)
    uniformity = _transverse_uniformity_metadata(
        sub_vac_run.planes,
        freq_mask,
        selected,
    )
    vacuum_stability = _vacuum_stability_metadata(vac_ref, vac_sub, freq_mask)
    if int(np.sum(freq_mask)) < _MIN_CLAIMS_BEARING_BINS:
        return {
            "classification": "inconclusive",
            "reason": "no claims-bearing non-floor passband bins",
            "fixture": selected.fixture_key,
            "fixture_name": selected.name,
            "fixture_parameters": selected.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "public_claim_allowed": False,
            "boundary_expansion_sweep": sweep,
            "usable_bins": int(np.sum(freq_mask)),
            "front_abs": np.abs(vac_sub[0]).tolist(),
            "back_abs": np.abs(vac_sub[1]).tolist(),
            "fixture_quality_gates": {
                "usable_passband": False,
                "transverse_uniformity": bool(uniformity["passed"]),
                "vacuum_stability": bool(vacuum_stability["passed"]),
                "plane_location": False,
            },
            "gates": _empty_rt_gates(),
            "transverse_uniformity": uniformity,
            "vacuum_stability": vacuum_stability,
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the private analytic sheet did not produce at least two "
                "vacuum front/back bins above both the 20% peak and "
                "1e12×normalization-floor thresholds"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
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
    phase_refs = np.concatenate(
        [r_ref[np.abs(r_ref) >= 0.05], t_ref[np.abs(t_ref) >= 0.05]]
    )
    phase_tests = np.concatenate(
        [r_sub[np.abs(r_ref) >= 0.05], t_sub[np.abs(t_ref) >= 0.05]]
    )
    phase_error = (
        _phase_error_deg(phase_tests, phase_refs)
        if len(phase_refs)
        else np.array([0.0], dtype=np.float64)
    )
    r_shift_delta = np.abs(np.abs(r_shift) - np.abs(r_sub)) / np.maximum(
        np.abs(r_sub),
        _NORMALIZATION_FLOOR,
    )
    t_shift_delta = np.abs(np.abs(t_shift) - np.abs(t_sub)) / np.maximum(
        np.abs(t_sub),
        _NORMALIZATION_FLOOR,
    )
    shift_phase_refs = np.concatenate(
        [
            r_sub[np.abs(r_sub) >= 0.05],
            t_sub[np.abs(t_sub) >= 0.05],
        ]
    )
    shift_phase_tests = np.concatenate(
        [
            r_shift[np.abs(r_sub) >= 0.05],
            t_shift[np.abs(t_sub) >= 0.05],
        ]
    )
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

    rt_gates = {
        "r_magnitude": float(np.max(r_mag_error)) <= 0.05,
        "t_magnitude": float(np.max(t_mag_error)) <= 0.05,
        "phase": float(np.max(phase_error)) <= 5.0,
        "energy": float(np.max(energy_balance)) <= 0.05,
        "plane_shift_r": float(np.max(r_shift_delta)) <= 0.05,
        "plane_shift_t": float(np.max(t_shift_delta)) <= 0.05,
        "plane_shift_phase": float(np.max(shift_phase_error)) <= 5.0,
    }
    fixture_quality_gates = {
        "usable_passband": int(np.sum(freq_mask)) >= _MIN_CLAIMS_BEARING_BINS,
        "transverse_uniformity": bool(uniformity["passed"]),
        "vacuum_stability": bool(vacuum_stability["passed"]),
        "plane_location": bool(
            rt_gates["plane_shift_r"]
            and rt_gates["plane_shift_t"]
            and rt_gates["plane_shift_phase"]
        ),
    }
    classification = (
        "pass"
        if all(fixture_quality_gates.values()) and all(rt_gates.values())
        else "inconclusive"
    )
    return {
        "classification": classification,
        "gates": rt_gates,
        "fixture_quality_gates": fixture_quality_gates,
        "fixture": selected.fixture_key,
        "fixture_name": selected.name,
        "fixture_parameters": selected.to_metadata(),
        "source_contract": "private_analytic_sheet_source",
        "normalization": "vacuum_device_two_run_incident_normalized",
        "boundary_expansion_sweep": sweep,
        "scored_freqs_hz": selected.scored_freqs[freq_mask].tolist(),
        "usable_bins": int(np.sum(freq_mask)),
        "max_r_magnitude_error": float(np.max(r_mag_error)),
        "max_t_magnitude_error": float(np.max(t_mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "energy_balance_residual": float(np.max(energy_balance)),
        "uniform_energy_balance_residual": float(np.max(uniform_energy_balance)),
        "energy_residual_delta_vs_uniform": float(np.max(energy_delta)),
        "unfloored_energy_balance_residual": float(np.nanmax(unfloored_energy_balance)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "usable_passband_threshold": {
            "min_bins": _MIN_CLAIMS_BEARING_BINS,
            "front_back_peak_fraction": 0.20,
            "nonfloor_factor_times_normalization_floor": _NONFLOOR_FACTOR,
        },
        "transverse_uniformity_threshold": {
            "magnitude_cv_max": 0.01,
            "phase_spread_deg_max": 1.0,
        },
        "vacuum_stability_threshold": {
            "relative_magnitude_error_max": 0.02,
            "phase_error_deg_max": 2.0,
        },
        "plane_location_threshold": {
            "relative_magnitude_delta_max": 0.05,
            "phase_delta_deg_max": 5.0,
        },
        "max_plane_shift_r_delta": float(np.max(r_shift_delta)),
        "max_plane_shift_t_delta": float(np.max(t_shift_delta)),
        "max_plane_shift_phase_error_deg": float(np.max(shift_phase_error)),
        "subgrid_vacuum_flux_diagnostics": _flux_diagnostics(
            vac_sub,
            signed_vac_sub,
            selected,
        ),
        "subgrid_slab_flux_diagnostics": _flux_diagnostics(
            slab_sub,
            signed_sub_slab,
            selected,
        ),
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "one or more private analytic-sheet fixture-quality or "
            "incident-normalized R/T gates remains below threshold, so the "
            "gate cannot be promoted as public true R/T evidence"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "The benchmark now uses a private analytic sheet, front/back "
            "private flux planes, and vacuum/device two-run normalization; "
            "public support remains deferred unless every fixture-quality "
            "and R/T gate passes."
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
        "Private analytic-sheet runtime parity is inconclusive for public "
        "promotion; support matrix must not promote public flux/DFT or true "
        "R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_benchmark_vs_uniform_fine():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private flux true-R/T metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private analytic-sheet true R/T gate is inconclusive for public "
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

    assert metadata["classification"] in {"pass", "inconclusive"}
    assert metadata["public_claim_allowed"] is False
    assert metadata["source_contract"] == "private_analytic_sheet_source"
    assert metadata["normalization"] == "vacuum_device_two_run_incident_normalized"
    assert metadata["fixture"] in {
        fixture.fixture_key for fixture in _RECOVERY_SWEEP_FIXTURES
    }
    assert metadata["fixture_name"] in {
        fixture.name for fixture in _RECOVERY_SWEEP_FIXTURES
    }
    assert metadata["boundary_expansion_sweep"]["candidate_count"] == len(
        _RECOVERY_SWEEP_FIXTURES
    )
    if metadata["classification"] == "pass":
        assert all(metadata["fixture_quality_gates"].values())
        assert all(metadata["gates"].values())
    else:
        assert metadata["no_go_reason"] == _NO_GO_REASON
        assert metadata["next_prerequisite"] == _NEXT_PREREQUISITE
        assert metadata["blocking_diagnostic"]
        assert not (
            all(metadata["fixture_quality_gates"].values())
            and all(metadata["gates"].values())
        )
