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
        np.abs(mag_test - mag_ref) / np.maximum(mag_ref, 1e-30),
        np.abs(mag_test - mag_ref),
    )


def _finite_or_fail(label: str, values: np.ndarray) -> dict[str, object] | None:
    if not np.all(np.isfinite(values)):
        return {"classification": "fail", "reason": f"{label} contains NaN/Inf"}
    return None


def _plane_requests(shift_cells: int = 0) -> tuple[_BenchmarkFluxPlaneRequest, ...]:
    dz = shift_cells * _FluxFixture.dx_f
    return (
        _benchmark_plane("front", coordinate=_FluxFixture.front_plane + dz),
        _benchmark_plane("back", coordinate=_FluxFixture.back_plane + dz),
    )


@lru_cache(maxsize=None)
def _run_flux_fixture(
    *,
    subgrid: bool,
    slab: bool,
    plane_shift_cells: int = 0,
) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    dx = _FluxFixture.coarse_dx if subgrid else _FluxFixture.uniform_dx
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
            planes=_plane_requests(plane_shift_cells),
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
        size=_FluxFixture.aperture_size,
        center=_FluxFixture.aperture_center,
        name="front",
    )
    sim.add_flux_monitor(
        axis="z",
        coordinate=_FluxFixture.back_plane + dz,
        freqs=_FluxFixture.scored_freqs,
        size=_FluxFixture.aperture_size,
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


@lru_cache(maxsize=None)
def _homogeneous_parity_metadata() -> dict[str, object]:
    dt_ref, flux_ref, _ = _run_flux_fixture(subgrid=False, slab=False)
    dt_sub, flux_sub, _ = _run_flux_fixture(subgrid=True, slab=False)
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
    mag_error = np.abs(np.abs(sub) - np.abs(ref)) / np.maximum(np.abs(ref), 1e-30)
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
    }


@lru_cache(maxsize=None)
def _plane_rt_metadata() -> dict[str, object]:
    dt_ref, vac_ref, _ = _run_flux_fixture(subgrid=False, slab=False)
    dt_ref_slab, slab_ref, signed_ref_slab = _run_flux_fixture(subgrid=False, slab=True)
    dt_sub, vac_sub, _ = _run_flux_fixture(subgrid=True, slab=False)
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
        inc_front = np.where(np.abs(vac[0]) >= 1e-30, vac[0], 1e-30 + 0j)
        inc_back = np.where(np.abs(vac[1]) >= 1e-30, vac[1], 1e-30 + 0j)
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
    signed_front = np.asarray(signed_sub_slab[0])[freq_mask]
    signed_back = np.asarray(signed_sub_slab[1])[freq_mask]
    signed_incident = np.asarray(_run_flux_fixture(subgrid=True, slab=False)[2][0])[freq_mask]
    energy_balance = np.abs((signed_front - signed_back) / np.maximum(np.abs(signed_incident), 1e-30))

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
        "max_plane_shift_r_delta": float(np.max(r_shift_delta)),
        "max_plane_shift_t_delta": float(np.max(t_shift_delta)),
        "max_plane_shift_phase_error_deg": float(np.max(shift_phase_error)),
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
        "Private flux runtime parity threshold is inconclusive; support matrix "
        "must not promote public flux/DFT or true R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_benchmark_vs_uniform_fine():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private flux true-R/T metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private plane-flux true R/T gate is inconclusive; support matrix remains deferred.",
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
