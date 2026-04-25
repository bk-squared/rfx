"""JIT runner for the canonical experimental SBP-SAT subgridding lane."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.yee import FDTDState, MaterialArrays, init_state
from rfx.core.dft_utils import dft_window_weight
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D,
    SubgridState3D,
    step_subgrid_3d,
    step_subgrid_3d_with_cpml,
    validate_subgrid_config_3d,
)


_BENCHMARK_FLUX_COMPONENTS = {
    0: ("ey", "ez", "hy", "hz"),
    1: ("ez", "ex", "hz", "hx"),
    2: ("ex", "ey", "hx", "hy"),
}


class _BenchmarkFluxPlaneSpec(NamedTuple):
    """Private fine-owned plane request for SBP-SAT benchmark evidence.

    This is deliberately not the public ``FluxMonitor`` contract.  It is
    consumed only by private benchmark helpers so Phase-1 public DFT/flux
    hard-fails can remain unchanged.
    """

    name: str
    axis: int
    index: int
    freqs: jnp.ndarray
    dx: float
    total_steps: int
    window: str = "rect"
    window_alpha: float = 0.25
    lo1: int = 0
    hi1: int = -1
    lo2: int = 0
    hi2: int = -1


class _BenchmarkFluxPlaneResult(NamedTuple):
    """Private raw DFT accumulators for one SBP-SAT benchmark plane."""

    name: str
    axis: int
    index: int
    freqs: jnp.ndarray
    dx: float
    e1_dft: jnp.ndarray
    e2_dft: jnp.ndarray
    h1_dft: jnp.ndarray
    h2_dft: jnp.ndarray
    lo1: int
    hi1: int
    lo2: int
    hi2: int


class SubgridResult(NamedTuple):
    """Result from the canonical JIT subgridded runner."""

    state_c: FDTDState
    state_f: FDTDState
    time_series: jnp.ndarray
    config: SubgridConfig3D
    dt: float
    benchmark_flux_planes: tuple[_BenchmarkFluxPlaneResult, ...] | None = None


def _benchmark_flux_spectrum(
    plane: _BenchmarkFluxPlaneResult,
) -> jnp.ndarray:
    """Compute signed private benchmark flux from raw DFT accumulators."""

    d_a = plane.dx * plane.dx
    integrand = (
        plane.e1_dft * jnp.conj(plane.h2_dft)
        - plane.e2_dft * jnp.conj(plane.h1_dft)
    )
    return jnp.real(jnp.sum(integrand, axis=(-2, -1))) * d_a


def _empty_benchmark_flux_accumulator(
    plane: _BenchmarkFluxPlaneSpec,
    shape_f: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create raw private flux accumulators for one fine-owned plane."""

    if plane.axis == 0:
        full1, full2 = shape_f[1], shape_f[2]
    elif plane.axis == 1:
        full1, full2 = shape_f[0], shape_f[2]
    else:
        full1, full2 = shape_f[0], shape_f[1]

    hi1 = full1 if plane.hi1 < 0 else plane.hi1
    hi2 = full2 if plane.hi2 < 0 else plane.hi2
    n1 = hi1 - plane.lo1
    n2 = hi2 - plane.lo2
    if n1 <= 0 or n2 <= 0:
        raise ValueError(
            "private SBP-SAT benchmark flux plane has empty tangential extent"
        )

    zeros = jnp.zeros((len(plane.freqs), n1, n2), dtype=jnp.complex128)
    return zeros, zeros, zeros, zeros


def _benchmark_flux_samples(
    state: SubgridState3D,
    axis: int,
    index: int,
    lo1: int,
    hi1: int,
    lo2: int,
    hi2: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample tangential E and co-located H like the uniform scan kernel."""

    e1_name, e2_name, h1_name, h2_name = _BENCHMARK_FLUX_COMPONENTS[axis]
    e1_field = getattr(state, f"{e1_name}_f")
    e2_field = getattr(state, f"{e2_name}_f")
    h1_field = getattr(state, f"{h1_name}_f")
    h2_field = getattr(state, f"{h2_name}_f")
    idx_m1 = index - 1
    if axis == 0:
        e1 = e1_field[index, lo1:hi1, lo2:hi2]
        e2 = e2_field[index, lo1:hi1, lo2:hi2]
        h1 = (
            h1_field[idx_m1, lo1:hi1, lo2:hi2]
            + h1_field[index, lo1:hi1, lo2:hi2]
        ) * 0.5
        h2 = (
            h2_field[idx_m1, lo1:hi1, lo2:hi2]
            + h2_field[index, lo1:hi1, lo2:hi2]
        ) * 0.5
    elif axis == 1:
        e1 = e1_field[lo1:hi1, index, lo2:hi2]
        e2 = e2_field[lo1:hi1, index, lo2:hi2]
        h1 = (
            h1_field[lo1:hi1, idx_m1, lo2:hi2]
            + h1_field[lo1:hi1, index, lo2:hi2]
        ) * 0.5
        h2 = (
            h2_field[lo1:hi1, idx_m1, lo2:hi2]
            + h2_field[lo1:hi1, index, lo2:hi2]
        ) * 0.5
    else:
        e1 = e1_field[lo1:hi1, lo2:hi2, index]
        e2 = e2_field[lo1:hi1, lo2:hi2, index]
        h1 = (
            h1_field[lo1:hi1, lo2:hi2, idx_m1]
            + h1_field[lo1:hi1, lo2:hi2, index]
        ) * 0.5
        h2 = (
            h2_field[lo1:hi1, lo2:hi2, idx_m1]
            + h2_field[lo1:hi1, lo2:hi2, index]
        ) * 0.5
    return e1, e2, h1, h2


def _accumulate_benchmark_flux_plane(
    acc: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    state: SubgridState3D,
    plane: _BenchmarkFluxPlaneSpec,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Accumulate one private fine-owned flux plane in the scan body."""

    e1_acc, e2_acc, h1_acc, h2_acc = acc
    hi1 = plane.hi1 if plane.hi1 >= 0 else e1_acc.shape[1]
    hi2 = plane.hi2 if plane.hi2 >= 0 else e1_acc.shape[2]
    e1, e2, h1, h2 = _benchmark_flux_samples(
        state,
        plane.axis,
        plane.index,
        plane.lo1,
        hi1,
        plane.lo2,
        hi2,
    )

    t_f64 = state.step.astype(jnp.float64) * jnp.float64(dt)
    freqs64 = plane.freqs.astype(jnp.float64)
    weight = dft_window_weight(
        state.step,
        plane.total_steps,
        plane.window,
        plane.window_alpha,
    ).astype(jnp.float64)
    phase_e = jnp.exp(-1j * 2.0 * jnp.pi * freqs64 * t_f64)
    phase_h = jnp.exp(
        -1j * 2.0 * jnp.pi * freqs64 * (t_f64 - jnp.float64(dt * 0.5))
    )
    kernel_e = (phase_e[:, None, None] * dt * weight).astype(jnp.complex128)
    kernel_h = (phase_h[:, None, None] * dt * weight).astype(jnp.complex128)
    return (
        e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
        e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
        h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
        h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
    )


def run_subgridded_jit(
    grid_c: Grid,
    mats_c: MaterialArrays,
    mats_f: MaterialArrays,
    config: SubgridConfig3D,
    n_steps: int,
    *,
    pec_mask_c=None,
    pec_mask_f=None,
    sources_f: list | None = None,
    probe_indices_f: list | None = None,
    probe_components: list | None = None,
    outer_pec_faces: frozenset[str] | None = None,
    outer_pmc_faces: frozenset[str] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    fine_periodic: tuple[bool, bool, bool] = (False, False, False),
    absorber_boundary: str = "pec",
    _benchmark_flux_planes: tuple[_BenchmarkFluxPlaneSpec, ...] | None = None,
) -> SubgridResult:
    """Run the canonical Phase-1 subgridding lane via ``jax.lax.scan``."""

    outer_pec_faces = outer_pec_faces or frozenset()
    outer_pmc_faces = outer_pmc_faces or frozenset()
    use_cpml = (
        absorber_boundary == "cpml"
        and grid_c.cpml_layers > 0
        and bool(getattr(grid_c, "cpml_axes", ""))
    )
    if grid_c.cpml_layers > 0 and absorber_boundary != "cpml":
        raise ValueError(
            "SBP-SAT subgridding supports only the bounded CPML absorbing subset; "
            "UPML remains unsupported"
        )
    if use_cpml and (outer_pec_faces or outer_pmc_faces):
        raise ValueError(
            "SBP-SAT subgridding does not yet support mixed reflector + CPML "
            "absorbing faces"
        )
    if use_cpml and (any(periodic) or any(fine_periodic)):
        raise ValueError(
            "SBP-SAT subgridding does not yet support mixed periodic + CPML "
            "absorbing faces"
        )
    validate_subgrid_config_3d(config)

    sources_f = sources_f or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []
    benchmark_flux_planes = tuple(_benchmark_flux_planes or ())
    use_benchmark_flux = bool(benchmark_flux_planes)

    shape_c = (config.nx_c, config.ny_c, config.nz_c)
    shape_f = (config.nx_f, config.ny_f, config.nz_f)

    init_c = init_state(shape_c)
    init_f = init_state(shape_f)
    state_init = SubgridState3D(
        ex_c=init_c.ex,
        ey_c=init_c.ey,
        ez_c=init_c.ez,
        hx_c=init_c.hx,
        hy_c=init_c.hy,
        hz_c=init_c.hz,
        ex_f=init_f.ex,
        ey_f=init_f.ey,
        ez_f=init_f.ez,
        hx_f=init_f.hx,
        hy_f=init_f.hy,
        hz_f=init_f.hz,
        step=0,
    )
    cpml_params = cpml_state_init = None
    cpml_grid_c = grid_c
    if use_cpml:
        import copy

        from rfx.boundaries.cpml import init_cpml

        # The subgridded lane advances both coarse and fine states with the
        # fine-grid shared timestep, so CPML ADE coefficients must be built
        # with config.dt rather than Grid's coarse-CFL dt.
        cpml_grid_c = copy.copy(grid_c)
        cpml_grid_c.dt = float(config.dt)
        cpml_params, cpml_state_init = init_cpml(cpml_grid_c)

    if sources_f:
        src_waveforms = jnp.stack([jnp.asarray(s[4], dtype=jnp.float32) for s in sources_f], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources_f]
    prb_meta = [(p[0], p[1], p[2], c) for p, c in zip(probe_indices_f, probe_components)]
    flux_acc_init = tuple(
        _empty_benchmark_flux_accumulator(plane, shape_f)
        for plane in benchmark_flux_planes
    )

    def _inject_sources(state: SubgridState3D, src_vals: jnp.ndarray) -> SubgridState3D:
        ex_f, ey_f, ez_f = state.ex_f, state.ey_f, state.ez_f
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            if sc == "ez":
                ez_f = ez_f.at[si, sj, sk].add(src_vals[idx_s])
            elif sc == "ex":
                ex_f = ex_f.at[si, sj, sk].add(src_vals[idx_s])
            elif sc == "ey":
                ey_f = ey_f.at[si, sj, sk].add(src_vals[idx_s])
        return state._replace(ex_f=ex_f, ey_f=ey_f, ez_f=ez_f)

    def _sample_probes(state: SubgridState3D) -> jnp.ndarray:
        if not prb_meta:
            return jnp.zeros(0, dtype=jnp.float32)

        samples = []
        for pi, pj, pk, comp in prb_meta:
            if comp == "ez":
                samples.append(state.ez_f[pi, pj, pk])
            elif comp == "ex":
                samples.append(state.ex_f[pi, pj, pk])
            elif comp == "ey":
                samples.append(state.ey_f[pi, pj, pk])
            elif comp == "hx":
                samples.append(state.hx_f[pi, pj, pk])
            elif comp == "hy":
                samples.append(state.hy_f[pi, pj, pk])
            else:
                samples.append(state.hz_f[pi, pj, pk])
        return jnp.stack(samples)

    def _advance(state: SubgridState3D, cpml_state):
        if use_cpml:
            return step_subgrid_3d_with_cpml(
                state,
                config,
                cpml_params=cpml_params,
                cpml_state=cpml_state,
                grid_c=cpml_grid_c,
                cpml_axes=cpml_grid_c.cpml_axes,
                mats_c=mats_c,
                mats_f=mats_f,
                pec_mask_c=pec_mask_c,
                pec_mask_f=pec_mask_f,
                outer_pec_faces=outer_pec_faces,
                outer_pmc_faces=outer_pmc_faces,
                periodic=periodic,
                fine_periodic=fine_periodic,
            )
        return (
            step_subgrid_3d(
                state,
                config,
                mats_c=mats_c,
                mats_f=mats_f,
                pec_mask_c=pec_mask_c,
                pec_mask_f=pec_mask_f,
                outer_pec_faces=outer_pec_faces,
                outer_pmc_faces=outer_pmc_faces,
                periodic=periodic,
                fine_periodic=fine_periodic,
            ),
            cpml_state,
        )

    def step_fn(carry, xs):
        _, src_vals = xs
        state, cpml_state = carry[0], carry[1]
        state, cpml_state = _advance(state, cpml_state)
        state = _inject_sources(state, src_vals)
        next_carry = (state, cpml_state)
        if use_benchmark_flux:
            flux_accs = tuple(
                _accumulate_benchmark_flux_plane(acc, state, plane, config.dt)
                for acc, plane in zip(carry[2], benchmark_flux_planes)
            )
            next_carry = (state, cpml_state, flux_accs)
        return next_carry, _sample_probes(state)

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    initial_carry = (state_init, cpml_state_init)
    if use_benchmark_flux:
        initial_carry = (state_init, cpml_state_init, flux_acc_init)
    final_carry, time_series = jax.lax.scan(
        step_fn, initial_carry, xs
    )
    final_state = final_carry[0]

    final_c = FDTDState(
        ex=final_state.ex_c,
        ey=final_state.ey_c,
        ez=final_state.ez_c,
        hx=final_state.hx_c,
        hy=final_state.hy_c,
        hz=final_state.hz_c,
        step=jnp.array(n_steps, dtype=jnp.int32),
    )
    final_f = FDTDState(
        ex=final_state.ex_f,
        ey=final_state.ey_f,
        ez=final_state.ez_f,
        hx=final_state.hx_f,
        hy=final_state.hy_f,
        hz=final_state.hz_f,
        step=jnp.array(n_steps, dtype=jnp.int32),
    )
    benchmark_flux_results = None
    if use_benchmark_flux:
        benchmark_flux_results = tuple(
            _BenchmarkFluxPlaneResult(
                name=plane.name,
                axis=plane.axis,
                index=plane.index,
                freqs=plane.freqs,
                dx=plane.dx,
                e1_dft=accs[0],
                e2_dft=accs[1],
                h1_dft=accs[2],
                h2_dft=accs[3],
                lo1=plane.lo1,
                hi1=plane.hi1 if plane.hi1 >= 0 else accs[0].shape[1],
                lo2=plane.lo2,
                hi2=plane.hi2 if plane.hi2 >= 0 else accs[0].shape[2],
            )
            for plane, accs in zip(benchmark_flux_planes, final_carry[2])
        )

    return SubgridResult(
        state_c=final_c,
        state_f=final_f,
        time_series=time_series,
        config=config,
        dt=config.dt,
        benchmark_flux_planes=benchmark_flux_results,
    )
