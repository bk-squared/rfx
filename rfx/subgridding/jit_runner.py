"""JIT runner for the canonical Phase-1 z-slab SBP-SAT lane."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.yee import FDTDState, MaterialArrays, init_state
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D,
    SubgridState3D,
    step_subgrid_3d,
    validate_subgrid_config_3d,
)


class SubgridResult(NamedTuple):
    """Result from the canonical JIT subgridded runner."""

    state_c: FDTDState
    state_f: FDTDState
    time_series: jnp.ndarray
    config: SubgridConfig3D
    dt: float


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
) -> SubgridResult:
    """Run the canonical Phase-1 subgridding lane via ``jax.lax.scan``."""

    if grid_c.cpml_layers > 0:
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding does not support CPML/UPML boundaries"
        )
    validate_subgrid_config_3d(config)

    sources_f = sources_f or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []

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

    if sources_f:
        src_waveforms = jnp.stack([jnp.asarray(s[4], dtype=jnp.float32) for s in sources_f], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources_f]
    prb_meta = [(p[0], p[1], p[2], c) for p, c in zip(probe_indices_f, probe_components)]

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

    def step_fn(state: SubgridState3D, xs):
        _, src_vals = xs
        state = step_subgrid_3d(
            state,
            config,
            mats_c=mats_c,
            mats_f=mats_f,
            pec_mask_c=pec_mask_c,
            pec_mask_f=pec_mask_f,
        )
        state = _inject_sources(state, src_vals)
        return state, _sample_probes(state)

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final_state, time_series = jax.lax.scan(step_fn, state_init, xs)

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
    return SubgridResult(
        state_c=final_c,
        state_f=final_f,
        time_series=time_series,
        config=config,
        dt=config.dt,
    )
