"""JIT-compiled subgridded FDTD runner via jax.lax.scan.

Replaces the Python-loop runner (runner.py) with a fully JIT-compiled
version that achieves 50-100x speedup. Both coarse and fine grids
are updated per step with SBP-SAT interface coupling.

Handles both CPML (absorbing) and PEC (reflecting) coarse-grid
boundaries. When cpml_layers == 0 the CPML subsystem is skipped
entirely and PEC is applied on all domain faces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_h, update_e,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D, _shared_node_coupling_3d,
)


class SubgridResult(NamedTuple):
    """Result from JIT-compiled subgridded simulation."""
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
    """Run subgridded FDTD via jax.lax.scan.

    Parameters
    ----------
    grid_c : Grid -- coarse grid (full domain, with or without CPML)
    mats_c : MaterialArrays -- coarse materials
    mats_f : MaterialArrays -- fine materials
    config : SubgridConfig3D
    n_steps : int
    pec_mask_c, pec_mask_f : boolean arrays or None
    sources_f : list of (i, j, k, component, waveform_array)
    probe_indices_f : list of (i, j, k)
    probe_components : list of component names
    """
    sources_f = sources_f or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []

    dt = config.dt
    dx_c = config.dx_c
    dx_f = config.dx_f

    # Override coarse grid dt to match subgrid global timestep
    grid_c.dt = dt

    # ---- Subsystem flags (resolved at trace time, not inside scan) ----
    use_cpml = grid_c.cpml_layers > 0

    cpml_params = None
    cpml_state = None
    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
        cpml_params, cpml_state = init_cpml(grid_c)

    # Initialize field states
    shape_c = (config.nx_c, config.ny_c, config.nz_c)
    shape_f = (config.nx_f, config.ny_f, config.nz_f)

    state_c = init_state(shape_c)
    state_f = init_state(shape_f)

    # Precompute source waveforms matrix (n_steps, n_sources)
    if sources_f:
        src_waveforms = jnp.stack([jnp.array(s[4]) for s in sources_f], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources_f]
    prb_meta = [(p[0], p[1], p[2], c) for p, c in
                zip(probe_indices_f, probe_components)]
    n_probes = len(prb_meta)

    use_pec_mask_c = pec_mask_c is not None
    use_pec_mask_f = pec_mask_f is not None

    # Pack carry — include CPML state only when CPML is active
    carry_init = {
        "c": (state_c.ex, state_c.ey, state_c.ez,
              state_c.hx, state_c.hy, state_c.hz),
        "f": (state_f.ex, state_f.ey, state_f.ez,
              state_f.hx, state_f.hy, state_f.hz),
    }
    if use_cpml:
        carry_init["cpml"] = cpml_state

    cpml_axes = "xyz"

    def step_fn(carry, xs):
        step_idx, src_vals = xs
        ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = carry["c"]
        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = carry["f"]

        # === Coarse H update ===
        st_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                         hx=hx_c, hy=hy_c, hz=hz_c,
                         step=step_idx)
        st_c = update_h(st_c, mats_c, dt, dx_c)
        if use_cpml:
            st_c, cpml_new = apply_cpml_h(st_c, cpml_params, carry["cpml"],
                                           grid_c, cpml_axes)

        # === Fine H update ===
        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=step_idx)
        st_f = update_h(st_f, mats_f, dt, dx_f)

        # === Coarse E update + boundary ===
        st_c = update_e(st_c, mats_c, dt, dx_c)
        if use_cpml:
            st_c, cpml_new = apply_cpml_e(st_c, cpml_params, cpml_new,
                                           grid_c, cpml_axes)
        st_c = apply_pec(st_c)
        if use_pec_mask_c:
            st_c = apply_pec_mask(st_c, pec_mask_c)

        # === Fine E update + PEC mask ===
        st_f = update_e(st_f, mats_f, dt, dx_f)
        if use_pec_mask_f:
            st_f = apply_pec_mask(st_f, pec_mask_f)

        # === SBP-SAT coupling ===
        (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
            _shared_node_coupling_3d(
                (st_c.ex, st_c.ey, st_c.ez),
                (st_f.ex, st_f.ey, st_f.ez),
                config,
            )

        # === Source injection on fine grid ===
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            if sc == "ez":
                ez_f_new = ez_f_new.at[si, sj, sk].add(src_vals[idx_s])
            elif sc == "ex":
                ex_f_new = ex_f_new.at[si, sj, sk].add(src_vals[idx_s])
            elif sc == "ey":
                ey_f_new = ey_f_new.at[si, sj, sk].add(src_vals[idx_s])

        # === Probe samples ===
        if n_probes > 0:
            def _get_field(comp, i, j, k):
                if comp == "ez": return ez_f_new[i, j, k]
                if comp == "ex": return ex_f_new[i, j, k]
                if comp == "ey": return ey_f_new[i, j, k]
                if comp == "hx": return st_f.hx[i, j, k]
                if comp == "hy": return st_f.hy[i, j, k]
                return st_f.hz[i, j, k]

            samples = [_get_field(pc, pi, pj, pk)
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        new_carry = {
            "c": (ex_c_new, ey_c_new, ez_c_new,
                  st_c.hx, st_c.hy, st_c.hz),
            "f": (ex_f_new, ey_f_new, ez_f_new,
                  st_f.hx, st_f.hy, st_f.hz),
        }
        if use_cpml:
            new_carry["cpml"] = cpml_new

        return new_carry, probe_out

    # Run scan
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final_carry, time_series = jax.lax.scan(step_fn, carry_init, xs)

    # Unpack final state
    ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = final_carry["c"]
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = final_carry["f"]

    final_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                        hx=hx_c, hy=hy_c, hz=hz_c,
                        step=jnp.array(n_steps, dtype=jnp.int32))
    final_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                        hx=hx_f, hy=hy_f, hz=hz_f,
                        step=jnp.array(n_steps, dtype=jnp.int32))

    return SubgridResult(
        state_c=final_c,
        state_f=final_f,
        time_series=time_series,
        config=config,
        dt=dt,
    )
