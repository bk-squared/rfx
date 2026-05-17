"""Research-only public runner for the Stage-2 disjoint topology."""

from __future__ import annotations

import math

import jax.numpy as jnp

from rfx.subgridding.disjoint_3d import (
    init_disjoint_subgrid_3d,
    step_disjoint_z_slab_sat_3d,
)
from rfx.subgridding.disjoint_runner_contract import build_disjoint_runner_contract


def _add_fine_source(state, component: str, index: tuple[int, int, int], value):
    name = f"{component}_f"
    field = getattr(state, name)
    return state._replace(**{name: field.at[index].add(value)})


def _sample_fine_probe(state, component: str, index: tuple[int, int, int]):
    return getattr(state, f"{component}_f")[index]


def run_disjoint_stage2_path(
    sim,
    grid_coarse,
    n_steps: int,
):
    """Run a minimal research-only Stage-2 disjoint z-slab smoke path.

    This consumes the public mapping contract and records fine-grid point
    probes.  It intentionally supports only soft sources and point probes; all
    production claims remain blocked by validation until waveform and external
    gates pass.
    """
    from rfx.api import Result

    unsupported_ports = [pe for pe in sim._ports if pe.impedance > 0.0]
    if unsupported_ports:
        raise NotImplementedError(
            "Stage-2 disjoint research runner currently supports only soft "
            "sources; impedance ports remain unintegrated."
        )
    if sim._ntff is not None:
        raise NotImplementedError(
            "Stage-2 disjoint research runner does not yet support NTFF."
        )

    contract = build_disjoint_runner_contract(sim, grid_coarse)
    ref = sim._refinement
    shape_convention = str(ref.get("disjoint_shape_convention", "cell_extent"))
    default_projection = "node_adjoint" if shape_convention == "endpoint_node" else "repeat_mean"
    config, state = init_disjoint_subgrid_3d(
        shape_c=contract.shape_c,
        fine_region=contract.fine_region,
        dx_c=contract.dx_c,
        ratio=contract.ratio,
        courant=float(ref.get("disjoint_courant", 0.45)),
        sat_strength=float(ref.get("disjoint_sat_strength", 0.02)),
        face_projection=str(ref.get("disjoint_face_projection", default_projection)),
        shape_convention=shape_convention,
    )
    if tuple(config.shape_f) != tuple(contract.shape_f):
        raise RuntimeError(
            "internal Stage-2 disjoint shape mismatch: "
            f"config.shape_f={config.shape_f!r}, contract.shape_f={contract.shape_f!r}"
        )
    stepper_name = str(ref.get("disjoint_stepper", "post_sat"))
    if stepper_name == "post_sat":
        stepper = step_disjoint_z_slab_sat_3d
    else:
        raise ValueError(
            "unknown Stage-2 disjoint stepper "
            f"{stepper_name!r}; expected 'post_sat'"
        )
    source_timing = str(ref.get("disjoint_source_timing", "pre_step"))
    if source_timing not in {"pre_step", "post_step"}:
        raise ValueError(
            "unknown Stage-2 disjoint source timing "
            f"{source_timing!r}; expected 'pre_step' or 'post_step'"
        )
    source_scale = float(ref.get("disjoint_source_scale", 1.0))
    if not math.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError(
            "Stage-2 disjoint source scale must be a positive finite value; "
            f"got {source_scale!r}"
        )

    source_entries = [
        (mapping, pe)
        for mapping, pe in zip(contract.source_mappings, sim._ports)
        if pe.impedance == 0.0 and pe.waveform is not None
    ]
    probe_entries = [
        (mapping, probe)
        for mapping, probe in zip(contract.probe_mappings, sim._probes)
    ]
    samples = []
    for step in range(int(n_steps)):
        t = float(step * config.dt)
        if source_timing == "pre_step":
            for mapping, pe in source_entries:
                state = _add_fine_source(
                    state,
                    mapping.component,
                    mapping.fine_index,
                    jnp.asarray(
                        source_scale * pe.waveform(t),
                        dtype=state.ex_f.dtype,
                    ),
                )
        state = stepper(state, config)
        if source_timing == "post_step":
            for mapping, pe in source_entries:
                state = _add_fine_source(
                    state,
                    mapping.component,
                    mapping.fine_index,
                    jnp.asarray(
                        source_scale * pe.waveform(t),
                        dtype=state.ex_f.dtype,
                    ),
                )
        samples.append(
            [
                _sample_fine_probe(state, mapping.component, mapping.fine_index)
                for mapping, _probe in probe_entries
            ]
        )
    if samples:
        time_series = jnp.asarray(samples, dtype=state.ex_f.dtype)
    else:
        time_series = jnp.zeros((int(n_steps), 0), dtype=state.ex_f.dtype)

    return Result(
        state=state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=None,
        dt=float(config.dt),
        freq_range=(sim._freq_max / 10, sim._freq_max, "pec"),
    )
