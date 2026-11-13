"""Whole-grid microstrip realization, before any slab owns an update."""

from typing import NamedTuple

import numpy as np


class MSLRealization(NamedTuple):
    materials: object
    materials_drive: object
    conductors: object
    drives: list


def realize_msl_ports(sim, grid, materials, conductors, *, eps_source,
                      stamp_drive_materials, edge_release, eigenmode,
                      geometry_edges, sources=None, n_steps=None):
    """Validate, profile, stamp, release, then defer electric drives.

    ``eps_source`` is the path's concrete fixture permittivity;
    ``stamp_drive_materials`` is the graded drive copy, or None;
    ``edge_release`` selects the run conductor release; ``eigenmode`` names
    the caller for the magnetic-channel refusal. Drives retain declaration
    and stamp order and contain (cell, component, waveform, sigma, field).
    """
    from rfx.model.conductors import clear_conductor_edges
    from rfx.sources.msl_port import (
        _msl_yz_cells, compute_msl_mode_profile, make_msl_port_sources,
        msl_cell, msl_cross_section_span, msl_normal_component,
        msl_port_from_entry, setup_msl_port, validate_msl_port_geometry,
    )

    drives = []
    for index, entry in enumerate(sim._msl_ports):
        port = msl_port_from_entry(entry)
        validate_msl_port_geometry(
            grid, port, pec_edge_masks=geometry_edges, conductors=conductors,
            periodic=sim._periodic_flags(),
            pec_faces=sim._boundary_spec.pec_faces(), name=entry.name)
        mode = entry.mode
        if mode == "eigenmode":
            raise NotImplementedError(
                f"{eigenmode} refuses microstrip mode='eigenmode': the J+M "
                "launch needs a magnetic-source channel, which slabs do not carry.")
        profile = None
        if mode == "laplace":
            span = msl_cross_section_span(grid, port)
            cell = msl_cell(entry.direction, span['i_feed'], span['w_centre'],
                            (span['n_lo'] + span['n_hi']) // 2)
            eps = float(entry.eps_r_sub) if entry.eps_r_sub is not None else float(np.asarray(eps_source[cell]))
            profile = compute_msl_mode_profile(grid, port, eps)
        elif mode != "uniform":
            raise ValueError(f"Unsupported microstrip mode: {mode!r}")
        materials = setup_msl_port(grid, port, materials, mode_profile=profile)
        if stamp_drive_materials is not None:
            stamp_drive_materials = setup_msl_port(
                grid, port, stamp_drive_materials, mode_profile=profile)
        if edge_release != "run":
            raise ValueError(f"Unsupported microstrip edge release: {edge_release!r}")
        cells = list(_msl_yz_cells(grid, port))
        conductors = clear_conductor_edges(
            conductors, cells, component=msl_normal_component(port),
            entity_id=f"msl_port[{index}]", port_kind="microstrip",
            released_edges=profile['cell_indices'] if profile is not None else cells)
        if entry.excite and entry.waveform is not None:
            drives.extend((cell, "ez", entry.waveform, sigma, field)
                          for cell, (sigma, field) in port._drive_stamps.items())
            if sources is not None:
                sources.defer(make_msl_port_sources, grid, port,
                              n_steps=n_steps, mode_profile=profile)
    return MSLRealization(materials, stamp_drive_materials, conductors, drives)
