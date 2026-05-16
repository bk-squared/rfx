"""Public-runner mapping contract for the Stage-2 disjoint topology.

This module is deliberately a contract layer, not the final production
disjoint runner.
It translates a ``Simulation.add_refinement(topology="stage2_disjoint_3d")``
setup into the coarse-hole/fine-block indexing conventions used by the
standalone Stage-2 prototype.  The public production path remains fail-closed
until the research smoke runner passes waveform/crossval gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

from rfx.subgridding.validation import build_stage2_disjoint_region


@dataclass(frozen=True)
class DisjointPointMapping:
    """Mapping of a public Simulation point observable/source into the fine block."""

    name: str
    component: str
    position_m: tuple[float, float, float]
    fine_index: tuple[int, int, int]


@dataclass(frozen=True)
class DisjointRunnerContract:
    """Serializable contract for future public Stage-2 disjoint execution."""

    topology: Literal["stage2_disjoint_3d"]
    shape_c: tuple[int, int, int]
    fine_region: tuple[int, int, int, int, int, int]
    shape_f: tuple[int, int, int]
    shape_convention: str
    dx_c: float
    dx_f: float
    ratio: int
    fine_origin_m: tuple[float, float, float]
    source_mappings: tuple[DisjointPointMapping, ...]
    probe_mappings: tuple[DisjointPointMapping, ...]
    coarse_active_cells: int
    fine_cells: int
    active_cells: int
    allocated_cells: int
    uniform_fine_cells: int
    cell_savings_factor: float
    allocated_cell_savings_factor: float


def _shape_c(grid) -> tuple[int, int, int]:
    return (int(grid.nx), int(grid.ny), int(grid.nz))


def _fine_origin(region, grid) -> tuple[float, float, float]:
    return (
        (region.fi_lo - int(getattr(grid, "pad_x_lo", grid.cpml_layers)))
        * region.dx_c,
        (region.fj_lo - int(getattr(grid, "pad_y_lo", grid.cpml_layers)))
        * region.dx_c,
        (region.fk_lo - int(getattr(grid, "pad_z_lo", grid.cpml_layers)))
        * region.dx_c,
    )


def _shape_f_from_region(region, shape_convention: str) -> tuple[int, int, int]:
    extents = (
        int(region.fi_hi - region.fi_lo),
        int(region.fj_hi - region.fj_lo),
        int(region.fk_hi - region.fk_lo),
    )
    ratio = int(region.ratio)
    if shape_convention == "cell_extent":
        return tuple(extent * ratio for extent in extents)
    if shape_convention == "endpoint_node":
        if any(extent < 2 for extent in extents):
            raise ValueError(
                "endpoint_node disjoint shape requires at least two coarse "
                f"nodes per refined axis, got extents={extents!r}"
            )
        return tuple((extent - 1) * ratio + 1 for extent in extents)
    raise ValueError(
        f"unknown disjoint_shape_convention={shape_convention!r}; "
        "expected 'cell_extent' or 'endpoint_node'"
    )


def _map_position(
    *,
    name: str,
    component: str,
    position: tuple[float, float, float],
    origin: tuple[float, float, float],
    shape_f: tuple[int, int, int],
    dx_f: float,
) -> DisjointPointMapping:
    idx = tuple(
        int(round((float(coord) - float(off)) / dx_f))
        for coord, off in zip(position, origin)
    )
    if not all(0 <= value < upper for value, upper in zip(idx, shape_f)):
        raise ValueError(
            f"{name} at {position!r} maps to fine index {idx!r}, outside "
            f"Stage-2 disjoint fine block shape {shape_f!r}"
        )
    return DisjointPointMapping(
        name=name,
        component=component,
        position_m=tuple(float(coord) for coord in position),
        fine_index=idx,
    )


def build_disjoint_runner_contract(sim, grid) -> DisjointRunnerContract:
    """Build the Stage-2 disjoint public-runner mapping contract.

    The returned shapes follow ``rfx.subgridding.disjoint_3d``: the coarse grid
    stores the full rectangular domain with a masked inactive hole, while the
    fine grid owns ``coarse_extent * ratio`` points along each axis.
    """
    ref = getattr(sim, "_refinement", None)
    if ref is None:
        raise ValueError("Simulation has no refinement to map")
    topology = ref.get("topology", "overlap_z_slab")
    if topology != "stage2_disjoint_3d":
        raise ValueError(
            "build_disjoint_runner_contract requires "
            "topology='stage2_disjoint_3d'"
        )

    region = build_stage2_disjoint_region(sim, grid)
    shape_convention = str(ref.get("disjoint_shape_convention", "cell_extent"))
    shape_c = _shape_c(grid)
    fine_region = (
        int(region.fi_lo),
        int(region.fi_hi),
        int(region.fj_lo),
        int(region.fj_hi),
        int(region.fk_lo),
        int(region.fk_hi),
    )
    shape_f = _shape_f_from_region(region, shape_convention)
    origin = _fine_origin(region, grid)
    source_mappings = tuple(
        _map_position(
            name=f"source_{idx}" if pe.impedance == 0.0 else f"port_{idx}",
            component=pe.component,
            position=pe.position,
            origin=origin,
            shape_f=shape_f,
            dx_f=region.dx_f,
        )
        for idx, pe in enumerate(getattr(sim, "_ports", ()))
    )
    probe_mappings = tuple(
        _map_position(
            name=f"probe_{idx}",
            component=probe.component,
            position=probe.position,
            origin=origin,
            shape_f=shape_f,
            dx_f=region.dx_f,
        )
        for idx, probe in enumerate(getattr(sim, "_probes", ()))
    )

    coarse_hole_cells = (
        (region.fi_hi - region.fi_lo)
        * (region.fj_hi - region.fj_lo)
        * (region.fk_hi - region.fk_lo)
    )
    coarse_total = int(prod(shape_c))
    fine_cells = int(prod(shape_f))
    coarse_active = int(coarse_total - coarse_hole_cells)
    active_cells = int(coarse_active + fine_cells)
    allocated_cells = int(coarse_total + fine_cells)
    uniform_shape = tuple(int(dim * region.ratio) for dim in shape_c)
    uniform_fine_cells = int(prod(uniform_shape))
    return DisjointRunnerContract(
        topology="stage2_disjoint_3d",
        shape_c=shape_c,
        fine_region=fine_region,
        shape_f=shape_f,
        shape_convention=shape_convention,
        dx_c=float(region.dx_c),
        dx_f=float(region.dx_f),
        ratio=int(region.ratio),
        fine_origin_m=origin,
        source_mappings=source_mappings,
        probe_mappings=probe_mappings,
        coarse_active_cells=coarse_active,
        fine_cells=fine_cells,
        active_cells=active_cells,
        allocated_cells=allocated_cells,
        uniform_fine_cells=uniform_fine_cells,
        cell_savings_factor=float(uniform_fine_cells / max(active_cells, 1)),
        allocated_cell_savings_factor=float(
            uniform_fine_cells / max(allocated_cells, 1)
        ),
    )
