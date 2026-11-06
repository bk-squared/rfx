"""Metadata for the coax stamp's existing node-sampled products.

This module never rasterizes or changes the stamp. Radii are area-equivalent
cross-section radii; declared radii retain the stamp's pin/bore/wall rule.
"""
from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class StampedEntity:
    entity_id: str
    port_id: str
    kind: str
    cells: object
    shape: object
    radii_m: tuple[float, float]
    declared_radii_m: tuple[float, float]
    provenance: str = 'coax stamp'
    sampling: str = 'closed cylinders at nodes, owned as cells'


def coax_stamp_entities(grid, *, center, height, a, b, outer_radius,
                        pin, shell, bore, outer, dielectric, port_id):
    """Retain the exact masks the stamp just used, including its clamped wall."""
    from rfx.geometry.csg import Cylinder
    area = np.asarray(grid.cells(0))[:, None] * np.asarray(grid.cells(1))[None, :]

    def radius(mask):
        cross = np.asarray(mask, dtype=bool).any(axis=2)
        return float(np.sqrt(np.sum(area[cross]) / np.pi))

    ra, rb, ro = radius(pin), radius(bore), radius(outer)
    return tuple(StampedEntity(f'{port_id}/{name}', port_id, kind,
        np.asarray(mask, dtype=bool), Cylinder(center, declared[1], height, axis='z'),
        radii, declared) for name, kind, mask, radii, declared in (
            ('shell', 'volume', shell, (rb, ro), (b, outer_radius)),
            ('pin', 'volume', pin, (0., ra), (0., a)),
            ('dielectric', 'material', dielectric, (ra, rb), (a, b))))


def stamp_check_findings(sim, conductors):
    """Read both validators against the owner; retain findings without judging.

    Cylindrical volumes do not receive the sheet free-edge size rule. The
    transition's declared sheets can still produce a verdict. Record it here
    rather than adding a new warning/refusal to a previously admitted lane.
    """
    from rfx.model.pad_fill import check_pad_fill
    from rfx.preflight.realization import context_from_conductors
    from rfx.preflight.pec_geometry import _warn_sheet_effective_size
    ctx = context_from_conductors(sim, conductors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_sheet_effective_size(warnings, ctx, ctx.interior_pec_entries())
    pad = []
    check_pad_fill(sim, conductors.grid, (), record=pad, conductors=conductors)
    return (('sheet-size', tuple(str(w.message) for w in caught)),
            ('pad-fill', tuple(pad)))


def stamped_geometry_entities(conductors, nodes, sizes, *, compact=False):
    """Describe captured masks using the record's cell ownership convention."""
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.realized_geometry import AxisGeometry, EntityGeometry, _mask_ranges, _readonly
    result = []
    for entity in conductors.stamped_entities:
        mask = np.asarray(entity.cells)
        bounds = entity.shape.bounding_box()
        axes = []
        occupied = np.where(mask)
        if occupied[0].size:
            for a, indices in enumerate(occupied):
                first, end = int(indices.min()), int(indices.max()) + 1
                lo, hi = float(nodes[a][first]), float(nodes[a][end])
                declared = (float(bounds[0][a]), float(bounds[1][a]))
                axes.append(AxisGeometry('xyz'[a], (first, end), (first, end),
                    declared, (lo, hi), hi - lo,
                    (lo - declared[0], hi - declared[1]),
                    float(np.mean(sizes[a][first:end]))))
        edges = (() if entity.kind == 'material' else realized_pec_edge_masks(
            mask, periodic=conductors.periodic))
        result.append(EntityGeometry(entity.entity_id, entity.entity_id, entity.kind,
            tuple(axes), int(mask.sum()), None,
            () if not edges else tuple(tuple(realized_wall_planes(edges, a,
                periodic=conductors.periodic)) for a in range(3)),
            None if compact else _readonly(mask),
            () if compact else tuple(_readonly(e) for e in edges),
            declared_bounds_m=bounds, edge_counts=tuple(int(np.asarray(e).sum()) for e in edges),
            edge_ranges=tuple(_mask_ranges(e) for e in edges),
            radii_m=entity.radii_m, declared_radii_m=entity.declared_radii_m,
            provenance=entity.provenance, port_id=entity.port_id, sampling=entity.sampling))
    return result
