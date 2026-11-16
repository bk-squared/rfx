"""Restrict node couplings to the step's realized free edges."""
from dataclasses import replace

import jax.numpy as jnp

from rfx.boundaries.pec import wall_edge_masks


def restrict_node_pec(materials, edge_masks, wall_faces):
    """Keep final conductors (including port releases) separate from walls.

    Bulk/interface activation is not an input mask: a free bulk edge may
    supply a curl to an interface edge. Only PEC excludes both endpoints.
    """
    tensor = materials.components.tensor
    if tensor is None or tensor.node_smoothing is None:
        return materials
    nodes = tensor.node_smoothing
    walls = wall_edge_masks(nodes.active[0].shape, wall_faces)
    keep = tuple(jnp.ones_like(wall, dtype=bool) if edge_masks is None else ~edge
                 for wall, edge in zip(walls, edge_masks or walls))
    nodes = replace(nodes, pec_keep=keep,
                    active=tuple(active & free & ~wall
                                 for active, free, wall in zip(nodes.active, keep, walls)))
    return materials._replace(components=replace(
        materials.components, tensor=replace(tensor, node_smoothing=nodes)))
