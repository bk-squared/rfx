"""Off-diagonal part of the mean of eight node triplets (WBC Eq. 10)."""
import jax.numpy as jnp

from rfx.core.yee import EPS_0, _shift_fwd, h_neighbor


def off_diagonal(curls, nodes, *, periodic, boundary=None):
    """Apply Eq. 10 pair coefficients at each of an edge's two endpoints."""
    if nodes.pec_keep is not None:
        curls = tuple(jnp.where(keep, curl, 0.0)
                      for curl, keep in zip(curls, nodes.pec_keep))
    if boundary is not None:
        periodic = boundary.periodic
        if boundary.pec_faces:
            from rfx.boundaries.pec import wall_edge_masks

            masks = wall_edge_masks(curls[0].shape, boundary.pec_faces)
            curls = tuple(jnp.where(mask, 0.0, curl)
                          for curl, mask in zip(curls, masks))
    incident = tuple((c, h_neighbor(c, a, periodic=periodic, boundary=boundary))
                     for a, c in enumerate(curls))
    ends = [[jnp.zeros_like(c), jnp.zeros_like(c)] for c in curls]
    reverse = nodes.cross if nodes.cross_reverse is None else nodes.cross_reverse
    for (a, b), coupling, backward in zip(((0, 1), (0, 2), (1, 2)), nodes.cross, reverse):
        for sa in (0, 1):
            for sb in (0, 1):
                ends[a][sa] = ends[a][sa] + coupling[sa, sb] * incident[b][sb]
                ends[b][sb] = ends[b][sb] + backward[sa, sb] * incident[a][sa]
    return tuple(jnp.where(nodes.active[a], plus + (
        jnp.roll(minus, -1, a) if periodic[a] else _shift_fwd(minus, a)), 0.0)
        for a, (plus, minus) in enumerate(ends))


def add_node_increment(updated, curls, nodes, eps, sigma, dt, *, periodic, boundary=None):
    """R-b: own-edge semi-implicit loss; R-e: lumped C leaves cross terms alone."""
    cross = off_diagonal(curls, nodes, periodic=periodic, boundary=boundary)
    return tuple(jnp.where(active, old + (dt / EPS_0) * value / (
        1.0 + sig * (dt / (2.0 * EPS_0)) / e), old).astype(old.dtype)
        for old, value, e, sig, active in zip(updated, cross, eps, sigma, nodes.active))
