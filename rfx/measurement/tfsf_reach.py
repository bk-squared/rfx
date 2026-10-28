"""Travel from the realized TF/SF launch to its box and recorded points."""
from itertools import product

import numpy as np


def tfsf_corner_paths(cfg, grid, *, inset=0):
    """Physical box corners and launch paths; inset samples interior corner nodes."""
    spans = []
    for axis, size in zip("xyz", grid.shape):
        lo, hi = getattr(cfg, axis + "_lo", 0), getattr(cfg, axis + "_hi", size - 1)
        if axis != "x" and lo == hi and size > 1:
            lo, hi = 0, size - 1
        spans.append((lo + inset, hi - inset))
    indices = np.asarray(list(product(*spans)))
    corners = np.asarray([[grid.node_of(a, int(i)) for a, i in zip("xyz", c)]
                          for c in indices])
    if hasattr(cfg, "source_p"):
        # Per-node window delays reference source_p on every sublattice.
        paths = (indices @ np.asarray(cfg.m) - cfg.source_p) * cfg.dx_1d / np.linalg.norm(cfg.m)
    elif hasattr(cfg, "d_aux"):
        # init_tfsf_methodB maps the projection of padded node 0 to src_idx.
        origin = np.asarray([grid.node_of(a, 0) for a in "xy"])
        paths = (corners[:, :2] - origin) @ np.asarray([cfg.cos_theta, cfg.sin_theta])
    elif hasattr(cfg, "i0_x"):
        # The Bloch source applies the envelope along the entire src_x plane.
        paths = (cfg.direction_sign * (cfg.i0_x + indices[:, 0] - cfg.x_lo - cfg.src_x)
                 * cfg.dx_1d / cfg.cos_theta)
    else:
        # The line runs on the realized x cells (a graded x axis is admitted
        # for the normal slab), so sum its widths from the source node.
        line = cfg.i0 + indices[:, 0] - cfg.x_lo
        n = int(max(line.max(), cfg.src_idx)) + 1
        widths = np.broadcast_to(np.asarray(cfg.primal_widths, dtype=float), (n,)) \
            if np.ndim(cfg.primal_widths) == 0 else np.asarray(cfg.primal_widths, dtype=float)
        nodes = np.concatenate([[0.0], np.cumsum(widths)])
        paths = np.abs(nodes[line] - nodes[cfg.src_idx])

    return corners, paths


def tfsf_reach(cfg, grid):
    """Return a reach callable; None reads incident transit through the box."""
    corners, paths = tfsf_corner_paths(cfg, grid)

    def reach(points):
        if points is None:
            return float(np.max(paths))
        distances = np.linalg.norm(corners[:, None, :] - np.asarray(points)[None, :, :], axis=-1)
        return float(np.max(paths + np.max(distances, axis=1)))

    return reach
