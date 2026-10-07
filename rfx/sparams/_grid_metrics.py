"""Physical lengths for S-parameter geometry advisories."""
import numpy as np


def node_distance(grid, axis, first, second):
    """Node separation; retain uniform multiplication's exact rounding."""
    if grid.is_constant(axis):
        return abs(second - first) * float(grid.boundary_cell(axis, "lo"))
    return abs(float(grid.node_of(axis, second)) - float(grid.node_of(axis, first)))


def absorber_depth(grid, axis, side, n_cells):
    """Sum physical pad cells, excluding the final bounding-node provider.

    Uniform multiplication preserves the old operation order. Graded pads
    use their own axis and face, including unequal low/high boundary cells.
    """
    if grid.is_constant(axis):
        return n_cells * float(grid.boundary_cell(axis, side))
    cells = np.asarray(grid.cells(axis), dtype=float)
    pad = cells[:n_cells] if side == "lo" else cells[len(cells) - 1 - n_cells:-1]
    return float(pad.sum())


def junction_absorber_depth(grid, axis, default_cells):
    """Thinnest active propagation pad, including per-face layer overrides."""
    depths = []
    for side in ("lo", "hi"):
        n = int(getattr(grid, f"pad_{axis}_{side}", default_cells))
        if n > 0:
            depths.append(absorber_depth(grid, axis, side, n))
    return min(depths, default=float("inf"))


def _calculator_source_plane(grid, port):
    """Keep the calculator's derived half-cell source on its original node.

    This is an internal one-cell descriptor, not a declared coordinate.
    Preserve its float64 half-to-even arithmetic independently of the
    declared-coordinate tie rule.
    """
    direction = 1 if port.face == "bottom" else -1
    centre = port.position[2] + direction * port.pin_length / 2.0
    dz = float(grid.cells(2)[0])   # uniform z only (coax calculators refuse a graded mesh)
    return int(round(centre / dz)) + grid.pad_z_lo
