"""Compatibility imports for conformal geometry and operand formation."""

from rfx.geometry.cell_table import build_cell_table
from rfx.geometry.cell_table.body import edge_fraction, face_fraction
from rfx.model.conformal_rules import (
    conformal_b_floor, conformal_fractions_from_table, sc_clamp,
)


def compute_conformal_fractions(grid, pec_shapes, *, clamp=True, conformal_pec=None):
    """Return E free lengths and H free areas on the uniform Yee lattice."""
    return conformal_fractions_from_table(
        build_cell_table(grid, pec_shapes), grid, conformal_pec, clamp=clamp)
