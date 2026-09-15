"""Tests for Holland thin-wire subcell model.

SCOPE UNDER #931 (lattice ownership contract). ``ThinWire`` is NOT one of
the contract's three declarations. §1.4's wire is ``PolylineWire``, which a
PEC declaration classifies into a filament (radius below half the local
cell: the E edges of the axis-aligned lattice path) or a volume
(centre-sampled tube). ``ThinWire`` is a different object: a subcell
MATERIAL model that returns eps/sigma corrections along the path and never
forms an edge set, so ``realized_pec_edge_masks`` never sees it and no
wall is realized for it.

That is a deliberate fence, not an oversight, and it is written here
because nothing else states it: the Holland model's whole point is to
represent a wire thinner than the mesh through the update coefficients
rather than by zeroing edges. A model that wants a PEC filament declares
``PolylineWire``; a model that wants the Holland correction declares
``ThinWire`` and gets no PEC edges. The two are compared head to head in
``test_thin_wire_is_not_a_pec_declaration`` below.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.geometry.thin_wire import ThinWire, compute_thin_wire_correction


def test_thin_wire_creates_correction():
    """Wire should produce non-zero eps/sigma along its path."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.02, 0.02), dx=0.002, cpml_layers=0)
    wire = ThinWire(start=(0.01, 0.01, 0.01), end=(0.04, 0.01, 0.01),
                    radius=0.0001)

    eps_corr, sigma_corr = compute_thin_wire_correction(grid, wire)

    assert float(jnp.max(eps_corr)) > 0, "Should have non-zero eps correction"
    assert float(jnp.max(sigma_corr)) > 0, "Should have non-zero sigma correction"

    n_modified = int(jnp.sum(eps_corr > 0))
    print("\nThin wire (x-aligned, r=0.1mm):")
    print(f"  Modified cells: {n_modified}")
    print(f"  Max eps_corr: {float(jnp.max(eps_corr)):.4f}")
    print(f"  Max sigma_corr: {float(jnp.max(sigma_corr)):.2e}")
    assert n_modified > 0


def test_thin_wire_axis_aligned():
    """X/Y/Z aligned wires should modify correct cells."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=0.003, cpml_layers=0)

    for axis, start, end in [
        ("x", (0.006, 0.015, 0.015), (0.024, 0.015, 0.015)),
        ("y", (0.015, 0.006, 0.015), (0.015, 0.024, 0.015)),
        ("z", (0.015, 0.015, 0.006), (0.015, 0.015, 0.024)),
    ]:
        wire = ThinWire(start=start, end=end, radius=0.0001)
        eps_corr, _ = compute_thin_wire_correction(grid, wire)
        n_mod = int(jnp.sum(eps_corr > 0))
        print(f"  {axis}-wire: {n_mod} cells modified")
        assert n_mod > 0, f"{axis}-aligned wire should modify cells"


def test_thin_wire_preserves_bulk():
    """Cells far from wire should have zero correction."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    wire = ThinWire(start=(0.025, 0.025, 0.01), end=(0.025, 0.025, 0.04),
                    radius=0.0001)

    eps_corr, sigma_corr = compute_thin_wire_correction(grid, wire)

    # Corner cells should be unmodified
    assert float(eps_corr[0, 0, 0]) == 0.0
    assert float(sigma_corr[0, 0, 0]) == 0.0
    assert float(eps_corr[-1, -1, -1]) == 0.0


def test_thin_wire_is_not_a_pec_declaration():
    """The fence in the module docstring, measured (#931 §1.4).

    Same physical filament, two declarations. ``PolylineWire`` with a
    sub-cell radius is classified as a WIRE and realizes exactly the Ex
    edges of its lattice path; ``ThinWire`` realizes no PEC edge at all and
    instead returns eps/sigma corrections on the cells it crosses. Neither
    is wrong; they are different models, and a change that quietly routed
    ``ThinWire`` through the edge rule would have to red this test.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks, WireSpec
    from rfx.geometry.csg import PolylineWire
    from rfx.geometry.rasterize_grid import (
        cell_centres_from_nodes, classify_pec_entry, coords_from_uniform_grid,
    )

    grid = Grid(freq_max=5e9, domain=(0.05, 0.02, 0.02), dx=0.002,
                cpml_layers=0)
    start, end, radius = (0.01, 0.01, 0.01), (0.04, 0.01, 0.01), 0.0001

    co = coords_from_uniform_grid(grid)
    cells, sheet, wire = classify_pec_entry(
        PolylineWire((start, end), radius=radius), co,
        cell_centres_from_nodes(co), name="filament")
    assert cells is None and sheet is None
    assert isinstance(wire, WireSpec)
    edges = realized_pec_edge_masks(None, wires=[wire])
    ex = np.asarray(edges[0], bool)
    assert int(ex.sum()) == 15, int(ex.sum())      # 30 mm / 2 mm cells
    assert not np.asarray(edges[1]).any() and not np.asarray(edges[2]).any()

    # the Holland model on the same filament: material corrections, no edges
    eps_corr, sigma_corr = compute_thin_wire_correction(
        grid, ThinWire(start=start, end=end, radius=radius))
    assert float(jnp.max(eps_corr)) > 0 and float(jnp.max(sigma_corr)) > 0
    assert not hasattr(ThinWire, "to_shapes"), (
        "ThinWire must not acquire a geometry-declaration path without a "
        "contract decision (#931 §1.4)")
