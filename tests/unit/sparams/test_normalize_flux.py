"""Tests for normalize="flux" (hybrid power-flux magnitude + modal phase).

Validates that the Poynting-flux S-matrix extraction:
  1. runs end-to-end and returns the right shape,
  2. gives |S11| ≥ 0.99 on a PEC-short in the TE20-clean band (5–6.5 GHz).

The PEC-short gate is the same physics criterion used by
``test_pec_short_s11_with_kottke_pec_path`` but exercises the flux
extraction path (``normalize="flux"``) rather than the modal normalize=False
path with kottke_pec subpixel smoothing.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.boundaries.spec import BoundarySpec, Boundary
from tests._realized_geometry import assert_wall_planes


def _wr90_pec_short_sim():
    """WR-90 waveguide with a PEC short at x = 84-87 mm, two ports.

    The short is a VOLUME (#931 §1.2): 3 mm at dx = 3 mm is one primal
    cell, and the contract realizes it as a filled cell with tangential
    walls on BOTH drawn faces, x = 84 mm and x = 87 mm, with the normal
    Ex between them shorted. Before the contract the far face was never
    a wall at any thickness, so the same declaration realized a single
    wall at 84 mm — a short is still a short either way (|S11| >= 0.99
    survives), but the reflection reference plane moves by one cell, and
    that is why the phase-sensitive siblings are re-read alongside this
    one. ``test_wr90_short_realizes_both_drawn_faces`` below is the
    build-time witness.
    """
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=10,
    )
    sim.add(Box((0.084, 0, 0), (0.087, 0.04, 0.02)), material="pec")
    freqs = jnp.linspace(5e9, 6.5e9, 6)
    sim.add_waveguide_port(
        0.010, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.090, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="right",
    )
    return sim


def test_normalize_flux_smoke():
    """normalize="flux" runs, shape is (2, 2, n_freqs), entries are finite."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    freqs = jnp.linspace(5e9, 6.5e9, 4)
    sim.add_waveguide_port(
        0.010, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.090, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="right",
    )
    res = sim.compute_waveguide_s_matrix(num_periods=8, normalize="flux")
    s = np.asarray(res.s_params)
    assert s.shape == (2, 2, 4), f"unexpected shape {s.shape}"
    assert np.all(np.isfinite(s)), "S-matrix contains non-finite entries"


def test_normalize_flux_pec_short_s11():
    """|S11| ≥ 0.99 in the TE20-clean 5–6.5 GHz window via normalize="flux".

    PEC box at x=[84, 87] mm: both faces are exact grid cells (dx=3 mm),
    so Kottke reduces to a binary mask.  The 5–6.5 GHz window keeps TE20
    evanescent contamination below 0.3 %.  A value below 0.99 signals that
    the Poynting-flux extraction is leaking energy or double-counting
    reflection.
    """
    res = _wr90_pec_short_sim().compute_waveguide_s_matrix(
        num_periods=40, normalize="flux"
    )
    s11 = np.abs(np.asarray(res.s_params)[0, 0, :])
    print(f"\n[flux pec-short] |S11| range [{s11.min():.4f}, {s11.max():.4f}] "
          f"mean={s11.mean():.4f}")
    assert s11.min() >= 0.99, (
        f"normalize='flux' PEC-short |S11| below gate: min={s11.min():.4f} "
        f"(gate 0.99). Flux extraction is leaking energy or mis-attributing "
        f"reflection."
    )


def test_wr90_short_realizes_both_drawn_faces():
    """Build-time witness (no solve): a volume's far face IS a wall.

    Nothing in the S-parameter or port suite asserted this before #931 —
    the nearest thing asserted the opposite (the hi node plane carries no
    cell). The short is drawn 84 -> 87 mm on node lines at dx = 3 mm, so
    the realized wall planes along x are exactly those two, and the Ex
    edge in the cell between them is PEC.
    """
    import numpy as _np

    from tests._realized_geometry import realized

    sim = _wr90_pec_short_sim()
    rz = realized(sim)
    assert_wall_planes(sim, 0, [0.084, 0.087], what="WR-90 PEC short")
    pad = rz.grid.axis_pads[0]
    k_lo = pad + int(round(0.084 / rz.grid.dx))
    assert bool(_np.asarray(rz.edge_masks[0])[k_lo, rz.grid.ny // 2,
                                              rz.grid.nz // 2]), (
        "the Ex edge inside the short must be PEC: a volume shorts every "
        "normal edge between its faces (#931 §1.2)")
