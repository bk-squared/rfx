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


def _wr90_pec_short_sim():
    """WR-90 waveguide with PEC short at x=84–87 mm, two ports."""
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
