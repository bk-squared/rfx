"""Forward path must carry DFT plane-probe accumulators on `ForwardResult`.

Issue: gap #4 in `docs/agent-memory/rfx-known-issues.md`
("Differentiable MSL inverse-design infrastructure gaps", 2026-05-05).
The scan body in `rfx/simulation.py` already accumulates `SimResult.dft_planes`
(line ~1253), but `_forward_from_materials` was repackaging into
`ForwardResult` without carrying the field — so `forward(...)` callers
could not access plane-resolved DFT data on the differentiable path,
even when an `add_dft_plane_probe` was registered.

This test pins the carry-through end-to-end.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec


C0 = 2.998e8


def _build_thin_box_sim(*, dx: float = 0.5e-3, lx: float = 0.020):
    """Tiny 3D sim with a single dielectric Box and one source/probe."""
    sim = Simulation(
        freq_max=10e9, domain=(lx, lx, 0.005), dx=dx, cpml_layers=4,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("eps2", eps_r=2.0)
    sim.add(Box((0, 0, 0), (lx, lx, 0.001)), material="eps2")
    sim.add_source((lx / 2, lx / 2, 0.0005), "ez")
    sim.add_probe((lx / 2 + dx, lx / 2, 0.0005), "ez")
    return sim


def test_forward_result_carries_dft_planes_when_registered():
    """Registering an `add_dft_plane_probe` should populate
    `ForwardResult.dft_planes` after `forward()`."""
    sim = _build_thin_box_sim()
    sim.add_dft_plane_probe(
        axis="z", coordinate=0.0005, component="ez",
        freqs=jnp.asarray([5e9, 8e9], dtype=jnp.float32),
        name="ez_mid",
    )
    fr = sim.forward(num_periods=2)
    assert fr.dft_planes is not None, (
        "forward() with an add_dft_plane_probe must populate "
        "ForwardResult.dft_planes"
    )
    # Name-keyed dict mirrors runners/uniform.py / Result.dft_planes.
    assert "ez_mid" in fr.dft_planes
    plane = fr.dft_planes["ez_mid"]
    assert plane.accumulator is not None
    arr = np.asarray(plane.accumulator)
    # accumulator shape: (n_freqs, *plane_shape) — non-zero after a
    # forward run with a real source.
    assert arr.shape[0] == 2, (
        f"expected 2 freqs in plane accumulator, got shape {arr.shape}"
    )
    assert np.any(np.abs(arr) > 0), (
        "plane accumulator must accrue non-zero DFT energy after forward"
    )


def test_forward_result_dft_planes_is_none_when_no_probe():
    """No plane probes registered → `ForwardResult.dft_planes is None`."""
    sim = _build_thin_box_sim()
    fr = sim.forward(num_periods=2)
    assert fr.dft_planes is None, (
        f"forward() with no plane probes must keep dft_planes=None; "
        f"got: {fr.dft_planes!r}"
    )


def test_forward_result_dft_planes_multiple_planes():
    """Multiple plane probes should all be carried, in registration order."""
    sim = _build_thin_box_sim()
    f = jnp.asarray([6e9], dtype=jnp.float32)
    sim.add_dft_plane_probe(axis="z", coordinate=0.0005, component="ez",
                            freqs=f, name="ez_mid")
    sim.add_dft_plane_probe(axis="z", coordinate=0.0010, component="ez",
                            freqs=f, name="ez_top")
    fr = sim.forward(num_periods=2)
    assert fr.dft_planes is not None
    assert set(fr.dft_planes.keys()) == {"ez_mid", "ez_top"}, (
        f"both plane probes must be carried by name; got {list(fr.dft_planes)}"
    )
    for nm in ("ez_mid", "ez_top"):
        assert fr.dft_planes[nm].accumulator is not None
        # Accumulator shape (n_freqs, *plane_shape) and non-trivial
        arr = np.asarray(fr.dft_planes[nm].accumulator)
        assert arr.ndim == 3 and arr.shape[0] == 1
