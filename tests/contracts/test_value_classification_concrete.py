"""Concrete conductor, Kerr, port and RLC meanings remain unchanged (Step B)."""
from itertools import combinations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.core.yee import init_materials, lumped_components
from rfx.lumped import build_rlc_meta, setup_rlc_materials


def scene():
    return Simulation(domain=(.012,) * 3, dx=.001, freq_max=10e9, boundary="pec")


@pytest.mark.parametrize("sigma,pec", [(2e6, True), (5e5, False)])
def test_concrete_conductivity_realizes_edges(sigma, pec):
    sim = scene()
    sim.add_material("body", sigma=sigma)
    sim.add(Box((.003,) * 3, (.009,) * 3), material="body")
    grid = sim._build_grid()
    sheets, wires = [], []
    materials, _, _, mask, _, _, kerr = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    edges = realized_pec_edge_masks(jnp.zeros(grid.shape, bool) if mask is None else mask, tuple(sheets), tuple(wires))
    assert any(np.any(e) for e in edges) == pec
    assert bool(np.any(np.asarray(materials.sigma))) != pec
    if not pec:
        assert np.max(materials.sigma) == sigma
    assert kerr is None


ELEMENTS = [tuple(k for k in "RLC" if k in selected)
            for count in (1, 2, 3) for selected in combinations("RLC", count)]


@pytest.mark.parametrize("topology", ["series", "parallel"])
@pytest.mark.parametrize("elements", ELEMENTS)
def test_concrete_rlc_realized_topology(topology, elements):
    sim = scene()
    values = {"R": 10.0, "L": 1e-9, "C": 1e-12}
    sim.add_lumped_rlc((.006,) * 3, topology=topology,
                       **{name: values[name] for name in elements})
    grid, spec = sim._build_grid(), sim._lumped_rlc[0]
    materials = setup_rlc_materials(grid, spec, init_materials(grid.shape))
    meta = build_rlc_meta(grid, spec, materials)
    series = topology == "series" and len(elements) >= 2
    assert meta.is_series == series
    assert meta.has_inductor == ("L" in elements)
    assert meta.has_capacitor == ("C" in elements)
    sigma = lumped_components(materials.sigma_lumped)[2]
    eps = lumped_components(materials.eps_r_lumped)[2]
    assert (sigma is not None and bool(np.any(sigma))) == ("R" in elements and not series)
    assert (eps is not None and bool(np.any(eps))) == ("C" in elements and not series)
    if "R" not in elements:
        assert meta.R == 0.0
        assert np.isfinite(np.asarray(materials.eps_r)).all()
        assert np.isfinite(np.asarray(materials.sigma)).all()


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_concrete_bad_impedance(value):
    with pytest.raises(ValueError, match=f"impedance must be positive, got {value}"):
        scene().add_port((.006,) * 3, impedance=value)


@pytest.mark.parametrize("element", list("RLC"))
def test_concrete_negative_element(element):
    values = dict(R=10.0, L=1e-9, C=1e-12)
    values[element] = -1.0
    with pytest.raises(ValueError, match="R, L, C must be non-negative"):
        scene().add_lumped_rlc((.006,) * 3, **values)
