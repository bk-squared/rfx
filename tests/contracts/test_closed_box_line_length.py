"""Closed-box record bounds cannot size the incident line."""
import math

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.tfsf import tfsf_from_declaration


def scene(name):
    if name == "S1":
        sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=.0025,
                         boundary="cpml", cpml_layers=8, snap="declared")
        source = dict(f0=.6e9, bandwidth=.95, polarization="ez")
    else:
        sim = Simulation(freq_max=15e9, domain=(.024,) * 3, dx=.001,
                         boundary="cpml", cpml_layers=8, snap="declared")
        theta, phi = (30, 20) if name == "S2" else (45, 0)
        source = dict(f0=10e9, bandwidth=.5, polarization="s",
                      theta_deg=theta, phi_deg=phi)
    sim.add_tfsf_source(**source, margin=3, direction="+x", closed_box=True,
                        waveform="differentiated_gaussian")
    return sim, sim._build_grid()


@pytest.mark.parametrize("name", ("S1", "S2", "S3"))
def test_bar1_line_length_independent_of_record(name):
    sim, grid = scene(name)
    sizes = [tfsf_from_declaration(sim._tfsf, grid, sim._freq_max,
                                  n_steps=n)[0].n_nodes for n in (1000, 100000)]
    assert sizes[0] == sizes[1], "record length changed the closed-box line"


@pytest.mark.parametrize("name", ("S1", "S2", "S3"))
def test_line_extent_from_read_range_and_physical_layer(name):
    sim, grid = scene(name)
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1000)
    reads = np.concatenate([t.source_index + cfg.origin for t in cfg.tables_e + cfg.tables_h]
                           + [np.array(cfg.source_nodes) + cfg.origin]
                           + [w.nodes + cfg.origin for w in cfg.source_windows])
    g = 2 * max(abs(v) for v in cfg.m) + 2
    length = 64 * math.ceil(math.sqrt(sum(v * v for v in cfg.m)))
    assert cfg.absorber_nodes == length
    assert cfg.origin == int(reads.min()) - g - length
    assert cfg.n_nodes == int(reads.max() - reads.min()) + 1 + 2 * (g + length)
    assert cfg.n_steps == 1000


def test_bar1_slab_line_length_independent_of_record():
    from jax.sharding import Mesh
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_tfsf import prepare
    sim, grid = scene("S1")
    devices = jax.devices("cpu")[:3]
    mesh = Mesh(np.array(devices), ("x",))
    layout = Slab(grid.nx, len(devices))
    sizes = [prepare(sim, grid, n, layout, mesh)[0].n_nodes for n in (1000, 100000)]
    assert sizes[0] == sizes[1], "record length changed the replicated line"


@pytest.mark.parametrize("name,radius,n_nodes,origin", (
    ("S1", 1, 2029, -998),
    ("S2", 81, 165781, -80586),
))
def test_identity_absorber_reproduces_old_padded_extent(monkeypatch, name, radius, n_nodes, origin):
    from rfx.sources import tfsf_discrete_plane_wave as plane
    original = plane.init_plane_wave

    def padded(*args, **kwargs):
        return original(*args, **kwargs, _zero_absorber=True,
                        _absorber_nodes=1000 * radius + 2 - (2 * radius + 2))

    monkeypatch.setattr(plane, "init_plane_wave", padded)
    sim, grid = scene(name)
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=1000)
    # Old ranges: S1 [4, 28], S2 [416, 4192]; padding = 1000 * radius + 2.
    assert (cfg.n_nodes, cfg.origin) == (n_nodes, origin)
