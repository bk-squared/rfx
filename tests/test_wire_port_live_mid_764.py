"""Issue #764 dead-mid-cell witness: the V/I reference cell is the
midpoint of the LIVE wire run, not of the raw extent.

A dead extent cell (inside PEC, issue #318) carries essentially no port
current (measured |I_dead|/|I_mid| = 0.003-0.03 on the #313 thru), so an
all-extent midpoint landing on a dead cell read a QUENCHED Ampere loop
as the port current. Both lanes now pin the reference cell (and the
uniform probe helpers) to the live-run midpoint; with no dead cells this
is bit-identical to the historical all-extent midpoint (covered by the
suite's unchanged no-dead-cell locks).

Fixture: an ez wire port whose extent starts two cells inside a PEC
block — 3 extent cells, the lower two dead, so the all-extent midpoint
(the middle cell) is DEAD and the live run is the single top cell.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

DX = 1e-3
DOMAIN = (12e-3, 12e-3, 12e-3)
PORT_X, PORT_Y = 6e-3, 6e-3
Z0_PORT = 4e-3            # port start (cell k=4); extent 2 mm -> cells 4,5,6
EXTENT = 2e-3
FREQS = jnp.array([1.0e9])
PULSE = GaussianPulse(f0=2e9, bandwidth=0.9)


def _build(nu: bool):
    kw = {"dz_profile": np.full(12, DX)} if nu else {}
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="pec",
                     **kw)
    # PEC block whose top two cell-layers (k=4,5) swallow the port's two
    # lower extent cells; k=6 stays vacuum (the single LIVE cell).
    sim.add(Box((4e-3, 4e-3, 2e-3), (8e-3, 8e-3, 6e-3)), material="pec")
    sim.add_port(position=(PORT_X, PORT_Y, Z0_PORT), component="ez",
                 impedance=50.0, extent=EXTENT, excite=True, waveform=PULSE)
    return sim


def test_uniform_spec_mid_is_live_run_midpoint():
    import rfx.simulation as sim_mod
    cap = {}
    orig = sim_mod.run

    def spy(grid, materials, n_steps, **kwargs):
        cap["specs"] = kwargs.get("wire_port_sparams")
        return orig(grid, materials, n_steps, **kwargs)

    sim_mod.run = spy
    try:
        _build(False).run(n_steps=8, compute_s_params=True,
                          s_param_freqs=FREQS, skip_preflight=True)
    finally:
        sim_mod.run = orig

    specs = cap["specs"]
    assert specs and len(specs) == 1
    spec = specs[0]
    # extent cells are k=4,5,6; 4 and 5 are dead (inside the PEC block).
    assert spec.live_cells == ((6, 6, 6),), spec.live_cells
    assert (spec.mid_i, spec.mid_j, spec.mid_k) == (6, 6, 6), (
        "all-extent midpoint (k=5, DEAD — a quenched Ampere loop) leaked "
        f"back in: got mid k={spec.mid_k}")


def test_nu_spec_mid_is_live_run_midpoint():
    import rfx.runners.nonuniform as nur
    cap = {}
    orig = nur.run_nonuniform

    def spy(grid, materials, n_steps, **kwargs):
        cap["wire_ports"] = kwargs.get("wire_ports")
        return orig(grid, materials, n_steps, **kwargs)

    nur.run_nonuniform = spy
    try:
        _build(True).run(n_steps=8, compute_s_params=True,
                         s_param_freqs=FREQS, skip_preflight=True)
    finally:
        nur.run_nonuniform = orig

    wps = cap["wire_ports"]
    assert wps and len(wps) == 1
    wp = wps[0]
    assert wp["n_live"] == 1, wp
    assert wp["live_cells"] == ((6, 6, 6),), wp["live_cells"]
    assert (wp["mid_i"], wp["mid_j"], wp["mid_k"]) == (6, 6, 6), (
        "all-extent midpoint (k=5, DEAD) leaked back in: got "
        f"mid k={wp['mid_k']}")


def test_probe_helpers_use_live_run_midpoint():
    """wire_port_voltage / wire_port_current / init_wire_sparam_probe pin
    to the live-run midpoint when given the assembled pec mask."""
    from rfx.grid import Grid
    from rfx.probes.probes import _wire_port_live_mid, init_wire_sparam_probe
    from rfx.sources.sources import WirePort

    grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=0)
    pec_mask = np.zeros(grid.shape, dtype=bool)
    pec_mask[4:8, 4:8, 2:6] = True
    wp = WirePort(start=(PORT_X, PORT_Y, Z0_PORT),
                  end=(PORT_X, PORT_Y, Z0_PORT + EXTENT),
                  component="ez", impedance=50.0, excitation=PULSE)
    mid = _wire_port_live_mid(grid, wp, jnp.asarray(pec_mask))
    assert tuple(mid) == (6, 6, 6), mid
    probe = init_wire_sparam_probe(grid, wp, np.asarray(FREQS),
                                   pec_mask=jnp.asarray(pec_mask))
    assert tuple(probe.port_index) == (6, 6, 6), probe.port_index
    # pec_mask=None (or no dead cells) stays bit-identical to the
    # historical all-extent midpoint.
    assert tuple(_wire_port_live_mid(grid, wp, None)) == (6, 6, 5)
