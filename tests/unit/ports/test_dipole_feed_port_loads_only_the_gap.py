"""A centre-fed dipole's 50 ohm port loads the feed gap and nothing else (#1236).

A half-wave dipole (thin PEC wire along z, L = 47.5 mm, 19 cells) is fed by a
50 ohm lumped port on the one-cell gap edge Ez between its arms. The port is a
resistor across that gap. The wire and its surroundings are symmetric under a
mirror through the wire axis and under a quarter turn about it, so the radial
field that leaves the lower arm's tip at the feed node must be the same on
every side: Ex on the +x edge equals minus Ex on the -x edge, and Ey on the +y
edge equals minus Ey on the -y edge.

Before #1236 the port's conductance was added to all three E components at its
node: the port was also a 50 ohm resistor on the +x and +y edges leaving the
feed node. Those two edges were pinned to 0.03 of the gap field while the
unloaded -x side kept 0.33 (resonance +0.34 % at lambda/43, Zin off by 12-28 %
at 3.5-4 GHz). With the load on the gap only, the loaded-side and mirror
records agree to the CPML's own lo/hi asymmetry (4.3e-4 of the peak here; a
PEC box gives exactly 0).

Fixture: the rfx-rcs measurement's dipole (``exp_lumped_transverse/run_case.py``
case D, mesh L/19: 12 cells of clearance, 10 CPML layers), run for 300 steps --
the tip's radial field reaches 0.38 of the gap field inside that window. The
gate threshold 1e-3 is the one pre-declared for it (G1).
"""
from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, PolylineWire, Simulation, realized_pec_edge_masks

L_DIP = 47.5e-3
N_CELLS = 19          # L/19 = 2.5 mm, lambda/43 at the realized resonance
N_LAT = 12            # 30 mm of clearance
N_CPML = 10
N_STEPS = 300


def _dipole():
    dx = L_DIP / N_CELLS
    n_arm = (N_CELLS - 1) // 2
    nz = 2 * N_LAT + N_CELLS
    sim = Simulation(freq_max=6e9,
                     domain=(2 * N_LAT * dx, 2 * N_LAT * dx, nz * dx),
                     dx=dx, boundary="cpml", cpml_layers=N_CPML)
    xc = yc = N_LAT * dx
    z_lo = N_LAT * dx
    z_port = (N_LAT + n_arm) * dx
    z_hi = (N_LAT + N_CELLS) * dx
    r = 0.1 * dx
    sim.add(PolylineWire(((xc, yc, z_lo), (xc, yc, z_port)), radius=r),
            material="pec")
    sim.add(PolylineWire(((xc, yc, z_port + dx), (xc, yc, z_hi)), radius=r),
            material="pec")
    sim.add_port(position=(xc, yc, z_port), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=3e9, bandwidth=0.8))
    probes = {
        "ex_plus": ((xc, yc, z_port), "ex"),        # edge the old rule loaded
        "ex_minus": ((xc - dx, yc, z_port), "ex"),  # its mirror, never loaded
        "ey_plus": ((xc, yc, z_port), "ey"),        # edge the old rule loaded
        "ey_minus": ((xc, yc - dx, z_port), "ey"),  # its mirror
        "ez_gap": ((xc, yc, z_port), "ez"),
    }
    for pos, comp in probes.values():
        sim.add_probe(pos, comp)
    return sim, list(probes), n_arm


def _assert_realized(sim, n_arm):
    """The wire and the gap are what the fixture declares (realized edges)."""
    grid = sim._build_grid()
    sheets, wires = [], []
    mats = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    mx, my, mz = (np.asarray(m) for m in realized_pec_edge_masks(
        mats[3], sheets, wires, periodic=sim._periodic_flags()))
    i, j, k = (int(v) for v in grid.position_to_index(sim._ports[0].position))
    col = np.nonzero(mz[i, j, :])[0].tolist()
    assert not mz[i, j, k], "the port's gap edge is shorted"
    assert col == list(range(k - n_arm, k)) + list(range(k + 1, k + 1 + n_arm)), (
        f"arms realized on Ez edges {col}, want {n_arm} each side of k={k}")
    assert int(mz.sum()) == len(col) and not mx.any() and not my.any(), (
        "PEC edges off the wire column")
    # The two edges the old rule loaded are live (the test needs them).
    assert not mx[i, j, k] and not my[i, j, k]
    # The fixture is centred: the mirror of node i about the wire is itself.
    nx, ny = grid.shape[0], grid.shape[1]
    assert (i, j) == ((nx - 1) // 2, (ny - 1) // 2) and nx % 2 == 1


def test_the_feed_tip_radial_field_is_the_same_on_the_loaded_and_free_sides():
    sim, names, n_arm = _dipole()
    _assert_realized(sim, n_arm)
    res = sim.run(n_steps=N_STEPS, skip_preflight=True, compute_s_params=False)
    ts = dict(zip(names, np.asarray(res.time_series).T))

    gap_peak = float(np.max(np.abs(ts["ez_gap"])))
    for plus, minus in (("ex_plus", "ex_minus"), ("ey_plus", "ey_minus")):
        peak = float(np.max(np.abs(ts[minus])))
        # The field under test is not trivially small: 0.38 of the gap field.
        assert peak > 0.2 * gap_peak, (
            f"{minus} peaks at {peak / gap_peak:.3f} of the gap field; the "
            f"fixture no longer carries a radial field at the feed")
        # Radial field is odd under the mirror: E(+side) = -E(-side).
        mismatch = float(np.max(np.abs(ts[plus] + ts[minus]))) / peak
        ratio = float(np.max(np.abs(ts[plus]))) / peak
        assert mismatch <= 1e-3, (
            f"{plus} vs its mirror {minus}: max |E+ + E-| = {mismatch:.3e} of "
            f"the peak, peak ratio {ratio:.4f}. The port's 50 ohm load is on "
            f"the {plus[:2].upper()} edge leaving the feed node as well as on "
            f"the gap (#1236: that edge was pinned to ~0.09 of its mirror).")
