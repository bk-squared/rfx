"""Aligned junction for API instrumentation tests, not RF qualification.

The historical attempt-3 fixture has a misplaced MSL ground and a partially
closed clearance hole under #931. Its frozen physics receipts must remain
attached to that geometry. This separate board tests ladder/flux payloads
and report-only advisories with short, deliberately unsettled records.
"""
import numpy as np

from rfx import Box, Cylinder, Simulation

DX = 100e-6
DOMAIN = (12.5e-3, 3.4e-3, 3.9e-3)
GROUND = 25*DX
HEIGHT = 3*DX
TRACE = GROUND+HEIGHT
JUNCTION_X = 10*DX
Y_C = 17*DX
FEED_X = 110*DX
CLEARANCE = 4*DX
FREQS = np.array([6e9, 8e9, 10e9])


def build_instrument_junction():
    sim = Simulation(freq_max=16e9, domain=DOMAIN, dx=DX,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, GROUND), (DOMAIN[0], DOMAIN[1], TRACE)), material="sub")
    # Four closed sheet footprints, with an explicit square clearance hole.
    # No volume-face padding or dielectric-over-PEC carving is involved.
    xl, xh = JUNCTION_X-CLEARANCE, JUNCTION_X+CLEARANCE
    yl, yh = Y_C-CLEARANCE, Y_C+CLEARANCE
    for x0, x1, y0, y1 in ((0, xl, 0, DOMAIN[1]), (xh, DOMAIN[0], 0, DOMAIN[1]),
                           (xl, xh, 0, yl), (xl, xh, yh, DOMAIN[1])):
        sim.add(Box((x0, y0, GROUND), (x1, y1, GROUND)), material="pec")
    sim.add(Box((JUNCTION_X, Y_C-3*DX, TRACE),
                (DOMAIN[0], Y_C+3*DX, TRACE)), material="pec")
    sim.add(Cylinder(center=(JUNCTION_X, Y_C, (GROUND+TRACE)/2),
                     radius=2*DX, height=HEIGHT, axis="z"), material="pec")
    sim.add_coaxial_port(position=(JUNCTION_X, Y_C, GROUND), face="bottom",
                         pin_radius=2*DX, outer_radius=6*DX, impedance=50.)
    sim.add_msl_port(position=(FEED_X, Y_C, GROUND), width=6*DX,
                     height=HEIGHT, direction="-x", impedance=50., eps_r_sub=3.66)
    return sim


def instrument_kwargs(n_steps=200):
    return dict(junction_x=JUNCTION_X, eps_r_sub=3.66, n_steps=n_steps,
                freqs=FREQS, probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
                msl_probe_count=9, msl_probe_start_cells=4, msl_probe_spacing_cells=10,
                skip_preflight=True, strict_passivity=False)
