"""Issue #636 advisory: dispersive pole material touching a CPML face.

The shipped pad extension replicates only the statics (#627a); pole masks
are not replicated (#627b reverted; #636 factorial measured naive
re-addition divergent, and Drude divergent even under the CFS alpha
rule). The advisory ``dispersive_pole_at_absorber_face`` flags the
configurations that live with the resulting band-limited pad mismatch —
narrowly: high-Q in-band Lorentz (Q >= 10, omega_0 <= 1.5*2*pi*freq_max)
or Drude-type poles, touching a face that actually carries an absorber.
Debye relaxations and inset structures must stay quiet.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.geometry.csg import Box
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole

DX = 1e-3
NA, NB, NZ = 20, 16, 10
F0 = 3e9
W0 = 2 * np.pi * F0


def _sim(boundary="cpml"):
    return Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary=boundary, cpml_layers=8)


def _findings(sim):
    report = sim.preflight()
    return report.by_code("dispersive_pole_at_absorber_face")


def _touching_box():
    # touches x-lo, x-hi and y-lo; interior in z
    return Box((0.0, 0.0, 3 * DX), (NA * DX, 8 * DX, 7 * DX))


def _inset_box():
    # >= 2 cells of vacuum before every face
    return Box((3 * DX, 3 * DX, 3 * DX), (10 * DX, 8 * DX, 7 * DX))


def test_high_q_lorentz_touching_face_warns():
    sim = _sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_touching_box(), material="slab")
    found = _findings(sim)
    assert len(found) == 1, found
    issue = found[0]
    assert issue.severity == "warning"
    assert "Q=60" in str(issue)
    assert "x-lo" in issue.loc and "y-lo" in issue.loc
    assert "#636" in str(issue)


def test_drude_touching_face_warns():
    sim = _sim()
    sim.add_material("metalish", eps_r=1.0,
                     lorentz_poles=[drude_pole(omega_p=W0, gamma=W0 / 100.0)])
    sim.add(_touching_box(), material="metalish")
    found = _findings(sim)
    assert len(found) == 1, found
    assert "Drude" in str(found[0])


def test_inset_structure_stays_quiet():
    sim = _sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_inset_box(), material="slab")
    assert _findings(sim) == []


def test_low_q_lorentz_stays_quiet():
    sim = _sim()
    sim.add_material("lossy_pole", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 4.0,
                                                kappa=3.0 * W0 ** 2)])  # Q=2
    sim.add(_touching_box(), material="lossy_pole")
    assert _findings(sim) == []


def test_out_of_band_high_q_lorentz_stays_quiet():
    sim = _sim()
    w_hi = 2 * np.pi * 40e9  # far above 1.5 * 2*pi*freq_max (7.5e9 band)
    sim.add_material("uv_pole", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=w_hi,
                                                delta=w_hi / 120.0,
                                                kappa=0.5 * w_hi ** 2)])
    sim.add(_touching_box(), material="uv_pole")
    assert _findings(sim) == []


def test_debye_touching_face_stays_quiet():
    sim = _sim()
    sim.add_material("fr4ish", eps_r=4.0,
                     debye_poles=[DebyePole(delta_eps=0.4, tau=1e-11)])
    sim.add(_touching_box(), material="fr4ish")
    assert _findings(sim) == []


def test_pec_boundary_stays_quiet():
    sim = _sim(boundary="pec")
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_touching_box(), material="slab")
    assert _findings(sim) == []


def test_two_touching_entries_aggregate_into_one_finding():
    sim = _sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(Box((0.0, 0.0, 3 * DX), (5 * DX, 8 * DX, 7 * DX)),
            material="slab")  # x-lo
    sim.add(Box((15 * DX, 0.0, 3 * DX), (NA * DX, 8 * DX, 7 * DX)),
            material="slab")  # x-hi
    found = _findings(sim)
    assert len(found) == 1, found
    assert "#0" in found[0].loc and "#1" in found[0].loc
