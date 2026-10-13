"""A wire port declared without ``radius`` is reported once per preflight.

``add_port(..., extent=l)`` with ``radius=None`` is a filament of the
lattice's own radius, about 0.20 d (d the transverse cell). Its series
inductance against a wire of fixed radius a is (mu0 l / 2 pi) ln(0.20 d / a),
so each halving of d adds omega (mu0 l / 2 pi) ln 2 of reactance: about
5.05 ohm at 5.8 GHz for a 1 mm port, and S never converges with the mesh
(rfx-archive ledger, "A wire port has no radius").

Expected values here are written from the formula with literals; the
check's own helpers are not called to produce them.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np

from rfx import GaussianPulse, Simulation

MM = 1e-3
CODE = "wire_port_no_radius"


def _dX_literal(f_c: float, length: float) -> float:
    # omega * (mu0 * l / (2 pi)) * ln 2, mu0 = 1.25663706212e-6 H/m.
    return 2 * math.pi * f_c * (1.25663706212e-6 * length / (2 * math.pi)) * math.log(2)


def _uniform():
    sim = Simulation(freq_max=11.6e9, domain=(6 * MM, 6 * MM, 6 * MM),
                     dx=0.25 * MM, boundary="cpml", cpml_layers=8)
    return sim


def _findings(sim):
    return [i for i in sim.preflight() if i.code == CODE]


def _port_line(text: str, index: int) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(f"port {index} (")]
    assert len(lines) == 1, text
    return lines[0]


def _dX_of(line: str) -> str:
    m = re.search(r"dX per mesh halving = ([0-9.eE+-]+) ohm", line)
    assert m, line
    return m.group(1)


def test_one_mm_wire_port_without_radius_reports_dX_per_halving():
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    found = _findings(sim)
    assert len(found) == 1
    issue = found[0]
    assert issue.severity == "warning"
    line = _port_line(str(issue), 0)
    expected = _dX_literal(5.8e9, 1e-3)
    assert abs(expected - 5.0519) < 1e-3  # the brief's ~5.05 ohm, by hand
    assert _dX_of(line) == f"{expected:.3g}"
    assert "l = 1mm" in line
    assert "d = 250µm (x) x 250µm (z)" in line
    assert "f_c = 5.8GHz (the port's pulse centre f0)" in line
    assert "does not converge" in str(issue)


def test_wire_port_with_radius_reports_nothing():
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM,
                 radius=0.02 * MM, waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    assert _findings(sim) == []


def test_single_cell_lumped_port_reports_nothing():
    sim = _uniform()
    sim.add_port((3 * MM, 3 * MM, 3 * MM), "ey")
    assert _findings(sim) == []


def test_two_radius_less_ports_give_one_finding_naming_both():
    sim = _uniform()
    sim.add_port((2 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    sim.add_port((4 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM, excite=False)
    found = _findings(sim)
    assert len(found) == 1
    text = str(found[0])
    assert _dX_of(_port_line(text, 0)) == f"{_dX_literal(5e9, 1e-3):.3g}"
    # The passive port has no pulse: freq_max/2 = 5.8 GHz.
    line1 = _port_line(text, 1)
    assert "passive" in line1
    assert "f_c = 5.8GHz (freq_max/2" in line1
    assert _dX_of(line1) == f"{_dX_literal(5.8e9, 1e-3):.3g}"


def test_non_uniform_transverse_mesh_reads_d_at_the_port():
    # y (the port axis) uniform 0.25 mm; x 0.5 mm; z graded 0.5/0.4/0.5 mm
    # with the port's z inside the 0.4 mm band. Transverse d is x and z at
    # the port, not the axial 0.25 mm and not z's coarsest 0.5 mm. PEC walls:
    # a graded profile against an absorber face is its own refusal.
    dy_profile = np.full(24, 0.25 * MM)
    dz_profile = np.array([0.5] * 4 + [0.4] * 5 + [0.5] * 4) * MM
    sim = Simulation(freq_max=11.6e9, domain=(6 * MM, 6 * MM, 6 * MM),
                     dx=0.5 * MM, dy_profile=dy_profile, dz_profile=dz_profile,
                     boundary="pec", cpml_layers=0)
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    found = _findings(sim)
    assert len(found) == 1
    line = _port_line(str(found[0]), 0)
    assert "d = 500µm (x) x 400µm (z)" in line, line
    assert "l = 1mm" in line, line
    assert _dX_of(line) == f"{_dX_literal(5.8e9, 1e-3):.3g}"


def test_run_reports_it():
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=1 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=4)
    text = "\n".join(str(w.message) for w in caught)
    assert "Wire port without radius" in text


def test_dX_uses_the_realized_length_not_the_declared_extent():
    # 1.1 mm from y = 2.5 mm on 0.25 mm cells realizes four edges, l = 1 mm.
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=1.1 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    line = _port_line(str(_findings(sim)[0]), 0)
    assert "l = 1mm (declared extent 1.1mm)" in line
    assert _dX_of(line) == f"{_dX_literal(5.8e9, 1e-3):.3g}"


def test_an_extent_realized_as_one_edge_is_not_reported():
    # l = d: the term shrinks with the mesh, like a single-cell port.
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=0.05 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    assert _findings(sim) == []


def test_an_extent_of_exactly_one_cell_is_reported():
    # Fixed physical length 0.25 mm: its term grows on refinement even
    # though this mesh realizes it as one edge.
    sim = _uniform()
    sim.add_port((3 * MM, 2.5 * MM, 3 * MM), "ey", extent=0.25 * MM,
                 waveform=GaussianPulse(f0=5.8e9, bandwidth=0.8))
    line = _port_line(str(_findings(sim)[0]), 0)
    assert _dX_of(line) == f"{_dX_literal(5.8e9, 0.25e-3):.3g}"
