"""The #1512 census predicate is independent of solver admission."""
import math

import pytest

from rfx import Box, Simulation
from rfx.preflight.line_stub import line_stub_findings


def make_sim(direction='+x', *, endpoint=0., terminates=None, thin=False):
    sim = Simulation(domain=(.028, .012, .006), dx=.0005,
                     freq_max=40e9, boundary='cpml', cpml_layers=3)
    sim.add_material('board', eps_r=3.66)
    sim.add(Box((0, 0, 0), (.028, .012, .001)), material='board')
    sim.add(Box((0, 0, 0), (.028, .012, 0)), material='pec')
    axis = 'xy'.index(direction[-1])
    lo, hi = [0, .005, .001], [.028, .007, .001]
    position = [.002 if direction[0] == '+' else .026, .006, 0]
    if axis == 1:
        lo[:2], hi[:2] = [.005, 0], [.007, .012]
        position = [.006, .002 if direction[0] == '+' else .010, 0]
    if direction[0] == '+':
        lo[axis] = endpoint
    else:
        hi[axis] -= endpoint
    strip = Box(tuple(lo), tuple(hi))
    if thin:
        sim.add_thin_conductor(strip)
    else:
        sim.add(strip, material='pec')
    sim.add_msl_port(tuple(position), width=.002, height=.001,
                     direction=direction, eps_r_sub=3.66, terminates=terminates)
    return sim


@pytest.mark.parametrize('direction', ['+x', '-x', '+y', '-y'])
@pytest.mark.parametrize('thin', [False, True])
def test_tail_length_and_frequency(direction, thin):
    findings = line_stub_findings(make_sim(direction, thin=thin))
    assert len(findings) == 1
    f = findings[0]
    assert f.overhang_m == pytest.approx(.002)
    assert f.frequency_hz == pytest.approx(299792458 / (.008 * math.sqrt(f.eps_eff)))
    assert 'Start the strip at the port plane' in f.message
    assert '2 mm' in f.message


def test_internal_endpoint_and_ground_exclusion():
    sim = make_sim(endpoint=.0005)
    assert line_stub_findings(sim)[0].overhang_m == pytest.approx(.0015)
    sim = make_sim(endpoint=.002)
    sim.add(Box((0, .009, .001), (.028, .010, .001)), material='pec')
    assert line_stub_findings(sim) == []


def test_detached_tail_and_continuation_are_not_internal_stubs():
    sim = make_sim(endpoint=.002)
    sim.add(Box((0, .005, .001), (.001, .007, .001)), material='pec')
    assert line_stub_findings(sim) == []
    assert line_stub_findings(make_sim(terminates=())) == []


def test_joined_declarations_measure_the_whole_tail():
    sim = make_sim(endpoint=.001)
    sim.add(Box((.0005, .005, .001), (.001, .007, .001)), material='pec')
    assert line_stub_findings(sim)[0].overhang_m == pytest.approx(.0015)


def test_registered_coax_signal_extension_excludes_outer_metal():
    sim = Simulation(domain=(.012, .012, .012), dx=.001,
                     freq_max=20e9, boundary='cpml', cpml_layers=2)
    sim.add(Box((.0055, .0055, .002), (.0065, .0065, .009)), material='pec')
    sim.add(Box((.008, 0, 0), (.010, .012, .012)), material='pec')
    sim.add_coaxial_port((.006, .006, .004), face='bottom', pin_length=.004)
    f, = line_stub_findings(sim)
    assert f.overhang_m == pytest.approx(.002)


def test_transverse_gap_cannot_join_an_unrelated_tail():
    sim = make_sim(endpoint=.002)
    # Narrow the actual strip to one side of the aperture and put an
    # axially overlapping but disconnected strip on the other side.
    sim._geometry.pop()
    sim.add(Box((.002, .005, .001), (.028, .0055, .001)), material='pec')
    sim.add(Box((.0005, .0065, .001), (.0015, .007, .001)), material='pec')
    assert line_stub_findings(sim) == []


def test_roundoff_at_the_trace_plane_is_not_a_gap():
    sim = make_sim(endpoint=.0005)
    from dataclasses import replace
    sim._msl_ports[0] = replace(sim._msl_ports[0], height=.001 + 2e-19)
    assert line_stub_findings(sim)[0].overhang_m == pytest.approx(.0015)


def test_coax_ground_wall_is_not_a_line_tail():
    sim = Simulation(domain=(.012, .012, .012), dx=.001,
                     freq_max=20e9, boundary='cpml', cpml_layers=2)
    sim.add(Box((0, 0, .002), (.012, .012, .004)), material='pec')
    sim.add_coaxial_port((.006, .006, .004), face='bottom', pin_length=.004)
    assert line_stub_findings(sim) == []


def test_node_pinned_signal_sheet_is_included():
    sim = make_sim(endpoint=.002)
    sim._geometry.pop()
    sim.add_pinned_sheet(plane_index=2, i_range=(1, 50), j_range=(10, 14))
    assert line_stub_findings(sim)[0].overhang_m == pytest.approx(.0015)


def test_cylindrical_coax_pin_uses_realized_axial_support():
    from rfx import Cylinder
    sim = Simulation(domain=(.012, .012, .012), dx=.0005,
                     freq_max=20e9, boundary='cpml', cpml_layers=2)
    sim.add(Cylinder(center=(.006, .006, .005), radius=.0005,
                     height=.006, axis='z'), material='pec')
    sim.add_coaxial_port((.006, .006, .004), face='bottom',
                         pin_radius=.0005, pin_length=.004)
    f, = line_stub_findings(sim)
    assert f.overhang_m == pytest.approx(.002)


def test_boundary_without_absorber_is_not_an_open_end():
    sim = Simulation(domain=(.010, .010, .004), dx=.001,
                     freq_max=10e9, boundary='pec', cpml_layers=0)
    sim.add(Box((0, .004, .001), (.010, .006, .001)), material='pec')
    sim.add_msl_port((.002, .005, 0), width=.002, height=.001)
    assert line_stub_findings(sim) == []
