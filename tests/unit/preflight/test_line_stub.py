"""The #1512 census predicate is independent of solver admission."""
import math

import pytest

from rfx import Box, Simulation
from rfx.preflight.line_stub import line_stub_findings


def closed_form(eps_r, width, height, length):
    """Independent of rfx: quasi-static eps_eff, Hammerstad open end, quarter wave."""
    u = width / height
    eps_eff = (eps_r + 1)/2 + (eps_r - 1)/(2*math.sqrt(1 + 12/u))
    extension = .412*height*(eps_eff + .3)*(u + .264)/((eps_eff - .258)*(u + .8))
    return eps_eff, extension, 299792458/(4*(length + extension)*math.sqrt(eps_eff))


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
    eps_eff, extension, frequency = closed_form(3.66, .002, .001, .002)
    assert f.overhang_m == pytest.approx(.002)
    assert f.eps_eff == pytest.approx(eps_eff, rel=1e-12)
    assert f.end_extension_m == pytest.approx(extension, rel=1e-12)
    assert f.effective_length_m == pytest.approx(.002 + extension, rel=1e-12)
    assert f.frequency_hz == pytest.approx(frequency, rel=1e-12)
    assert f.frequency_hz < 299792458 / (.008 * math.sqrt(eps_eff))
    assert f"open-end extension {extension*1e3:.6g} mm" in f.message
    assert f"effective length is {(.002 + extension)*1e3:.6g} mm" in f.message
    assert 'start the signal strip at that coordinate' in f.message
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


@pytest.mark.parametrize('nonuniform', [False, True])
def test_realized_port_and_endpoint_coordinates_own_length(nonuniform):
    from dataclasses import replace
    import numpy as np
    from rfx.sources.msl_port import _msl_grid_geometry, _msl_position_to_index

    sim = make_sim(endpoint=.0006, thin=True)
    if nonuniform:
        # Exercise a genuinely graded propagation axis, not only NU dispatch.
        sim._dx_profile = np.r_[.0005, np.full(27, .00025), np.full(27, .00075), .0005]
    port = sim._msl_ports[0]
    sim._msl_ports[0] = replace(port, position=(.0021, *port.position[1:]))
    grid = sim._build_realized_grid()
    nodes, _ = _msl_grid_geometry(grid)
    p = _msl_position_to_index(grid, sim._msl_ports[0].position)[0]
    endpoint = next(float(x) for x in nodes[0] if x >= .0006)
    finding, = line_stub_findings(sim, grid)
    assert finding.overhang_m == pytest.approx(float(nodes[0][p]) - endpoint)
    assert finding.declared_overhang_m == pytest.approx(.0015)
    assert finding.port_node_m == pytest.approx(float(nodes[0][p]))
    assert f"x={float(nodes[0][p])*1e3:.9g} mm" in finding.message
    assert finding.overhang_m != pytest.approx(finding.declared_overhang_m)
    assert 'realized L; declared 1.5 mm' in finding.message


@pytest.mark.parametrize('width_over_height,eps_r', [(2., 3.55), (4., 3.55), (1., 2.2), (3., 9.8)])
def test_eps_eff_and_quarter_wave_have_an_independent_closed_form(width_over_height, eps_r):
    """Independent quasi-static expression plus full HJ1980 cross-check.

    Full HJ1980: https://qucs.sourceforge.net/tech/node75.html, eqs. 11.15–18.
    The production helper retains the HJ name but implements the simplified
    expression. The 1% comparison is to the full model, not a solver bound.
    The open-end extension is Hammerstad's closed form, written out here and
    not read from rfx; the stub's quarter wave is taken on realized L + extension.
    """
    u = width_over_height
    sim = Simulation(domain=(.028, .012, .006), dx=.0005,
                     freq_max=40e9, boundary='cpml', cpml_layers=3)
    sim.add_material('board', eps_r=eps_r)
    sim.add(Box((0, 0, 0), (.028, .012, .001)), material='board')
    sim.add(Box((0, 0, 0), (.028, .012, 0)), material='pec')
    width = u * .001
    sim.add(Box((0, .006-width/2, .001), (.028, .006+width/2, .001)), material='pec')
    sim.add_msl_port((.002, .006, 0), width=width, height=.001, eps_r_sub=eps_r)
    finding, = line_stub_findings(sim)
    expected = (eps_r + 1)/2 + (eps_r - 1)/(2*math.sqrt(1 + 12/u))
    a = 1 + math.log((u**4 + (u/52)**2)/(u**4 + .432))/49 + math.log(1 + (u/18.1)**3)/18.7
    b = .564*((eps_r - .9)/(eps_r + 3))**.053
    full_hj = (eps_r + 1)/2 + (eps_r - 1)/2*(1 + 10/u)**(-a*b)
    def extension(eps_eff):
        return .412*.001*(eps_eff + .3)*(u + .264)/((eps_eff - .258)*(u + .8))
    assert finding.eps_eff == pytest.approx(expected, rel=1e-12)
    assert finding.overhang_m == pytest.approx(.002, rel=1e-9)
    assert finding.end_extension_m == pytest.approx(extension(expected), rel=1e-12)
    assert .2e-3 < finding.end_extension_m < .6e-3
    assert finding.frequency_hz == pytest.approx(
        299792458/(4*(.002 + extension(expected))*math.sqrt(expected)), rel=1e-9)
    assert finding.eps_eff == pytest.approx(full_hj, rel=.012)
    assert finding.frequency_hz == pytest.approx(
        299792458/(4*(.002 + extension(full_hj))*math.sqrt(full_hj)), rel=.006)


def test_invalid_realization_stays_blocking_without_aborting_other_preflight_checks():
    from tests.unit.ports.test_msl_port_preflight import _build_sim, W_TRACE, H_SUB
    sim = _build_sim(dx=1e-3, ly=W_TRACE + 8*H_SUB)
    report = sim.preflight()
    assert any(issue.code == 'line_stub_realization' for issue in report.errors)
    assert any('danger zone' in str(issue) for issue in report)
    with pytest.raises(ValueError, match='ZERO nodes'):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize('nonuniform', [False, True])
@pytest.mark.parametrize('declared', [2.2, 9.8])
def test_realized_substrate_owns_frequency_and_decision(nonuniform, declared):
    from dataclasses import replace
    import numpy as np
    from rfx.preflight.line_stub import resonant_odd_orders, stub_message
    sim = make_sim()
    sim.add_material('actual', eps_r=2.2)
    sim._geometry[0] = replace(sim._geometry[0], material_name='actual')
    sim._msl_ports[0] = replace(sim._msl_ports[0], eps_r_sub=declared)
    # Registration-time HJ values must not override the realized substrate.
    sim._msl_auto_probe_spacing[sim._msl_ports[0].name] = 7.38
    if nonuniform:
        sim._dx_profile = np.full(56, .0005)
    f, = line_stub_findings(sim)
    expected, _, frequency = closed_form(2.2, .002, .001, .002)
    assert f.substrate_eps_r == 2.2
    assert f.eps_eff == pytest.approx(expected)
    assert f.frequency_hz == pytest.approx(frequency)
    # 22.6 GHz with the open-end extension (27.7 GHz on the metal length alone).
    assert resonant_odd_orders(f, (4e9, 14e9)) is None
    assert resonant_odd_orders(f, (4e9, 18e9)) is not None
    assert resonant_odd_orders(f, (20e9, 30e9)) is not None
    message = stub_message(f, (4e9, 14e9))
    assert ('Realized substrate eps_r=2.2; declared port eps_r_sub=9.8' in message) == (declared == 9.8)


def test_substrate_is_sampled_under_stub_with_realized_masks_and_precedence():
    from dataclasses import replace
    sim = make_sim()
    sim._msl_ports[0] = replace(sim._msl_ports[0], eps_r_sub=9.8)
    sim.add_material('stub_only', eps_r=2.2)
    # Ends before the port; starts above z=.5 mm but realizes onto that node.
    sim.add(Box((0, 0, .00051), (.0016, .012, .00065)), material='stub_only')
    f, = line_stub_findings(sim)
    assert f.substrate_eps_r == 2.2
    # Confirm the independently assembled material at the sampled stub node.
    from rfx.sources.msl_port import _msl_position_to_index
    grid = sim._build_realized_grid()
    materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    index = _msl_position_to_index(grid, (.001, .006, .0005))
    assert float(materials.eps_r[index]) == pytest.approx(f.substrate_eps_r)
    # With no realized dielectric, use the declaration.
    sim._geometry = [e for e in sim._geometry if e.material_name == 'pec']
    f, = line_stub_findings(sim)
    assert f.substrate_eps_r == 9.8


@pytest.mark.parametrize('thin', [False, True])
@pytest.mark.parametrize('port_x', [.00213, .00237])
def test_the_placement_the_refusal_recommends_is_admitted(thin, port_x):
    """Follow the message literally: redraw the strip from the printed coordinate.

    The port plane is off the node lines on purpose (nearest node behind it for
    2.13 mm, ahead of it for 2.37 mm): a strip drawn from the declared plane is
    what port preflight rejects, so the remedy must name the node, not the plane.
    """
    import re

    def build(start):
        sim = Simulation(domain=(.028, .012, .006), dx=.0005,
                         freq_max=40e9, boundary='cpml', cpml_layers=3)
        sim.add_material('board', eps_r=3.66)
        sim.add(Box((0, 0, 0), (.028, .012, .001)), material='board')
        sim.add(Box((0, 0, 0), (.028, .012, 0)), material='pec')
        strip = Box((start, .005, .001), (.028, .007, .001))
        if thin:
            sim.add_thin_conductor(strip)
        else:
            sim.add(strip, material='pec')
        sim.add_msl_port((port_x, .006, 0), width=.002, height=.001,
                         direction='+x', eps_r_sub=3.66)
        return sim

    finding, = line_stub_findings(build(0.))
    printed = float(re.search(r"grid node is x=([0-9.eE+-]+) mm", finding.message).group(1)) * 1e-3
    assert printed != pytest.approx(port_x, abs=1e-5)
    sim = build(printed)
    assert line_stub_findings(sim) == []
    report = sim.preflight()
    # The fixture's 2 mm strip on 0.5 mm cells also trips the unrelated sheet-size
    # check; only the port-attachment and stub errors are this test's subject.
    blocking = [str(i) for i in getattr(report, 'errors', [])
                if 'MSL port' in str(i) or 'stub' in str(i)]
    assert not blocking, blocking
    # The declared plane itself is the placement that fails, which is why the
    # message names the node.
    rejected = [str(i) for i in getattr(build(port_x).preflight(), 'errors', [])
                if 'MSL port' in str(i)]
    assert bool(rejected) == (printed < port_x)


def test_an_uninspectable_shape_does_not_switch_the_refusal_off():
    """A shape the lattice cannot place is skipped alone; the strip is still judged."""
    from dataclasses import replace
    from rfx.preflight.line_stub import resonant_odd_orders

    class _NoBBox:  # the lattice refuses to place it (cf. the reflector-scan double)
        def mask(self, grid):
            raise NotImplementedError

        def mask_on_coords(self, x, y, z):
            raise NotImplementedError

    def build(extra):
        sim = make_sim()  # 2 mm open tail behind the port, band 0..40 GHz
        if extra:
            sim._geometry.append(replace(sim._geometry[-1], shape=_NoBBox()))
        return sim

    clean, = line_stub_findings(build(False))
    assert resonant_odd_orders(clean, (0., 40e9)) is not None
    skipped = []
    found, = line_stub_findings(build(True), uninspectable=skipped)
    assert found.overhang_m == pytest.approx(clean.overhang_m)
    assert len(skipped) == 1 and '_NoBBox' in skipped[0]
    report = [str(i) for i in build(True).preflight(strict=False, check_ntff=False)]
    assert any('could not inspect 1 conductor shape' in m and '_NoBBox' in m for m in report)
    assert any('is an open stub that shorts the port' in m for m in report)
    with pytest.raises(ValueError, match='is an open stub that shorts the port'):
        build(True).run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize('method', ['run', 'forward', 'compute_msl_s_matrix'])
def test_uninspectable_shape_is_named_at_the_solve_entry_without_preflight(method):
    """skip_preflight must not make the skipped shape silent; the caller owns the warning."""
    import warnings
    from dataclasses import replace

    class _NoBBox:
        def mask(self, grid):
            raise NotImplementedError

        def mask_on_coords(self, x, y, z):
            raise NotImplementedError

    sim = make_sim(endpoint=.002)  # no tail: nothing to refuse, only the skip to report
    sim._geometry.append(replace(sim._geometry[-1], shape=_NoBBox()))
    kwargs = {'n_steps': 1}
    if method != 'compute_msl_s_matrix':
        kwargs['skip_preflight'] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            getattr(sim, method)(**kwargs)
        except Exception:  # the shape cannot be assembled either; admission ran first
            pass
    skipped = [w for w in caught
               if getattr(w.message, 'code', '') == 'line_stub_inspection_unavailable']
    assert skipped, [str(w.message) for w in caught]
    assert all('_NoBBox' in str(w.message) for w in skipped)
    assert skipped[0].filename == __file__, (skipped[0].filename, skipped[0].lineno)


def test_coax_permittivity_is_read_between_pin_and_shield():
    """A fill that leaves the pin-centre cell uncovered still sets the stub frequency."""
    def build(fill):
        sim = Simulation(domain=(.012, .012, .012), dx=.001,
                         freq_max=20e9, boundary='cpml', cpml_layers=2)
        sim.add_material('fill', eps_r=2.1)
        if fill:  # one side of the pin only; the pin-centre column stays unfilled
            sim.add(Box((.0065, .004, .001), (.009, .008, .005)), material='fill')
        sim.add(Box((.0055, .0055, .002), (.0065, .0065, .009)), material='pec')
        sim.add_coaxial_port((.006, .006, .004), face='bottom', pin_length=.004)
        return sim

    f, = line_stub_findings(build(True))
    assert f.overhang_m == pytest.approx(.002)
    assert f.substrate_eps_r == 2.1 and f.eps_eff == 2.1
    assert f.end_extension_m == 0.  # no open-end formula is applied to a coax pin
    assert f.frequency_hz == pytest.approx(299792458/(4*.002*math.sqrt(2.1)), rel=1e-12)
    empty, = line_stub_findings(build(False))
    assert empty.eps_eff == 1.
