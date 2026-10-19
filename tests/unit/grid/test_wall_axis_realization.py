"""Independent arithmetic pins for #1887, including both grid lanes."""
import math
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.grid import Grid, realized_cells
from rfx.runners.nonuniform import build_nonuniform_grid


def model(dx, *, snap='strict', guide=False, graded=False, length=.060):
    kwargs = dict(dx=dx, snap=snap, cpml_layers=2)
    if not guide:
        kwargs['boundary'] = dict(x='cpml', y='pec', z='pec')
    if graded:
        kwargs['dz_profile'] = np.full(8, .01016 / 8)
    sim = Simulation(299792458.0 / .0015 / 20, (length, .02286, .01016), **kwargs)
    if guide:
        sim.add_waveguide_port(.015, direction='+x', mode=(1, 0),
                               freqs=np.array([10e9]), f0=10e9)
    return sim


def spans(grid):
    if hasattr(grid, 'dx_arr_f64'):
        return tuple(sum(widths[getattr(grid, f'pad_{a}_lo'):
                                len(widths) - getattr(grid, f'pad_{a}_hi') - 1])
                     for a, widths in zip('xyz', (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)))
    return tuple((sl.stop - sl.start - 1) * grid.dx for sl in grid.interior)


@pytest.mark.parametrize('guide', [False, True])
def test_wr90_strict_refuses_and_names_both_axes(guide):
    with pytest.raises(ValueError) as caught:
        model(.0015, guide=guide)._build_realized_grid()
    message = str(caught.value)
    assert "axis 'y'" in message and "axis 'z'" in message
    assert 'nearest dx=1.524 mm' in message
    assert "snap='declared'" in message


@pytest.mark.parametrize('guide', [False, True])
def test_wr90_declared_rounds_walls(guide):
    with pytest.warns(UserWarning, match="wall axis 'y'"):
        grid = model(.0015, snap='declared', guide=guide)._build_realized_grid()
    assert spans(grid) == pytest.approx((40 * .0015, 15 * .0015, 7 * .0015), abs=1e-12)


def test_declared_wall_warning_is_coded_once_per_axis_per_simulation():
    sim = model(.0015, snap='declared')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(3):
            sim._build_realized_grid()
    assert len(caught) == 2
    assert [(w.message.code, w.message.loc, w.message.source) for w in caught] == [
        ('wall_axis_off_node', axis, 'judge_wall_axes') for axis in 'yz']
    assert all(w.message.severity == 'warning' for w in caught)

    # A new Simulation owns a separate history; preflight preserves the code
    # while its internal validators repeatedly build the same grid.
    other = model(.0015, snap='declared')
    report = other.preflight()
    assert [issue.loc for issue in report.by_code('wall_axis_off_node')] == ['y', 'z']
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter('always')
        other._build_realized_grid()
        other._build_realized_grid()
    assert not repeated

    # A changed tuple warns again; restoring a previously warned tuple does not.
    original = sim._domain
    sim._domain = (original[0], .0232, original[2])
    with pytest.warns(UserWarning) as changed:
        sim._build_realized_grid()
    assert [(w.message.code, w.message.loc) for w in changed] == [('wall_axis_off_node', 'y')]
    sim._domain = original
    with warnings.catch_warnings(record=True) as restored:
        warnings.simplefilter('always')
        sim._build_realized_grid()
    assert not restored
    sim._snap = 'strict'
    with pytest.raises(ValueError, match="axis 'y'.*nearest dx=1.524 mm"):
        sim._build_realized_grid()


@pytest.mark.parametrize('guide', [False, True])
def test_wr90_exact_spacing_has_no_length_warning(guide):
    with warnings.catch_warnings(record=True) as caught:
        grid = model(.00127, guide=guide)._build_realized_grid()
    assert not caught
    assert spans(grid)[1:] == pytest.approx((18 * .00127, 8 * .00127), abs=1e-12)


@pytest.mark.parametrize('guide', [False, True])
def test_automatic_fit_never_coarsens(guide):
    with pytest.warns(UserWarning, match='snapped'):
        grid = model(None, guide=guide)._build_realized_grid()
    lengths = (.02286, .01016)
    requested = .0015
    candidates = sorted({length / n for length in lengths
                         for n in range(math.ceil(length / requested),
                                        math.floor(length / (.75 * requested)) + 1)}, reverse=True)
    expected = next(dx for dx in candidates
                    if all(abs(math.floor(length / dx + .5) * dx / length - 1) <= .01
                           for length in lengths))
    assert grid.dx == expected and .75 * requested <= grid.dx <= requested
    assert spans(grid)[0] == pytest.approx(math.ceil(.060 / expected) * expected, abs=1e-12)
    for solved, length in zip(spans(grid)[1:], lengths):
        assert abs(solved / length - 1) <= .01


def test_graded_wall_judgement_uses_interior_widths():
    with pytest.raises(ValueError, match="axis 'y'.*nearest dx=1.524 mm"):
        model(.0015, graded=True)._build_realized_grid()
    with pytest.warns(UserWarning, match="wall axis 'y'"):
        grid = model(.0015, graded=True, snap='declared')._build_realized_grid()
    assert spans(grid) == pytest.approx((40 * .0015, 15 * .0015, .01016), abs=1e-12)
    with warnings.catch_warnings(record=True) as caught:
        grid = model(.00127, graded=True)._build_realized_grid()
    assert not caught
    assert spans(grid)[1] == pytest.approx(18 * .00127, abs=1e-12)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('length,count', [(.060, 40), (.0607, 41)])
def test_absorber_count_is_ceiling_in_both_lanes(graded, length, count):
    with pytest.warns(UserWarning, match='wall axis'):
        grid = model(.0015, graded=graded, snap='declared', length=length)._build_realized_grid()
    assert spans(grid)[0] == pytest.approx(count * .0015, abs=1e-12)


def test_synthesized_graded_z_uses_absorber_ceiling():
    grid = build_nonuniform_grid(10e9, (.006, .006, .0607), .0015, 2,
                                None, dx_profile=np.full(4, .0015))
    assert spans(grid)[2] == pytest.approx(41 * .0015, abs=1e-12)


@pytest.mark.parametrize('ratio,count', [(2.5, 3), (3.5, 4), (2.5 - 1e-10, 3), (2.5 - 1e-4, 2), (.1, 1)])
def test_half_ties_and_node_band(ratio, count):
    assert realized_cells(ratio, 1., 'wall') == count


def test_direct_grid_counts_without_simulation_judgement():
    grid = Grid(10e9, (.060, .02286, .01016), dx=.0015,
                pec_faces={f'{a}_{side}' for a in 'yz' for side in ('lo', 'hi')})
    assert spans(grid)[1:] == pytest.approx((15 * .0015, 7 * .0015), abs=1e-12)


@pytest.mark.parametrize('snap', ['strict', 'declared'])
def test_automatic_fit_refuses_below_cost_bound(snap):
    sim = Simulation(299792458.0 / .001 / 20, (.0007, .007, .007),
                     boundary='pec', snap=snap)
    with pytest.raises(ValueError, match='largest fitting dx=0.0007 m'):
        sim._build_realized_grid()


def test_periodic_and_wall_fit_share_one_search():
    sim = Simulation(299792458.0 / .001 / 20, (.0096, .0072, .0056),
                     boundary=dict(x='periodic', y='pec', z='pec'))
    with pytest.warns(UserWarning, match='snapped') as caught:
        grid = sim._build_realized_grid()
    assert len(caught) == 1
    assert grid.dx == pytest.approx(.0008, abs=1e-15)
    assert grid.shape == (12, 10, 8)


@pytest.mark.parametrize('mesh,entry', [
    (mesh, entry) for mesh, entries in [
        ('uniform', ['run', 'forward', 'compute_waveguide_s_matrix',
                     'compute_mixed_s_matrix', 'compute_msl_s_matrix',
                     'compute_lumped_wire_s_matrix_via_scan']),
        ('graded', ['run', 'forward', 'compute_waveguide_s_matrix']),
    ] for entry in entries
])
@pytest.mark.parametrize('explicit_ranges', [False, True])
def test_simulation_entry_refuses_wr90_before_stepping(entry, explicit_ranges, mesh, monkeypatch):
    """Every drive entry must reach the common wall judgement before stepping."""
    import jax
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    from rfx.stepping import graded, uniform

    stepping = []

    def unexpected_step(*args, **kwargs):
        stepping.append(True)
        pytest.fail('time stepping reached before the wall-axis refusal')

    # Guard both step factories and scan execution; leave all builders and
    # admission checks real. A registration-time exception is outside raises.
    monkeypatch.setattr(uniform, 'make_uniform_step', unexpected_step)
    monkeypatch.setattr(graded, 'make_graded_step', unexpected_step)
    monkeypatch.setattr(jax.lax, 'scan', unexpected_step)
    sim = model(.0015, graded=mesh == 'graded')
    if explicit_ranges and entry in ('run', 'forward', 'compute_waveguide_s_matrix'):
        # Uniform guide allocation leaves transverse absorber faces at depth 0.
        if mesh == 'uniform':
            sim = Simulation(299792458.0 / .0015 / 20, (.060, .02286, .01016),
                             dx=.0015, cpml_layers=8)
        for position, direction in ((.015, '+x'), (.045, '-x')):
            sim.add_waveguide_port(position, direction=direction, mode=(1, 0),
                                   freqs=np.array([10e9]), f0=10e9,
                                   y_range=(0., .02286), z_range=(0., .01016))
    freqs = np.array([10e9])
    if entry in ('run', 'forward'):
        sim.add_source((.015, .0105, .0045), component='ez', amplitude_kind='current')
    elif entry == 'compute_waveguide_s_matrix' and not explicit_ranges:
        sim.add_waveguide_port(.015, direction='+x', mode=(1, 0), freqs=freqs, f0=10e9)
        sim.add_waveguide_port(.045, direction='-x', mode=(1, 0), freqs=freqs, f0=10e9)
    else:
        if entry in ('compute_mixed_s_matrix', 'compute_lumped_wire_s_matrix_via_scan'):
            sim.add_port((.015, .0105, .0045), component='ez', impedance=50.)
        if entry in ('compute_mixed_s_matrix', 'compute_msl_s_matrix'):
            sim.add_msl_port((.030, .0105, 0.), width=.003, height=.0015,
                             n_probe_offset=3, n_probe_spacing=2, n_probes=3)

    with pytest.raises(ValueError) as caught:
        if entry == 'compute_lumped_wire_s_matrix_via_scan':
            compute_lumped_wire_s_matrix_via_scan(sim, freqs, n_steps=1)
        elif entry in ('compute_mixed_s_matrix', 'compute_msl_s_matrix'):
            getattr(sim, entry)(n_steps=1, freqs=freqs)
        elif entry == 'run':
            sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
        elif entry == 'forward':
            sim.forward(n_steps=1, skip_preflight=True)
        else:
            sim.compute_waveguide_s_matrix(n_steps=1, normalize=True)
    assert "axis 'y'" in str(caught.value)
    # 22.86 mm / 15 cells = 1.524 mm.
    assert "nearest dx=1.524 mm" in str(caught.value)
    assert not stepping


@pytest.mark.parametrize('entry', [
    'run', 'run-skip-preflight', 'forward', 'compute_waveguide_s_matrix',
])
def test_graded_declared_full_aperture_refuses(entry, monkeypatch):
    import jax

    def unexpected_step(*args, **kwargs):
        pytest.fail('time stepping reached before guide admission')

    monkeypatch.setattr(jax.lax, 'scan', unexpected_step)
    sim = Simulation(12e9, (.05, .02286, .01016), dx=.002, cpml_layers=8,
                     boundary='cpml',
                     dx_profile=np.array([.002]*8 + [.0019, .0021]*4 + [.002]*9))
    freqs = np.array([10e9])
    for position, direction in ((.01, '+x'), (.04, '-x')):
        sim.add_waveguide_port(position, direction=direction, mode=(1, 0),
                               freqs=freqs, f0=10e9,
                               y_range=(0., .02286), z_range=(0., .01016))
    with pytest.raises(ValueError, match='full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo'):
        if entry == 'run-skip-preflight':
            sim.run(n_steps=1, skip_preflight=True)
        elif entry in ('run', 'forward'):
            getattr(sim, entry)(n_steps=1)
        else:
            sim.compute_waveguide_s_matrix(n_steps=1, normalize=True)


def test_zero_depth_absorbers_round_and_are_judged():
    # 10.4 / 1 = 10.4 cells -> 10 cells -> 10 mm on each wall axis.
    sim = Simulation(10e9, (.0104, .0104, .0104), dx=.001, cpml_layers=0)
    with pytest.raises(ValueError, match="wall axis 'x'.*solved 10 mm"):
        sim._build_realized_grid()
    sim._snap = 'declared'
    with pytest.warns(UserWarning) as caught:
        grid = sim._build_realized_grid()
    assert grid.shape == (11, 11, 11)
    assert spans(grid) == pytest.approx((.010, .010, .010), abs=1e-12)
    assert [w.message.loc for w in caught] == ['x', 'y', 'z']
    assert all("set snap='declared' to accept" not in str(w.message) for w in caught)


def test_guide_admission_records_only_the_final_fitted_grid(monkeypatch):
    from rfx.boundaries import features

    seen = []
    record = features._record_defaults

    def observe(sim, grid):
        seen.append(grid)
        return record(sim, grid)

    monkeypatch.setattr(features, '_record_defaults', observe)
    with pytest.warns(UserWarning, match='snapped'):
        grid = model(None, guide=True)._build_realized_grid()
    assert seen == [grid]
    assert hasattr(grid, '_waveguide_admission')
