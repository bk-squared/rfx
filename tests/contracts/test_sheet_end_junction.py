"""A sheet end that lies on a perpendicular conductor over its whole width is
joined, not a free edge (#1787): a shorting wall ending on its patch or ground,
and the patch edge it shorts. Build only.

All expectations below are drawing/node literals in mm. U has 1 mm cells,
planes 3 and 5; G has 0.2 mm cells from 3 to 3.8 and 1 mm outside.
Free ends extend by 0.35 times their outside cell; joined ends stay on nodes.
"""
import contextlib
import io
import re
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.mesh_edges import EDGE_OFFSET, edge_aware_profiles

MM = 1e-3
# case: wall (lo, hi, free_lo, free_hi), patch x or None, for U and G.
# Case 3 bottom is one cell above ground (4 U / 3.2 G): offsets .35 / .07.
# Case 6 patch is one cell ABOVE wall: 6 U / 4.8 G, leaving a true gap.
EXPECTED = {
    1: ((3, 5, False, False), (3, 3.8, False, False), (5.65, 14.35, True, True)),
    2: ((3, 5, False, False), (3, 3.8, False, False), (8, 14.35, False, True)),
    3: ((3.65, 5, True, False), (3.13, 3.8, True, False), (8, 14.35, False, True)),
    4: ((2.65, 5.35, True, True), (2.65, 4.15, True, True), None),
    5: ((2.65, 5, True, False), (2.65, 3.8, True, False), None),
    6: ((3, 5.35, False, True), (3, 4.15, False, True), (7.65, 14.35, True, True)),
    7: ((3, 5.35, False, True), (3, 4.15, False, True), (8, 14.35, False, True)),
    8: ((3, 5, False, False), (3, 3.8, False, False), (7.65, 14.35, True, True)),
}


def _box(lo, hi):
    return Box(tuple(v * MM for v in lo), tuple(v * MM for v in hi))


def _drawing(case, top, bottom=3, gap=1):
    shapes = {}
    if case not in (4, 5):
        shapes['ground'] = _box((4, 4, 3), (16, 16, 3))
    if case not in (4, 5):
        pz = top + gap if case == 6 else top
        shapes['patch'] = _box((6 if case == 1 else 8, 6, pz), (14, 14, pz))
    wy = (5, 15) if case == 7 else (7, 9) if case == 8 else (6, 14)
    shapes['wall'] = _box((8, wy[0], bottom), (8, wy[1], top))
    if case == 5:
        shapes['block'] = _box((7, 5, top), (10, 15, top + 1))
    return shapes


def _build(shapes, domain, dx, **profiles):
    sim = Simulation(freq_max=10e9, domain=domain, dx=dx, boundary='pec',
                     snap='strict', **profiles)
    labels = {}
    for name, shape in shapes.items():
        labels[name] = f'geometry[{len(labels)}]'
        sim.add(shape, material='pec')
    return sim, labels


def _reports(sim):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), pytest.MonkeyPatch.context() as patch:
        warnings.simplefilter('ignore')
        # The public message lists only five offenders by default. Lift only
        # its display cap so every sheet face can be compared in this contract.
        patch.setattr('rfx.preflight.pec_geometry._CAMPAIGN_MAX_OFFENDERS', 100)
        issues = [issue.to_dict() for issue in sim.preflight()]
        fidelity = sim.fidelity_report(print_report=False)
        record = sim.realized_geometry()
    return issues, fidelity, record


_ROW = re.compile(r"(geometry\[\d+\]) '[^']*' \(sheet\) ([xyz]): extent "
                  r"[^,]+, worst face residual ([\d.]+)(m|mm|µm|nm) ")
_UNIT_UM = {'m': 1e6, 'mm': 1e3, 'µm': 1, 'nm': .001}


def _census_matches(issues, fidelity, record):
    rows = {}
    for issue in issues:
        if issue['code'] == 'off_lattice_design_edges':
            for m in _ROW.finditer(issue['message'].split('. OBSERVED')[0] + ' '):
                rows[m[1], m[2]] = float(m[3]) * _UNIT_UM[m[4]]
    for entity in record.entities:
        if entity.kind != 'sheet' or all(a.declared_bounds_m is None for a in entity.axes):
            continue
        item, = [item for item in fidelity if str(item.get('entity', '')).split(' ')[0] == entity.label]
        for a, axis in zip(entity.axes, item['axes'], strict=True):
            if a.declared_bounds_m[0] == a.declared_bounds_m[1]:
                continue
            residual = np.abs(np.array(a.face_residual_m)) * 1e6
            assert axis['face_residual_um'] == pytest.approx(residual, abs=1e-6)
            worst = max(residual)
            key = (entity.label, a.axis)
            if worst / axis['declared_extent_um'] > .005:
                assert rows[key] == pytest.approx(worst, rel=1e-3)
            else:
                assert key not in rows


def _assert_span(issues, fidelity, entity, axis, expected):
    lo, hi, free_lo, free_hi = expected
    a = entity.axes['xyz'.index(axis)]
    assert a.free_ends == (free_lo, free_hi)
    assert a.bounds_m == pytest.approx((lo * MM, hi * MM), abs=1e-12)
    assert a.comparison_bounds_m == pytest.approx(a.declared_bounds_m, abs=1e-12)
    item, = [item for item in fidelity if str(item.get('entity', '')).split(' ')[0] == entity.label]
    fa, = [a for a in item['axes'] if a['axis'] == axis]
    assert fa['realized_um'] == pytest.approx((lo * 1000, hi * 1000), abs=1e-6)
    for i, free in enumerate((free_lo, free_hi)):
        if not free:
            assert fa['face_residual_um'][i] == pytest.approx(0, abs=1e-6)
    if not free_lo and not free_hi:
        for issue in issues:
            if issue['code'] in ('sheet_effective_size', 'off_lattice_design_edges'):
                assert not re.search(re.escape(entity.label) + r" '[^']*'(?: \(sheet\))? " + axis + ':',
                                     issue['message'].split('. OBSERVED')[0])
    print(f'{entity.label} {axis}: {a.bounds_m}, free={a.free_ends}')


@pytest.mark.parametrize('mesh', ['U', 'G'])
@pytest.mark.parametrize('case', range(1, 9))
def test_case(case, mesh):
    graded = mesh == 'G'
    top = 3.8 if graded else 5
    bottom = (3.2 if graded else 4) if case == 3 else 3
    profiles = {'dz_profile': np.array([1, 1, 1] + [.2] * 4 + [1] * 4) * MM} if graded else {}
    shapes = _drawing(case, top, bottom)
    sim, labels = _build(shapes, (20 * MM, 20 * MM, (7.8 if graded else 9) * MM), MM, **profiles)
    issues, fidelity, record = _reports(sim)
    entities = {entity.label: entity for entity in record.entities}
    wall_u, wall_g, patch = EXPECTED[case]
    _assert_span(issues, fidelity, entities[labels['wall']], 'z', wall_g if graded else wall_u)
    if patch is not None:
        _assert_span(issues, fidelity, entities[labels['patch']], 'x', patch)
    _census_matches(issues, fidelity, record)


def _registered(case):
    shapes = _drawing(case, 4.6)
    domain = (30 * MM, 30 * MM, 12 * MM)
    profiles = edge_aware_profiles(domain, .5 * MM, sheets=list(shapes.values()), axes='xyz')
    return shapes, domain, profiles


@pytest.mark.parametrize('case', [1, 2])
def test_helper_junction_planes(case):
    shapes, domain, profiles = _registered(case)
    nodes = {a: np.r_[0., np.cumsum(profiles[f'd{a}_profile'])] for a in 'xyz'}
    for axis, plane in [('z', .003), ('z', .0046), ('x', .008)]:
        assert min(abs(nodes[axis] - plane)) <= 1e-12
    # Open patch x-hi=14 mm: final interior node plus .35 outside cell.
    x = nodes['x']
    i = np.searchsorted(x, .014) - 1
    assert abs(x[i] + EDGE_OFFSET * (x[i + 1] - x[i]) - .014) <= 1e-12


@pytest.mark.parametrize('case', [1, 2])
def test_strict_admission(case):
    shapes, domain, profiles = _registered(case)
    sim, _ = _build(shapes, domain, .5 * MM, **profiles)
    issues, fidelity, record = _reports(sim)
    assert not [i for i in issues if i['code'] == 'sheet_effective_size' and i['severity'] == 'error']
    assert not [i for i in issues if i['code'] == 'off_lattice_design_edges']
    _census_matches(issues, fidelity, record)


@pytest.mark.parametrize('case', ['wider_wall', 'beside_end'])
def test_helper_free_top(case):
    shapes = _drawing(2, 4.6)
    if case == 'wider_wall':
        shapes['wall'] = _box((8, 5, 3), (8, 15, 4.6))
    else:
        shapes['patch'] = _box((9, 6, 4.6), (14, 14, 4.6))
    profiles = edge_aware_profiles((.030, .030, .012), .0005,
                                   sheets=list(shapes.values()), axes='xyz')
    z = np.r_[0., np.cumsum(profiles['dz_profile'])]
    # Drawn wall top 4.6 mm is a free edge: last node + .35 outside cell.
    i = np.searchsorted(z, .0046) - 1
    assert z[i] + EDGE_OFFSET * (z[i + 1] - z[i]) == pytest.approx(.0046, abs=1e-12)
    if case == 'wider_wall':
        # Partial contact leaves the top free, so its edge displaces the
        # coincident patch face from every node.
        assert min(abs(z - .0046)) > 1e-12


def test_pinned_patch_metric_wall():
    shapes = {
        'ground': _box((4, 4, 3), (16, 16, 3)),
        'wall': _box((8, 6, 3), (8, 14, 5)),
    }
    sim, labels = _build(shapes, (.020, .020, .009), .001)
    sim.add_pinned_sheet(plane_index=5, i_range=(8, 14), j_range=(6, 14))
    issues, fidelity, record = _reports(sim)
    entities = {entity.label: entity for entity in record.entities}
    _assert_span(issues, fidelity, entities[labels['wall']], 'z', (3, 5, False, False))
    # Pinned record axes expose node bounds; solved spans are on its sheet.
    # x-lo=8 mm meets the wall; x-hi=14 mm has a 1 mm outside cell.
    patch = entities['pinned_sheet[0]'].sheet.solved_spans[0]
    assert (patch.free_lo, patch.free_hi) == (False, True)
    assert (patch.lo, patch.hi) == pytest.approx((.008, .01435), abs=1e-12)
    _census_matches(issues, fidelity, record)


def test_closed_box_every_end_joined():
    """Six sheets close a box (x 4..12, y 5..14, z 3..7 mm, unequal on purpose):
    every in-plane end of every face lies on a perpendicular face, so every
    span is node to node and no end is free. Covers x-, y- and z-normal sheets
    along each of their in-plane axes."""
    x, y, z = (4, 12), (5, 14), (3, 7)
    faces = {
        'x_lo': _box((x[0], y[0], z[0]), (x[0], y[1], z[1])),
        'x_hi': _box((x[1], y[0], z[0]), (x[1], y[1], z[1])),
        'y_lo': _box((x[0], y[0], z[0]), (x[1], y[0], z[1])),
        'y_hi': _box((x[0], y[1], z[0]), (x[1], y[1], z[1])),
        'z_lo': _box((x[0], y[0], z[0]), (x[1], y[1], z[0])),
        'z_hi': _box((x[0], y[0], z[1]), (x[1], y[1], z[1])),
    }
    sim, labels = _build(faces, (20 * MM, 20 * MM, 9 * MM), MM)
    issues, fidelity, record = _reports(sim)
    entities = {entity.label: entity for entity in record.entities}
    span = {'x': x, 'y': y, 'z': z}
    for name in faces:
        for axis in 'xyz':
            if axis != name[0]:
                _assert_span(issues, fidelity, entities[labels[name]], axis,
                             (*span[axis], False, False))
    _census_matches(issues, fidelity, record)
