"""Static closure of the port-owned line, independently read from its cells."""
from collections import deque

import numpy as np
import pytest

from rfx import Box, Simulation
from tests.contracts.test_coax_port_model_and_drive import capture


@pytest.fixture(params=[(dx, face) for dx in (.00030, .00025, .0001875)
                        for face in ('top', 'bottom')])
def shielded(request, monkeypatch):
    dx, face = request.param
    reference = .002 if face == 'top' else .038
    sim = Simulation(freq_max=11e9, domain=(.012, .012, .040), dx=dx,
                     boundary='cpml', cpml_layers=4, snap='declared')
    sim.add(Box((0., 0., reference), (.012, .012, reference)), material='pec')
    sim.add_coaxial_port((.006, .006, reference), face=face, pin_length=0.)
    captured = capture(monkeypatch, sim, 'run')
    root = sim.realized_geometry().conductors
    grid = captured['grid']
    port = sim._coaxial_ports[0]
    layout = root.coax_bore_release.layout
    direction = 1 if face == 'top' else -1
    cap = layout.z_hi_index if direction == 1 else layout.z_lo_index
    gap = max(2, int(np.ceil(port.outer_radius / float(grid.dx))))
    gap_indices = cap - direction * np.arange(1, gap + 1)
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * grid.dx - port.position[0]
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * grid.dx - port.position[1]
    radius = np.hypot(x[:, None], y[None, :])
    print(f'shield dx={dx * 1e3:g} mm face={face} g={gap} cap={cap} '
          f'pin_end={cap-direction*(gap+1)} feed={layout.feed_index} '
          f'source={layout.source_index} planes={layout.plane_indices}')
    return grid, port, root, layout, direction, cap, gap, gap_indices, radius, captured


def test_closure(shielded):
    grid, port, root, layout, direction, cap, gap, indices, radius, captured = shielded
    cells = np.asarray(root.pec_cells)
    shell = next(e for e in root.stamped_entities if e.entity_id.endswith('/shell'))
    outer = shell.declared_radii_m[1]
    low = layout.z_lo_index - (direction == -1)
    high = layout.z_hi_index + (direction == 1)
    seeds = np.argwhere((radius > port.pin_radius) & (radius <= port.outer_radius)
                        & ~cells[:, :, layout.source_index])
    assert seeds.size
    queue = deque((int(i), int(j), layout.source_index) for i, j in seeds)
    seen = np.zeros(grid.shape, bool)
    for cell in queue:
        seen[cell] = True
    while queue:
        i, j, k = queue.popleft()
        assert direction * (k - cap) <= 0, 'flood escaped beyond cap'
        assert radius[i, j] <= outer, 'flood escaped realized clamped shell'
        for a, b, c in ((i-1,j,k), (i+1,j,k), (i,j-1,k), (i,j+1,k), (i,j,k-1), (i,j,k+1)):
            if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and low <= c <= high:
                if not seen[a,b,c] and not cells[a,b,c]:
                    seen[a,b,c] = True
                    queue.append((a,b,c))


def test_gap(shielded):
    grid, port, root, layout, direction, cap, gap, indices, radius, captured = shielded
    pin = next(e for e in root.stamped_entities if e.entity_id.endswith('/pin'))
    assert len(indices) == gap
    assert not np.any(np.asarray(pin.cells)[:, :, indices]), 'pin cell left in gap'
    assert not np.any(np.asarray(root.pec_cells)[:, :, indices][radius <= port.pin_radius]), 'PEC pin cell left in gap'
    pin_end = np.where(pin.cells)[2].max() if direction == 1 else np.where(pin.cells)[2].min()
    assert direction * (cap - pin_end) - 1 == gap
    eps = np.asarray(captured['materials'].eps_r)
    np.testing.assert_array_equal(eps[:, :, indices][radius <= port.pin_radius],
                                  np.asarray(2.1, dtype=eps.dtype))


def test_cap(shielded):
    grid, port, root, layout, direction, cap, gap, indices, radius, captured = shielded
    entity = next(e for e in root.stamped_entities if '/cap/' in e.entity_id)
    expected = np.zeros(grid.shape, bool)
    expected[:, :, cap] = radius <= port.outer_radius
    np.testing.assert_array_equal(entity.cells, expected)
    assert np.all(np.asarray(root.pec_cells)[:, :, cap][radius <= port.outer_radius])


def test_layout_keeps_old_offsets_and_planes(shielded):
    grid, port, root, layout, direction, cap, gap, indices, radius, captured = shielded
    ref = int(grid.position_to_index(port.position)[2])
    far = grid.shape[2] - grid.pad_z_hi - 2 if direction == 1 else grid.pad_z_lo + 1
    distance = direction * (far - ref)
    nominal = ref + distance - 3
    old = int(round(((nominal-grid.pad_z_lo)*grid.dx-grid.dx/2)
                    /float(np.asarray(grid.cells(2))[0]))) + grid.pad_z_lo
    old_source = ref + direction * (old-ref)
    old_feed = far-direction
    old_pin = far if direction == 1 else far-1
    pin_end = cap-direction*(gap+1)
    assert layout.source_index == old_source-direction*(gap+1)
    assert layout.feed_index == old_feed-direction*(gap+1)
    assert layout.feed_index-layout.source_index == old_feed-old_source
    assert pin_end-layout.feed_index == old_pin-old_feed
    assert layout.plane_indices == tuple(ref+direction*(8+4*k) for k in range(12))
    assert layout.reference_index == ref
    assert cap == (far if direction == 1 else far-1)
