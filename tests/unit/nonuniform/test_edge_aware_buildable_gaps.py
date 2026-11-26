"""The strip/absorber gap set follows closed-form ratio bounds, not helper output."""

import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxRefusal
from rfx.mesh_edges import EDGE_OFFSET, SheetEdge, edge_aware_profile, edge_aware_profiles


H, W, LX = 254e-6, 600e-6, 12e-3
DX, LZ = H / 3.0, H + 1.5e-3
RATIO = 1.3
OUTSIDE = 1.0 - EDGE_OFFSET
LOWER = 1.0 + OUTSIDE
UPPER = 1.0 + 1.0 / RATIO + OUTSIDE / RATIO**2
INTERVALS = (
    (1.0 + OUTSIDE / RATIO, LOWER),
    (UPPER, 2.0 + OUTSIDE),
    (1.0 + 1.0 / RATIO + 1.0 / RATIO**2 + OUTSIDE / RATIO**3, 3.0 + OUTSIDE),
)


def _drawing(c):
    ly = W + 2.0 * c * DX
    strip = Box((0.0, ly / 2.0 - W / 2.0, H), (LX, ly / 2.0 + W / 2.0, H))
    slab = Box((0.0, 0.0, 0.0), (LX, ly, H))
    return (LX, ly, LZ), strip, slab


def _profiles(c, *, dx=DX, boundary_cell=None):
    domain, strip, slab = _drawing(c)
    return edge_aware_profiles(domain, dx, sheets=[strip], solids=[slab], axes="y",
                               boundary_cell=boundary_cell)


def _check_preflight(c, profiles):
    domain, strip, slab = _drawing(c)
    sim = Simulation(10e9, domain, dx=DX, boundary="pec", **profiles)
    sim.add_material("substrate", eps_r=2.2)
    sim.add(slab, material="substrate")
    sim.add(strip, material="pec")
    sim.add_source((LX / 2.0, domain[1] / 2.0, H + 3.0 * DX), component="ez")
    assert not sim.preflight().by_code("sheet_effective_size")


@pytest.mark.parametrize("c", np.round(np.arange(1.5, 3.001, 0.05), 2))
def test_the_strip_absorber_gaps_follow_the_ratio_rule(c):
    expected = any(low - 1e-9 <= c <= high + 1e-9 for low, high in INTERVALS)
    domain, strip, _ = _drawing(c)
    if expected:
        profiles = _profiles(c)
        cells = np.asarray(profiles["dy_profile"], dtype=np.float64)
        nodes = np.concatenate(([0.0], np.cumsum(cells)))
        assert np.all(cells > 0.0)
        assert np.all(cells >= DX / 10.0 * (1.0 - 1e-9))
        assert np.all(cells <= DX * (1.0 + 1e-9))
        assert np.all(cells[1:] <= RATIO * cells[:-1] * (1.0 + 1e-9))
        assert np.all(cells[:-1] <= RATIO * cells[1:] * (1.0 + 1e-9))
        assert cells[0] == cells[-1] == DX
        assert abs(nodes[-1] - domain[1]) <= 1e-9 * DX
        indices = [int(np.searchsorted(nodes, position, side="right")) - 1
                   for position in (strip.corner_lo[1], strip.corner_hi[1])]
        low, high = indices
        assert low != high
        assert abs(nodes[low + 1] - EDGE_OFFSET * cells[low] - strip.corner_lo[1]) <= 1e-9 * DX
        assert abs(nodes[high] + EDGE_OFFSET * cells[high] - strip.corner_hi[1]) <= 1e-9 * DX
        width = nodes[high] - nodes[low + 1] + EDGE_OFFSET * (cells[low] + cells[high])
        assert abs(width - W) <= 1e-9 * DX
        _check_preflight(c, profiles)
        return
    with pytest.raises(RfxRefusal) as caught:
        _profiles(c)
    refusal = caught.value
    assert refusal.axis == "y"
    assert refusal.diagnostic_kind == "local"
    assert len(refusal.blocked_runs) == 2
    coordinates = [(0.0, strip.corner_lo[1]), (strip.corner_hi[1], domain[1])]
    for run, (lo, hi) in zip(refusal.blocked_runs, coordinates):
        assert abs(run["lo"] - lo) <= 1e-9 * DX
        assert abs(run["hi"] - hi) <= 1e-9 * DX
        assert abs(run["gap"] - c * DX) <= 1e-9 * DX
        assert abs(run["nearest_below"] - LOWER * DX) <= 1e-6 * DX
        assert abs(run["nearest_above"] - UPPER * DX) <= 1e-6 * DX
        for name in ("nearest_below", "nearest_above"):
            _profiles(run[name] / DX)
    text = str(refusal)
    assert text.count("this gap closes for no cell sizes at its ends; it closes at") == 2
    assert "with everything to its left as drawn" not in text
    printed_low = [float(value) for value in re.findall(r"nearest_below=([\deE.+-]+) m", text)]
    printed_high = [float(value) for value in re.findall(r"nearest_above=([\deE.+-]+) m", text)]
    assert len(printed_low) == len(printed_high) == 2
    for exact, low, high in zip(refusal.blocked_runs, printed_low, printed_high):
        assert low <= exact["nearest_below"]
        assert high >= exact["nearest_above"]
        _profiles(low / DX)
        _profiles(high / DX)
    threshold = c * DX / UPPER
    assert refusal.route_dx is not None
    assert abs(refusal.route_dx / threshold - 1.0) <= 1e-5
    assert refusal.route_dx <= threshold * (1.0 + 1e-9)
    route = _profiles(c, dx=refusal.route_dx, boundary_cell=refusal.route_boundary_cell)
    _check_preflight(c, route)
    printed_route = re.search(r"edge_aware_profiles\(\.\.\., dx=([\deE.+-]+), boundary_cell=([\deE.+-]+)\)", text)
    assert printed_route is not None
    printed_dx, printed_boundary = map(float, printed_route.groups())
    assert printed_dx <= refusal.route_dx
    _check_preflight(c, _profiles(c, dx=printed_dx, boundary_cell=printed_boundary))


def test_locally_closable_runs_can_have_incompatible_shared_cells():
    """The left run forces s=1; the right forces s=1/rho; no fill fits either.

    The shortest one-fill left run is UPPER; the shortest one-fill right
    run is 1 + 1/rho + EDGE_OFFSET/rho**2. Both exceed the drawn gaps, so
    zero fills are forced. Each run alone obeys the ratio with its pinned
    end cell 1, but their shared edge cell cannot take both sizes.
    """
    left = 1.0 + OUTSIDE
    right = 1.0 + EDGE_OFFSET / RATIO
    assert left < UPPER
    assert right < 1.0 + 1.0 / RATIO + EDGE_OFFSET / RATIO**2
    assert (left - 1.0) / OUTSIDE == pytest.approx(1.0)
    assert (right - 1.0) / EDGE_OFFSET == pytest.approx(1.0 / RATIO)
    with pytest.raises(RfxRefusal) as caught:
        edge_aware_profile(0.0, left + right, 1.0, boundary_cell=1.0,
                           sheet_edges=[SheetEdge(left, +1)])
    refusal = caught.value
    assert refusal.diagnostic_kind == "forward"
    assert len(refusal.blocked_runs) == 1
    run = refusal.blocked_runs[0]
    assert run["lo"] == left
    assert run["hi"] == left + right
    assert run["nearest_below"] is None
    assert run["nearest_above"] == pytest.approx(1.0 + EDGE_OFFSET, abs=1e-9)
    assert "closes this run with everything to its left as drawn" in str(refusal)
    assert "this gap closes for no cell sizes at its ends" not in str(refusal)
