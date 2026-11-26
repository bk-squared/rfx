"""The strip/absorber gap set follows closed-form ratio bounds, not helper output."""

import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxRefusal
from rfx.mesh_edges import EDGE_OFFSET, SheetEdge, edge_aware_profile, edge_aware_profiles


H, W, LX = 254e-6, 600e-6, 12e-3
DX = H / 3.0
LZ = H + 18.0 * DX
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


def _profiles(c, *, dx=DX, boundary_cell=None, axes="y", max_ratio=RATIO):
    domain, strip, slab = _drawing(c)
    return edge_aware_profiles(domain, dx, sheets=[strip], solids=[slab], axes=axes,
                               boundary_cell=boundary_cell, max_ratio=max_ratio)


def _preflight(c, profiles):
    domain, strip, slab = _drawing(c)
    sim = Simulation(10e9, domain, dx=DX, boundary="pec", **profiles)
    sim.add_material("substrate", eps_r=2.2)
    sim.add(slab, material="substrate")
    sim.add(strip, material="pec")
    sim.add_source((LX / 2.0, domain[1] / 2.0, H + 3.0 * DX), component="ez")
    report = sim.preflight()
    assert not report.by_code("assembly_refusal"), str(report.by_code("assembly_refusal"))
    assert not report.by_code("assembly_fault"), str(report.by_code("assembly_fault"))
    return report


def _check_preflight(c, profiles):
    findings = _preflight(c, profiles).by_code("sheet_effective_size")
    cells = {name: np.asarray(values).tolist() for name, values in profiles.items()}
    assert not findings, f"c={c!r}, findings={findings!r}, profiles={cells!r}"


def test_uniform_y_cells_expose_the_conductor_sheet_size_finding():
    domain, _, _ = _drawing(2.0)
    cells = np.full(int(np.ceil(domain[1] / DX)), DX)
    findings = _preflight(2.0, {"dy_profile": cells}).by_code("sheet_effective_size")
    assert findings


def _direction_changes(c, cells, dx=DX):
    """Count turns between drawn events, including the cells on both sides."""
    domain, strip, _ = _drawing(c)
    nodes = np.concatenate(([0.0], np.cumsum(cells)))
    positions = (0.0, strip.corner_lo[1], strip.corner_hi[1], domain[1])
    indices = [max(0, min(len(cells) - 1, int(np.searchsorted(nodes, p, side="right")) - 1))
               for p in positions]
    turns = []
    for start, stop in zip(indices, indices[1:]):
        difference = np.diff(cells[start:stop + 1])
        scale = np.maximum(np.abs(cells[start:stop]), np.abs(cells[start + 1:stop + 1]))
        signs = np.sign(difference[np.abs(difference) > 1e-9 * scale])
        turns.append(int(np.count_nonzero(signs[1:] != signs[:-1])))
    return turns


@pytest.mark.parametrize("c,axes", [
    (c, axes) for c in np.round(np.arange(1.5, 3.001, 0.05), 2)
    for axes in (("y", "xy") if LOWER < c < UPPER else ("y",))
])
def test_the_strip_absorber_gaps_follow_the_ratio_rule(c, axes):
    expected = any(low - 1e-9 <= c <= high + 1e-9 for low, high in INTERVALS)
    domain, strip, _ = _drawing(c)
    if expected:
        profiles = _profiles(c, axes=axes)
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
        if c not in (2.30, 2.40, 2.45, 2.50, 2.85, 2.90, 2.95, 3.00):
            assert max(_direction_changes(c, cells)) <= 1
        _check_preflight(c, profiles)
        return
    with pytest.raises(RfxRefusal) as caught:
        _profiles(c, axes=axes)
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
    assert max(_direction_changes(c, np.asarray(route["dy_profile"]), refusal.route_dx)) <= 1
    if axes == "xy":
        route.update(edge_aware_profiles(domain, DX, sheets=[strip], solids=[_drawing(c)[2]],
                                         axes="x", boundary_cell=DX))
        assert f"axes='x', dx={DX!r}, boundary_cell={DX!r}" in text
    _check_preflight(c, route)
    printed_route = re.search(r"edge_aware_profiles\(\.\.\., axes='y', dx=([\deE.+-]+), boundary_cell=([\deE.+-]+)\)", text)
    assert printed_route is not None
    printed_dx, printed_boundary = map(float, printed_route.groups())
    assert printed_dx <= refusal.route_dx
    printed_profiles = _profiles(c, dx=printed_dx, boundary_cell=printed_boundary)
    assert max(_direction_changes(c, np.asarray(printed_profiles["dy_profile"]), printed_dx)) <= 1
    if axes == "xy":
        printed_profiles["dx_profile"] = route["dx_profile"]
    _check_preflight(c, printed_profiles)


def test_an_x_oriented_strip_has_no_route_with_pinned_x_ends():
    domain, strip, slab = _drawing(2.0)
    swap = lambda point: (point[1], point[0], point[2])
    with pytest.raises(RfxRefusal) as caught:
        edge_aware_profiles(swap(domain), DX,
                            sheets=[Box(swap(strip.corner_lo), swap(strip.corner_hi))],
                            solids=[Box(swap(slab.corner_lo), swap(slab.corner_hi))], axes="xy")
    refusal = caught.value
    assert refusal.axis == "x"
    assert refusal.route_dx is None
    assert len(refusal.blocked_runs) == 2
    for run in refusal.blocked_runs:
        assert run["nearest_below"] == pytest.approx(LOWER * DX, abs=1e-6 * DX)
        assert run["nearest_above"] == pytest.approx(UPPER * DX, abs=1e-6 * DX)
    assert "No dx' builds group axes='x' down to dx/2" in str(refusal)
    assert "move the drawing to one of the named nearest gaps" in str(refusal)
    assert "hand" not in str(refusal)


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


@pytest.mark.parametrize("face", [-0.5, 2.5])
def test_an_outside_face_is_an_input_error_before_search(monkeypatch, face):
    from rfx import _edge_profile_search as search

    def unexpected(*args, **kwargs):
        pytest.fail("the search must not run for an outside face")

    monkeypatch.setattr(search, "_fallback", unexpected)
    with pytest.raises(ValueError, match=rf"face at {face:g} m is outside"):
        edge_aware_profile(0.0, 2.0, 1.0, faces=[face])


def test_a_face_one_ulp_beyond_the_axis_keeps_todays_cells():
    plain = edge_aware_profile(0.0, 10.0, 1.0, boundary_cell=1.0)
    tolerance_face = edge_aware_profile(0.0, 10.0, 1.0, boundary_cell=1.0,
                                        faces=[np.nextafter(10.0, np.inf)])
    assert len(tolerance_face.cells) == 10
    assert np.array_equal(tolerance_face.cells, plain.cells), (
        f"without face={plain.cells.tolist()!r}, with face={tolerance_face.cells.tolist()!r}")


def test_the_refusal_names_blocked_runs_under_each_refused_axis():
    side = W + 4.0 * DX
    strip = Box((2.0 * DX, 2.0 * DX, H), (side - 2.0 * DX, side - 2.0 * DX, H))
    slab = Box((0.0, 0.0, 0.0), (side, side, H))
    with pytest.raises(RfxRefusal) as caught:
        edge_aware_profiles((side, side, LZ), DX, sheets=[strip], solids=[slab], axes="xy")
    refusal = caught.value
    assert refusal.axis == "x"
    assert len(refusal.blocked_runs) == 2
    sections = str(refusal).split("Axis x:\n", 1)[1].split("Axis y:\n", 1)
    for section in (sections[0], sections[1].split("Rule:", 1)[0]):
        assert section.count("gap=") == 2
        assert section.count("nearest_below=") == 2
        assert section.count("nearest_above=") == 2


@pytest.mark.parametrize("boundary_cell", [0.01, 2.0])
def test_no_named_gap_explains_the_pinned_end_and_offers_no_gap_move(boundary_cell):
    with pytest.raises(RfxRefusal) as caught:
        edge_aware_profile(0.0, 10.0, 1.0, boundary_cell=boundary_cell,
                           sheet_edges=[SheetEdge(0.001, +1)])
    refusal = caught.value
    assert refusal.blocked_runs
    assert all(run["nearest_below"] is None and run["nearest_above"] is None
               for run in refusal.blocked_runs)
    text = str(refusal)
    assert text.count("no gap closes this run under the rule") == len(refusal.blocked_runs)
    assert f"The pinned end cell {boundary_cell!r} m" in text
    assert ("below the dx/10 floor" if boundary_cell < 0.1 else "above the dx ceiling") in text
    assert "move the drawing" not in text


@pytest.mark.parametrize("max_ratio", [RATIO, 1.01])
def test_the_cell_ceiling_is_preserved_before_any_search_lp(monkeypatch, max_ratio):
    from rfx import _edge_profile_search as search
    from rfx.model import graded_profile

    monkeypatch.setattr(graded_profile, "_MAX_PROFILE_CELLS", 500)

    def unexpected(*args, **kwargs):
        pytest.fail("no search LP may run above the cell ceiling")

    monkeypatch.setattr(search, "_solve", unexpected)
    message = ("band profile would need 600 cells, more than the 500 cell ceiling "
               "(a sliver segment or a ratio close to 1 is driving the count)")
    with pytest.raises(ValueError) as caught:
        edge_aware_profile(0.0, 600.0, 1.0, max_ratio=max_ratio)
    assert str(caught.value) == message


def test_a_single_axis_route_prints_its_own_call_and_a_building_float(monkeypatch):
    from rfx import _edge_profile_search as search
    from rfx.errors import EdgeAwareProfileRefusal

    route = 0.92857123456789
    checked = []

    def builds(drawings, candidate, ratio, pin, edge_offset, *, exact=False):
        checked.append((candidate, pin, exact))
        return candidate == route

    monkeypatch.setattr(search, "_find_route", lambda *args, **kwargs: (route, route))
    monkeypatch.setattr(search, "_route_builds", builds)
    refusal = EdgeAwareProfileRefusal()
    refusal.axis, refusal.diagnostic_kind, refusal.blocked_runs = "y", "local", []
    refusal.route_dx = refusal.route_boundary_cell = None
    refusal._rule = (1.0, RATIO, 1.0, EDGE_OFFSET)
    search._complete_refusal(refusal, [("y", 0.0, 10.0, [], [])], 1.0, RATIO, None,
                             EDGE_OFFSET, single=True)
    assert checked == [(0.928571, 0.928571, True)]
    assert f"dx={route!r}, boundary_cell={route!r}" in str(refusal)
    assert "Use edge_aware_profile(..., dx=" in str(refusal)
    assert "Use edge_aware_profiles(" not in str(refusal)


def test_an_empty_final_set_names_the_drawing(monkeypatch):
    from rfx import _edge_profile_search as search

    def forward(events, floor, ratio, *, costs=False):
        return [[] if costs else [search._State(0.1, 1.0)] for event in events]

    monkeypatch.setattr(search, "_forward", forward)
    with pytest.raises(RuntimeError, match="final interval set is empty: drawing="):
        search._search_profile(0.0, 2.0, 1.0, [], [], RATIO, 1.0, EDGE_OFFSET)


def test_below_the_search_cutoff_keeps_the_original_error_and_explains_it(monkeypatch):
    from rfx import _edge_profile_search as search
    from rfx.mesh_edges import _today_profile

    domain, strip, _ = _drawing(2.0)
    edges = [SheetEdge(strip.corner_lo[1], +1), SheetEdge(strip.corner_hi[1], -1)]
    with pytest.raises(ValueError) as original:
        _today_profile(0.0, domain[1], DX, [0.0, domain[1]], edges, 1.01, DX, EDGE_OFFSET)

    def unexpected(*args, **kwargs):
        pytest.fail("the exact search must not run below max_ratio 1.02")

    monkeypatch.setattr(search, "_fallback", unexpected)
    with pytest.raises(ValueError) as caught:
        _profiles(2.0, max_ratio=1.01)
    assert type(caught.value) is ValueError
    sentence = ("the exact search runs for max_ratio >= 1.02; "
                "raise max_ratio to let it look for a profile.")
    assert str(caught.value) == str(original.value) + "\n" + sentence
