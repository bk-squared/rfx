"""Linear interval projections for edge-aware axes, in units of target dx."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from math import ceil, log

import numpy as np
from scipy.optimize import linprog

from rfx.errors import EdgeAwareProfileRefusal, RfxRefusal


_FLOOR = 0.1
_WINDOW_FLOOR = 0.1
_CEILING = 1.0
_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
_REQUEST = ContextVar("edge_profile_request", default=("unspecified", False))


@contextmanager
def _axis_call(axis):
    token = _REQUEST.set((axis, True))
    try:
        yield
    finally:
        _REQUEST.reset(token)


@dataclass(frozen=True)
class _Event:
    position: float
    left: float = 0.0
    right: float = 0.0
    pin: float | None = None
    edge: int | None = None
    drawn: float | None = None

    @property
    def face(self):
        return self.pin is None and self.edge is None


@dataclass
class _State:
    low: float
    high: float
    cost: int = 0
    parent: _State | None = None
    count: int = 0


def _solve(objective, inequalities, limits, equalities, targets, bounds, *, allow_unbounded=False):
    try:
        result = linprog(
            np.asarray(objective, dtype=np.float64),
            A_ub=np.asarray(inequalities, dtype=np.float64) if len(inequalities) else None,
            b_ub=np.asarray(limits, dtype=np.float64) if len(limits) else None,
            A_eq=np.asarray(equalities, dtype=np.float64) if len(equalities) else None,
            b_eq=np.asarray(targets, dtype=np.float64) if len(targets) else None,
            bounds=bounds, method="highs-ds", options=_OPTIONS,
        )
    except ValueError as exc:
        raise RuntimeError(f"edge-aware linear solve rejected its structure: {exc}") from exc
    if result.status == 2:
        return None
    if result.status == 3 and allow_unbounded:
        return result
    if not result.success:
        raise RuntimeError(f"edge-aware linear solve failed: {result.message}")
    return result


def _bounds(event, floor):
    return (event.pin, event.pin) if event.pin is not None else (floor, _CEILING)


def _events(lo, hi, dx, faces, edges, boundary_cell, edge_offset):
    pin = None if boundary_cell is None else boundary_cell / dx
    events = [_Event(0.0, right=float(pin is not None), pin=pin, drawn=lo)]
    events.extend(_Event((face - lo) / dx, drawn=float(face))
                  for face in sorted(set(faces)) if lo < face < hi)
    for index, edge in enumerate(edges):
        left = edge_offset if edge.metal_side == -1 else 1.0 - edge_offset
        events.append(_Event((edge.position - lo) / dx, left, 1.0 - left,
                             edge=index, drawn=float(edge.position)))
    events.append(_Event((hi - lo) / dx, left=float(pin is not None), pin=pin, drawn=hi))
    def order(event):
        side = -1 if event.edge is not None and event.left == 1.0 else 0
        if event.edge is not None and event.left == 0.0:
            side = 1
        return event.position, side

    return sorted(events, key=order)


def _count_window(left, right, state, floor, ratio):
    """Shortest/longest residual fills, plus a full floor-to-cap ramp per end."""
    gap = right.position - left.position
    right_low, right_high = _bounds(right, floor)
    longest = gap - left.right * state.low - right.left * right_low
    if not np.isfinite(_CEILING):
        return range(max(0, int(np.floor(longest / max(floor, _WINDOW_FLOOR) + 1e-9))) + 1)
    shortest = gap - left.right * state.high - right.left * right_high
    ramp = ceil(log(_CEILING / _WINDOW_FLOOR) / log(ratio))

    def fewest(length):
        count = max(0, ceil(length / _CEILING - 1e-12))
        while True:
            largest = sum(min(_CEILING, state.high * ratio**min(i + 1, ramp),
                              right_high * ratio**min(count - i, ramp))
                          for i in range(count))
            if largest >= length - 1e-12:
                return count
            count += 1

    first = fewest(shortest)
    last = fewest(longest) + 2 * ramp + 1
    if floor > 0.0:
        last = min(last, max(0, int(np.floor(longest / floor + 1e-9))))
    return range(first, last + 1)


def _run_problem(left, right, state, count, floor, ratio, *, external=False, length=True):
    size = count + 2
    bounds = [(state.low, state.high), *[(floor, _CEILING)] * count, _bounds(right, floor)]
    inequalities, limits = [], []
    for index in range(size - 1):
        for a, b in ((index, index + 1), (index + 1, index)):
            row = np.zeros(size)
            row[a], row[b] = 1.0, -ratio
            inequalities.append(row)
            limits.append(0.0)
    equalities, targets = [], []
    span = np.ones(size)
    span[0], span[-1] = left.right, right.left
    if length:
        equalities.append(span)
        targets.append(right.position - left.position)
    if right.face and not external:
        row = np.zeros(size)
        row[-1], row[-2] = 1.0, -1.0
        equalities.append(row)
        targets.append(0.0)
    return inequalities, limits, equalities, targets, bounds, span


def _project(left, right, state, count, floor, ratio):
    inequalities, limits, equalities, targets, bounds, _ = _run_problem(
        left, right, state, count, floor, ratio,
    )
    objective = np.zeros(count + 2)
    objective[-1] = 1.0
    low = _solve(objective, inequalities, limits, equalities, targets, bounds)
    if low is None:
        return None
    high = _solve(-objective, inequalities, limits, equalities, targets, bounds)
    if high is None:
        raise RuntimeError("edge-aware projection lost a feasible interval")
    lower, upper = float(low.x[-1]), float(high.x[-1])
    if lower > upper + 1e-9:
        raise RuntimeError("edge-aware interval has reversed endpoints")
    return _State(min(lower, upper), max(lower, upper),
                  state.cost + count + int(not right.face), state, count)


def _union(states, costs):
    if costs:
        kept: list[_State] = []
        for state in sorted(states, key=lambda s: (s.cost, s.low, -s.high)):
            if not any(old.cost <= state.cost and old.low <= state.low and old.high >= state.high
                       for old in kept):
                kept.append(state)
        return kept
    merged: list[_State] = []
    for state in sorted(states, key=lambda s: s.low):
        if merged and state.low <= merged[-1].high:
            merged[-1].high = max(merged[-1].high, state.high)
        else:
            merged.append(_State(state.low, state.high))
    return merged


def _forward(events, floor, ratio, *, costs=False):
    low, high = _bounds(events[0], floor)
    if low < floor or high < low:
        return [[] for _ in events]
    stages = [[_State(low, high, int(not events[0].face))]]
    for left, right in zip(events, events[1:]):
        projected = []
        for state in stages[-1]:
            for count in _count_window(left, right, state, floor, ratio):
                interval = _project(left, right, state, count, floor, ratio)
                if interval is not None:
                    projected.append(interval)
        stages.append(_union(projected, costs))
    return stages


def _backward(events, floor, ratio):
    mirrored = [_Event(-event.position, event.right, event.left, event.pin, event.edge, event.drawn)
                for event in reversed(events)]
    return list(reversed(_forward(mirrored, floor, ratio)))


def _exact_vertex(result, inequalities, limits, equalities, targets, bounds):
    """Re-solve every equality and active ratio/cell bound as one float64 system."""
    rows, rhs = list(equalities), list(targets)
    for row, limit in zip(inequalities, limits):
        scale = max(abs(limit), float(np.max(np.abs(np.asarray(row) * result.x))), np.finfo(float).tiny)
        if abs(float(np.dot(row, result.x)) - limit) <= 1e-9 * scale:
            rows.append(row)
            rhs.append(limit)
    for index, (low, high) in enumerate(bounds):
        for bound in (low, high):
            if bound is not None and np.isfinite(bound) and abs(result.x[index] - bound) <= 1e-9 * max(abs(bound), np.finfo(float).tiny):
                row = np.zeros(len(bounds))
                row[index] = 1.0
                rows.append(row)
                rhs.append(bound)
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    try:
        exact, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"edge-aware active vertex solve failed: {exc}") from exc
    if rank != len(bounds) or np.max(np.abs(matrix @ exact - target)) > 1e-9:
        raise RuntimeError("edge-aware active vertex system is inconsistent or singular")
    return exact


def _whole_axis(events, counts, floor, ratio):
    bounds, spans, edge_indices = [], [], {}
    if not events[0].face:
        bounds.append(_bounds(events[0], floor))
    for left, right, count in zip(events, events[1:], counts):
        terms = {}
        if left.right:
            terms[len(bounds) - 1] = left.right
        for _ in range(count):
            terms[len(bounds)] = 1.0
            bounds.append((floor, _CEILING))
        if not right.face:
            terms[len(bounds)] = right.left
            if right.edge is not None:
                edge_indices[right.edge] = len(bounds)
            bounds.append(_bounds(right, floor))
        spans.append((terms, right.position - left.position))
    cell_count = len(bounds)
    bounds.append((floor, _CEILING))
    equalities, targets = [], []
    for terms, gap in spans:
        row = np.zeros(cell_count + 1)
        for index, coefficient in terms.items():
            row[index] = coefficient
        equalities.append(row)
        targets.append(gap)
    inequalities, limits = [], []
    for index in range(cell_count - 1):
        for a, b in ((index, index + 1), (index + 1, index)):
            row = np.zeros(cell_count + 1)
            row[a], row[b] = 1.0, -ratio
            inequalities.append(row)
            limits.append(0.0)
    for index in range(cell_count):
        row = np.zeros(cell_count + 1)
        row[-1], row[index] = 1.0, -1.0
        inequalities.append(row)
        limits.append(0.0)
    objective = np.zeros(cell_count + 1)
    objective[-1] = -1.0
    result = _solve(objective, inequalities, limits, equalities, targets, bounds)
    if result is None:
        raise RuntimeError("edge-aware interval path has no whole-axis vertex")
    exact = _exact_vertex(result, inequalities, limits, equalities, targets, bounds)
    return exact[:-1], edge_indices


def _search_cells(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset, *, feasible_only=False):
    """Return normalized cells and edge indices, or None for an empty feasible set."""
    events = _events(lo, hi, dx, faces, edges, boundary_cell, edge_offset)
    forward = _forward(events, _FLOOR, ratio)
    backward = _backward(events, _FLOOR, ratio)
    if bool(forward[-1]) != bool(backward[0]):
        raise RuntimeError("forward and mirrored backward feasibility disagree")
    if not forward[-1]:
        return None
    if feasible_only:
        return True
    low, high = _FLOOR, min(_CEILING, (hi - lo) / dx)
    while high - low > 1e-6:
        middle = (low + high) / 2.0
        if _forward(events, middle, ratio)[-1]:
            low = middle
        else:
            high = middle
    final = _forward(events, low, ratio, costs=True)[-1]
    state = min(final, key=lambda item: item.cost)
    counts = []
    while state.parent is not None:
        counts.append(state.count)
        state = state.parent
    cells, edge_indices = _whole_axis(events, list(reversed(counts)), low, ratio)
    return cells, edge_indices


def _search_profile(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset, *, axis="unspecified"):
    from rfx.mesh_edges import EdgeAwareProfile

    drawing = f"[{lo!r}, {hi!r}], faces={faces!r}, sheet_edges={edges!r}, dx={dx!r}"
    try:
        found = _search_cells(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset)
        if found is None:
            return None
        normalized, indices = found
        cells = np.asarray(normalized * dx, dtype=np.float64)
        if boundary_cell is not None:
            cells[0] = cells[-1] = boundary_cell
        nodes = lo + np.concatenate(([0.0], np.cumsum(cells)))
        tol = 1e-9 * dx
        allowed = np.full(cells.size, _CEILING * dx)
        if boundary_cell is not None:
            allowed[0] = allowed[-1] = max(dx, boundary_cell)
        valid = (
            np.all(np.isfinite(cells)) and np.all(cells >= _FLOOR * dx * (1.0 - 1e-9))
            and np.all(cells <= allowed + tol)
            and np.all(cells[1:] <= ratio * cells[:-1] * (1.0 + 1e-9))
            and np.all(cells[:-1] <= ratio * cells[1:] * (1.0 + 1e-9))
            and abs(nodes[-1] - hi) <= tol
            and all(np.min(np.abs(nodes - face)) <= tol for face in faces)
        )
        ends, edge_cells = [], []
        for index, edge in enumerate(edges):
            cell_index = indices[index]
            inside = nodes[cell_index + int(edge.metal_side == 1)]
            size = cells[cell_index]
            valid = valid and abs(inside - edge.metal_side * edge_offset * size - edge.position) <= tol
            ends.append(float(inside))
            edge_cells.append(float(size))
        if not valid or len(set(indices.values())) != len(edges):
            raise RuntimeError("the feasible vertex failed the profile checks")
        return EdgeAwareProfile(cells, nodes, tuple(ends), tuple(edge_cells))
    except RuntimeError as exc:
        raise RuntimeError(f"edge-aware search internal error on axis {axis}: {drawing}: {exc}") from exc


def _find_route(drawings, dx, ratio, boundary_cell, edge_offset, *, pins_x=False):
    """Scan the whole requested drawing; keep the building end of 12 bisections."""
    from rfx.mesh_edges import _today_profile, edge_aware_profile

    def builds(candidate):
        pin = dx if pins_x else None if boundary_cell is None else float(boundary_cell)
        if pin is None:
            pin = candidate
        for axis, lo, hi, faces, edges in drawings:
            ordered = sorted(edges, key=lambda edge: edge.position)
            try:
                _today_profile(lo, hi, candidate, sorted({lo, hi, *faces}), ordered,
                               ratio, pin, edge_offset)
            except ValueError:
                try:
                    feasible = _search_cells(lo, hi, candidate, faces, ordered, ratio, pin,
                                             edge_offset, feasible_only=True)
                except RuntimeError as exc:
                    raise RuntimeError(f"route search failed on axis {axis}, drawing="
                                       f"{(lo, hi, faces, edges)!r}: {exc}") from exc
                if not feasible:
                    return False
        return True

    refused = dx
    for step in range(1, 51):
        candidate = dx * (1.0 - 0.01 * step)
        if builds(candidate):
            building = candidate
            for _ in range(12):
                middle = (building + refused) / 2.0
                if builds(middle):
                    building = middle
                else:
                    refused = middle
            pin = dx if pins_x else None if boundary_cell is None else float(boundary_cell)
            if pin is None:
                pin = building
            for axis, lo, hi, faces, edges in drawings:
                try:
                    with _axis_call(axis):
                        edge_aware_profile(lo, hi, building, faces=faces, sheet_edges=edges,
                                           max_ratio=ratio, boundary_cell=pin,
                                           edge_offset=edge_offset)
                except RfxRefusal:
                    raise RuntimeError(f"route feasibility lost its profile on axis {axis}")
            return building, pin
        refused = candidate
    return None, None


def _gap_intervals(left, right, states, ratio):
    """Project run length with its neighbouring sizes free inside their intervals."""
    if (left.pin is not None and left.pin < _FLOOR) or (right.pin is not None and right.pin < _FLOOR):
        return []
    largest = max(0.0, right.position - left.position)
    cap = _CEILING if np.isfinite(_CEILING) else max(1.0, largest)
    ramp = ceil(log(cap / _WINDOW_FLOOR) / log(ratio))
    last = ceil(largest / cap) + 2 * ramp + 2
    intervals = []
    for state in states:
        for count in range(last + 1):
            inequalities, limits, equalities, targets, bounds, span = _run_problem(
                left, right, state, count, _FLOOR, ratio, length=False,
            )
            low = _solve(span, inequalities, limits, equalities, targets, bounds)
            if low is None:
                continue
            high = _solve(-span, inequalities, limits, equalities, targets, bounds, allow_unbounded=True)
            if high is None:
                raise RuntimeError("run length projection lost a feasible interval")
            minimum = _exact_vertex(low, inequalities, limits, equalities, targets, bounds)
            maximum = float("inf") if high.status == 3 else float(span @ _exact_vertex(
                high, inequalities, limits, equalities, targets, bounds,
            ))
            intervals.append(_State(float(span @ minimum), maximum))
    return _union(intervals, False)


def _diagnose(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset):
    events = _events(lo, hi, dx, faces, edges, boundary_cell, edge_offset)
    forward = _forward(events, _FLOOR, ratio)
    _backward(events, _FLOOR, ratio)
    blocked = []
    for index, (left, right) in enumerate(zip(events, events[1:])):
        intervals = _gap_intervals(left, right, [_State(*_bounds(left, _FLOOR))], ratio)
        gap = right.position - left.position
        if not any(interval.low - 1e-10 <= gap <= interval.high + 1e-10 for interval in intervals):
            blocked.append((index, intervals))
    kind = "local"
    if not blocked:
        kind = "forward"
        index = next(i for i in range(len(events) - 1) if forward[i] and not forward[i + 1])
        blocked = [(index, _gap_intervals(events[index], events[index + 1], forward[index], ratio))]
    runs = []
    for index, intervals in blocked:
        left, right = events[index:index + 2]
        gap = right.position - left.position
        below = [interval.high for interval in intervals if interval.high < gap]
        above = [interval.low for interval in intervals if interval.low > gap]
        drawn_lo, drawn_hi = left.drawn, right.drawn
        runs.append({"lo": float(drawn_lo), "hi": float(drawn_hi),
                     "gap": float(drawn_hi - drawn_lo),
                     "nearest_below": float(max(below) * dx) if below else None,
                     "nearest_above": float(min(above) * dx) if above else None})
    return kind, runs


def _inward(value, *, upward=False):
    if value is None:
        return "none"
    with localcontext() as context:
        context.prec = 40
        number = Decimal.from_float(float(value))
        quantum = Decimal(1).scaleb(number.adjusted() - 5)
        rounded = number.quantize(quantum, rounding=ROUND_CEILING if upward else ROUND_FLOOR)
        return str(rounded)


def _refusal_text(refusal):
    dx, ratio, boundary_cell, edge_offset = refusal._rule
    lines = [f"edge-aware axis {refusal.axis} cannot close under the cell rule."]
    for run in refusal.blocked_runs:
        below = _inward(run["nearest_below"])
        above = _inward(run["nearest_above"], upward=True)
        where = f"[{run['lo']!r}, {run['hi']!r}] m, gap={run['gap']!r} m ({run['gap'] / dx:.9g} dx)"
        limits = f"nearest_below={below} m and nearest_above={above} m"
        if refusal.diagnostic_kind == "local":
            lines.append(f"{where}: this gap closes for no cell sizes at its ends; it closes at {limits}.")
        else:
            lines.append(f"{where}: {limits} closes this run with everything to its left as drawn.")
    ends = "free" if boundary_cell is None else repr(boundary_cell)
    lines.append(f"Rule: max_ratio={ratio!r}, edge_offset={edge_offset!r}, end cells={ends}, "
                 f"dx ceiling={dx!r} m (pinned ends exempt), dx/10 floor={dx / 10!r} m.")
    if refusal.route_dx is None:
        profile_name = "d<axis>_profile" if refusal.axis == "unspecified" else f"d{refusal.axis}_profile"
        lines.append(f"No dx' builds down to dx/2; supply a hand {profile_name} "
                     "obeying this rule and the drawn faces and sheet-edge offsets.")
    else:
        route = _inward(refusal.route_dx)
        boundary = route if refusal._route_follows_dx else repr(refusal.route_boundary_cell)
        lines.append(f"Use edge_aware_profiles(..., dx={route}, boundary_cell={boundary}), "
                     "the largest dx' found below dx on a 1 % grid (12 bisection steps); "
                     f"keep Simulation(dx={dx!r}) unchanged. This route builds the whole requested call.")
    return "\n".join(lines)


def _complete_refusal(refusal, drawings, dx, ratio, boundary_cell, edge_offset, *, pins_x=False):
    try:
        refusal.route_dx, refusal.route_boundary_cell = _find_route(
            drawings, dx, ratio, boundary_cell, edge_offset, pins_x=pins_x,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"edge-aware route internal error on axis {refusal.axis}, drawing={drawings!r}: {exc}") from exc
    refusal._route_follows_dx = not pins_x and boundary_cell is None
    if refusal.route_dx is not None:
        from rfx.mesh_edges import edge_aware_profile

        printed_dx = float(_inward(refusal.route_dx))
        printed_pin = printed_dx if refusal._route_follows_dx else refusal.route_boundary_cell
        for axis, lo, hi, faces, edges in drawings:
            try:
                with _axis_call(axis):
                    edge_aware_profile(lo, hi, printed_dx, faces=faces, sheet_edges=edges,
                                       max_ratio=ratio, boundary_cell=printed_pin,
                                       edge_offset=edge_offset)
            except RfxRefusal as exc:
                raise RuntimeError(f"rounded route failed on axis {axis}, drawing={drawings!r}") from exc
    refusal.args = (_refusal_text(refusal),)
    return refusal


def _fallback(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset):
    axis, defer_route = _REQUEST.get()
    profile = _search_profile(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset, axis=axis)
    if profile is not None:
        return profile
    try:
        kind, runs = _diagnose(lo, hi, dx, faces, edges, ratio, boundary_cell, edge_offset)
    except RuntimeError as exc:
        raise RuntimeError(f"edge-aware diagnostic internal error on axis {axis}, drawing="
                           f"{(lo, hi, faces, edges)!r}: {exc}") from exc
    refusal = EdgeAwareProfileRefusal()
    refusal.axis, refusal.diagnostic_kind, refusal.blocked_runs = axis, kind, runs
    refusal.route_dx = refusal.route_boundary_cell = None
    refusal._rule = (dx, ratio, boundary_cell, edge_offset)
    refusal._route_follows_dx = False
    if not defer_route:
        _complete_refusal(refusal, [(axis, lo, hi, faces, edges)], dx, ratio,
                          boundary_cell, edge_offset)
    raise refusal
