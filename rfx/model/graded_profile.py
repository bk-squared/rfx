"""The graded cell-profile planner: from declared bands to a cell profile.

A declaration-time tool. Moved verbatim from ``rfx/nonuniform.py``, which
re-exports every name here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Band profile builder — axis-agnostic, interface-exact, ratio-law-exact.
# Design note: docs/design_notes/20260907_nu_band_profile_predeclaration.md
# ---------------------------------------------------------------------------

class _BandSeg(NamedTuple):
    """One declared segment of a band profile (private engine input).

    ``length``/``target`` in metres. A protected segment is realized as
    uniform cells (``n = max(min_cells, ceil(length/target - 1e-9))``) and
    never hosts a ramp; a free segment may host ramps on either side.
    ``split_lo``/``split_hi``/``verbatim`` exist for the auto-z path only:
    ``_make_dz_profile`` applies the thirds rule BEFORE smoothing, so a
    protected block there carries 1/3 + 2/3 edge sub-cells. ``verbatim`` is
    the post-thirds block passed through bit-identically while the block is
    unrefined; a block the seam rule refines is re-derived with the same
    arithmetic ``apply_thirds_rule`` uses. ``emax_lo``/``emax_hi`` (free
    only) bound the first/last cell of the run (``make_z_profile``'s
    "fine cell at every feature plane").
    """
    length: float
    target: float
    protected: bool
    min_cells: int = 1
    split_lo: bool = False
    split_hi: bool = False
    verbatim: tuple | None = None
    emax_lo: float | None = None
    emax_hi: float | None = None


class _Anchor(NamedTuple):
    """What a free run ramps from on one side.

    ``kind``: ``"nb"`` — the neighbouring segment's seam cell (the run ramps
    UP from it when the run is coarser; when the run is finer the neighbour
    is responsible — refine if protected, ramp if free); ``"pin"`` — a fixed
    boundary cell (the run ramps up OR down from it, it can never change);
    ``"emax"`` — a bound on the run's own first cell, which becomes the
    ramp's first cell.
    """
    value: float
    kind: str


# Engine constants — see make_band_profile's docstring for the numbers.
_RATIO_FLOOR = 1.0001        # a 2x transition already needs 6932 cells here
_MAX_PROFILE_CELLS = 1_000_000
_STEP_TOL = 1e-10            # relative slack on the ramp step count (F5 fix)
_EDGE_TOL = 1e-12            # the I1 node tolerance, metres


def _ramp_steps(ratio: float, cap: float) -> int:
    """Minimal step count ``m`` with ``cap**m >= ratio`` (``ratio > 1``).

    The ceiling carries a 1e-10 relative slack on the logarithm: a seam
    cell on the ratio law exactly (``ratio == cap**k``) arrives with float
    noise of order ``ulp(column) / cell`` — 5.5e-13 relative on a 39 m
    column of mm cells — and a 1e-12 slack flipped ``k`` to ``k + 1``
    there, which broke the exact-fit branch (review finding, 2026-09-07).
    The per-step ratio may then exceed ``cap`` by at most
    ``cap x ln(cap) x 1e-10 / m`` — 4.7e-11 absolute at 1.4, far inside
    the I2 tolerance of 1e-9.
    """
    return max(1, int(np.ceil(np.log(ratio) / np.log(cap) - _STEP_TOL)))


def _ramp_spec(anchor: _Anchor | None, u: float, cap: float) -> tuple[str, int]:
    """Classify the ramp one side needs to reach the plateau ``u``.

    Returns ``(mode, m)``: ``"none"`` (no ramp on this side), ``"up"``
    (``m`` cells from the neighbour's seam cell up to ``u``, the seam cell
    excluded), ``"down"`` (from a pinned cell down to ``u``, pin
    excluded), ``"emax"`` (the run's own first cell bounded by ``anchor``,
    ``m + 1`` cells including that first cell). ``m`` is the minimal
    step count for the anchor/``u`` pair.
    """
    if anchor is None:
        return "none", 0
    v, kind = float(anchor.value), anchor.kind
    tol = 1e-12
    if kind == "emax":
        if u <= v * (1.0 + tol):
            return "none", 0            # thin run: its cells are <= emax
        return "emax", _ramp_steps(u / v, cap)
    if u > v * (1.0 + tol):
        return "up", _ramp_steps(u / v, cap)
    if kind == "pin" and v > u * (1.0 + tol):
        return "down", _ramp_steps(v / u, cap)
    return "none", 0


def _ramp_sum(anchor: _Anchor | None, u: float, mode: str, m: int) -> float:
    """Closed-form sum of the ramp ``(mode, m)`` from ``anchor`` to ``u``.

    Geometric series; O(1) whatever ``m`` is (the old cell-by-cell sum made
    the plateau solve quadratic in the ramp length — 8.5 s at
    ``max_ratio = 1.0001`` on a 40-cell profile). ``m == 1`` is returned
    directly (the closed form divides by ``rho - 1``, which vanishes as
    the one-step ratio approaches 1).
    """
    if mode == "none":
        return 0.0
    v = float(anchor.value)
    if mode == "up":
        if m == 1:
            return u
        rho = (u / v) ** (1.0 / m)
        return rho * (u - v) / (rho - 1.0)
    if mode == "down":
        if m == 1:
            return u
        rho = (v / u) ** (1.0 / m)
        return (v - u) / (rho - 1.0)
    # emax: v, v rho, ..., u  (m + 1 cells)
    if m == 1:
        return v + u
    rho = (u / v) ** (1.0 / m)
    return (u * rho - v) / (rho - 1.0)


def _ramp_cells(anchor: _Anchor | None, u: float, mode: str, m: int) -> list[float]:
    """The cells of the ramp ``(mode, m)``, edge-adjacent cell first; the
    last cell is ``u`` itself (the ramp ends ON the first plateau cell, so
    the ramp/plateau junction ratio is the ramp's own ratio whatever the
    plateau count)."""
    if mode == "none":
        return []
    v = float(anchor.value)
    if mode == "up":
        rho = (u / v) ** (1.0 / m)
        cells = [v * rho ** i for i in range(1, m + 1)]
    elif mode == "down":
        rho = (v / u) ** (1.0 / m)
        cells = [v / rho ** i for i in range(1, m + 1)]
    else:
        rho = (u / v) ** (1.0 / m)
        cells = [v * rho ** i for i in range(0, m + 1)]
    cells[-1] = float(u)
    return cells


def _band_realize_free(length: float, target: float, cap: float,
                       lo: _Anchor | None, hi: _Anchor | None,
                       what: str = "free segment") -> list[float]:
    """Exact-fit realization of one free run.

    Cells are ``ramp(lo -> u) + [u] * n + ramp(u <- hi)`` with a plateau
    value ``u <= min(target, length)`` and an integer ``n >= 0`` chosen so
    the run sums to ``length`` EXACTLY: ``G(u) = (length - S_lo(u) -
    S_hi(u)) / u`` must be an integer. The candidate range ``[u_floor, P]``
    is cut into pieces at every value where a ramp's step count changes
    (``v cap^k`` for an ascending side, ``v / cap^k`` for a descent from a
    pin); on each CLOSED piece the step counts are held at the piece's
    interior value, so ``G`` is continuous and strictly decreasing there
    and its ends can be evaluated exactly (no inset, no snap — the inset
    version missed roots inside its own 1e-10 band on long plateaus).
    Scanned from the largest ``u`` down, so the first solution has the
    fewest cells. Never rescales, never places a cell above ``target``,
    never moves a pinned cell.

    Tolerance on "G is an integer": ``1e-10 x (|G| + ramp cells + 1)``,
    the float error of ``(length - S) / u`` scaled the same way; the
    residual is spread over the plateau cells (or, with no plateau, as a
    uniform scale of the run), so every ratio moves by at most 1e-10
    relative.
    """
    if length <= 0.0:
        return []
    P = min(float(target), float(length))
    u_floor = P * 1e-9

    th: set[float] = set()
    for anchor in (lo, hi):
        if anchor is None:
            continue
        v = float(anchor.value)
        k = 0
        while True:
            t = v * cap ** k
            if t >= P * (1.0 - _STEP_TOL):
                break
            if t > u_floor * (1.0 + _STEP_TOL):
                th.add(t)
            k += 1
        if anchor.kind == "pin":
            k = 1
            while True:
                t = v / cap ** k
                if t <= u_floor * (1.0 + _STEP_TOL):
                    break
                if t < P * (1.0 - _STEP_TOL):
                    th.add(t)
                k += 1
    # Merge thresholds closer than the step slack (a seam cell on the law
    # exactly produces two thresholds 1e-13 apart from the two sides).
    bounds = [P]
    for t in sorted(th, reverse=True):
        if t < bounds[-1] * (1.0 - _STEP_TOL):
            bounds.append(t)
    bounds.append(u_floor)

    for b_hi, b_lo in zip(bounds[:-1], bounds[1:]):
        mid = float(np.sqrt(b_hi * b_lo))
        mode_lo, m_lo = _ramp_spec(lo, mid, cap)
        mode_hi, m_hi = _ramp_spec(hi, mid, cap)

        def _G(u):
            s = _ramp_sum(lo, u, mode_lo, m_lo) + _ramp_sum(hi, u, mode_hi, m_hi)
            return (length - s) / u

        g_hi = _G(b_hi)
        g_lo = _G(b_lo)
        has_ramp = mode_lo != "none" or mode_hi != "none"
        tol_g = 1e-10 * (abs(g_hi) + m_lo + m_hi + 1.0)
        n = max(int(np.ceil(g_hi - tol_g)), 0 if has_ramp else 1)
        if n > g_lo + tol_g:
            continue                      # no integer reachable on this piece
        if g_hi >= n - tol_g:
            u = b_hi                      # root at the top (u == P exact)
        elif g_lo <= n + tol_g:
            u = b_lo
        else:
            a, b = b_lo, b_hi             # G(a) > n > G(b)
            for _ in range(200):
                mid = 0.5 * (a + b)
                if _G(mid) - n > 0.0:
                    a = mid
                else:
                    b = mid
                if b - a <= 1e-15 * b_hi:
                    break
            u = 0.5 * (a + b)
        if abs(_G(u) - n) > tol_g:
            continue
        left = _ramp_cells(lo, u, mode_lo, m_lo)
        right = _ramp_cells(hi, u, mode_hi, m_hi)
        cells = left + [float(u)] * n + right[::-1]
        if not cells:
            continue
        # Float dust: onto the plateau cells; with no plateau, spread
        # over the ramp cells that are not pinned by construction (an
        # ``emax`` first/last cell is ``dx_fine`` exactly and stays so).
        dust = length - float(np.sum(cells))
        if n > 0:
            per = dust / n
            for j in range(len(left), len(left) + n):
                cells[j] += per
        else:
            free_idx = list(range(len(cells)))
            if mode_lo == "emax" and len(free_idx) > 1:
                free_idx = free_idx[1:]
            if mode_hi == "emax" and len(free_idx) > 1:
                free_idx = free_idx[:-1]
            base = sum(cells[j] for j in free_idx)
            scale = 1.0 + dust / base
            for j in free_idx:
                cells[j] *= scale
        return [float(c) for c in cells]
    raise ValueError(
        f"{what}: span {length*1e3:.6g} mm cannot hold its ramps at ratio "
        f"<= {cap:g} (pinned boundary cell too large for the span, or the "
        f"segment is too thin for the neighbouring cells it must reach)."
    )


def _band_realize_protected(seg: _BandSeg, lim_lo: float | None,
                            lim_hi: float | None) -> list[float]:
    """Uniform realization of a protected segment under seam limits.

    ``lim_lo``/``lim_hi`` are the maximum seam cells the seam rule has
    imposed (None = unconstrained). With no limit the ``verbatim`` block
    (auto-z post-thirds cells) passes through bit-identically.
    """
    if lim_lo is None and lim_hi is None and seg.verbatim is not None:
        return [float(c) for c in seg.verbatim]
    f_lo = 1.0 / 3.0 if seg.split_lo else 1.0
    f_hi = 1.0 / 3.0 if seg.split_hi else 1.0
    c = float(seg.target)
    if lim_lo is not None:
        c = min(c, lim_lo / f_lo)
    if lim_hi is not None:
        c = min(c, lim_hi / f_hi)
    n = max(int(seg.min_cells), int(np.ceil(seg.length / c - 1e-9)))
    if seg.split_lo or seg.split_hi:
        n = max(n, 2)
    if n > _MAX_PROFILE_CELLS:
        raise ValueError(
            f"protected segment of span {seg.length*1e3:.6g} mm would need "
            f"{n} cells (seam limit {c*1e6:.4g} um) — more than the "
            f"{_MAX_PROFILE_CELLS} cell ceiling. A neighbouring segment is "
            "a sliver (edges from float noise?) and the ratio law is "
            "pulling this one down to its size; merge or widen it.")
    # Seam guard: the realized seam cell must respect the limit itself (the
    # 1e-9 ceil tolerance above is for float dust on the declared quotient,
    # not a licence to overshoot a seam limit).
    while True:
        d = seg.length / n
        if lim_lo is not None and d * f_lo > lim_lo * (1.0 + 1e-13):
            n += 1
            continue
        if lim_hi is not None and d * f_hi > lim_hi * (1.0 + 1e-13):
            n += 1
            continue
        break
    cells = [d] * n
    if seg.split_lo:
        cells = [d / 3, d * 2 / 3] + cells[1:]
    if seg.split_hi:
        cells = cells[:-1] + [d * 2 / 3, d / 3]
    return cells


def _build_band_profile(segs: list[_BandSeg], cap: float,
                        pin_lo: float | None = None,
                        pin_hi: float | None = None) -> np.ndarray:
    """Engine behind ``make_band_profile`` / ``make_z_profile`` /
    ``_make_dz_profile``.

    Fixed-point iteration over segment realizations: free runs ramp from
    their neighbours' ACTUAL seam cells; a protected segment that is the
    coarser side of a seam violating ``cap`` gets its uniform cell reduced
    (more cells) until the seam complies — including a seam between two
    protected segments, which no ramp can ever serve. Seam limits only
    ever tighten, so the iteration is monotone. Returns float64 cells.
    """
    n_seg = len(segs)
    if n_seg == 0:
        return np.zeros(0, dtype=np.float64)
    if not (np.isfinite(cap) and cap > 1.0):
        raise ValueError(f"max_ratio must be a finite number > 1 (got {cap})")
    if cap < _RATIO_FLOOR:
        raise ValueError(
            f"max_ratio must be >= {_RATIO_FLOOR} (got {cap!r}): a 2x "
            f"transition at that ratio already needs "
            f"{int(np.ceil(np.log(2.0) / np.log(_RATIO_FLOOR)))} cells, and "
            "the plateau solve's cost grows with the ramp length")
    for k, s in enumerate(segs):
        if not (s.length > 0.0):
            raise ValueError(f"segment {k}: span must be > 0 (got {s.length})")
        if not (s.target > 0.0):
            raise ValueError(f"segment {k}: cell size must be > 0 (got {s.target})")
    if pin_lo is not None and segs[0].protected:
        raise ValueError("boundary_cell needs a FREE first segment (a "
                         "protected end segment cannot host the pinned cell)")
    if pin_hi is not None and segs[-1].protected:
        raise ValueError("boundary_cell needs a FREE last segment (a "
                         "protected end segment cannot host the pinned cell)")

    def _free_len(k):
        L = float(segs[k].length)
        if k == 0 and pin_lo is not None:
            L -= float(pin_lo)
        if k == n_seg - 1 and pin_hi is not None:
            L -= float(pin_hi)
        if L < -1e-12 * float(segs[k].length):
            raise ValueError(
                f"segment {k}: span {segs[k].length*1e3:.6g} mm is shorter "
                f"than the pinned boundary cell(s) it must hold")
        # A remainder inside the node tolerance is float dust from the
        # edge subtraction (10 mm - 9 mm - 1 mm = 8.7e-19 m), not a run.
        return L if L > _EDGE_TOL else 0.0

    lens = [_free_len(k) if not s.protected else float(s.length)
            for k, s in enumerate(segs)]
    seam_lim: list[list[float | None]] = [[None, None] for _ in segs]

    real: list[list[float]] = []
    for k, s in enumerate(segs):
        if s.protected:
            real.append(_band_realize_protected(s, None, None))
        elif lens[k] <= 0.0:
            real.append([])
        else:
            P = min(float(s.target), lens[k])
            m = max(1, int(np.ceil(lens[k] / P - 1e-9)))
            real.append([lens[k] / m] * m)

    # An end segment whose span EQUALS its pinned cell has no free cells:
    # the pin is then that segment's seam cell on BOTH sides (its lo side
    # for k == 0, its hi side for k == n_seg - 1), so the neighbour ramps
    # from it and the seam audit sees it. Before the 2026-09-07 review the
    # pin was visible from the outside only, and
    # ``make_band_profile([0, 1e-3, 5e-3], [1e-3, 1e-4], boundary_cell=1e-3)``
    # returned a ratio-10 seam with no error.
    def _lo_cell(k, cur):
        if cur[k]:
            return cur[k][0]
        if k == 0 and pin_lo is not None:
            return float(pin_lo)
        if k == n_seg - 1 and pin_hi is not None:
            return float(pin_hi)
        return None

    def _hi_cell(k, cur):
        if cur[k]:
            return cur[k][-1]
        if k == n_seg - 1 and pin_hi is not None:
            return float(pin_hi)
        if k == 0 and pin_lo is not None:
            return float(pin_lo)
        return None

    def _anchor_lo(k, new, cur):
        s = segs[k]
        if k == 0:
            base = _Anchor(float(pin_lo), "pin") if pin_lo is not None else None
        else:
            c = _hi_cell(k - 1, new)
            if c is None:
                base = None
            elif not new[k - 1] and k - 1 == 0 and pin_lo is not None:
                base = _Anchor(c, "pin")
            else:
                base = _Anchor(c, "nb")
        if s.emax_lo is not None and base is not None and base.kind == "nb":
            if s.emax_lo <= base.value * cap:
                return _Anchor(float(s.emax_lo), "emax")
            return base
        if s.emax_lo is not None and base is None:
            return _Anchor(float(s.emax_lo), "emax")
        return base

    def _anchor_hi(k, new, cur):
        s = segs[k]
        if k == n_seg - 1:
            base = _Anchor(float(pin_hi), "pin") if pin_hi is not None else None
        else:
            c = _lo_cell(k + 1, cur)
            if c is None:
                base = None
            elif not cur[k + 1] and k + 1 == n_seg - 1 and pin_hi is not None:
                base = _Anchor(c, "pin")
            else:
                base = _Anchor(c, "nb")
        if s.emax_hi is not None and base is not None and base.kind == "nb":
            if s.emax_hi <= base.value * cap:
                return _Anchor(float(s.emax_hi), "emax")
            return base
        if s.emax_hi is not None and base is None:
            return _Anchor(float(s.emax_hi), "emax")
        return base

    max_iter = 60 + 6 * n_seg
    for _it in range(max_iter):
        new: list[list[float]] = []
        for k, s in enumerate(segs):
            if s.protected:
                new.append(_band_realize_protected(s, *seam_lim[k]))
            elif lens[k] <= 0.0:
                new.append([])
            else:
                new.append(_band_realize_free(
                    lens[k], float(s.target), cap,
                    _anchor_lo(k, new, real), _anchor_hi(k, new, real),
                    what=f"segment {k}"))
        n_cells = sum(len(x) for x in new)
        if n_cells > _MAX_PROFILE_CELLS:
            raise ValueError(
                f"band profile would need {n_cells} cells, more than the "
                f"{_MAX_PROFILE_CELLS} cell ceiling (a sliver segment or a "
                "ratio close to 1 is driving the count)")
        # Seam audit on the ACTUAL seam cells.
        viol = []
        for k in range(n_seg - 1):
            a = _hi_cell(k, new)
            b = _lo_cell(k + 1, new)
            if a is None or b is None:
                continue
            r = max(a / b, b / a)
            if r > cap * (1.0 + 1e-11):
                viol.append((k, a, b))
        # Convergence is judged with a 1e-11 relative tolerance: two adjacent
        # free runs feed each other their seam cells, and the ramp
        # arithmetic can flip those by one ulp in a two-cycle (measured on
        # the fuzz family: 1e-18 m swings, every seam compliant), which an
        # exact-equality test never settles.
        changed = any(
            len(x) != len(y)
            or any(abs(p - q) > 1e-11 * max(abs(p), abs(q))
                   for p, q in zip(x, y))
            for x, y in zip(new, real))
        if not viol and not changed:
            real = new
            break
        for k, a, b in viol:
            coarse = k if a > b else k + 1
            fine_cell = min(a, b)
            side = 1 if coarse == k else 0
            s = segs[coarse]
            if s.protected:
                lim = fine_cell * cap
                cur = seam_lim[coarse][side]
                if cur is None or lim < cur * (1.0 - 1e-13):
                    seam_lim[coarse][side] = lim
                else:
                    raise RuntimeError(
                        f"band profile: protected segment {coarse} cannot "
                        f"satisfy its seam (limit {cur}, needed {lim})")
            else:
                empty_pin = (
                    (coarse == 0 and pin_lo is not None and not new[0])
                    or (coarse == n_seg - 1 and pin_hi is not None
                        and not new[-1]))
                if empty_pin:
                    raise ValueError(
                        f"segment {coarse}: the pinned boundary cell "
                        f"{max(a, b)*1e3:.6g} mm faces a {fine_cell*1e3:.6g} "
                        f"mm cell with no room for a ramp (ratio "
                        f"{max(a, b)/fine_cell:.3f} > {cap:g}); widen the "
                        f"segment so it can hold the ramp")
                # free coarse side re-ramps from the fresh neighbour cell
        real = new
    else:
        raise RuntimeError("band profile: fixed-point iteration did not "
                           f"converge in {max_iter} passes")

    parts: list[float] = []
    if pin_lo is not None:
        parts.append(float(pin_lo))
    for cells in real:
        parts.extend(cells)
    if pin_hi is not None:
        parts.append(float(pin_hi))
    out = np.asarray(parts, dtype=np.float64)
    # Self-audit of the invariants the caller relies on.
    total = float(sum(float(s.length) for s in segs))
    if abs(float(np.sum(out)) - total) > 1e-12:
        raise RuntimeError("band profile: column length not realized "
                           f"({float(np.sum(out))} vs {total})")
    return out


def make_band_profile(
    edges,
    cell_sizes,
    *,
    max_ratio: float = 1.4,
    protected=None,
    boundary_cell: float | None = None,
    min_cells: int = 1,
) -> np.ndarray:
    """Build a 1-D cell-size profile whose interfaces land on node planes
    and whose adjacent-cell ratios obey ``max_ratio`` everywhere.

    Axis-agnostic: the same output serves ``dx_profile``, ``dy_profile``
    or ``dz_profile`` of :func:`make_nonuniform_grid`.

    Parameters
    ----------
    edges : sequence of float
        Sorted physical coordinates (m) of every interface that must lie
        on a node plane, INCLUDING the domain start and end.
    cell_sizes : sequence of float
        Target cell size (m) per segment, ``len(edges) - 1`` entries.
    max_ratio : float
        Cap on every adjacent-cell ratio (default 1.4 — the validated
        z-axis multi-band cap; pass 1.3 for an in-plane profile, the
        in-plane threshold, see docs/guides/support_matrix.md). Must be
        finite and ``>= 1.0001`` (a 2x transition at that ratio already
        needs 6932 cells; the plateau solve's cost grows with the ramp
        length — measured 0.03 s at 1.0001 on a 40-cell profile and
        0.4 s for a 1 um pin descending into 10 mm of 100 um cells, an
        8110-cell profile).
    protected : sequence of bool, optional
        Per segment. A protected segment is realized UNIFORM —
        ``n = max(min_cells, ceil(span / target - 1e-9))`` equal cells —
        and never hosts a ramp cell; a free segment (default) may host the
        ramps its neighbours need.
    boundary_cell : float, optional
        Pins ``cells[0] == cells[-1] == boundary_cell`` exactly (the x/y
        CPML contract of :func:`make_nonuniform_grid`). Both end segments
        must be free and wide enough to hold the pinned cell plus its ramp
        (``ValueError`` otherwise).
    min_cells : int
        Minimum cell count of a PROTECTED segment (default 1).

    Returns
    -------
    np.ndarray
        float64 cells. Invariants, for any valid input: every edge is a
        cumulative node coordinate to 1e-12 m; every adjacent ratio is
        ``<= max_ratio`` (to 1e-9), INCLUDING seams between two protected
        segments; the sum equals ``edges[-1] - edges[0]`` to 1e-12 m;
        with ``boundary_cell``, both end cells equal it bit-exactly.
        The two 1e-12 m figures are float64 accumulation of the cells
        (a running sum, as the rasterizer positions nodes) and hold for
        columns up to about 1 m with up to 1e4 cells; the miss grows
        with column length and cell count (measured 1e-11 m on a 10 m
        column of 1e5 cells, 2.2e-9 m on 100 m / 1e6 cells), while each
        cell itself matches its declared value to 1e-15 relative.

    Raises
    ------
    ValueError
        On inconsistent input: edges not strictly increasing or two edges
        closer than 1e-12 m (the node tolerance — merge them), a
        non-positive cell size, ``max_ratio`` not finite or below 1.0001,
        a protected end segment with ``boundary_cell``, an end segment
        shorter than the pinned cell or too short to hold the pin plus
        its ramp, and a profile that would exceed 1,000,000 cells (a
        sliver segment pulls its neighbours down to its size under the
        ratio law: a 1e-9 m protected sliver between 1 mm blocks asks
        for 1.4e6 cells).

    Notes
    -----
    Ramps are geometric (per-step ratio spread evenly, ``<= max_ratio``)
    and placed INSIDE the coarser free segment on whichever side needs
    one; the free run's plateau value is solved so the run sums to its
    declared span exactly (no cell above its target, no rescale). Two
    adjacent PROTECTED segments whose seam exceeds the cap cannot be
    served by a ramp: the coarser one's uniform cell is reduced (more
    cells) until the seam complies, iterated to a fixed point. The cost
    is cells AND, whenever the refined block was the coarsest declared
    cell, a smaller minimum cell than declared — the time step follows
    the global minimum, so dt drops with it. Measured on 3000 seeded
    stacks (rng 1; 2-6 segments, spans 20 um-3 mm, targets 0.05-2x the
    span, 70 % protected, caps 1.2-1.4): the realized minimum sits below
    the declared minimum in 410 of them, down to 0.31x in the worst case
    (two single-cell protected blocks whose span ratio just exceeds the
    cap at 1.2). Hand case: ``[0, 0.1, 0.1355] mm, [25, 100] um, both
    protected, 1.4`` realizes 5 x 20 um + 2 x 17.75 um, minimum 0.71x
    the declared 25 um. A free segment too thin for any ramp is a
    refinement source for its neighbours in the same way.
    """
    e = np.asarray(edges, dtype=np.float64).ravel()
    cs = np.asarray(cell_sizes, dtype=np.float64).ravel()
    if e.size < 2:
        raise ValueError("edges needs at least two entries (domain start and end)")
    if cs.size != e.size - 1:
        raise ValueError(
            f"cell_sizes must have len(edges)-1 = {e.size - 1} entries "
            f"(got {cs.size})")
    spans = np.diff(e)
    if np.any(spans <= 0.0):
        raise ValueError("edges must be strictly increasing")
    if np.any(spans <= _EDGE_TOL):
        k = int(np.argmax(spans <= _EDGE_TOL))
        raise ValueError(
            f"edges[{k}] and edges[{k + 1}] are {spans[k]:.3g} m apart, "
            f"closer than the {_EDGE_TOL:g} m node tolerance: merge them")
    if np.any(cs <= 0.0):
        raise ValueError("cell_sizes must be > 0")
    if protected is None:
        prot = [False] * int(cs.size)
    else:
        prot = [bool(p) for p in protected]
        if len(prot) != int(cs.size):
            raise ValueError("protected must have one entry per segment")
    if int(min_cells) < 1:
        raise ValueError("min_cells must be >= 1")
    if boundary_cell is not None and not (float(boundary_cell) > 0.0):
        raise ValueError("boundary_cell must be > 0")
    segs = [
        _BandSeg(length=float(spans[k]), target=float(cs[k]),
                 protected=prot[k], min_cells=int(min_cells))
        for k in range(int(cs.size))
    ]
    pin = None if boundary_cell is None else float(boundary_cell)
    return _build_band_profile(segs, float(max_ratio), pin_lo=pin, pin_hi=pin)


def make_z_profile(
    features: list[float],
    domain_z: float,
    dx_fine: float,
    dx_coarse: float | None = None,
    grading: float = 1.4,
) -> np.ndarray:
    """Generate a z-profile that snaps to feature planes: fine cells at
    every feature plane, coarse cells away from them.

    Every segment between consecutive features (plus 0 and ``domain_z``)
    is realized as fine -> coarse -> fine: a cell of at most ``dx_fine`` on
    each end, a geometric ramp with per-step ratio ``<= grading`` up to a
    plateau of at most ``dx_coarse`` and back down, the run summing to the
    segment exactly. Segments too thin for grading (``span <= 4 dx_fine``,
    or ``dx_coarse <= 1.01 dx_fine``) are realized uniform at about
    ``dx_fine``. Built on :func:`make_band_profile`'s engine, so every
    feature coordinate is a node plane (1e-12 m), every adjacent ratio is
    ``<= grading`` and ``sum == domain_z``.

    Parameters
    ----------
    features : list of z-positions that must align to cell boundaries
    domain_z : total z domain height
    dx_fine : fine cell size (at feature planes)
    dx_coarse : coarse cell size (away from features). If None, uses dx_fine
        everywhere (no grading).
    grading : max ratio between adjacent cells (default 1.4). A value
        ``<= 1`` means no grading (uniform ``dx_fine`` everywhere, as
        before); otherwise it must be ``>= 1.0001`` (see
        :func:`make_band_profile`).

    Raises
    ------
    ValueError
        A feature outside ``[0, domain_z]`` (the column must sum to
        ``domain_z``), non-positive sizes, or a non-finite ``grading``.
        Features closer than 1e-12 m are merged into one plane.
    """
    if dx_coarse is None:
        dx_coarse = dx_fine
    if not (dx_fine > 0.0) or not (dx_coarse > 0.0):
        raise ValueError("dx_fine and dx_coarse must be > 0")
    grading = float(grading)
    if not np.isfinite(grading):
        raise ValueError(f"grading must be a finite number (got {grading})")
    cap = grading
    if grading <= 1.0:
        # No grading: uniform dx_fine everywhere, which is what the old
        # loop produced for grading <= 1 (its plateau never grew). The
        # seams between the uniform segments are then held at the
        # default 1.4.
        dx_coarse = dx_fine
        cap = 1.4
    if not (domain_z > 0.0):
        raise ValueError(f"domain_z must be > 0 (got {domain_z})")

    # Features outside [0, domain_z] cannot be node planes of a column
    # that sums to domain_z (the old loop silently extended the column);
    # features closer together than the 1e-12 m node tolerance are one
    # plane (0.1 mm + 0.2 mm vs 0.3 mm differ by 5e-20 m and used to
    # become a 5e-20 m cell).
    pts = sorted(float(f) for f in features)
    for f in pts:
        if f < -_EDGE_TOL or f > domain_z + _EDGE_TOL:
            raise ValueError(
                f"feature {f} lies outside [0, domain_z = {domain_z}]")
    feats = [0.0]
    for f in pts:
        if f - feats[-1] > _EDGE_TOL:
            feats.append(f)
    if domain_z - feats[-1] > _EDGE_TOL:
        feats.append(float(domain_z))
    else:
        feats[-1] = float(domain_z)

    segs: list[_BandSeg] = []
    for i in range(len(feats) - 1):
        span = float(feats[i + 1] - feats[i])
        if span <= 0:
            continue
        if dx_coarse <= dx_fine * 1.01 or span <= 4 * dx_fine:
            # Uniform fine cells for thin segments or when no grading needed
            n = max(1, int(round(span / dx_fine)))
            segs.append(_BandSeg(length=span, target=span / n, protected=False))
        else:
            segs.append(_BandSeg(length=span, target=float(dx_coarse),
                                 protected=False,
                                 emax_lo=float(dx_fine), emax_hi=float(dx_fine)))
    if not segs:
        return np.array([])
    return _build_band_profile(segs, cap)
