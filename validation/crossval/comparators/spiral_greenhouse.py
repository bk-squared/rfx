"""Quasi-static inductance referee for planar rectangular spirals
(Greenhouse summation of Hoer-Love partial inductances).

Physics. At low frequency the current density in a good conductor is
uniform over its cross-section and the magnetic energy of a set of
conductors is a quadratic form in their currents; the coefficient matrix
is the matrix of partial inductances (Ruehli 1972). For a series-connected
chain of straight segments carrying the same current the terminal
inductance is the double sum

    L = sum_i L_ii + sum_{i != j} s_i s_j M_ij ,

with L_ii the partial self-inductance of segment i, M_ij the partial mutual
inductance of the pair (zero for perpendicular segments because the Neumann
integrand is dl . dl') and s_i = +-1 the orientation of the current along
the common axis (Greenhouse 1974). A perfectly conducting ground plane is
handled by images: a copy of every segment mirrored in the plane carrying
the opposite current, whose mutual couplings with the real segments are
subtracted.

Partial inductances. Both L_ii and M_ij are the six-fold integral
mu0/(4 pi A_i A_j) int int dV dV' / |r - r'| over two axis-parallel
rectangular bars. Hoer and Love (J. Res. NBS 69C, 1965, eq. (10)) give the
exact closed form of that integral as a triple alternating sum of a sextuple
antiderivative f(x, y, z); Ruehli (IBM J. Res. Dev. 16, 1972) popularised
it as the partial-inductance formula. ``mutual_inductance_bars`` implements
it; ``self_inductance_bar`` is the coincident-bar special case. The closed
form is exact for every offset and overlap (coincident, touching, end to
end: the cases where a quadrature rule meets the 1/R and log singularities)
and it is validated here against an independent numerical evaluation of the
same integral (tests R1/R2, agreement <= 2e-8). Its weakness is
cancellation: f is O(D^5) in the largest distance D while the sum is
O(l w^2 t^2 ln), so digits are lost as (D / cross-section)^4 -- fatal for a
thin bar coupled to a far image (measured: a 1 mm x 20 um x 2 um bar and
its image 0.2 m away gives -5e-7 H from a true 1e-13 H). Therefore, WHEN
the two cross-sections are separated (gap >= the largest cross-section
dimension) the mutual is instead integrated by a tensor Gauss-Legendre rule
over the two cross-sections of the analytic thin-filament mutual (Grover's
double line integral, exact in the length direction). There the integrand
is analytic with its nearest singularity at least one half-interval away,
so the 16-point rule converges geometrically (measured 8 -> 16 -> 32 point
changes below 1e-12 at the marginal gap, test R2b). Every evaluation also
returns an estimated relative error (round-off amplification x 1e-15 for
Hoer-Love; the 8- vs 16-point change for Gauss-Legendre) and
``greenhouse_terms`` reports the worst one, so a run that lost precision
says so instead of returning a wrong number.

Scope fence. Magnetoquasistatic, uniform current density, non-magnetic
conductors, no substrate eddy currents, no skin effect (the internal
inductance of a bar is included by the uniform-current assumption and is
the DC value). Segments must be axis-aligned and lie in horizontal planes
(x- or y-directed); vias are not modelled. The Greenhouse sum uses
centreline segment lengths that meet at the corner points, so the corner
squares are counted twice by the self terms; that is the convention of
Greenhouse 1974 and of the published loop formulas compared against, and
it costs O(w / l) per corner. This module deliberately shares no code with
``rfx``: it is a referee for the 3-D FDFD extraction, so it must be an
independent implementation (numpy + scipy only; ``rfx`` is never imported).

Units: SI everywhere (metres, henries).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

MU0 = 4e-7 * math.pi  # vacuum permeability; differs from the 2019 SI value by 5e-10 relative


# --------------------------------------------------------------------------
# 1. Hoer-Love / Ruehli partial inductances
# --------------------------------------------------------------------------

def _hoer_love_f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Sextuple antiderivative of 1/rho, rho = sqrt(x^2 + y^2 + z^2)
    (Hoer & Love 1965 eq. (10)); d^6 f / dx^2 dy^2 dz^2 = 1/rho.

    Every term is of the form (polynomial prefactor) * (log or arctan that
    can be singular where a coordinate vanishes); wherever the transcendental
    factor is singular its prefactor is zero and the product's limit is
    zero, which is what the ``where`` guards return. The logarithm is written
    as ``x * asinh(x / sqrt(y^2 + z^2))``, an even function of x with no
    cancellation for x < 0 (the textbook ``x ln((x + rho) / sqrt(y^2+z^2))``
    subtracts nearly equal numbers there).
    """
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    x2, y2, z2 = x * x, y * y, z * z
    rho = np.sqrt(x2 + y2 + z2)
    out = np.zeros_like(rho)

    def log_term(u: np.ndarray, u2: np.ndarray, v2: np.ndarray, w2: np.ndarray) -> np.ndarray:
        pref = (v2 * w2 / 4.0 - v2 * v2 / 24.0 - w2 * w2 / 24.0) * u
        s = np.sqrt(v2 + w2)
        ok = (s > 0) & (pref != 0)
        ratio = np.divide(u, s, out=np.zeros_like(u), where=ok)
        return np.where(ok, pref * np.arcsinh(ratio), 0.0)

    out += log_term(x, x2, y2, z2)
    out += log_term(y, y2, x2, z2)
    out += log_term(z, z2, x2, y2)
    out += (x2 * x2 + y2 * y2 + z2 * z2 - 3.0 * (x2 * y2 + y2 * z2 + z2 * x2)) * rho / 60.0

    def atan_term(pref: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        ok = (den != 0) & (pref != 0)
        arg = np.divide(num, den, out=np.zeros_like(num), where=ok)
        return np.where(ok, pref * np.arctan(arg), 0.0)

    xyz = x * y * z
    out -= atan_term(xyz * z2 / 6.0, x * y, z * rho)
    out -= atan_term(xyz * y2 / 6.0, x * z, y * rho)
    out -= atan_term(xyz * x2 / 6.0, y * z, x * rho)
    return out


def _hoer_love_bars(l1: float, w1: float, t1: float, l2: float, w2: float, t2: float,
                    dw: float, dt: float, dl: float) -> tuple[float, float]:
    """Hoer-Love closed form; returns (M, estimated relative error). The
    error estimate is max |term| / |sum| times 1e-15 (64 terms of double
    precision round-off in the alternating sum)."""
    scale = max(l1, w1, t1, l2, w2, t2, abs(dw), abs(dt), abs(dl))  # work in O(1) units
    q = np.array([dw - w1, dw + w2 - w1, dw + w2, dw]) / scale
    r = np.array([dt - t1, dt + t2 - t1, dt + t2, dt]) / scale
    s = np.array([dl - l1, dl + l2 - l1, dl + l2, dl]) / scale
    sign = np.array([1.0, -1.0, 1.0, -1.0])  # (-1)^(i+1): +f(q1) - f(q2) + f(q3) - f(q4)
    f = _hoer_love_f(q[:, None, None], r[None, :, None], s[None, None, :])
    terms = f * sign[:, None, None] * sign[None, :, None] * sign[None, None, :]
    total = float(terms.sum())
    m = MU0 / (4.0 * math.pi) * total * scale ** 5 / (w1 * t1 * w2 * t2)
    err = float(np.abs(terms).max() / abs(total)) * 1e-15 if total != 0 else math.inf
    return m, err


def _filament_mutual_array(d: np.ndarray, l1: float, l2: float, s: float) -> np.ndarray:
    """Vectorised :func:`filament_mutual` for an array of distances d > 0."""
    def g(u: float) -> np.ndarray:
        return u * np.arcsinh(u / d) - np.hypot(u, d)

    return MU0 / (4.0 * math.pi) * (g(s + l2) - g(s + l2 - l1) - g(s) + g(s - l1))


def _gauss_legendre_bars(l1: float, w1: float, t1: float, l2: float, w2: float, t2: float,
                         dw: float, dt: float, dl: float, n: int) -> float:
    """n^4-point tensor Gauss-Legendre integration of the thin-filament
    mutual over both cross-sections (the length direction is analytic)."""
    x, wx = np.polynomial.legendre.leggauss(n)
    xs1, ws1 = 0.5 * w1 * (x + 1.0), 0.5 * w1 * wx
    ys1, wy1 = 0.5 * t1 * (x + 1.0), 0.5 * t1 * wx
    xs2, ws2 = dw + 0.5 * w2 * (x + 1.0), 0.5 * w2 * wx
    ys2, wy2 = dt + 0.5 * t2 * (x + 1.0), 0.5 * t2 * wx
    u = xs2[None, :] - xs1[:, None]
    v = ys2[None, :] - ys1[:, None]
    d = np.sqrt(u[:, :, None, None] ** 2 + v[None, None, :, :] ** 2)
    m = _filament_mutual_array(d, l1, l2, dl)
    wt = ws1[:, None, None, None] * ws2[None, :, None, None] * wy1[None, None, :, None] * wy2[None, None, None, :]
    return float((m * wt).sum() / (w1 * t1 * w2 * t2))


def cross_section_gap(w1: float, t1: float, w2: float, t2: float, dw: float, dt: float) -> float:
    """Largest of the two axis gaps between the cross-section rectangles
    (negative when they overlap in both directions)."""
    gap_w = max(dw - w1, -(dw + w2))
    gap_t = max(dt - t1, -(dt + t2))
    return max(gap_w, gap_t)


def mutual_inductance_bars(l1: float, w1: float, t1: float, l2: float, w2: float, t2: float,
                           dw: float, dt: float, dl: float, method: str = "auto",
                           return_error: bool = False) -> float | tuple[float, float]:
    """Partial mutual inductance (H) of two parallel rectangular bars.

    Bar 1 occupies ``[0, w1] x [0, t1] x [0, l1]`` (width, thickness,
    length); bar 2 occupies ``[dw, dw + w2] x [dt, dt + t2] x [dl, dl + l2]``.
    Any offsets are allowed, including overlap and coincidence (the self
    case). The result is the unsigned Neumann coefficient for currents
    flowing in the same +length direction; the caller applies the sign for
    antiparallel currents.

    ``method``: ``"hoer_love"`` (exact closed form, loses digits for far
    thin pairs), ``"gauss_legendre"`` (16^4-point filament rule, converged
    only for separated cross-sections; raises ``ValueError`` when the
    cross-sections overlap or touch, because the 1/R kernel is then
    singular inside the integration domain and the rule returns an
    unconverged number or NaN) or ``"auto"``: Gauss-Legendre when
    :func:`cross_section_gap` >= the largest cross-section dimension, else
    Hoer-Love -- see the module docstring for the measurements behind the
    rule. With ``return_error=True`` also returns the estimated relative
    error of the value (round-off amplification for Hoer-Love, the 8- vs
    16-point change for Gauss-Legendre).
    """
    if min(l1, w1, t1, l2, w2, t2) <= 0:
        raise ValueError("bar dimensions must be > 0")
    gap = cross_section_gap(w1, t1, w2, t2, dw, dt)
    if method == "auto":
        method = "gauss_legendre" if gap >= max(w1, t1, w2, t2) else "hoer_love"
    if method == "hoer_love":
        m, err = _hoer_love_bars(l1, w1, t1, l2, w2, t2, dw, dt, dl)
    elif method == "gauss_legendre":
        if gap <= 0:
            raise ValueError(
                f"method='gauss_legendre' needs separated cross-sections (gap {gap:g} <= 0); "
                "use method='hoer_love' or 'auto' for overlapping or touching bars")
        m = _gauss_legendre_bars(l1, w1, t1, l2, w2, t2, dw, dt, dl, 16)
        m8 = _gauss_legendre_bars(l1, w1, t1, l2, w2, t2, dw, dt, dl, 8)
        err = abs(m - m8) / abs(m) if m != 0 else math.inf
    else:
        raise ValueError(f"unknown method {method!r}")
    return (m, err) if return_error else m


def self_inductance_bar(length: float, w: float, t: float) -> float:
    """Partial self-inductance (H) of a straight bar of ``length``,
    width ``w``, thickness ``t`` with uniform current density: the
    Hoer-Love integral of the bar with itself."""
    m = mutual_inductance_bars(length, w, t, length, w, t, 0.0, 0.0, 0.0, method="hoer_love")
    assert isinstance(m, float)
    return m


def grover_bar(length: float, w: float, t: float) -> float:
    """Published approximate closed form for the same bar (Grover 1946
    eq. (7); Greenhouse 1974 eq. (3)):
    ``L = mu0 l / (2 pi) [ln(2l/(w+t)) + 0.50049 + (w+t)/(3l)]``.
    It is the thin-filament formula with the geometric mean distance of the
    cross-section, ``GMD = 0.22313 (w+t)`` (exact between the strip and
    square limits to 0.2 %), plus the DC internal term 1/4 and an arithmetic
    mean distance correction. Used only as the published reference of test
    R3 and for reporting; the exact Hoer-Love form is what the referee uses.
    """
    ell = length
    return MU0 * ell / (2.0 * math.pi) * (math.log(2.0 * ell / (w + t)) + 0.50049 + (w + t) / (3.0 * ell))


def filament_mutual(d: float, l1: float, l2: float, s: float) -> float:
    """Mutual inductance (H) of two parallel thin filaments a distance
    ``d > 0`` apart: filament 1 spans ``[0, l1]``, filament 2 spans
    ``[s, s + l2]`` along the common axis (Grover 1946 ch. 5; the analytic
    double line integral of 1/R).
    ``M = mu0/(4 pi) [G(s + l2) - G(s + l2 - l1) - G(s) + G(s - l1)]`` with
    ``G(u) = u asinh(u/d) - sqrt(u^2 + d^2)``.
    """
    if d <= 0:
        raise ValueError("filament distance must be > 0")

    def g(u: float) -> float:
        return u * math.asinh(u / d) - math.hypot(u, d)

    return MU0 / (4.0 * math.pi) * (g(s + l2) - g(s + l2 - l1) - g(s) + g(s - l1))


# --------------------------------------------------------------------------
# 2. Greenhouse summation over axis-aligned segments
# --------------------------------------------------------------------------

class Segment(NamedTuple):
    """One straight rectangular conductor. ``start``/``end`` are the
    centreline end points (x, y, z) in metres, ``width`` the in-plane
    transverse size, ``thickness`` the vertical (z) size, ``current`` the
    signed current (+1 flows from ``start`` to ``end``)."""
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    width: float
    thickness: float
    current: float = 1.0


def segment_axis(seg: Segment) -> int:
    """0 for an x-directed segment, 1 for y-directed; ``ValueError`` for
    anything else (vertical or oblique segments are outside the model)."""
    d = np.asarray(seg.end, float) - np.asarray(seg.start, float)
    nz = np.flatnonzero(d != 0)
    if len(nz) != 1 or nz[0] == 2:
        raise ValueError(f"segment must be x- or y-directed and non-degenerate, got direction {d}")
    return int(nz[0])


def _bar_frame(seg: Segment, axis: int) -> tuple[float, float, float, float, float, float, float]:
    """(length, width, thickness, corner_w, corner_t, corner_l, orientation)
    of a segment in the (width, thickness, length) frame of its axis."""
    a = np.asarray(seg.start, float)
    b = np.asarray(seg.end, float)
    perp = 1 - axis
    length = float(abs(b[axis] - a[axis]))
    corner_l = float(min(a[axis], b[axis]))
    corner_w = float(a[perp] - 0.5 * seg.width)
    corner_t = float(a[2] - 0.5 * seg.thickness)
    orient = math.copysign(1.0, b[axis] - a[axis]) * seg.current
    return length, seg.width, seg.thickness, corner_w, corner_t, corner_l, orient


def pair_mutual(si: Segment, sj: Segment, return_error: bool = False) -> float | tuple[float, float]:
    """Signed partial mutual inductance of two segments: zero when
    perpendicular, else :func:`mutual_inductance_bars` (``method="auto"``)
    times the product of the current orientations.
    """
    ai, aj = segment_axis(si), segment_axis(sj)
    if ai != aj:
        return (0.0, 0.0) if return_error else 0.0
    li, wi, ti, cwi, cti, cli, oi = _bar_frame(si, ai)
    lj, wj, tj, cwj, ctj, clj, oj = _bar_frame(sj, aj)
    res = mutual_inductance_bars(li, wi, ti, lj, wj, tj, cwj - cwi, ctj - cti, clj - cli,
                                 return_error=return_error)
    if return_error:
        m, err = res  # type: ignore[misc]
        return oi * oj * m, err
    assert isinstance(res, float)
    return oi * oj * res


class GreenhouseResult(NamedTuple):
    """``total`` = ``self_sum + mutual_sum + image_sum`` (H).
    ``mutual_sum`` is the signed sum over ordered pairs of real segments,
    ``image_sum`` the (negative) contribution of the ground-plane images,
    ``worst_error`` the largest estimated relative error of any single
    partial-inductance term (see :func:`mutual_inductance_bars`)."""
    total: float
    self_sum: float
    mutual_sum: float
    image_sum: float
    worst_error: float


def greenhouse_terms(segments: Sequence[Segment], ground_height: float | None = None) -> GreenhouseResult:
    """Greenhouse sum with its parts, see :func:`greenhouse_inductance`."""
    segs = list(segments)
    worst = 0.0
    self_sum = 0.0
    for s in segs:
        ax = segment_axis(s)
        ell, w, t, *_ = _bar_frame(s, ax)
        m, c = mutual_inductance_bars(ell, w, t, ell, w, t, 0.0, 0.0, 0.0, method="hoer_love",
                                      return_error=True)  # type: ignore[misc]
        self_sum += m * s.current ** 2
        worst = max(worst, c)
    mutual_sum = 0.0
    for i, si in enumerate(segs):
        for j, sj in enumerate(segs):
            if i == j:
                continue
            m, c = pair_mutual(si, sj, return_error=True)  # type: ignore[misc]
            mutual_sum += m
            worst = max(worst, c)
    image_sum = 0.0
    if ground_height is not None:
        if ground_height <= 0:
            raise ValueError("ground_height must be > 0 (distance from the PEC plane to z = 0)")
        # PEC plane at z = -h: the image of a point at z is at -2h - z with opposite current.
        images = []
        for s in segs:
            a, b = np.asarray(s.start, float), np.asarray(s.end, float)
            a[2] = -2.0 * ground_height - a[2]
            b[2] = -2.0 * ground_height - b[2]
            images.append(Segment(tuple(a), tuple(b), s.width, s.thickness, -s.current))
        for si in segs:
            for sj in images:
                m, c = pair_mutual(si, sj, return_error=True)  # type: ignore[misc]
                image_sum += m
                worst = max(worst, c)
    return GreenhouseResult(self_sum + mutual_sum + image_sum, self_sum, mutual_sum, image_sum, worst)


def greenhouse_inductance(segments: Sequence[Segment], ground_height: float | None = None) -> float:
    """Total low-frequency inductance (H) of series-connected axis-aligned
    rectangular segments: partial self terms plus signed mutual terms of
    every parallel pair (antiparallel currents subtract). With
    ``ground_height=h`` a PEC plane lies at ``z = -h`` (h below the z = 0
    reference in which the segment coordinates are given; put the spiral's
    centreline at z = 0 and h is then the ground-to-metal-centre distance)
    and its effect is the image method: mirror segments at ``z -> -2h - z``
    with opposite current, coupled to the real segments.
    """
    return greenhouse_terms(segments, ground_height).total


# --------------------------------------------------------------------------
# 3. Square spiral geometry (rfx.fdfd.gds.rect_spiral convention, copied)
# --------------------------------------------------------------------------

def rect_spiral_centreline(n_turns: int, r_out: float, width: float, spacing: float,
                           lead_length: float = 0.0) -> np.ndarray:
    """Centreline points (M, 2) of the square spiral of
    ``rfx.fdfd.gds.rect_spiral`` -- the convention is copied, not imported.

    Side ``k`` lies on the line with outward normal at angle ``k pi/2``
    (right, top, left, bottom, ...) and apothem ``a0 - pitch (k // 4)``,
    ``a0 = r_out - width/2``, ``pitch = width + spacing``; vertex ``k`` is
    the corner of lines ``k-1`` and ``k``, vertex 0 is the bottom-right
    corner ``(a0, -a0)``. Travel is counter-clockwise, the lead (if any)
    extends side 0 downwards by ``lead_length`` to the outer port, and the
    strip ends with an upward extension of ``1.5 width`` on the inner
    vertical line ``x = a_in = a0 - pitch n_turns``. Raises ``ValueError``
    for the same infeasible cases as the rfx generator (inner end reaching
    the axis, last side shorter than ``width``).
    """
    if n_turns < 1:
        raise ValueError("n_turns >= 1")
    if r_out <= 0 or width <= 0 or spacing <= 0:
        raise ValueError("r_out, width, spacing must be > 0")
    if lead_length < 0:
        raise ValueError("lead_length must be >= 0")
    pitch = width + spacing
    a0 = r_out - 0.5 * width
    a_in = a0 - pitch * n_turns
    if a_in <= 0.5 * width:
        raise ValueError("too many turns for r_out / width / spacing: the spiral would reach the centre")
    c_last = a0 - pitch * (n_turns - 1)
    last_side = 2.0 * c_last - pitch
    if last_side < width:
        raise ValueError(f"last side {last_side:.3g} m shorter than the width {width:.3g} m")

    def apothem(k: int) -> float:
        return a0 - pitch * (k // 4)

    verts = []
    for k in range(4 * n_turns + 1):
        c_prev = apothem(k - 1) if k > 0 else a0
        c = apothem(k)
        m = k % 4  # normal of line k: 0 -> +x, 1 -> +y, 2 -> -x, 3 -> -y
        if m == 0:      # line k: x = c; line k-1: y = -c_prev
            verts.append((c, -c_prev))
        elif m == 1:    # line k: y = c; line k-1: x = c_prev
            verts.append((c_prev, c))
        elif m == 2:    # line k: x = -c; line k-1: y = c_prev
            verts.append((-c, c_prev))
        else:           # line k: y = -c; line k-1: x = -c_prev
            verts.append((-c_prev, -c))
    end = (a_in, verts[-1][1] + 1.5 * width)
    head = [(verts[0][0], verts[0][1] - lead_length)] if lead_length > 0 else []
    return np.array(head + verts + [end], dtype=np.float64)


def rect_spiral_segments(n_turns: int, r_out: float, width: float, spacing: float,
                         thickness: float, lead_length: float = 0.0,
                         underpass: tuple[float, float] | None = None) -> list[Segment]:
    """Segment list of the square spiral (centreline at z = 0, current
    entering at the outer port and flowing along the strip to the inner
    end). ``underpass=(dz, t_u)`` adds the return strip of
    ``rfx.fdfd.gds.rect_spiral``: from the inner end straight down (-y) on a
    layer whose centre is ``dz`` below the spiral, thickness ``t_u``, to the
    port line, with the current continuing in the series sense (the via is
    not modelled).
    """
    pts = rect_spiral_centreline(n_turns, r_out, width, spacing, lead_length)
    segs = [Segment((float(a[0]), float(a[1]), 0.0), (float(b[0]), float(b[1]), 0.0), width, thickness)
            for a, b in zip(pts[:-1], pts[1:])]
    if underpass is not None:
        dz, t_u = underpass
        x_in, y_end = float(pts[-1][0]), float(pts[-1][1])
        y_port = float(pts[0][1])
        segs.append(Segment((x_in, y_end, -dz), (x_in, y_port, -dz), width, t_u))
    return segs


# --------------------------------------------------------------------------
# 4. Published closed forms for cross-checks
# --------------------------------------------------------------------------

def mohan_wheeler(n_turns: float, d_out: float, d_in: float, width: float, spacing: float) -> float:
    """Mohan modified-Wheeler estimate for a square spiral (Mohan, del Mar
    Hershenson, Boyd, Lee, JSSC 34(10), 1999, eq. (2)):
    ``L = K1 mu0 n^2 d_avg / (1 + K2 rho)``, ``K1 = 2.34``, ``K2 = 2.75``,
    ``d_avg = (d_out + d_in)/2``, ``rho = (d_out - d_in)/(d_out + d_in)``.
    Mohan reports typical errors of 2-3 % and worst cases of ~8 % against
    field-solver results for ``spacing <= 3 width``; ``width`` and
    ``spacing`` are accepted only to check that condition (a warning-free
    ``ValueError`` if violated). This is a coarse sanity band, not a
    reference.
    """
    if spacing > 3.0 * width:
        raise ValueError("Mohan's fit is stated for spacing <= 3 width")
    if d_in <= 0 or d_out <= d_in:
        raise ValueError("need 0 < d_in < d_out")
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)
    return 2.34 * MU0 * n_turns ** 2 * d_avg / (1.0 + 2.75 * rho)


def square_loop_closed_form(side: float, w: float, t: float) -> float:
    """Published closed form for a closed square loop of side ``side``
    (centreline) made of rectangular conductor ``w x t``: Grover 1946 ch. 8
    (rectangle formula, eq. (60), with the rectangular-section substitution
    of ch. 8 sec. "rectangular section", i.e. Greenhouse 1974 eqs. (3)-(5)):
    four straight sides by the Grover/Greenhouse bar formula
    (:func:`grover_bar`) minus twice the thin-filament mutual of each pair of
    opposite sides at the geometric mean distance ``side``:

        L = 4 L_bar(side, w, t) - 4 M_fil(side, side)
        M_fil(d, l) = mu0/(2 pi) [l asinh(l/d) - sqrt(l^2 + d^2) + d].

    Perpendicular sides do not couple. Accuracy: Grover gives the bar
    formula for a length large compared with the section without a numeric
    bound; its error comes from the GMD/AMD expansion (the 0.2 % GMD range
    between strip and square sections enters L divided by
    ``ln(2l/(w+t)) ~ 5``, the AMD term is a first-order correction) and the
    opposite-side mutual at the filament level neglects O((w/side)^2); the
    corner double counting is the same in both forms and cancels. Measured
    against the exact Hoer-Love Greenhouse sum (this module): -5.6e-4
    (square section) and -6.5e-4 (4:1 section) at ``(w+t)/side = 1/100``,
    -1.3e-3 at 1/25, -1.6e-3 at 1/16.7. Test R3 gates at 1e-3 for
    ``(w+t)/side = 1/100``.
    """
    m = MU0 / (2.0 * math.pi) * (side * math.asinh(1.0) - math.sqrt(2.0) * side + side)
    return 4.0 * grover_bar(side, w, t) - 4.0 * m
