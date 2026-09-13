"""Gates for the independent Greenhouse / Hoer-Love inductance referee
(``validation/crossval/comparators/spiral_greenhouse.py``).

Everything here is host numpy/scipy; no JAX, no x64 scope. The references
are written in this file from scratch (they do not call the comparator's
own helpers): the analytic double line integral of 1/R between parallel
filaments and a scipy adaptive double quadrature of that kernel over the
reduced cross-section coordinates, so the closed-form sextuple
antiderivative and the tensor Gauss-Legendre rule are judged by a different
route through the same Neumann integral.

Gate tolerances are the measured values with a stated margin; the task's
bands (1e-4 for R1/R2, 15 % for R5) are looser than what is asserted.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest
from scipy.integrate import dblquad

MU0 = 4e-7 * math.pi


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "validation/crossval/comparators/spiral_greenhouse.py"
    spec = importlib.util.spec_from_file_location("spiral_greenhouse_referee", path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sg = _load()
S = sg.Segment


# --------------------------------------------------------------------------
# Independent Neumann references (written here, not imported)
# --------------------------------------------------------------------------

def _g(u: float, d: float) -> float:
    return u * math.asinh(u / d) - math.hypot(u, d)


def ref_filament(d: float, l1: float, l2: float, s: float) -> float:
    """mu0/4pi int_0^l1 dz int_s^{s+l2} dz' / sqrt((z-z')^2 + d^2), done by hand:
    the inner integral is asinh((z'-z)/d), the outer one of asinh is G."""
    return MU0 / (4 * math.pi) * (_g(s + l2, d) - _g(s + l2 - l1, d) - _g(s, d) + _g(s - l1, d))


def _overlap(u: float, w1: float, w2: float, e: float) -> float:
    """Length of {x in [0, w1] : x + u in [e, e + w2]} -- the density of the
    transverse difference u = x' - x between two intervals."""
    return max(0.0, min(w1, e + w2 - u) - max(0.0, e - u))


def ref_neumann(l1, w1, t1, l2, w2, t2, dw, dt, dl, epsrel=1e-10):
    """Neumann integral of two bars by scipy quadrature: the 4-D integral
    over both cross-sections of the filament mutual depends only on the
    transverse differences (u, v), whose joint density is the product of
    two trapezoids, so it reduces to a 2-D integral. The (u, v) domain is
    split at every kink of the density and at u = 0, v = 0 so that the log
    singularity of coincident filaments sits on cell corners, where the
    adaptive Gauss-Kronrod rule handles it. Returns (value, error estimate).
    """
    bu = sorted({dw - w1, dw, dw + w2 - w1, dw + w2, 0.0})
    bu = [b for b in bu if dw - w1 <= b <= dw + w2]
    bv = sorted({dt - t1, dt, dt + t2 - t1, dt + t2, 0.0})
    bv = [b for b in bv if dt - t1 <= b <= dt + t2]

    def f(v, u):
        return ref_filament(math.hypot(u, v), l1, l2, dl) * _overlap(u, w1, w2, dw) * _overlap(v, t1, t2, dt)

    tot = err = 0.0
    for ua, ub in zip(bu[:-1], bu[1:]):
        for va, vb in zip(bv[:-1], bv[1:]):
            r, e = dblquad(f, ua, ub, va, vb, epsabs=0.0, epsrel=epsrel)
            tot += r
            err += e
    return tot / (w1 * t1 * w2 * t2), err / (w1 * t1 * w2 * t2)


# --------------------------------------------------------------------------
# R0: the filament kernel itself against a brute-force double quadrature of 1/R
# --------------------------------------------------------------------------

def test_r0_filament_formula_vs_double_quadrature():
    d, l1, l2, s = 0.3, 1.0, 0.7, 0.4
    num, err = dblquad(lambda zp, z: 1.0 / math.hypot(z - zp, d), 0.0, l1, s, s + l2,
                       epsabs=0.0, epsrel=1e-12)
    num *= MU0 / (4 * math.pi)
    ana = ref_filament(d, l1, l2, s)
    # measured |rel| = 1e-15; quadrature error estimate 1e-13
    assert abs(num - ana) / abs(ana) < 1e-10
    assert abs(sg.filament_mutual(d, l1, l2, s) - ana) / abs(ana) < 1e-14


# --------------------------------------------------------------------------
# R1: self-inductance of a bar vs Neumann quadrature (three aspect ratios)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("length, w, t, grover_tol", [
    (1.0, 0.1, 0.05, 6e-3),     # l/w = 10, w/t = 2, l/(w+t) = 6.7
    (1.0, 0.01, 0.002, 1e-3),   # l/w = 100, w/t = 5 (spiral-like), l/(w+t) = 83
    (0.3, 0.1, 0.1, 5e-2),      # square section, l/w = 3, l/(w+t) = 1.5
])
def test_r1_self_inductance_vs_neumann(length, w, t, grover_tol):
    ref, err = ref_neumann(length, w, t, length, w, t, 0.0, 0.0, 0.0)
    val = sg.self_inductance_bar(length, w, t)
    rel = abs(val - ref) / abs(ref)
    # measured rel: 2.8e-14, 1.2e-8, 7.8e-15 with quadrature error estimates
    # 2.2e-9, 2.6e-8, 2.1e-9 -- the second case is quadrature-limited.
    # Gate 1e-6 (40x the largest quadrature estimate); task band 1e-4.
    assert rel < 1e-6, (val, ref, rel, err)
    # the published Grover/Greenhouse bar formula is a GMD/AMD expansion in
    # (w+t)/l; measured deviation from the exact value: +4.1e-3 (l/(w+t)=6.7),
    # +6.2e-4 (83), +3.4e-2 (1.5) -- per-case tolerance 1.5x that
    assert abs(sg.grover_bar(length, w, t) - val) / val < grover_tol


# --------------------------------------------------------------------------
# R2: mutual inductance of parallel bars vs Neumann quadrature
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, args", [
    ("side by side, same length", (1.0, 0.1, 0.05, 1.0, 0.1, 0.05, 0.3, 0.0, 0.0)),
    ("partial overlap, all offsets", (1.0, 0.1, 0.05, 0.7, 0.08, 0.03, 0.05, 0.02, 0.4)),
    ("end to end, collinear", (1.0, 0.1, 0.05, 0.5, 0.1, 0.05, 0.0, 0.0, 1.0)),
    ("stacked on another layer", (1.0, 0.1, 0.05, 1.0, 0.1, 0.05, 0.0, 0.2, 0.0)),
    ("spiral far pair (auto -> GL)", (300e-6, 15e-6, 2e-6, 260e-6, 15e-6, 2e-6, 250e-6, 0.0, 20e-6)),
])
def test_r2_mutual_inductance_vs_neumann(name, args):
    ref, err = ref_neumann(*args)
    hl, e_hl = sg.mutual_inductance_bars(*args, method="hoer_love", return_error=True)
    auto, e_auto = sg.mutual_inductance_bars(*args, return_error=True)
    # measured Hoer-Love rel: 5.6e-13, 1.2e-13, 1.9e-12, 9.6e-14, 1.6e-11
    # (quadrature estimates <= 1.9e-9); gate 1e-6, task band 1e-4
    assert abs(hl - ref) / abs(ref) < 1e-6, (name, hl, ref, err)
    assert abs(auto - ref) / abs(ref) < 1e-6, (name, auto, ref, err)
    assert e_hl < 1e-6 and e_auto < 1e-6


def test_r2b_gauss_legendre_converged_and_auto_rule():
    # marginal case of the switching rule: cross-section gap == largest cross-section dimension
    marginal = (1.0, 0.1, 0.05, 0.8, 0.1, 0.05, 0.2, 0.02, 0.3)
    g8, g16, g32 = (sg._gauss_legendre_bars(*marginal, n) for n in (8, 16, 32))
    ref, _ = ref_neumann(*marginal)
    # measured: 8->16 change 3.3e-16, 16->32 change 0.0, GL16 vs quadrature 0.0 (quad est 3e-13)
    assert abs(g16 - g8) / abs(g16) < 1e-10
    assert abs(g32 - g16) / abs(g16) < 1e-12
    assert abs(g16 - ref) / abs(ref) < 1e-9
    assert sg.cross_section_gap(0.1, 0.05, 0.1, 0.05, 0.2, 0.02) == pytest.approx(0.1)

    # where Hoer-Love collapses: 1 mm x 20 um x 2 um bar and its image 0.2 m away
    far = (1e-3, 20e-6, 2e-6, 1e-3, 20e-6, 2e-6, 0.0, -0.2, 0.0)
    ref, _ = ref_neumann(*far)
    hl, e_hl = sg.mutual_inductance_bars(*far, method="hoer_love", return_error=True)
    auto, e_auto = sg.mutual_inductance_bars(*far, return_error=True)
    # measured: Hoer-Love rel error +1.1e6 with its own estimate 0.6 (flagged);
    # auto (Gauss-Legendre) rel -2.3e-12 with estimate 4e-13
    assert abs(hl - ref) / abs(ref) > 1.0 and e_hl > 1e-3
    assert abs(auto - ref) / abs(ref) < 1e-8 and e_auto < 1e-9
    assert ref == pytest.approx(MU0 / (4 * math.pi) * (1e-3) ** 2 / 0.2, rel=2e-5)  # dipole limit l^2/d


@pytest.mark.parametrize("geom", [
    (0.4, 0.02, 0.01, 0.4, 0.02, 0.01, 0.0, 0.0, 0.3),      # coincident sections (gap -0.01)
    (1.0, 0.1, 0.05, 1.0, 0.1, 0.05, 0.05, 0.0, 0.0),       # partial overlap (gap -0.05)
    (1.0, 0.1, 0.05, 1.0, 0.1, 0.05, 0.1, 0.0, 0.0),        # touching faces (gap 0)
])
def test_gauss_legendre_refuses_overlapping_sections(geom):
    """Forcing the filament rule where the 1/R kernel is singular inside the
    domain used to return NaN (node on d = 0) or an unconverged finite value
    with divide-by-zero warnings; it now raises. 'auto' and 'hoer_love' still
    handle the same geometries and agree with the independent quadrature."""
    with pytest.raises(ValueError, match="separated cross-sections"):
        sg.mutual_inductance_bars(*geom, method="gauss_legendre")
    ref, est = ref_neumann(*geom)
    auto = sg.mutual_inductance_bars(*geom)
    # measured rel (auto = Hoer-Love on all three): -2.8e-11 (coincident,
    # Hoer-Love self-estimate 5.9e-11), 3.6e-15, -5.8e-14; quadrature
    # estimates <= 1.1e-15. Gate 1e-8 (task band 1e-4).
    assert abs(auto - ref) / abs(ref) < 1e-8, (auto, ref, est)


# --------------------------------------------------------------------------
# Segment bookkeeping: signs, perpendicular pairs, orientation independence
# --------------------------------------------------------------------------

def test_segment_signs_and_orientation():
    w, t = 0.01, 0.002
    a, b = S((0, 0, 0), (1, 0, 0), w, t), S((0, 0.1, 0), (1, 0.1, 0), w, t)
    m = sg.mutual_inductance_bars(1, w, t, 1, w, t, 0.1, 0, 0)
    assert sg.pair_mutual(a, b) == pytest.approx(m)
    assert sg.pair_mutual(a, S((1, 0.1, 0), (0, 0.1, 0), w, t)) == pytest.approx(-m)  # reversed
    assert sg.pair_mutual(a, S((0, 0.1, 0), (1, 0.1, 0), w, t, current=-1.0)) == pytest.approx(-m)
    assert sg.pair_mutual(a, S((0.5, -1, 0), (0.5, 1, 0), w, t)) == 0.0  # perpendicular
    ls = sg.self_inductance_bar(1, w, t)
    assert sg.greenhouse_inductance([a, b]) == pytest.approx(2 * ls + 2 * m)
    assert sg.greenhouse_inductance([a, S((1, 0.1, 0), (0, 0.1, 0), w, t)]) == pytest.approx(2 * ls - 2 * m)
    with pytest.raises(ValueError):
        sg.segment_axis(S((0, 0, 0), (0, 0, 1), w, t))
    # a closed loop traversed the other way round has the same inductance
    loop = [S((1, -1, 0), (1, 1, 0), w, t), S((1, 1, 0), (-1, 1, 0), w, t),
            S((-1, 1, 0), (-1, -1, 0), w, t), S((-1, -1, 0), (1, -1, 0), w, t)]
    rev = [S(s.end, s.start, w, t) for s in loop[::-1]]
    assert sg.greenhouse_inductance(rev) == pytest.approx(sg.greenhouse_inductance(loop), rel=1e-12)


# --------------------------------------------------------------------------
# R3: closed square loop vs the published Grover/Greenhouse closed form
# --------------------------------------------------------------------------

@pytest.mark.parametrize("w, t", [(0.005, 0.005), (0.008, 0.002)])
def test_r3_square_loop_vs_published_closed_form(w, t):
    """Cross-check against a published closed form, not a field reference:
    Grover's loop formula is itself 4 x (approximate bar self) - 4 x (filament
    mutual of opposite sides) with centreline lengths meeting at the corners,
    i.e. the same Greenhouse decomposition. The residual therefore measures
    the Grover bar approximation (GMD/AMD expansion) against the exact
    Hoer-Love bar, and this test does not validate the corner convention
    (see the module docstring's scope fence)."""
    side = 1.0
    a = side / 2
    loop = [S((a, -a, 0), (a, a, 0), w, t), S((a, a, 0), (-a, a, 0), w, t),
            S((-a, a, 0), (-a, -a, 0), w, t), S((-a, -a, 0), (a, -a, 0), w, t)]
    res = sg.greenhouse_terms(loop)
    ref = sg.square_loop_closed_form(side, w, t)
    rel = (res.total - ref) / ref
    # Source: Grover 1946 ch. 8 rectangle formula with the rectangular-section
    # GMD substitution (= Greenhouse 1974 eqs. (3)-(5)); its bar term is a
    # GMD/AMD expansion valid for l >> (w + t). Measured residual at
    # (w+t)/side = 1/100: -5.6e-4 (square section), -6.5e-4 (4:1 section);
    # gate 1e-3 (1.5x the larger). The mutual part is -3.737e-7 H of 4.27e-6.
    assert abs(rel) < 1e-3, (res, ref, rel)
    assert res.mutual_sum < 0 and res.self_sum > 0 and res.worst_error < 1e-8


# --------------------------------------------------------------------------
# R4: PEC ground by images -- monotone in h, free-space limit
# --------------------------------------------------------------------------

def test_r4_image_method_monotone_and_limit():
    bar = [S((0, 0, 0), (1e-3, 0, 0), 20e-6, 2e-6)]
    l0 = sg.greenhouse_inductance(bar)
    hs = [2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6, 1e-3, 1e-2, 1e-1, 1.0]
    ls = [sg.greenhouse_inductance(bar, ground_height=h) for h in hs]
    ratios = np.array(ls) / l0
    # measured ratios: 0.0802, 0.1821, 0.2908, 0.4152, 0.5832, 0.7025, 0.8066,
    # 0.9068, 0.9511, 0.99502, 0.99950, 0.999950 (free space 1.00290e-9 H)
    assert l0 == pytest.approx(1.00290e-9, rel=1e-4)
    assert np.all(np.diff(ratios) > 0) and np.all(ratios > 0) and np.all(ratios < 1)
    assert ratios[0] == pytest.approx(0.0802, abs=2e-4)
    assert ratios[-1] == pytest.approx(1.0, abs=1e-4)
    # h -> large: image coupling tends to the filament value mu0/4pi * l^2 / (2h)
    m_img = l0 - ls[-1]
    assert m_img == pytest.approx(MU0 / (4 * math.pi) * (1e-3) ** 2 / 2.0, rel=1e-4)
    with pytest.raises(ValueError):
        sg.greenhouse_inductance(bar, ground_height=0.0)


# --------------------------------------------------------------------------
# Spiral geometry: same centreline as rfx.fdfd.gds.rect_spiral (rfx imported
# only inside this test, so the referee module stays rfx-free)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n, r_out, w, s, lead", [
    (2, 150e-6, 15e-6, 5e-6, 0.0), (3, 150e-6, 15e-6, 5e-6, 30e-6),
    (1, 100e-6, 10e-6, 10e-6, 0.0), (2, 200e-6, 30e-6, 14e-6, 50e-6),
])
def test_spiral_geometry_matches_rfx_rect_spiral(n, r_out, w, s, lead):
    from rfx.fdfd.gds import rect_spiral  # the only rfx import in this file

    sp = rect_spiral(n, r_out, w, s, lead=lead)
    cl = sg.rect_spiral_centreline(n, r_out, w, s, lead)
    assert cl.shape == sp.centreline.shape
    # measured max |diff| = 0.0 (1.4e-20 for the lead case)
    assert np.abs(cl - sp.centreline).max() < 1e-18
    segs = sg.rect_spiral_segments(n, r_out, w, s, 2e-6, lead, underpass=(3e-6, 2e-6))
    length = sum(abs(np.subtract(q.end, q.start)).sum() for q in segs[:-1])
    assert length == pytest.approx(sp.length, rel=1e-14)
    assert abs(np.subtract(segs[-1].end, segs[-1].start)).sum() == pytest.approx(sp.underpass_length, rel=1e-12)
    assert segs[-1].start[2] == -3e-6 and segs[-1].thickness == 2e-6
    for q in segs:
        sg.segment_axis(q)  # all axis-aligned
    with pytest.raises(ValueError):
        sg.rect_spiral_centreline(10, r_out, w, s)


# --------------------------------------------------------------------------
# R5: spiral Greenhouse vs Mohan modified Wheeler (sanity band)
# --------------------------------------------------------------------------

_R5_GEOM = (150e-6, 15e-6, 5e-6, 2e-6)  # r_out, w, s, t


def _r5_case(n):
    r_out, w, s, t = _R5_GEOM
    segs = sg.rect_spiral_segments(n, r_out, w, s, t)
    d_out = 2 * r_out
    d_in = d_out - 2 * n * w - 2 * (n - 1) * s
    return segs, sg.greenhouse_terms(segs), d_out, d_in


@pytest.mark.parametrize("n", [2, 3])
def test_r5_spiral_vs_mohan_wheeler(n):
    """Physics gate: Greenhouse total vs Mohan's modified-Wheeler estimate
    (an independent empirical fit, quoted accuracy a few percent) inside the
    task's 15 % sanity band. Nothing here is compared with the module's own
    output; the regression pins live in the next test."""
    segs, res, d_out, d_in = _r5_case(n)
    r_out, w, s, t = _R5_GEOM
    mw = sg.mohan_wheeler(n, d_out, d_in, w, s)
    rel = (res.total - mw) / mw
    # measured: n=2 GH 2.2967 nH vs Mohan 2.2865 nH (+0.4 %);
    #           n=3 GH 3.9085 nH vs Mohan 4.0090 nH (-2.5 %). Sanity band 15 %.
    assert abs(rel) < 0.15, (res.total, mw, rel)
    assert res.worst_error < 1e-8  # measured 1.6e-10 (n=2), 6.9e-11 (n=3)
    # ground plane reduces it (measured h = 10 um: 0.8490 nH / 1.2766 nH)
    assert sg.greenhouse_inductance(segs, ground_height=10e-6) < 0.5 * res.total
    with pytest.raises(ValueError):
        sg.mohan_wheeler(n, d_out, d_in, w, 4 * w)


@pytest.mark.parametrize("n, pinned_gh_nH, pinned_mw_nH", [
    (2, 2.2967, 2.2865),
    (3, 3.9085, 4.0090),
])
def test_r5_regression_pins(n, pinned_gh_nH, pinned_mw_nH):
    """Regression guard only (not a physics gate): the Greenhouse totals are
    the module's own outputs recorded when the R1/R2/R4 gates and the
    reviewer's independent bookkeeping agreed; the Mohan values were
    reproduced by hand-evaluating the published formula."""
    _, res, d_out, d_in = _r5_case(n)
    _, w, s, _ = _R5_GEOM
    assert res.total * 1e9 == pytest.approx(pinned_gh_nH, rel=1e-4)
    assert sg.mohan_wheeler(n, d_out, d_in, w, s) * 1e9 == pytest.approx(pinned_mw_nH, rel=1e-4)
