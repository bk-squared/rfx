"""Where a coaxial TEM line measures, computed from its band and higher modes.

KAPPA is the phase between neighbouring planes at freq_max, in radians.
TOLERANCE is the target reflection error at the reference plane in every bin.
JUNCTION_TOLERANCE_FRACTION is the share for the junction's higher modes.
SOURCE_TOLERANCE_FRACTION is the share for the source's own near field.
JUNCTION_AMPLITUDE bounds a higher mode relative to the incident TEM wave.
SOURCE_TE11_AMPLITUDE and SOURCE_TE21_AMPLITUDE are the modes left by the source.
TE21_RATE_FACTOR is the realized TE21 decay rate divided by its closed form.
BIN_COUNT frequencies span freq_min to freq_max; PLANE_COUNT planes measure them.
DECAY_LENGTHS bounds both searches in TE11 decay lengths at freq_max.
SPACING_ROUNDING keeps an exact multiple of dz from rounding up.
CLEARANCE_SCAN_FACTOR bounds the closed-form clearance in search-bound cells.
GAP_FLOOR_CELLS is the smallest source setback in cells.
LINE_EXTRA_CELLS covers three feed cells, one setback cell and source rounding.
RADIUS_RATIO_LIMIT bounds the outer to inner radius ratio of the cutoff scan.
LOAD_PHASES and LOAD_MAGNITUDES form the junction reflection grid.
TE11_PHASES and TE21_PHASES form the higher-mode phase grids.
CUTOFF_SCAN gives start, stop and step in k_c times the inner radius.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import jvp, yvp

from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.grid import C0


KAPPA = 0.54
TOLERANCE = 1e-2
JUNCTION_TOLERANCE_FRACTION = 0.8
SOURCE_TOLERANCE_FRACTION = 0.1
JUNCTION_AMPLITUDE = 1.0
SOURCE_TE11_AMPLITUDE = 1.44e-3
SOURCE_TE21_AMPLITUDE = 6.64e-3
TE21_RATE_FACTOR = 0.9313
BIN_COUNT = 41
PLANE_COUNT = 12
DECAY_LENGTHS = 40
SPACING_ROUNDING = 1e-9
CLEARANCE_SCAN_FACTOR = 50
GAP_FLOOR_CELLS = 2
LINE_EXTRA_CELLS = 5
RADIUS_RATIO_LIMIT = 100.0
LOAD_PHASES = 24
LOAD_MAGNITUDES = (1.0, 0.3)
TE11_PHASES = 12
TE21_PHASES = 6
CUTOFF_SCAN = (0.01, 12, 1e-3)

_ang = lambda n: np.exp(2j*np.pi*np.arange(n)/n)
_GL = np.concatenate([m*_ang(LOAD_PHASES) for m in LOAD_MAGNITUDES])
G, P1, P2 = [x.ravel() for x in np.meshgrid(
    _GL, _ang(TE11_PHASES), _ang(TE21_PHASES), indexing="ij")]


class CoaxPlaneRule(NamedTuple):
    """Cell offsets, higher-mode clearances and TE cutoffs of a coaxial line."""

    start: int
    spacing: int
    clearance_closed_form: int
    clearance_model: int
    clearance: int
    gap: int
    min_cells: int
    start_fallback: bool
    clearance_fallback: bool
    search_bound: int
    cutoff_wavenumber_te11: float
    cutoff_wavenumber_te21: float
    cutoff_frequency_te11: float


class _LineInputs(NamedTuple):
    dz: float
    frequencies: np.ndarray
    beta: np.ndarray
    alpha_te11: np.ndarray
    alpha_te21: np.ndarray
    spacing: int
    search_bound: int
    gap: int
    cutoff_wavenumber_te11: float
    cutoff_wavenumber_te21: float
    cutoff_frequency_te11: float


def _cutoff_wavenumber(inner, outer, m):
    cr = outer/inner
    start, stop, step = CUTOFF_SCAN
    x = np.arange(start, stop, step)
    fn = lambda t: jvp(m, t)*yvp(m, cr*t) - jvp(m, cr*t)*yvp(m, t)
    y = fn(x)
    ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
    roots = np.nonzero(ok & (np.sign(y[:-1]) != np.sign(y[1:])))[0]
    if not roots.size:
        raise RfxUnsupported(
            f"coaxial line: the cutoff scan from {start:g} to {stop:g} in k_c times the inner radius holds no TE{m}1 root for the outer to inner radius ratio {cr:.6g}, so the cutoff is not known; this line is not supported")
    i = roots[0]
    return brentq(fn, x[i], x[i + 1])/inner


def _fit_reflection(z, V):
    D = float(np.diff(z).mean())
    z0 = float(z.mean())
    h = np.sum((V[..., :-2] + V[..., 2:])*V[..., 1:-1].conj(), -1)/np.sum(abs(V[..., 1:-1])**2, -1)/2
    g = np.arccosh(h + 0j)/D
    g = np.where(g.imag < 0, g.conj(), g)
    e1 = np.exp(g[..., None]*(z - z0))
    e2 = np.exp(-g[..., None]*(z - z0))
    a11 = np.sum(abs(e1)**2, -1)
    a22 = np.sum(abs(e2)**2, -1)
    a12 = np.sum(e1.conj()*e2, -1)
    b1 = np.sum(e1.conj()*V, -1)
    b2 = np.sum(e2.conj()*V, -1)
    det = a11*a22 - abs(a12)**2
    A = (a22*b1 - a12*b2)/det
    B = (a11*b2 - a12.conj()*b1)/det
    return B*np.exp(g*z0)/(A*np.exp(-g*z0))


def _line_inputs(inner, outer, eps_r, dz, freq_max, freq_min, kappa):
    kc1 = _cutoff_wavenumber(inner, outer, 1)
    kc2 = _cutoff_wavenumber(inner, outer, 2)
    fc1 = kc1*C0/(2*np.pi*np.sqrt(eps_r))
    if freq_max >= fc1:
        raise RfxUnsupported(
            f"coaxial line: freq_max {freq_max/1e9:.6g} GHz is at or above the TE11 cutoff {fc1/1e9:.6g} GHz of this line (inner radius {inner*1e3:.6g} mm, outer radius {outer*1e3:.6g} mm, eps_r {eps_r:.6g}); only the TEM mode is measured, so lower freq_max below {fc1/1e9:.6g} GHz")
    f = np.linspace(freq_min, freq_max, BIN_COUNT)
    b = 2*np.pi*f*np.sqrt(eps_r)/C0
    a1 = np.sqrt(kc1**2 - b**2)
    a2 = np.sqrt(kc2**2 - b**2)
    D_c = int(np.ceil(kappa/b[-1]/dz - SPACING_ROUNDING))
    n_hi = int(np.ceil(DECAY_LENGTHS/(np.sqrt(kc1**2 - b[-1]**2)*dz)))
    g_s = max(GAP_FLOOR_CELLS, int(np.ceil(outer/dz)))
    return _LineInputs(dz, f, b, a1, a2, D_c, n_hi, g_s, kc1, kc2, fc1)


def _start_error(z, beta, alpha_te11, alpha_te21):
    V = (np.exp(1j*beta[:, None, None]*z)
         + G[:, None]*np.exp(-1j*beta[:, None, None]*z)
         + JUNCTION_AMPLITUDE*P1[:, None]*np.exp(-alpha_te11[:, None, None]*z)
         + JUNCTION_AMPLITUDE*P2[:, None]*np.exp(-alpha_te21[:, None, None]*z))
    return abs(_fit_reflection(z, V) - G).max(-1)


def _source_error(z, beta, alpha_te11, alpha_te21, source_position):
    b = beta[:, None, None]
    zs = source_position
    V = (np.exp(1j*b*z) + G[:, None]*np.exp(-1j*b*z)
         + SOURCE_TE11_AMPLITUDE*P1[:, None]*np.exp(-alpha_te11[:, None, None]*(zs - z))
         + SOURCE_TE21_AMPLITUDE*P2[:, None]*np.exp(-alpha_te21[:, None, None]*(zs - z)))
    return abs(_fit_reflection(z, V) - G).max(-1)


def _start_errors(inputs):
    b = inputs.beta
    a1 = inputs.alpha_te11
    a2 = TE21_RATE_FACTOR*inputs.alpha_te21
    search = "first-plane offset"

    def checked(n, value):
        if not np.all(np.isfinite(value)):
            raise RfxUnsupported(
                f"coaxial line: the plane fit error is not finite at {n} cells in the {search} search, so the layout cannot be judged; this line is not supported")
        return value

    def one(n):
        z = inputs.dz*(n + inputs.spacing*np.arange(PLANE_COUNT))
        return checked(n, float(_start_error(z, b[:1], a1[:1], a2[:1]).max()))

    def every(n):
        z = inputs.dz*(n + inputs.spacing*np.arange(PLANE_COUNT))
        return checked(n, _start_error(z, b, a1, a2))

    return one, every


def _source_errors(inputs, start):
    b = inputs.beta
    a1 = inputs.alpha_te11
    a2 = TE21_RATE_FACTOR*inputs.alpha_te21
    z = inputs.dz*(start + inputs.spacing*np.arange(PLANE_COUNT))
    search = "source clearance"

    def checked(n, value):
        if not np.all(np.isfinite(value)):
            raise RfxUnsupported(
                f"coaxial line: the plane fit error is not finite at {n} cells in the {search} search, so the layout cannot be judged; this line is not supported")
        return value

    def one(n):
        zs = z[-1] + n*inputs.dz
        return checked(n, float(_source_error(z, b[:1], a1[:1], a2[:1], zs).max()))

    def every(n):
        zs = z[-1] + n*inputs.dz
        return checked(n, _source_error(z, b, a1, a2, zs))

    return one, every


def _bisect(error, threshold, bound):
    if error(bound) > threshold:
        return None
    if error(0) <= threshold:
        return 0
    lo, hi = 0, bound
    while hi - lo > 1:
        m = (lo + hi)//2
        if error(m) > threshold:
            lo = m
        else:
            hi = m
    return hi


def _search(one, every, threshold, bound) -> tuple[int | None, bool]:
    n1 = _bisect(one, threshold, bound)
    if n1 is None:
        return None, False
    if every(n1).max() <= threshold:
        return n1, False
    return _bisect(lambda n: every(n).max(), threshold, bound), True


def _closed_form_clearance(inputs, start, junction_threshold, source_threshold):
    dz = inputs.dz
    a1t = float(inputs.alpha_te11[-1])
    a2t = TE21_RATE_FACTOR*float(inputs.alpha_te21[-1])
    g = a1t*start*dz - np.log(JUNCTION_AMPLITUDE/junction_threshold)
    cc = np.arange(0, CLEARANCE_SCAN_FACTOR*inputs.search_bound + 1)
    two = np.exp(g)*(SOURCE_TE11_AMPLITUDE*np.exp(-a1t*cc*dz)
                    + SOURCE_TE21_AMPLITUDE*np.exp(-a2t*cc*dz))
    ok = np.nonzero(two <= source_threshold)[0]
    return int(cc[ok[0]]) if ok.size else None


@functools.lru_cache(maxsize=128)
def _coax_plane_rule(inner, outer, eps_r, dz, freq_max, freq_min, kappa, tolerance):
    inputs = _line_inputs(inner, outer, eps_r, dz, freq_max, freq_min, kappa)
    D_c = inputs.spacing
    n_hi = inputs.search_bound
    if D_c < 1:
        raise RfxRefusal(
            f"coaxial line: kappa {kappa:.6g} rad gives a plane spacing of 0 cells at dz {dz*1e3:.6g} mm; use a larger kappa")
    phase = inputs.beta[-1]*D_c*dz
    if phase >= np.pi:
        raise RfxRefusal(
            f"coaxial line: the phase step between planes at freq_max is {phase:.6g} rad, at or above pi, so the plane recurrence cannot separate the wavenumber from its alias; use a smaller kappa or a finer dz")
    t_c = JUNCTION_TOLERANCE_FRACTION*tolerance
    t_s = SOURCE_TOLERANCE_FRACTION*tolerance
    one, every = _start_errors(inputs)
    s_c, start_fallback = _search(one, every, t_c, n_hi)
    if s_c is None:
        raise RfxUnsupported(
            f"coaxial line: no first-plane offset up to {n_hi} cells ({n_hi*dz*1e3:.6g} mm, {DECAY_LENGTHS} TE11 decay lengths at freq_max) keeps the fit error at or below {t_c:.6g}; use a larger tolerance=")
    c_a = _closed_form_clearance(inputs, s_c, t_c, t_s)
    if c_a is None:
        raise RfxUnsupported(
            f"coaxial line: the closed-form source clearance exceeds its scan bound of {CLEARANCE_SCAN_FACTOR*n_hi} cells; use a larger tolerance=")
    one, every = _source_errors(inputs, s_c)
    c_b, clearance_fallback = _search(one, every, t_s, n_hi)
    if c_b is None:
        raise RfxUnsupported(
            f"coaxial line: no source clearance up to {n_hi} cells ({n_hi*dz*1e3:.6g} mm) keeps the fit error at or below {t_s:.6g}; use a larger tolerance=")
    C_c = max(c_a, c_b)
    L_min = s_c + (PLANE_COUNT - 1)*D_c + C_c + inputs.gap + LINE_EXTRA_CELLS
    return CoaxPlaneRule(s_c, D_c, c_a, c_b, C_c, inputs.gap, L_min,
                         start_fallback, clearance_fallback, n_hi,
                         inputs.cutoff_wavenumber_te11, inputs.cutoff_wavenumber_te21,
                         inputs.cutoff_frequency_te11)


def coax_plane_rule(*, inner_radius: float, outer_radius: float, eps_r: float,
                    dz: float, freq_max: float, freq_min: float,
                    kappa: float = KAPPA, tolerance: float = TOLERANCE) -> CoaxPlaneRule:
    """TEM measurement planes with bounded junction and source higher-mode errors."""
    values = dict(inner_radius=inner_radius, outer_radius=outer_radius, eps_r=eps_r,
                  dz=dz, freq_max=freq_max, freq_min=freq_min, kappa=kappa,
                  tolerance=tolerance)
    for name, value in values.items():
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            raise RfxRefusal(
                f"coaxial line: {name} must be a concrete real number, because the plane layout is fixed before tracing; got {type(value).__name__}") from None
    for name in ("inner_radius", "outer_radius", "eps_r", "dz", "freq_max", "kappa"):
        value = values[name]
        if not np.isfinite(value) or value <= 0:
            raise RfxRefusal(
                f"coaxial line: {name} must be a finite positive number; got {value!r}")
    inner, outer = values["inner_radius"], values["outer_radius"]
    eps_r, dz = values["eps_r"], values["dz"]
    freq_max, freq_min = values["freq_max"], values["freq_min"]
    kappa, tolerance = values["kappa"], values["tolerance"]
    if outer <= inner:
        raise RfxRefusal(
            f"coaxial line: outer_radius must exceed inner_radius; got inner {inner*1e3:.6g} mm and outer {outer*1e3:.6g} mm")
    if not np.isfinite(freq_min) or freq_min <= 0 or freq_min >= freq_max:
        raise RfxRefusal(
            f"coaxial line: freq_min must satisfy 0 < freq_min < freq_max; got freq_min {freq_min:.6g} Hz and freq_max {freq_max:.6g} Hz; pass freq_min= inside that band")
    if not np.isfinite(tolerance) or tolerance <= 0 or tolerance >= 1:
        raise RfxRefusal(
            f"coaxial line: tolerance must satisfy 0 < tolerance < 1; got {tolerance!r}")
    if outer/inner >= RADIUS_RATIO_LIMIT:
        raise RfxUnsupported(
            f"coaxial line: the outer to inner radius ratio {outer/inner:.6g} is at or above {RADIUS_RATIO_LIMIT:g}, where the cutoff scan can miss the TE11 root; use a ratio below {RADIUS_RATIO_LIMIT:g}")
    return _coax_plane_rule(inner, outer, eps_r, dz, freq_max, freq_min, kappa, tolerance)
