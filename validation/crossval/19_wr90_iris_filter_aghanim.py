"""WR-90 4th-order inductive-iris bandpass filter vs TEn0 mode-matching (item 3, S3).

The multi-iris case that stage S1 (`18_wr90_iris_modematch.py`, merged) exists to
make possible: five symmetric inductive irises forming four coupled cavities, i.e.
a resonant structure where a per-face geometry error is a PASSBAND SHIFT rather
than a magnitude tolerance.

PUBLISHED DESIGN — Aghanim, Zbitou, Errkik, Tajmouati, Latrach, "Design and
simulation of a bandpass filter based on inductive irises in rectangular
waveguide", E3S Web of Conferences 351, 01059 (2022), open access CC BY 4.0,
DOI 10.1051/e3sconf/202235101059.  Table 6 (optimized), WR-90 a = 22.86 mm,
b = 10.16 mm, iris thickness t = 2.00 mm (all five), apertures
d = [10.27, 6.65, 6.18, 6.65, 10.27] mm, cavities
l = [14.29, 15.73, 15.73, 14.29] mm.  Their Table 5 pre-optimization set is
explicitly reported as FAILING the spec and is not what this case builds.

REFERENCE POSTURE — three distinct comparisons, kept separate on purpose:

  (1) GATED: rfx vs the mode-matching oracle evaluated on the AS-RASTERIZED
      geometry — same apertures, same snapped cavity lengths, same aperture
      offsets rfx actually built.  This is a solver-accuracy comparison on
      identical geometry, exactly the S1 posture.
  (2) REPORTED: the oracle on the NOMINAL (paper) geometry vs the oracle on the
      as-rasterized geometry — our own snap, quantified rather than hidden.
  (3) REPORTED ANCHOR: the oracle on nominal geometry vs the paper's digitized
      curve.  The paper's own two solvers (HFSS, CST) disagree by 21.9 MHz in
      centre frequency while agreeing on bandwidth to 0.4 MHz, so 21.9 MHz is
      the reference's own uncertainty and nothing tighter is supportable.
      Measured (interpolated -10 dB crossings): the oracle sits 6.2 MHz from
      CST and 28.2 MHz from HFSS, i.e. inside that spread on CST.

WHAT IS GATED vs WHAT IS REPORTED
---------------------------------------------------------------------------
GATED (exit-1), gate = round-UP(measured envelope x 1.5) over a NINE-config
population, enforced as EXACT equality by the --write-fixture self-check:
  * centre frequency f0 of the -10 dB |S11| span, rfx vs oracle@as-realized,
    at the gated mesh a/90.  f0 is the least convention-sensitive continuous
    observable (~2.4 MHz per cell of the unsettled iris-thickness leg, vs
    ~40 MHz for bandwidth and 22-30 MHz for individual edges), which is why
    it carries the gate and the edges/bandwidth do not.
  * structural reflection-zero COUNT inside the passband (a depth-independent
    local-minimum count).  This is the topology check.
REPORTED, NOT GATED:
  * band edges and bandwidth: their comparator-input uncertainty from the
    iris-thickness convention (about half a cell) exceeds any defensible gate
    on them, and d_bw is identically d_hi - d_lo, so they are one fact.
  * worst in-band return loss.  The reference is NOT self-consistent here:
    HFSS ripple peaks are -19.3/-14.9/-18.4 dB and CST's are
    -24.9/-18.7/-14.2 dB, and the two tools disagree on WHICH peak is worst.
  * individual ripple-peak levels, and every reflection-zero DEPTH.  Four
    nominally identical equiripple zeros bottom out across a 16 dB spread in
    the published figure, which proves the paper's frequency step — not
    physics — sets those depths.  Zero FREQUENCIES are meaningful; zero
    depths are not values.
  * the coarse diagnostic mesh a/60, whose snap moves f0 by +120.2 MHz and
    destroys two of the four reflection zeros.  Committed as data because it
    is the evidence that the gated mesh had to be a/90.
  * phase / group delay: not claimed (magnitude-only lane posture).
FENCED: nothing here promotes the waveguide-obstacle lane beyond what S1
established.  Multi-iris filters remain EXPERIMENTAL per
docs/guides/support_matrix.md; this case measures one published design against
an analytic oracle, it does not certify arbitrary filters.

GEOMETRY DISCIPLINE (inherited from S1, extended)
  * Mesh a/90 (dx = 0.254 mm) is the GATED rung.  It was chosen by
    measurement, not convenience: at a/60 the snap moves f0 +120.2 MHz (33% of
    the passband) and drops the reflection-zero count 4 -> 2; at a/90 it moves
    f0 +3.3 MHz (inside the reference's own 21.9 MHz CST-vs-HFSS spread) and
    the count is 4 -> 3, with BW 18.2 MHz narrow.  The remaining lost zero is
    recorded, not glossed: the rasterized geometry is a SNAPPED version of the
    Aghanim filter — its centre frequency is the paper's to within the
    reference's solver scatter, but its ripple structure is perturbed.
    Both snap figures are quoted for the COMPENSATED cell counts (see
    rasterized_geometry); the uncompensated counts move f0 -101.4 MHz at a/90.
  * At a/90 every aperture lands on an EVEN cell count (40/26/24/26/40), so
    symmetric fins are realizable with zero offset.  That is luck of this
    geometry, not a rule: the realizable electrical aperture is
    (cells - 2*fin_c)*dx, whose parity follows `cells`, so an odd-cell
    aperture on an even-cell guide is NOT representable symmetrically.  The
    a/60 diagnostic rung hits exactly that case (27/17/16 cells) and therefore
    uses asymmetric fin placement.  The qualitative reason is parity: with
    symmetric fins the realizable aperture is (cells - 2*fin_c)*dx, whose parity
    follows `cells`, so an odd-cell aperture is not representable symmetrically
    at all.  An earlier version of this docstring claimed a measured "4 zeros vs
    2" advantage for asymmetric placement; that number came from a scratch run
    predating both the cell-count compensation and the interpolated band edges,
    it is contradicted by the committed a/60 rung (2 zeros in BOTH rfx and its
    oracle), and no symmetric-a/60 variant is committed anywhere.  It is
    withdrawn rather than restated.
  * Fin corners sit half a cell off the node planes (the S1 midpoint recipe).
    Drawing to the nominal dimension instead leaves the electrical aperture
    one to two cells too wide, and WHICH is not predictable from the nominal
    dimensions: node coordinates are built in float32 as f32(f32(i)*f32(dx))
    while box corners arrive as an f64 value cast once, so algebraically equal
    values land on opposite sides of the comparison (issue #493).
  * Guide height is REDUCED to 4 cells.  For TE10 with y-invariant inductive
    fins the S-matrix is b-independent, and this is a measured witness, not an
    assumption: b = 13/8/4/2 cells give |S11| identical to 0.00000 (bit-level)
    on the S1 single-iris geometry.  That is an 8x cell-count saving and it is
    what makes this case local-CPU affordable.
  * CPML depth = 0.75 * lambda_g at the low band edge (issue #494 / the S1
    rule): 0.5 lambda_g removes only about half the residual ripple.

Usage:
  python validation/crossval/19_wr90_iris_filter_aghanim.py            # gated set
  python validation/crossval/19_wr90_iris_filter_aghanim.py --write-fixture
      # + coarse diagnostic rung; ring-down, b-invariance, feed-clearance and
      # absorber-depth witnesses (each axis with an interior sample); the
      # formulation-independent 2-D H-plane FDFD check (three levels
      # r=2,3,4, both Richardson estimates and their consistency witness);
      # snap decomposition and the
      # paper anchor; regenerates
      # validation/crossval/_19_iris_filter_results/rfx.json AND
      # tests/fixtures/wr90_iris_filter/fixture.json (+ the artifact under _19_iris_filter_results/)

Exit codes: 0 = all configured gates passed; 1 = an oracle self-check, a raster
assert, or a gate failed.  Failure prints "SOME CHECKS FAILED".
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box, rasterize  # noqa: E402

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "comparators"))
import fdfd_hplane  # noqa: E402  (numpy/scipy only, zero rfx dependency)

C0 = 299792458.0
MU0 = 4e-7 * np.pi

# --- Aghanim 2022 Table 6 (optimized) --------------------------------------
A_WR90 = 22.86e-3
APERTURES_NOM = np.array([10.27, 6.65, 6.18, 6.65, 10.27]) * 1e-3
CAVITIES_NOM = np.array([14.29, 15.73, 15.73, 14.29]) * 1e-3
T_IRIS_NOM = 2.00e-3

# --- reference scalars digitized from the paper's Fig. 5 -------------------
# (calibration 2.234187 MHz/px RMS 0.547 MHz, -0.2135972 dB/px RMS 0.078 dB;
#  zero frequencies are calibration-invariant by construction, worst-RL moves
#  <= 0.13 dB across three independent calibrations)
PAPER = {
    "hfss": {"lo": 10.8064e9, "hi": 11.1603e9, "f0": 10.9834e9, "bw": 354e6,
             "worst_rl_db": 14.9, "zeros_ghz": [10.8307, 10.9066, 11.0519, 11.1323]},
    "cst": {"lo": 10.7847e9, "hi": 11.1382e9, "f0": 10.9614e9, "bw": 354e6,
            "worst_rl_db": 14.2, "zeros_ghz": [10.8173, 10.8754, 10.9982, 11.1144]},
    "solver_spread_f0_hz": 21.9e6,
    "solver_spread_bw_hz": 0.4e6,
}

FREQS = np.linspace(10.40e9, 11.70e9, 131)      # 10 MHz — the digitized grid
GATED_CELLS = 90
COARSE_CELLS = 60
B_CELLS = 4
FEED_CELLS = 40
PORT_CELLS = 12
CPML_FRACTION = 0.75                             # of lambda_g at the low edge
# CPML is EXTERIOR to the requested domain in rfx (measured: a 200-cell domain
# yields 200 + 2*cpml_layers + 1 nodes), so FEED_CELLS is clearance between the
# absorber interface and the first iris, not a budget the absorber eats into.
# The port then sits PORT_CELLS*dx = 3.05 mm from that interface at a/90, about
# 0.08*lambda_g, where the S1 recipe used roughly 1*lambda_g — so the clearance
# is witnessed by re-running the gate geometry with a generous feed rather than
# assumed adequate (see the feed-clearance witness in main).
FEED_CELLS_WITNESS = 100
PORT_CELLS_WITNESS = 60
CPML_FRACTION_WITNESS = 1.25
# INTERIOR samples. Two endpoints per axis cannot detect non-monotonic
# sensitivity; PR #475 was exactly that failure (three sampled clearances
# passed while 9 of 13 exceeded the gate). Each setup axis therefore gets a
# sample strictly between the gated value and the generous/deep one.
FEED_CELLS_MID = 70
PORT_CELLS_MID = 30
CPML_FRACTION_MID = 1.0

GATE_F0_MHZ = 19.0        # centre-frequency agreement, rfx vs oracle@as-realized
# = ceil(12.1230 x 1.5), the envelope over the NINE-configuration population
# (four setup axes, each with an interior sample as well as an endpoint).
# When the interior samples were first added this constant was a PREDICTION
# from the earlier five-configuration envelope; the nine-configuration
# regeneration then measured every interior sample inside its endpoint range
# (envelope unchanged at 12.1230), so it is now a measurement. If a future
# regeneration reports an envelope/gate mismatch, that IS a non-monotonicity
# finding and the constant must be raised to the measured value -- it is a
# result, not a nuisance. PR #475 is the precedent: three sampled clearances
# passed while 9 of 13 exceeded the gate. That tightness is the point: the
# residual is a reproducible systematic difference, not a setup artifact, and it
# corresponds to about 0.12 cell of cavity length at -105 MHz/cell. The
# write-fixture self-check demands exact equality with ceil(env x 1.5).
# Band edges and bandwidth are REPORTED, not gated: see the gated block in main
# for the measured sensitivity that decides which observable can carry a gate.


# --------------------------------------------------------------------------- #
# Oracle: TEn0 mode-matching cascade with arbitrary aperture position.
# The centred limit reproduces the merged single-iris oracle of case 18 to
# 1.05e-04 in the ODD-MODE formulation of the frozen gate test (a 1.8e-16
# agreement is obtainable only by calling this same cascade with N=1, which
# proves nothing), so this inherits S1's validation (including the PR #480 review's
# formulation-independent FDFD confirmation at 5.8e-4).
# --------------------------------------------------------------------------- #
def _gamma(n, width, k):
    # GUARD, not a note: at exact cutoff (k == n*pi/width) the argument
    # vanishes, gamma is 0, the modal admittance is 0, and the power
    # normalisation divides by sqrt(0). Unreachable at this case's band
    # edges, but a future band or guide width could land there, and a
    # silent nan is worse than a stop. Raise rather than warn.
    arg = (n * np.pi / width) ** 2 - k * k
    if abs(arg) < 1e-6 * max((n * np.pi / width) ** 2, k * k):
        raise ValueError(
            f"mode n={n} is at exact cutoff for width={width:.6e} at "
            f"k={k:.6e}: the modal admittance vanishes and the power "
            "normalisation is singular. Shift the frequency grid or the "
            "guide width off the cutoff.")
    return np.sqrt(complex(arg))


def _overlap(a, d, x0, n, m):
    al, be = n * np.pi / a, m * np.pi / d

    def iss(p, q, L):
        if abs(p - q) < 1e-30:
            return L / 2 - np.sin(2 * p * L) / (4 * p)
        return (np.sin((p - q) * L) / (p - q) - np.sin((p + q) * L) / (p + q)) / 2

    def ics(p, q, L):
        if abs(p - q) < 1e-30:
            return (1 - np.cos(2 * q * L)) / (4 * q) if q > 0 else 0.0
        return ((1 - np.cos((q + p) * L)) / (q + p)
                + (1 - np.cos((q - p) * L)) / (q - p)) / 2

    return (np.sqrt(2 / a) * np.sqrt(2 / d)
            * (np.cos(al * x0) * iss(al, be, d) + np.sin(al * x0) * ics(al, be, d)))


def _step(a, d, x0, k, n_a, n_b):
    Na, Nb = np.arange(1, n_a + 1), np.arange(1, n_b + 1)
    gA = np.array([_gamma(n, a, k) for n in Na])
    gB = np.array([_gamma(m, d, k) for m in Nb])
    w = k * C0
    YA, YB = gA / (1j * w * MU0), gB / (1j * w * MU0)
    C = np.array([[_overlap(a, d, x0, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + C.T @ YAd @ C)
    T_ba = 2 * Minv @ C.T @ YAd
    R_aa = C @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - C.T @ YAd @ C)
    T_ab = C @ (np.eye(n_b) + R_bb)
    # DECORATIVE for this script's outputs, and verified so: sqrt(Y) is an
    # exact diagonal gauge that cancels in the reported (0,0) entries --
    # substituting an arbitrary random complex diagonal changes |S11| by
    # <= 1.5e-15, and flipping the branch on the evanescent modes changes it
    # by exactly 0. Do NOT 'fix' this believing the physics changes; it would
    # matter only for off-diagonal or higher-mode entries, which nothing here
    # reads. (Independent review, 2026-07-31.)
    sA, sB = np.sqrt(YA), np.sqrt(YB)
    return ((sA[:, None] * R_aa) / sA[None, :], (sA[:, None] * T_ab) / sB[None, :],
            (sB[:, None] * T_ba) / sA[None, :], (sB[:, None] * R_bb) / sB[None, :])


def _line(width, length, k, n):
    g = np.array([_gamma(m, width, k) for m in np.arange(1, n + 1)])
    P = np.diag(np.exp(-g * length))
    z = np.zeros((n, n), dtype=complex)
    return (z, P, P, z)


def _star(sa, sb):
    A11, A12, A21, A22 = sa
    B11, B12, B21, B22 = sb
    n = A22.shape[0]
    i1 = np.linalg.inv(np.eye(n) - A22 @ B11)
    i2 = np.linalg.inv(np.eye(n) - B11 @ A22)
    return (A11 + A12 @ B11 @ i1 @ A21, A12 @ i2 @ B12,
            B21 @ i1 @ A21, B22 + B21 @ A22 @ i2 @ B12)


def _iris(a, d, x0, t, k, n_a, nb_scale=1.0):
    # nb_scale exists so the APERTURE mode count can be scanned independently
    # of n_a. An independent review measured that this is the more sensitive
    # knob (nb x2 moves the lower band edge 2.3 MHz where n_a 90 -> 130 moves
    # it 0.4 MHz), so an n_a-only convergence witness understates truncation.
    n_b = max(6, int(round(nb_scale * n_a * d / a)))
    fwd = _step(a, d, x0, k, n_a, n_b)
    rev = (fwd[3], fwd[2], fwd[1], fwd[0])
    return _star(_star(fwd, _line(d, t, k, n_b)), rev)


def filter_s11_s21(a, apertures, offsets, thicknesses, cavities, freq, n_a=90,
                   nb_scale=1.0):
    """(S11, S21) of the N-iris filter; evanescent modes carried across cavities."""
    k = 2 * np.pi * freq / C0
    tot = _iris(a, apertures[0], offsets[0], thicknesses[0], k, n_a, nb_scale)
    for i, L in enumerate(cavities):
        tot = _star(tot, _line(a, L, k, n_a))
        tot = _star(tot, _iris(a, apertures[i + 1], offsets[i + 1],
                               thicknesses[i + 1], k, n_a, nb_scale))
    return tot[0][0, 0], tot[2][0, 0]


def _f0_by_bisection(a, n_a=90, nb_scale=1.0, threshold_db=10.0, tol=1e3):
    """Centre frequency from the -10 dB crossings, located by bisection.

    Bisection rather than a swept grid so a convergence witness costs ~40 oracle
    evaluations instead of 131, and so the crossing is not quantised by the
    frequency grid.
    """
    def rl(f):
        s11 = abs(filter_s11_s21(a, list(APERTURES_NOM),
                                 list((a - APERTURES_NOM) / 2),
                                 [T_IRIS_NOM] * 5, list(CAVITIES_NOM), f,
                                 n_a=n_a, nb_scale=nb_scale)[0])
        return -20 * np.log10(max(s11, 1e-12)) - threshold_db

    def cross(f_out, f_in):
        assert rl(f_out) < 0 < rl(f_in), "bracket does not straddle the crossing"
        lo, hi = f_out, f_in
        while abs(hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rl(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)

    mid = 10.95e9
    return 0.5 * (cross(10.60e9, mid) + cross(11.40e9, mid))


def validate_oracle() -> dict:
    """Self-witnesses; raises on failure (never gate against a broken oracle)."""
    a, f = A_WR90, 10e9
    w = {}
    # W1 single iris, centred: unitarity + reciprocity
    s11, s21 = filter_s11_s21(a, [12.192e-3], [(a - 12.192e-3) / 2], [1.524e-3], [], f)
    w["unitarity_single"] = abs(abs(s11) ** 2 + abs(s21) ** 2 - 1)
    assert w["unitarity_single"] < 1e-9
    # W2 full filter: unitarity across the band
    devs = []
    for fg in (10.6e9, 11.0e9, 11.4e9):
        s11, s21 = filter_s11_s21(a, list(APERTURES_NOM), list((a - APERTURES_NOM) / 2),
                                  [T_IRIS_NOM] * 5, list(CAVITIES_NOM), fg)
        devs.append(abs(abs(s11) ** 2 + abs(s21) ** 2 - 1))
    w["unitarity_filter"] = max(devs)
    assert w["unitarity_filter"] < 1e-9
    # W3 mode convergence on the filter
    v90 = abs(filter_s11_s21(a, list(APERTURES_NOM), list((a - APERTURES_NOM) / 2),
                             [T_IRIS_NOM] * 5, list(CAVITIES_NOM), 11.0e9, n_a=90)[0])
    v130 = abs(filter_s11_s21(a, list(APERTURES_NOM), list((a - APERTURES_NOM) / 2),
                              [T_IRIS_NOM] * 5, list(CAVITIES_NOM), 11.0e9, n_a=130)[0])
    # |S11| at a single in-band frequency is a BAD convergence metric and the
    # earlier revision used it: at 11.0 GHz |S11| is 0.019, so an absolute
    # deviation of 9.4e-3 on the aperture axis is a ~50% RELATIVE change, while
    # the same axis moves the out-of-band points by 1e-4 to 7e-6. The witness is
    # therefore measured on the GATED observable -- the centre frequency -- by
    # bisecting the -10 dB crossings, on BOTH truncation axes.
    w["mode_convergence_s11_n_a_11ghz"] = abs(v90 - v130)   # reported, not gated
    f0_ref = _f0_by_bisection(a, n_a=90, nb_scale=1.0)
    w["f0_convergence_n_a_hz"] = abs(_f0_by_bisection(a, n_a=130) - f0_ref)
    w["f0_convergence_nb_hz"] = abs(
        _f0_by_bisection(a, n_a=90, nb_scale=2.0) - f0_ref)
    # Both axes must stay far under the gate (19 MHz), so oracle truncation
    # cannot be what the gate is measuring. The aperture axis is the larger one.
    assert w["f0_convergence_n_a_hz"] < 2e6, w["f0_convergence_n_a_hz"]
    assert w["f0_convergence_nb_hz"] < 2e6, w["f0_convergence_nb_hz"]
    # W4 L -> 0 collapse: two irises with a vanishing cavity == one thick iris
    d0, t0 = 8e-3, 1.524e-3
    s_pair = filter_s11_s21(a, [d0, d0], [(a - d0) / 2] * 2, [t0, t0], [0.2e-3], f)
    s_thick = filter_s11_s21(a, [d0], [(a - d0) / 2], [2 * t0 + 0.2e-3], [], f)
    w["collapse_limit"] = abs(abs(s_pair[0]) - abs(s_thick[0]))
    assert w["collapse_limit"] < 5e-3
    # W5 offset symmetry: mirroring an off-centre aperture is invariant
    d1 = 12.192e-3
    x0 = (a - d1) / 2 + 0.19e-3
    s_a = filter_s11_s21(a, [d1], [x0], [t0], [], f)
    s_b = filter_s11_s21(a, [d1], [a - d1 - x0], [t0], [], f)
    w["mirror_symmetry"] = abs(abs(s_a[0]) - abs(s_b[0]))
    assert w["mirror_symmetry"] < 1e-9
    return w


# --------------------------------------------------------------------------- #
# Geometry: nominal -> as-rasterized, with the S1 conventions.
# --------------------------------------------------------------------------- #
def rasterized_geometry(cells: int, allow_asymmetric: bool):
    """Return the geometry rfx will actually build at this mesh.

    Cell counts are chosen so the ELECTRICAL dimensions land on the nominal
    ones, because the electrical length of a region is the distance between its
    bounding zeroed node planes (see raster_assert): a cavity drawn with L_c
    cells of clear space is electrically (L_c + 1)*dx and an iris drawn t_c
    cells thick is electrically (t_c - 1)*dx.  Hence the -1 / +1 below.  The
    aperture needs no correction: d_c*dx already IS the electrical aperture.

    This is a statement of intent, not a fit to the reference. Its residual is
    the sub-cell rounding error, and that residual is NOT monotone in the mesh:
    at a/90 it puts f0 +3.3 MHz from the paper's exact design (inside the
    reference's own 21.9 MHz solver spread) while at a/60 it lands +120.2 MHz,
    worse than the uncompensated -35.8 MHz there. Compensation chooses which
    side of the rounding you land on; only measurement says where.
    """
    dx = A_WR90 / cells
    d_c = np.round(APERTURES_NOM / dx).astype(int)
    t_c = int(round(T_IRIS_NOM / dx)) + 1
    L_c = np.round(CAVITIES_NOM / dx).astype(int) - 1
    if not allow_asymmetric:
        # symmetric fins can only realize apertures with the parity of `cells`
        d_c = d_c + ((cells - d_c) % 2)
    fin_l = np.floor((cells - d_c) / 2).astype(int)
    aps = d_c * dx
    offs = fin_l * dx
    # `thicknesses` / `cavities` are the ELECTRICAL targets, i.e. what the drawn
    # cell counts will realize. electrical_geometry() re-reads them off the
    # rasterized metal and asserts they match, so these are a target and the
    # measurement is the authority — never two independent sources of truth.
    return dict(dx=dx, d_cells=d_c, t_cells=t_c, L_cells=L_c,
                apertures=aps, offsets=offs,
                thicknesses=np.full(5, (t_c - 1) * dx),
                cavities=(L_c + 1) * dx,
                offset_from_centre=offs - (A_WR90 - aps) / 2)


def build(geo, b_cells=B_CELLS, freqs=FREQS, f0=11.0e9, bandwidth=0.14,
          feed_cells=FEED_CELLS, port_cells=PORT_CELLS,
          cpml_fraction=CPML_FRACTION):
    dx = geo["dx"]
    lam_g_lo = C0 / float(freqs[0]) / np.sqrt(
        1.0 - (C0 / (2 * A_WR90) / float(freqs[0])) ** 2)
    cpml_c = int(np.ceil(cpml_fraction * lam_g_lo / dx))
    span = int(geo["t_cells"] * 5 + geo["L_cells"].sum())
    glen_c = 2 * feed_cells + span
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(glen_c * dx, A_WR90, b_cells * dx), dx=dx,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=cpml_c)
    big, cur = 1.0, feed_cells
    for i in range(5):
        fin_c = int(np.floor((int(round(A_WR90 / dx)) - geo["d_cells"][i]) / 2))
        fy_lo = (fin_c + 0.5) * dx                      # midpoint recipe
        fy_hi = A_WR90 - (int(round(A_WR90 / dx)) - fin_c - geo["d_cells"][i] + 0.5) * dx
        sim.add(Box(((cur - 0.5) * dx, -big, -big),
                    ((cur + geo["t_cells"] - 0.5) * dx, fy_lo, big)), material="pec")
        sim.add(Box(((cur - 0.5) * dx, fy_hi, -big),
                    ((cur + geo["t_cells"] - 0.5) * dx, big, big)), material="pec")
        cur += geo["t_cells"] + (geo["L_cells"][i] if i < 4 else 0)
    for x, dr, nm in ((port_cells * dx, "+x", "P1"),
                      ((glen_c - port_cells) * dx, "-x", "P2")):
        sim.add_waveguide_port(x, mode=(1, 0), mode_type="TE", direction=dr,
                               f0=f0, bandwidth=bandwidth,
                               waveform="modulated_gaussian", freqs=freqs, name=nm)
    return sim, cpml_c, glen_c


def raster_assert(sim, geo):
    """Exact-footprint asserts per iris (the S1 discipline, extended to N).

    Also returns the LONGITUDINAL node runs, because the oracle must be fed the
    structure rfx actually built.  S1 established transversely that the
    electrical aperture is the distance between the two bounding zeroed node
    planes, i.e. (n_open + 1)*dx.  The same rule along x makes each cavity
    (L_c + 1)*dx and each iris (t_c - 1)*dx.  Total electrical length is a
    face-continuity CHECK across region types, not a uniqueness argument (an
    interface at sigma*dx beyond the outermost metal node conserves it for
    EVERY sigma); what the check does catch is MIXING sigma between region
    types, which overshoots by 4 or undershoots by 5 cells here.

    Feeding the oracle the DRAWN cell counts instead is worth +107.5 MHz of f0 at
    the shipped geometry (measured 2026-07-29: drawn t=9/L=55,61,61,55 fed as if
    electrical gives 10.9054-11.2267 GHz, the realized t=8/L=56,62,62,56 gives
    10.7833-11.1337), i.e. FIVE times the paper's own 21.9 MHz CST-vs-HFSS
    spread. The defect was first found as +90.0 MHz on the uncompensated cell
    counts; compensation changes the pair being confused, not the class. Either
    way it would have had to be absorbed by a gate of ~162 MHz -- 46% of the
    passband -- which pins nothing.
    """
    grid = sim._build_grid()
    cells = grid.shape[1] - 1
    sig = np.asarray(rasterize(grid, [(e.shape, 1.0, 1e7)
                                      for e in sim._geometry])[1])
    xs = np.where(sig.max(axis=(1, 2)) > 1e6)[0]
    runs = np.split(xs, np.where(np.diff(xs) != 1)[0] + 1)
    assert len(runs) == 5, ("iris count", len(runs))
    realized = []
    for run, d_c in zip(runs, geo["d_cells"]):
        assert len(run) == geo["t_cells"], ("thickness", len(run), geo["t_cells"])
        open_y = np.where(sig[run[0]].max(axis=1) < 1e6)[0]
        assert bool(np.all(np.diff(open_y) == 1)), "aperture not contiguous"
        # electrical aperture = distance between the bounding zeroed planes
        assert len(open_y) == d_c - 1, ("aperture nodes", len(open_y), d_c - 1)
        realized.append((int(open_y[0]), int(open_y[-1])))
    x_runs = [(int(r[0]), int(r[-1])) for r in runs]
    return cells, realized, x_runs


def electrical_geometry(geo, x_runs):
    """The oracle's inputs, MEASURED from the rasterized metal (not assumed).

    Raises if the measured lengths disagree with the (L_c + 1, t_c - 1) rule, so
    a future layout change cannot silently re-introduce the 90 MHz mismatch.
    """
    dx = geo["dx"]
    # Node indices are integers, so the electrical lengths are integer cell
    # counts EXACTLY; deriving the counts back from a float length instead lets
    # 62 arrive as 61.99999999999999 and int() truncate it to 61.
    th_cells = [hi - lo for lo, hi in x_runs]
    cav_cells = [x_runs[i + 1][0] - x_runs[i][1] for i in range(4)]
    assert all(c == geo["t_cells"] - 1 for c in th_cells), ("thickness", th_cells)
    assert cav_cells == [int(v) + 1 for v in geo["L_cells"]], ("cavities", cav_cells)
    span = geo["t_cells"] * 5 + int(geo["L_cells"].sum())
    assert sum(th_cells) + sum(cav_cells) == span - 1, (
        "total electrical length", sum(th_cells) + sum(cav_cells), span - 1)
    out = dict(geo)
    out.update(thicknesses=np.array([c * dx for c in th_cells]),
               cavities=np.array([c * dx for c in cav_cells]),
               electrical_thickness_cells=int(th_cells[0]),
               electrical_cavity_cells=[int(c) for c in cav_cells])
    return out


def measured_electrical_geometry(geo, freqs=FREQS):
    """Build (no time stepping) and read back the structure rfx will simulate."""
    sim, _, _ = build(geo, freqs=freqs)
    _, _, x_runs = raster_assert(sim, geo)
    return electrical_geometry(geo, x_runs)


def measure(geo, num_periods, b_cells=B_CELLS, freqs=FREQS,
            feed_cells=FEED_CELLS, port_cells=PORT_CELLS,
            cpml_fraction=CPML_FRACTION):
    sim, cpml_c, glen_c = build(geo, b_cells=b_cells, freqs=freqs,
                                feed_cells=feed_cells, port_cells=port_cells,
                                cpml_fraction=cpml_fraction)
    cells, realized, x_runs = raster_assert(sim, geo)
    grid = sim._build_grid()
    t0 = time.time()
    # Extractor warnings ARE part of the record (CLAUDE.md: quote every preflight
    # warning before reporting any |S| number). The previous revision asserted
    # that in prose and committed no warnings field at all; case 18 captures
    # them per row and this case had dropped it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_waveguide_s_matrix(normalize="flux",
                                             num_periods=num_periods)
    wall = time.time() - t0
    s = np.asarray(res.s_params)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    # The repo-mandated ring-down witness is ENERGY-based: terminal energy in dB
    # below the post-source peak, rule < -40 dB. The extractor already provides
    # it (PR #468), so the previous revision's num_periods sweep was a
    # confirmation axis substituted for the mandated number, not the number.
    settling = getattr(res, "settling_db", None)
    correction = getattr(res, "passivity_correction", None)
    colpow = s11 ** 2 + s21 ** 2
    over = np.where(colpow > 1.02)[0]
    return dict(cells_per_a=cells, dx_mm=round(geo["dx"] * 1e3, 4),
                b_cells=b_cells, cpml_cells=cpml_c, grid=list(grid.shape),
                glen_cells=glen_c, num_periods=num_periods,
                feed_cells=feed_cells, port_cells=port_cells,
                cpml_fraction=cpml_fraction,
                aperture_nodes=realized, iris_x_nodes=x_runs,
                s11=[round(float(v), 6) for v in s11],
                s21=[round(float(v), 6) for v in s21],
                max_colpow=round(float(np.max(colpow)), 4),
                # Passivity FOOTPRINT, not just the scalar max: the rule wants the
                # violating bins quoted as an artifact.
                colpow_over_102_bins=[int(i) for i in over],
                colpow_over_102_ghz=[round(float(FREQS[i]) / 1e9, 4) for i in over],
                settling_db=(None if settling is None
                             else [round(float(v), 2) for v in np.atleast_1d(settling)]),
                passivity_correction_max=(
                    None if correction is None
                    else round(float(np.max(np.atleast_1d(correction))), 6)),
                extractor_warnings=[str(w.message)[:400] for w in caught],
                wall_s=round(wall, 1))


def band_analysis(s11, freqs=FREQS, threshold_db=10.0):
    """-threshold_db |S11| band, plus a DEPTH-INDEPENDENT structural zero count.

    WINDOW GUARD (a failed probe in this campaign, 2026-07-28): if the passband
    touches either end of the frequency window, `f[inb][0/-1]` is the WINDOW
    edge, not the band edge, and every derived quantity (BW, f0, and any
    sensitivity read from them) is silently wrong.  The probe that taught this
    reported every |S21| peak at exactly the window's lower bound and therefore
    a sensitivity of +0 MHz for every perturbation.  An extremum pinned to a
    scan boundary is not a measurement, so this raises rather than returns.

    That guard checked only the two window ENDPOINTS, which is why it missed the
    interior version of the same defect; see the contiguity block below.
    """
    s11 = np.asarray(s11, dtype=float)
    f = np.asarray(freqs, dtype=float)
    rl = -20 * np.log10(np.clip(s11, 1e-12, None))
    inb = rl >= threshold_db
    if inb.sum() < 2:
        return dict(lo=None, hi=None, f0=None, bw=None, worst_rl_db=None, zeros=[])
    if bool(inb[0]) or bool(inb[-1]):
        raise ValueError(
            f"passband touches the frequency window edge "
            f"[{f[0]/1e9:.3f}, {f[-1]/1e9:.3f}] GHz — band edges would be window "
            f"edges and BW/f0 would be meaningless. Widen FREQS."
        )
    # Band edges are the -threshold_db CROSSINGS, linearly interpolated in dB
    # against frequency, not the first/last in-band SAMPLE.  On the 10 MHz
    # digitized grid a sampled edge is quantized to +/-one bin, which inflates
    # the rfx-vs-oracle envelope by up to 10 MHz per edge and would set a gate
    # out of grid spacing rather than out of physics.  The crossing is
    # well-defined here because |S11| is monotone through each band edge.
    i_lo, i_hi = int(np.argmax(inb)), int(len(inb) - 1 - np.argmax(inb[::-1]))

    def _cross(i_in, i_out):
        y_in, y_out = rl[i_in] - threshold_db, rl[i_out] - threshold_db
        if y_in == y_out:
            return float(f[i_in])
        w = y_in / (y_in - y_out)
        return float(f[i_in] + w * (f[i_out] - f[i_in]))

    lo, hi = _cross(i_lo, i_lo - 1), _cross(i_hi, i_hi + 1)
    zeros = [float(f[i]) for i in range(1, len(f) - 1)
             if lo <= f[i] <= hi and s11[i] < s11[i - 1] and s11[i] < s11[i + 1]]

    # CONTIGUITY. `lo`/`hi` are the OUTERMOST crossings, so the span between
    # them is a "-threshold_db span", not necessarily a passband: the region can
    # contain interior excursions back above the threshold. That is measured, not
    # assumed away, because a snapped filter genuinely can fail its own ripple
    # spec. Two reviewers found the earlier version reporting a bridged span as
    # a passband (2026-07-29): the a/60 rung's committed "band" was two separated
    # resonances with a 2.73 dB trough between them.
    span = slice(i_lo, i_hi + 1)
    holes = int((~inb[span]).sum())
    runs = np.split(np.where(inb[span])[0],
                    np.where(np.diff(np.where(inb[span])[0]) != 1)[0] + 1)
    longest = max(runs, key=len)
    longest_hz = float(f[i_lo + longest[-1]] - f[i_lo + longest[0]])

    # WORST RETURN LOSS is computed over the whole span, NOT over `rl[inb]`.
    # `rl[inb].min()` is filtered to `rl >= threshold_db` and therefore cannot
    # report a threshold violation at all — it is a clamped statistic, and it
    # published 10.1 dB for a trace whose true in-span worst is 9.80 dB.
    return dict(lo=lo, hi=hi, f0=(lo + hi) / 2, bw=hi - lo,
                worst_rl_db=float(rl[span].min()), zeros=zeros,
                span_holes=holes, n_span_bins=int(inb[span].size),
                longest_contiguous_hz=longest_hz,
                contiguous=bool(holes == 0))


def oracle_curve(geo=None, nominal=False, freqs=FREQS, n_a=90):
    if nominal:
        aps, offs = APERTURES_NOM, (A_WR90 - APERTURES_NOM) / 2
        ths, cav = np.full(5, T_IRIS_NOM), CAVITIES_NOM
    else:
        aps, offs = geo["apertures"], geo["offsets"]
        ths, cav = geo["thicknesses"], geo["cavities"]
    return [float(abs(filter_s11_s21(A_WR90, list(aps), list(offs), list(ths),
                                     list(cav), f, n_a=n_a)[0])) for f in freqs]


def main(argv):
    write_fixture = "--write-fixture" in argv
    ok = True

    w = validate_oracle()
    print("[oracle] mode-matching self-witnesses PASS:",
          {k: float(f"{v:.3e}") for k, v in w.items()})

    # --- reference decompositions (oracle only, no FDTD) -------------------
    nom_curve = oracle_curve(nominal=True)
    nom = band_analysis(nom_curve)
    print(f"\n[anchor] oracle @ NOMINAL: band {nom['lo']/1e9:.4f}-{nom['hi']/1e9:.4f} "
          f"f0={nom['f0']/1e9:.4f} BW={nom['bw']/1e6:.0f} MHz "
          f"worst RL={nom['worst_rl_db']:.1f} dB zeros={len(nom['zeros'])}")
    for tag, ref in (("HFSS", PAPER["hfss"]), ("CST", PAPER["cst"])):
        print(f"          vs paper {tag}: df0={(nom['f0']-ref['f0'])/1e6:+.1f} MHz "
              f"dBW={(nom['bw']-ref['bw'])/1e6:+.0f} MHz "
              f"dRL={nom['worst_rl_db']-ref['worst_rl_db']:+.1f} dB "
              f"(solver spread {PAPER['solver_spread_f0_hz']/1e6:.1f} MHz)")

    geo_g = rasterized_geometry(GATED_CELLS, allow_asymmetric=False)
    # The oracle is fed the geometry read back off the rasterized metal, not the
    # drawn cell counts: the difference is +107.5 MHz of f0 (see raster_assert).
    geo_g_e = measured_electrical_geometry(geo_g)
    print(f"[electrical] iris {geo_g_e['electrical_thickness_cells']} cells "
          f"(drawn {geo_g['t_cells']}), cavities "
          f"{geo_g_e['electrical_cavity_cells']} cells "
          f"(drawn {[int(v) for v in geo_g['L_cells']]}) — the S1 "
          f"bounding-zeroed-plane rule along x")
    ras_curve = oracle_curve(geo_g_e)
    ras = band_analysis(ras_curve)
    print(f"[snap]   oracle @ a/{GATED_CELLS} rasterized: f0={ras['f0']/1e9:.4f} "
          f"BW={ras['bw']/1e6:.0f} MHz zeros={len(ras['zeros'])} "
          f"-> snap df0={(ras['f0']-nom['f0'])/1e6:+.1f} MHz "
          f"({(ras['f0']-nom['f0'])/nom['bw']*100:+.0f}% of the passband)")

    # --- GATED: rfx vs oracle @ as-rasterized -----------------------------
    # WHAT IS GATED AND WHY THIS AND NOT THAT (revised after the #499 cold
    # review). The iris-thickness leg of the node-plane convention is NOT
    # settled: four independent FDTD runs at drawn t_c = 2/4/6/8 give a flat
    # offset of -0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec ~ (t_c - 0.68)*dx,
    # matching neither (t_c - 1)*dx nor case 18's t_c*dx. That leaves an
    # irreducible comparator-input uncertainty of order half a cell, and the
    # gated observable therefore has to be chosen by its SENSITIVITY to that
    # uncertainty rather than by convenience. Measured, per cell of convention
    # error: f0 ~2.4 MHz, bandwidth ~40 MHz, individual band edges ~22-30 MHz.
    # So f0 (17x less sensitive than BW) and the zero COUNT (an integer) are
    # gated; edges and bandwidth are REPORTED with the ambiguity stated, since
    # a +/-20 MHz input uncertainty cannot sit under a 15 MHz gate.
    # Adopting the fitted -0.68 instead would absorb the disagreement into a
    # free parameter, after which the residual measures nothing.
    print(f"\n== GATED rfx a/{GATED_CELLS} vs oracle@as-rasterized "
          f"(f0 {GATE_F0_MHZ} MHz + zero count; edges/BW reported) ==")
    row = measure(geo_g, 400.0)
    meas = band_analysis(row["s11"])
    d_f0 = (meas["f0"] - ras["f0"]) / 1e6
    d_lo = (meas["lo"] - ras["lo"]) / 1e6
    d_hi = (meas["hi"] - ras["hi"]) / 1e6
    d_bw = (meas["bw"] - ras["bw"]) / 1e6
    row.update(oracle_s11=ras_curve, band=meas, oracle_band=ras,
               d_f0_mhz=round(d_f0, 2),
               d_lo_mhz=round(d_lo, 2), d_hi_mhz=round(d_hi, 2),
               d_bw_mhz=round(d_bw, 2))
    zeros_ok = len(meas["zeros"]) == len(ras["zeros"])
    print(f"  grid {tuple(row['grid'])} cpml={row['cpml_cells']} "
          f"({row['wall_s']:.0f}s): band {meas['lo']/1e9:.4f}-{meas['hi']/1e9:.4f} "
          f"f0={meas['f0']/1e9:.4f} BW={meas['bw']/1e6:.0f} MHz "
          f"worst RL={meas['worst_rl_db']:.1f} dB zeros={len(meas['zeros'])} "
          f"holes={meas['span_holes']}/{meas['n_span_bins']}")
    if row["settling_db"] is not None:
        print(f"  settling witness (energy): {row['settling_db']} dB "
              f"(rule < -40); passivity corr max "
              f"{row['passivity_correction_max']}; "
              f"colpow>1.02 in {len(row['colpow_over_102_bins'])} bins")
    for wmsg in row["extractor_warnings"]:
        print(f"  [extractor] {wmsg}")
    f0_ok = abs(d_f0) <= GATE_F0_MHZ
    print(f"  GATED f0 {d_f0:+.2f} MHz (gate {GATE_F0_MHZ}, "
          f"{'PASS' if f0_ok else 'FAIL'}), zeros "
          f"{len(meas['zeros'])} vs {len(ras['zeros'])} "
          f"({'PASS' if zeros_ok else 'FAIL'})")
    print(f"  REPORTED (not gated): edges {d_lo:+.1f}/{d_hi:+.1f} MHz "
          f"[d_bw == d_hi - d_lo identically, so the edge asymmetry and the BW "
          f"deficit are ONE fact, not two], BW {d_bw:+.1f} MHz, "
          f"colpow {row['max_colpow']:.4f}")
    ok &= f0_ok and zeros_ok

    coarse = ring = binv = clearance = absorber = fdfd = None
    if write_fixture:
        print(f"\n== DIAGNOSTIC coarse rung a/{COARSE_CELLS} (asymmetric fins) ==")
        geo_c = rasterized_geometry(COARSE_CELLS, allow_asymmetric=True)
        cur = oracle_curve(measured_electrical_geometry(geo_c))
        cb = band_analysis(cur)
        coarse = measure(geo_c, 400.0)
        cm = band_analysis(coarse["s11"])
        coarse.update(oracle_s11=cur, band=cm, oracle_band=cb,
                      d_lo_mhz=round((cm["lo"] - cb["lo"]) / 1e6, 2),
                      d_hi_mhz=round((cm["hi"] - cb["hi"]) / 1e6, 2),
                      d_bw_mhz=round((cm["bw"] - cb["bw"]) / 1e6, 2))
        print(f"  offsets {np.round(geo_c['offset_from_centre']*1e3, 3).tolist()} mm; "
              f"snap f0 {(cb['f0']-nom['f0'])/1e6:+.0f} MHz, zeros {len(cb['zeros'])}; "
              f"rfx edges {coarse['d_lo_mhz']:+.1f}/{coarse['d_hi_mhz']:+.1f} MHz")

        print("\n== ring-down witness (num_periods) — GATED for convergence ==")
        ring = []
        # 600 is an INTERIOR point, and it is the whole reason this is a scan
        # rather than a pair. A one-alternative-per-axis envelope cannot detect
        # NON-MONOTONIC sensitivity, which is exactly what bit PR #475: three
        # sampled clearances passed while 9 of 13 exceeded the gate, and the
        # passing three were the sampled ones. Every witness axis here therefore
        # carries an interior sample.
        for npd in (200.0, 400.0, 600.0, 800.0):
            # num_periods=400 IS the gated configuration, so reuse that row
            # rather than paying for a bit-identical repeat run.
            r = row if npd == 400.0 else measure(geo_g, npd)
            b = meas if npd == 400.0 else band_analysis(r["s11"])
            # the trace rides with the row: an envelope member with no trace
            # cannot be recomputed, so its integrity would be borrowed from
            # asserts living in other tests.
            ring.append({"num_periods": npd, "f0": b["f0"], "bw": b["bw"],
                         "max_colpow": r["max_colpow"], "wall_s": r["wall_s"],
                         # BOTH components: membership is decided by column
                         # power = s11^2 + s21^2, so committing s11 alone would
                         # leave the criterion's input outside the anchor.
                         "s11": r["s11"], "s21": r["s21"]})
            print(f"  np={npd:5.0f}: f0={b['f0']/1e9:.4f} BW={b['bw']/1e6:.0f} MHz "
                  f"colpow={r['max_colpow']:.4f} ({r['wall_s']:.0f}s)", flush=True)
        # The gated run uses num_periods=400; a resonant structure's DFT window
        # must be shown settled rather than assumed. Truncation shows up first
        # as non-passivity (a 2-iris Q~87 cavity at num_periods=100 fired the
        # extractor self-check with column power 1.58) and then as a shifted
        # band, so BOTH are gated here: the 400 -> 800 doubling must not move
        # f0 or BW by more than one frequency bin, and the gated run must be
        # passivity-clean.
        bin_hz = float(FREQS[1] - FREQS[0])
        r400 = next(x for x in ring if x["num_periods"] == 400.0)
        r800 = next(x for x in ring if x["num_periods"] == 800.0)
        d_f0 = abs(r400["f0"] - r800["f0"])
        d_bw = abs(r400["bw"] - r800["bw"])
        settled = d_f0 <= bin_hz and d_bw <= bin_hz and r400["max_colpow"] <= 1.02
        ok &= settled
        print(f"  settling gate: |df0| {d_f0/1e6:.1f} MHz, |dBW| {d_bw/1e6:.1f} MHz "
              f"(one bin = {bin_hz/1e6:.0f} MHz), colpow {r400['max_colpow']:.4f} "
              f"-> {'PASS' if settled else 'FAIL — raise num_periods'}")

        print("\n== b-invariance witness (TE10 y-invariance) ==")
        binv = []
        for bc in (4, 6, 8):          # 6 is the interior sample
            # b=B_CELLS at num_periods=400 is the gated row; reuse it.
            r = row if bc == B_CELLS else measure(geo_g, 400.0, b_cells=bc)
            b = meas if bc == B_CELLS else band_analysis(r["s11"])
            binv.append({"b_cells": bc, "f0": b["f0"], "bw": b["bw"],
                         "max_dev_vs_b4": None, "wall_s": r["wall_s"],
                         "max_colpow": r["max_colpow"],
                         "s11": r["s11"], "s21": r["s21"]})
            print(f"  b={bc} cells: f0={b['f0']/1e9:.4f} BW={b['bw']/1e6:.0f} MHz "
                  f"({r['wall_s']:.0f}s)", flush=True)
        for entry in binv[1:]:
            entry["max_dev_vs_b4"] = abs(entry["f0"] - binv[0]["f0"])

        # Feed clearance. CPML is exterior to the requested domain, so the
        # absorber never overlaps the irises; what IS thin is the standoff
        # between the absorber interface and the port plane — PORT_CELLS*dx is
        # about 0.08*lambda_g at a/90 where the S1 recipe used roughly one
        # guide wavelength. Deviating from a validated recipe is witnessed,
        # not assumed: re-run the SAME geometry with a generous feed and
        # require the band edges to hold to one frequency bin, since the only
        # thing that changed is standoff.
        print("\n== feed-clearance witness (port standoff from the absorber) ==")
        mid = measure(geo_g, 400.0, feed_cells=FEED_CELLS_MID,
                      port_cells=PORT_CELLS_MID)
        mb = band_analysis(mid["s11"])
        gen = measure(geo_g, 400.0, feed_cells=FEED_CELLS_WITNESS,
                      port_cells=PORT_CELLS_WITNESS)
        gb = band_analysis(gen["s11"])
        d_lo_c = max(abs(mb["lo"] - meas["lo"]), abs(gb["lo"] - meas["lo"]))
        d_hi_c = max(abs(mb["hi"] - meas["hi"]), abs(gb["hi"] - meas["hi"]))
        clear_ok = d_lo_c <= bin_hz and d_hi_c <= bin_hz
        ok &= clear_ok

        def _leg(fc, pc, band, r):
            return {"feed_cells": fc, "port_cells": pc,
                    "standoff_mm": round(pc * geo_g["dx"] * 1e3, 3),
                    "lo": band["lo"], "hi": band["hi"],
                    "max_colpow": r["max_colpow"], "wall_s": r["wall_s"],
                    "s11": r["s11"], "s21": r["s21"],
                    "d_lo_mhz": round(abs(band["lo"] - meas["lo"]) / 1e6, 3),
                    "d_hi_mhz": round(abs(band["hi"] - meas["hi"]) / 1e6, 3)}

        clearance = {
            "gated": {"feed_cells": FEED_CELLS, "port_cells": PORT_CELLS,
                      "standoff_mm": round(PORT_CELLS * geo_g["dx"] * 1e3, 3),
                      "lo": meas["lo"], "hi": meas["hi"]},
            "mid": _leg(FEED_CELLS_MID, PORT_CELLS_MID, mb, mid),
            "generous": _leg(FEED_CELLS_WITNESS, PORT_CELLS_WITNESS, gb, gen),
            "d_lo_mhz": round(d_lo_c / 1e6, 2),
            "d_hi_mhz": round(d_hi_c / 1e6, 2),
            "worst_of_legs": "max over the interior and generous samples",
            "passed": bool(clear_ok),
        }
        print(f"  standoff {clearance['gated']['standoff_mm']:.2f} -> "
              f"{clearance['generous']['standoff_mm']:.2f} mm: edges move "
              f"{clearance['d_lo_mhz']:.1f}/{clearance['d_hi_mhz']:.1f} MHz "
              f"(one bin = {bin_hz/1e6:.0f} MHz) -> "
              f"{'PASS' if clear_ok else 'FAIL — use the generous feed'}")

        # Absorber depth. In S1 (PR #480) this was THE envelope-limiting term:
        # 0.5*lambda_g left the measured envelope absorber-limited rather than
        # discretization-limited, and the gate was set on an artifact until the
        # depth was scanned. The low band edge sits nearest the CPML's design
        # frequency, so an asymmetric edge error is exactly what a thin absorber
        # produces. Deepening the absorber must not move the edges by more than
        # one bin; if it does, the gated configuration is absorber-limited and
        # the deeper run becomes the gated one.
        print("\n== absorber-depth witness (CPML fraction of lambda_g) ==")
        amid = measure(geo_g, 400.0, cpml_fraction=CPML_FRACTION_MID)
        ab = band_analysis(amid["s11"])
        deep = measure(geo_g, 400.0, cpml_fraction=CPML_FRACTION_WITNESS)
        db = band_analysis(deep["s11"])
        d_lo_a = max(abs(ab["lo"] - meas["lo"]), abs(db["lo"] - meas["lo"]))
        d_hi_a = max(abs(ab["hi"] - meas["hi"]), abs(db["hi"] - meas["hi"]))
        absorber_ok = d_lo_a <= bin_hz and d_hi_a <= bin_hz
        ok &= absorber_ok
        absorber = {
            "gated": {"cpml_fraction": CPML_FRACTION,
                      "cpml_cells": row["cpml_cells"],
                      "lo": meas["lo"], "hi": meas["hi"]},
            "mid": {"cpml_fraction": CPML_FRACTION_MID,
                    "cpml_cells": amid["cpml_cells"],
                    "lo": ab["lo"], "hi": ab["hi"],
                    "max_colpow": amid["max_colpow"], "wall_s": amid["wall_s"],
                    "s11": amid["s11"], "s21": amid["s21"],
                    "d_lo_mhz": round(abs(ab["lo"] - meas["lo"]) / 1e6, 3),
                    "d_hi_mhz": round(abs(ab["hi"] - meas["hi"]) / 1e6, 3)},
            "deep": {"cpml_fraction": CPML_FRACTION_WITNESS,
                     "cpml_cells": deep["cpml_cells"],
                     "lo": db["lo"], "hi": db["hi"],
                     "max_colpow": deep["max_colpow"], "wall_s": deep["wall_s"],
                     "s11": deep["s11"], "s21": deep["s21"],
                     "d_lo_mhz": round(abs(db["lo"] - meas["lo"]) / 1e6, 3),
                     "d_hi_mhz": round(abs(db["hi"] - meas["hi"]) / 1e6, 3)},
            "d_lo_mhz": round(d_lo_a / 1e6, 2),
            "d_hi_mhz": round(d_hi_a / 1e6, 2),
            "passed": bool(absorber_ok),
        }
        print(f"  {CPML_FRACTION}->{CPML_FRACTION_WITNESS} lambda_g "
              f"({row['cpml_cells']}->{deep['cpml_cells']} cells): edges move "
              f"{absorber['d_lo_mhz']:.1f}/{absorber['d_hi_mhz']:.1f} MHz "
              f"(one bin = {bin_hz/1e6:.0f} MHz) -> "
              f"{'PASS' if absorber_ok else 'FAIL — absorber-limited'}")

        # FORMULATION-INDEPENDENT CHECK. The mode-matching cascade and the frozen
        # gate test's re-typed copy share their formulation, so neither can find a
        # formulation-level error. This solves the same ELECTRICAL geometry as a
        # 2-D H-plane FDFD -- different discretization, different aperture
        # treatment, different solve, sharing only numpy/scipy -- with every
        # dimension grid-exact at every refinement level (fdfd_hplane condition 3).
        #
        # It is FIRST-order convergent, so a single level is meaningless
        # (condition 2), and the handoff's protocol is carried in full: THREE
        # levels, two Richardson estimates, and their agreement committed as the
        # extrapolation's own consistency witness before either is trusted.
        print("\n== formulation-independent FDFD check (2-D H-plane) ==")
        d_e = [int(v) for v in geo_g["d_cells"]]
        cav_e = [int(v) for v in geo_g_e["electrical_cavity_cells"]]
        th_e = int(geo_g_e["electrical_thickness_cells"])
        gate_w = fdfd_hplane.self_test(A_WR90, 11.0e9, GATED_CELLS, 1,
                                       d_e, cav_e, th_e, 45)
        print(f"  [gate] empty guide |S11|={gate_w['empty_s11']:.2e} "
              f"|S21|={gate_w['empty_s21']:.12f} unitarity={gate_w['unitarity']:.1e}")
        fdfd_levels = {}
        for rr in (2, 3, 4):
            t0 = time.time()
            worst_u, curve = 0.0, []
            for fq in FREQS:
                a11, a21, _ = fdfd_hplane.solve(A_WR90, float(fq), GATED_CELLS,
                                                rr, d_e, cav_e, th_e, 45)
                worst_u = max(worst_u, abs(abs(a11) ** 2 + abs(a21) ** 2 - 1))
                curve.append(abs(a11))
            assert worst_u < 1e-6, ("FDFD unitarity gate", rr, worst_u)
            # round FIRST: the committed band must be exactly recomputable from
            # the committed 6-decimal trace, or the frozen test's recomputation
            # fails on a 10-50 Hz rounding shift (found by the consistency
            # review BEFORE the regeneration would have surfaced it).
            curve = [round(v, 6) for v in curve]
            bb = band_analysis(curve)
            fdfd_levels[rr] = dict(band=bb, s11=curve,
                                   worst_unitarity=worst_u,
                                   wall_s=round(time.time() - t0, 1))
            print(f"  r={rr}: lo={bb['lo']/1e9:.5f} hi={bb['hi']/1e9:.5f} "
                  f"f0={bb['f0']/1e9:.5f} BW={bb['bw']/1e6:.2f} "
                  f"zeros={len(bb['zeros'])} ({fdfd_levels[rr]['wall_s']:.0f}s)",
                  flush=True)
        rich23 = {k: fdfd_hplane.richardson_first_order(
            fdfd_levels[2]["band"][k], 2, fdfd_levels[3]["band"][k], 3)
            for k in ("lo", "hi", "f0", "bw")}
        rich34 = {k: fdfd_hplane.richardson_first_order(
            fdfd_levels[3]["band"][k], 3, fdfd_levels[4]["band"][k], 4)
            for k in ("lo", "hi", "f0", "bw")}
        consistency = {k: round(abs(rich34[k] - rich23[k]) / 1e6, 3)
                       for k in ("lo", "hi", "f0", "bw")}
        # the FINER pair carries the headline; the coarser one is the witness
        d_fdfd_oracle = (rich34["f0"] - ras["f0"]) / 1e6
        d_rfx_fdfd = (meas["f0"] - rich34["f0"]) / 1e6
        print(f"  Richardson(2,3): f0={rich23['f0']/1e9:.5f} BW={rich23['bw']/1e6:.2f}")
        print(f"  Richardson(3,4): f0={rich34['f0']/1e9:.5f} BW={rich34['bw']/1e6:.2f}")
        print(f"  two-estimate consistency: df0 {consistency['f0']:.2f} MHz, "
              f"dBW {consistency['bw']:.2f} MHz")
        print(f"  FDFD(3,4) - cascade: df0 {d_fdfd_oracle:+.2f} MHz, "
              f"dBW {(rich34['bw']-ras['bw'])/1e6:+.2f} MHz")
        print(f"  rfx  - FDFD(3,4)   : df0 {d_rfx_fdfd:+.2f} MHz "
              f"(rfx - cascade was {row['d_f0_mhz']:+.2f})")
        fdfd = {
            "levels": {str(k): v for k, v in fdfd_levels.items()},
            "richardson_23": rich23,
            "richardson_34": rich34,
            "richardson_consistency_mhz": consistency,
            "d_f0_vs_cascade_mhz": round(d_fdfd_oracle, 3),
            "d_bw_vs_cascade_mhz": round((rich34["bw"] - ras["bw"]) / 1e6, 3),
            "d_f0_rfx_vs_fdfd_mhz": round(d_rfx_fdfd, 3),
            "self_test": {k: float(v) for k, v in gate_w.items()
                          if isinstance(v, (int, float))},
            "note": ("2-D H-plane FDFD on the same electrical geometry, "
                     "grid-exact at every level, sharing only numpy/scipy with "
                     "the cascade. FIRST-order convergent, so single levels are "
                     "meaningless; three levels are committed and the two "
                     "Richardson estimates' agreement is the extrapolation's "
                     "own consistency witness, per the porting handoff. An "
                     "earlier revision's mask realized apertures 2h wide, which "
                     "Richardson cancelled but which made per-level geometry "
                     "claims false and produced a spurious fourth reflection "
                     "zero at r=2,3; with the exact mask every level shows "
                     "three zeros, matching the cascade and rfx."),
        }

        # ENVELOPE POPULATION. The previous revision took the envelope as
        # max(|d_lo|, |d_hi|) of the SINGLE gated run, i.e. the two band edges
        # of the datum being gated, while calling it a "measured envelope" in
        # the same public table where case 18 means a max over 8 configurations
        # and case 16 a 7-point scan. A gate derived from the one number it
        # gates has 50% headroom by construction and bounds nothing.
        # Every setup variation already paid for carries its own f0, so the
        # gated observable has a real population at no extra cost. All of these
        # are compared against the SAME oracle f0, so each entry is an
        # independent rfx-vs-oracle residual.
        f0_pop = [("gated a/90 np400 b4", meas["f0"])]
        # Membership follows a CRITERION, not a value list. A hardcoded
        # exclusion tuple lets a future failing row be dropped by adding its
        # num_periods to the tuple, and a guard keyed to the literal "200"
        # would not notice. So: the gated row enters separately (it is already
        # first), and every other leg enters UNLESS it fails the settling
        # criterion -- column power > 1.02 -- in which case folding it in would
        # inflate the envelope with a known truncation artifact and buy slack
        # for free. The excluded rows stay committed as the evidence that the
        # settling criterion can fire, with their reason recorded.
        SETTLED_MAX_COLPOW = 1.02
        excluded = [(f"ring np{int(r['num_periods'])}", r["max_colpow"])
                    for r in ring
                    if r["num_periods"] != 400.0
                    and r["max_colpow"] > SETTLED_MAX_COLPOW]
        f0_pop += [(f"ring np{int(r['num_periods'])}", r["f0"]) for r in ring
                   if r["num_periods"] != 400.0
                   and r["max_colpow"] <= SETTLED_MAX_COLPOW]
        f0_pop += [(f"b={r['b_cells']} cells", r["f0"]) for r in binv
                   if r["b_cells"] != B_CELLS]
        f0_pop += [("mid feed", mb["f0"]), ("generous feed", gb["f0"]),
                   ("mid absorber", ab["f0"]), ("deep absorber", db["f0"])]
        residuals = [(tag, (v - ras["f0"]) / 1e6) for tag, v in f0_pop]
        env_f0 = max(abs(d) for _, d in residuals)
        print(f"\n  f0 envelope population ({len(residuals)} configurations, "
              f"all vs the same oracle f0):")
        for tag, d in residuals:
            print(f"    {tag:24s} d_f0 = {d:+7.2f} MHz")
        for tag, cp in excluded:
            print(f"    EXCLUDED {tag:22s} column power {cp:.4f} > "
                  f"{SETTLED_MAX_COLPOW} (fails settling)")
        if not excluded:
            print("    WARNING: no leg fails the settling criterion, so the "
                  "criterion is untested by this record")
        print(f"  envelope {env_f0:.2f} MHz -> gate {GATE_F0_MHZ} MHz")
        required = np.ceil(max(env_f0, 1e-9) * 1.5)
        if abs(GATE_F0_MHZ - required) > 1e-9:
            print(f"  ENVELOPE/GATE MISMATCH (f0): gate {GATE_F0_MHZ} must equal "
                  f"round-up(env x 1.5) = {required}")
            ok = False
        # Reported alongside, never gated: the edge/BW residuals, whose
        # comparator-input uncertainty exceeds any defensible gate on them.
        env_edge = max(abs(row["d_lo_mhz"]), abs(row["d_hi_mhz"]))
        env_bw = abs(row["d_bw_mhz"])

        payload = {
            "schema": "rfx.wr90_iris_filter_aghanim",
            "schema_version": 1,
            "campaign": ("cross-solver validation campaign, item 3 stage S3: a "
                         "published 4th-order WR-90 inductive-iris bandpass "
                         "filter vs a TEn0 mode-matching cascade oracle"),
            "reference": {
                "citation": ("Aghanim, Zbitou, Errkik, Tajmouati, Latrach, E3S Web "
                             "of Conferences 351, 01059 (2022), CC BY 4.0, "
                             "DOI 10.1051/e3sconf/202235101059, Table 6 (optimized)"),
                "apertures_mm": (APERTURES_NOM * 1e3).tolist(),
                "cavities_mm": (CAVITIES_NOM * 1e3).tolist(),
                "iris_thickness_mm": T_IRIS_NOM * 1e3,
                "digitized_scalars": PAPER,
                "digitization_note": (
                    "Fig. 5 digitized at 10 MHz (native 2.234 MHz/px); zero "
                    "FREQUENCIES are calibration-invariant by construction "
                    "(affine-equivariant argmin, verified bit-identical across "
                    "three calibrations) while zero DEPTHS are sampling "
                    "artifacts — four nominally identical equiripple zeros "
                    "bottom out across a 16 dB spread. The reference is NOT "
                    "equiripple (HFSS -19.3/-14.9/-18.4 dB vs CST "
                    "-24.9/-18.7/-14.2 dB, disagreeing on which peak is worst), "
                    "so individual ripple levels are not gateable."),
            },
            "claim_scope": "A published 4th-order WR-90 inductive-iris bandpass filter (Aghanim "
                           "et al., E3S Web of Conferences 351, 01059 (2022), CC BY 4.0, Table "
                           "6 optimized: five irises t = 2.00 mm, apertures "
                           "10.27/6.65/6.18/6.65/10.27 mm, cavities 14.29/15.73/15.73/14.29 mm) "
                           "built at dx = a/90 and compared against a TEn0 mode-matching "
                           "cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. Stage "
                           "S3 of the waveguide-obstacle campaign and the first RESONANT "
                           "multi-obstacle case in the lane: unlike the single iris of S1, a "
                           "per-face geometry error here is a passband shift rather than a "
                           "magnitude tolerance. GATED: centre frequency f0 within 19 MHz = "
                           "round-up(measured envelope 12.1230 x 1.5), and the structural "
                           "reflection-zero COUNT (an integer, depth-independent), both against "
                           "the oracle evaluated on the AS-REALIZED geometry. Measured d_f0 = "
                           "+12.08 MHz, zeros 3 vs 3. The envelope is a population of NINE "
                           "configurations over four setup axes, not a single run, and each "
                           "axis carries an INTERIOR sample as well as an endpoint: guide "
                           "height b = 4/6/8 cells, run length num_periods 400/600/800, port "
                           "standoff 3.05/7.62/15.24 mm, absorber depth 0.75/1.00/1.25 "
                           "lambda_g. The interior samples are the point rather than "
                           "decoration: a one-alternative-per-axis envelope cannot detect "
                           "NON-MONOTONIC sensitivity, which is exactly the failure of PR #475, "
                           "where three sampled clearances passed while 9 of 13 exceeded the "
                           "gate and the passing three were the sampled ones. Every population "
                           "member carries its own committed |S11| trace, so each residual is "
                           "recomputable rather than a free-floating scalar whose integrity is "
                           "borrowed from asserts living in other tests. WHAT THAT GATE IS AND "
                           "IS NOT, stated because the phrasing invites more than it delivers: "
                           "the population makes the envelope ROBUST rather than resting on one "
                           "datum, but it does not make the gate independent of the datum. The "
                           "spread is 0.06 MHz while every member's |d_f0| is about 12.08 MHz, "
                           "so the envelope is dominated by the RESIDUAL and not by the "
                           "scatter, and gate = round-up(env x 1.5) is therefore 1.5x the "
                           "measured agreement. This is a REGRESSION LOCK with 50 percent "
                           "headroom, not an independent accuracy bound, exactly as the merged "
                           "case 18's gate is; what gives the measured agreement meaning is not "
                           "the gate but the comparison of that agreement against an external "
                           "scale, namely the reference's own 21.9 MHz f0 spread between two "
                           "independent commercial codes. That tightness is the substance of "
                           "the result: the residual is a reproducible systematic difference "
                           "rather than a setup artifact, and at the measured cavity "
                           "sensitivity of -105 MHz/cell it corresponds to about 0.12 cell of "
                           "cavity length. The num_periods = 200 run is EXCLUDED from the "
                           "envelope rather than folded in, because it fails the settling "
                           "criterion at column power 1.207; it stays committed as the evidence "
                           "that the settling gate can fire. WHY f0 AND NOT BANDWIDTH, which is "
                           "the correction this case exists to record: the oracle must be fed "
                           "the geometry that was BUILT, not the geometry that was DRAWN, and "
                           "the three legs of that convention are not equally settled. The "
                           "transverse aperture leg d_c*dx is confirmed to better than 0.05 "
                           "cell by an independent refit of 16 committed case-18 configurations "
                           "during the #499 review -- a session measurement; the committed "
                           "corroboration is the per-run raster assert on the open-node count "
                           "and the exact-mask FDFD agreement. The cavity leg (L_c + 1)*dx - "
                           "the distance between the bounding zeroed node planes - is confirmed "
                           "to 0.04-0.17 cell and carries about 105 of the 107.5 MHz that "
                           "separates a drawn-count oracle from a realized-geometry one. But "
                           "the IRIS-THICKNESS leg is NOT (t_c - 1)*dx: four independent FDTD "
                           "runs at drawn t_c = 2/4/6/8 give a flat offset of "
                           "-0.66/-0.68/-0.68/-0.70 cell, i.e. t_elec is about (t_c - 0.68)*dx, "
                           "matching neither this case's earlier rule nor the merged case 18's "
                           "t_c*dx, with a residual 10-33x below both. That leaves an "
                           "irreducible comparator-input uncertainty of order half a cell, and "
                           "the gated observable is therefore chosen by SENSITIVITY to it: per "
                           "cell of convention error, f0 moves about 2.4 MHz, bandwidth about "
                           "40 MHz, and individual band edges 22-30 MHz. So f0 and the zero "
                           "count are gated; band edges and bandwidth are REPORTED, because a "
                           "+/-20 MHz input uncertainty cannot honestly sit under a 15 MHz "
                           "gate. Adopting the fitted -0.68 cell would absorb the disagreement "
                           "into a free parameter, after which the residual would measure "
                           "nothing - the tautological-validation failure this campaign has hit "
                           "repeatedly - so the offset is recorded as an uncertainty and NOT "
                           "adopted. Handing the oracle the drawn cell counts instead of the "
                           "realized ones biases f0 by +107.5 MHz, five times the reference's "
                           "own 21.9 MHz CST-vs-HFSS spread, and the envelope-times-1.5 rule "
                           "does NOT catch that class because the rule bounds SCATTER and this "
                           "is BIAS. The realized lengths are read back off the rasterized "
                           "metal and re-derived again from the committed node indices in the "
                           "frozen gate test. Total electrical length is a face-continuity "
                           "CHECK across region types, NOT a uniqueness argument: putting the "
                           "metal/open interface at sigma*dx beyond the outermost metal node "
                           "gives total = span - 1 + 2*sigma = the outer extent measured at the "
                           "same sigma, conserved for EVERY sigma, and sigma = 0.5 is the drawn "
                           "pairing itself. An earlier revision claimed this was \"the only "
                           "pairing that conserves total electrical length\"; that is false and "
                           "is withdrawn. Drawn counts are COMPENSATED (t_c = round(t/dx) + 1, "
                           "L_c = round(L/dx) - 1) so the stated electrical dimensions land on "
                           "nominal, which is nearest-representable rounding with zero free "
                           "parameters and no reference number entering - that, not any mesh "
                           "comparison, is why it is a statement of intent rather than a fit. "
                           "It is NOT a monotone improvement: at a/60 it moves f0 from -35.8 to "
                           "+120.2 MHz, because it converts a uniformly-signed set of "
                           "per-cavity errors into a mixed-sign one. REPORTED, NEVER GATED: "
                           "individual band edges (+17.08 / +7.09 MHz) and bandwidth (-9.99 "
                           "MHz), which are ONE fact and not two - d_bw is identically d_hi - "
                           "d_lo - so the earlier framing of an \"unexplained asymmetric edge "
                           "residual\" separate from a bandwidth deficit was an algebraic error; "
                           "worst in-band return loss; individual ripple levels; every "
                           "reflection-zero DEPTH (four nominally identical equiripple zeros "
                           "bottom out across a wide spread in the published figure, so the "
                           "paper's frequency step and not physics sets those depths - zero "
                           "FREQUENCIES are meaningful, depths are not values); passband "
                           "contiguity; the coarse a/60 rung; and phase. PASSBAND CONTIGUITY IS "
                           "RECORDED, NOT ASSUMED: lo and hi are the OUTERMOST interpolated -10 "
                           "dB crossings, so the span between them is not necessarily a "
                           "passband. The built filter has one 10 MHz bin at 9.80 dB inside its "
                           "340 MHz span (longest contiguous -10 dB run 270 MHz), while the "
                           "oracle on the same geometry is contiguous over 35 bins - a real "
                           "difference that an earlier revision hid behind a clamped statistic, "
                           "because worst-RL had been computed as the minimum over samples "
                           "already filtered to >= 10 dB and therefore could not report a "
                           "violation at all. The coarse a/60 rung has no meaningful passband: "
                           "16 of 24 bins in its nominal span are above -10 dB with an interior "
                           "trough at 2.73 dB, i.e. two separated resonances. That, and not any "
                           "gate comparison, is the evidence that the gated mesh had to be "
                           "a/90. The a/60 rung's own numbers do not disqualify it cleanly: its "
                           "zero count matches its oracle (2 vs 2) and its f0 residual (+19.85 "
                           "MHz) is the same ~0.12-cell offset seen at a/90; against the "
                           "committed 19 MHz constant it happens to fail by 0.85 MHz, but a "
                           "self-derived envelope-times-1.5 gate would pass it. The broken "
                           "passband is the disqualifier. SETUP IS GATED SEPARATELY FROM "
                           "PHYSICS, because a resonant band read off an unsettled or "
                           "absorber-limited run is not a measurement. The repo's preferred "
                           "ENERGY-BASED ring-down witness (terminal energy in dB below the "
                           "post-source peak, rule < -40 dB) is NOT AVAILABLE on this path: it "
                           "is implemented for the lumped/MSL S-matrix extractor, but "
                           "compute_waveguide_s_matrix returns no settling_db, and the null "
                           "fields are committed in every row rather than papered over (filed "
                           "as an rfx capability gap). The ENFORCED settling criterion is "
                           "therefore the independent axis: the num_periods scan 400/600/800 "
                           "holds f0 and BW to under 0.1 MHz (the gate is the 400 -> 800 "
                           "doubling within one 10 MHz bin), the gated run is passivity-clean "
                           "at column power 1.0065, and the criterion demonstrably fires -- the "
                           "np=200 run is excluded non-passive at 1.207. The feed-clearance and "
                           "absorber-depth scans each hold the edges to one 10 MHz bin across "
                           "their interior and outer samples (standoff 3.05 -> 7.62 -> 15.24 "
                           "mm: 0.0/0.1 MHz; absorber 0.75 -> 1.00 -> 1.25 lambda_g: 0.0/0.1 "
                           "MHz). Absorber depth is scanned because in S1 it was the "
                           "envelope-limiting term at 0.5 lambda_g; here 0.75 lambda_g is "
                           "measurably sufficient, which is a negative result worth recording "
                           "rather than a rule inherited. Extractor warnings and the passivity "
                           "footprint (the bins where column power exceeds 1.02, not merely the "
                           "scalar maximum) are committed per row. Guide height is reduced to 4 "
                           "cells on a MEASURED b-invariance witness: b = 4 and b = 8 agree to "
                           "152 Hz in f0 on THIS resonant five-iris filter, not merely on the "
                           "single iris where it was first measured, which is the 8x saving "
                           "that makes the case affordable to generate. THE ORACLE HAS HAD AN "
                           "ADVERSARIAL PASS, and it did not find the residual. Its own "
                           "witnesses are unitarity 2.2e-15, reciprocity and mirror symmetry "
                           "exact, and an L -> 0 collapse closing two thin irises onto one "
                           "thick one; its N=1 centred limit reproduces the merged case-18 "
                           "oracle to 1.05e-04 in an independent odd-mode formulation, and THAT "
                           "object is what PR #480 confirmed against a formulation-independent "
                           "2-D H-plane FDFD at 5.8e-4 - a comparison the frozen gate test now "
                           "EXECUTES rather than asserting in prose. Those witnesses have known "
                           "limits, stated because an earlier revision leaned on them too hard: "
                           "unitarity constrains only the propagating sub-block, mirror "
                           "symmetry holds by construction for a symmetric geometry, and "
                           "injected overlap-integral errors leave the reduction and collapse "
                           "axes silent because they share the overlap routine, so the gate "
                           "test's re-typed cascade agreeing to 0.0e+00 is a REGRESSION LOCK "
                           "and not a second opinion. An independent review then closed the "
                           "real gaps. What is COMMITTED: the overlap integral is validated "
                           "against direct numerical quadrature IN CI (216 combinations across "
                           "three apertures including the as-realized 6.096 mm -- where n*pi/a "
                           "equals m*pi/d exactly for six mode pairs -- centred and 0.19 mm "
                           "off-centre, worst deviation bounded at 1e-12 with the "
                           "small-denominator guard exercised), and the oracle's truncation is "
                           "witnessed on the GATED observable at generation time: f0 by "
                           "bisection moves 0.33 MHz for n_a 90 -> 130 and 1.85 MHz for an "
                           "aperture-mode-count doubling -- the aperture axis is the sensitive "
                           "one, and both are an order under the 19 MHz gate. Session "
                           "measurements recorded in the research notes but NOT in this record "
                           "put the gauge invariance of the sqrt(Y) normalisation at machine "
                           "precision, truncation saturation near 1 MHz in bandwidth, and "
                           "inter-cavity evanescent transport near 1.6 MHz; they are "
                           "corroborating colour rather than load-bearing, because the "
                           "formulation-level check below subsumes their role. THE "
                           "FORMULATION-LEVEL CHECK IS NOW DONE, and it lands the residual on "
                           "the rfx side. A 2-D H-plane FDFD -- scalar Helmholtz on a "
                           "finite-difference grid with an exact discrete transparent port "
                           "condition, sharing only numpy and scipy with the cascade and no rfx "
                           "code path at all (validation/crossval/comparators/fdfd_hplane.py) "
                           "-- was run on the same electrical geometry, grid-exact at every "
                           "refinement level. It is FIRST-order convergent, so no single level "
                           "is meaningful; the record carries THREE levels (r = 2, 3, 4, whose "
                           "bandwidth deviations from the extrapolate shrink as 1/r: measured "
                           "ratios 1.55 and 1.33 against the first-order 1.50 and 1.33) and "
                           "BOTH Richardson estimates, which agree to 0.37 MHz in centre "
                           "frequency and 0.36 MHz in bandwidth -- the two-estimate consistency "
                           "protocol the porting handoff mandates before either estimate is "
                           "trusted. The finer pair gives f0 = 10.95742 GHz and BW = 351.42 MHz "
                           "against the cascade's 10.95851 and 350.43: agreement to 1.09 MHz in "
                           "centre frequency and 0.98 MHz in bandwidth between two formulations "
                           "that share nothing but their numerical libraries, with the "
                           "extrapolation's own consistency (0.37/0.36 MHz) bounding how much "
                           "of that gap the FDFD itself owns. rfx differs from the FDFD by "
                           "+13.17 MHz in f0 and -10.97 MHz in bandwidth, essentially the same "
                           "as it differs from the cascade (+12.08 / -9.99), so the 12 MHz "
                           "residual is not an oracle error. The FDFD's gates: lossless "
                           "unitarity is enforced on EVERY evaluation (worst 4.6e-07 across all "
                           "levels), and the empty-guide transparency gate -- |S11| = 5.0e-14 "
                           "with |S21| = 1.000000000000, the test that originally caught a "
                           "missing 1/h in the discrete propagation constant -- runs once per "
                           "generation and once per CI pass. One defect in the comparator "
                           "itself was found by an independent port review and fixed before "
                           "this record was generated: its aperture mask realized every "
                           "aperture two fine cells wide of the stated convention, a "
                           "first-order bias that Richardson cancelled -- making the "
                           "extrapolated numbers right for the wrong per-level geometry -- and "
                           "that produced a spurious FOURTH reflection zero at the coarser "
                           "levels. With the mask exact, every level shows THREE zeros, "
                           "matching the cascade and rfx. Independently, the cascade's zero "
                           "count was checked against its own aperture-mode truncation, which "
                           "nobody had done for the COUNT: it is 3 at nb_scale 1.0, 1.5, 2.0 "
                           "and 3.0, with f0 moving 1.7 MHz over that 3x range. What remains "
                           "genuinely unexplained is the ~12 MHz rfx residual itself: it is "
                           "mesh-invariant when expressed in cells (-0.1169 cell at a/90 "
                           "against -0.1241 at a/60, where dispersion would have given 0.083), "
                           "so it behaves like a fixed geometric offset rather than a "
                           "frequency-dependent solver error, but attributing it to a specific "
                           "convention leg has FAILED: propagating the independently measured "
                           "iris thickness (t_c - 0.68)*dx through node-plane length "
                           "conservation overshoots and flips the sign, taking a/90 from +12.08 "
                           "to -30.62 MHz. That attribution is recorded as falsified, not as "
                           "pending. _gamma at exact cutoff (k equal to n*pi/w, where the sqrt "
                           "argument vanishes) is unreachable at these band edges and untested. "
                           "FENCED: nothing here promotes the lane beyond S1. Multi-iris "
                           "filters, posts and septa remain EXPERIMENTAL; this measures one "
                           "published design on one mesh with one gated observable, and "
                           "certifies neither arbitrary filters nor the a/60 rung. Says nothing "
                           "about phase, group delay, loss, higher-order-mode ports, or "
                           "fabrication tolerance. The reference is an ANCHOR, not a solver "
                           "run: the CST and HFSS scalars are digitized from the paper's Fig. "
                           "5, no external solver is invoked here, and the case does not "
                           "compare rfx against either commercial code on any geometry - so the "
                           "12.1 MHz f0 residual against this case's own analytic oracle must "
                           "not be read as an accuracy claim relative to CST or HFSS. Reported "
                           "for context and not as a yardstick beaten: the oracle on nominal "
                           "dimensions sits -6.2 MHz from CST and -28.2 MHz from HFSS in f0, "
                           "+14.7 MHz from both in bandwidth (against a published inter-solver "
                           "bandwidth spread of only 0.4 MHz), -0.4 and -1.1 dB in worst return "
                           "loss, and up to 25.4 and 61.9 MHz in individual reflection-zero "
                           "frequencies. The bandwidth and zero-frequency misses are larger "
                           "than the f0 miss and are stated here because quoting only f0 would "
                           "be selective. The built structure is a SNAPPED Aghanim filter: its "
                           "centre frequency is within the reference's own solver scatter of "
                           "CST (though not of HFSS), one of the four structural reflection "
                           "zeros is lost (4 -> 3, confirmed grid-robust by refining the oracle "
                           "to 1 MHz, with the loss occurring in the upper band), and worst "
                           "in-band return loss degrades from 13.82 dB to 10.65 dB by "
                           "rasterization alone, oracle to oracle, with rfx at 9.80 dB. Say "
                           "snapped, not equivalent. OBSERVABLE PRIORITY for a resonant "
                           "structure, as this case measures it: the structural reflection-zero "
                           "COUNT first (an integer, depth-independent, and shown grid-robust), "
                           "then centre frequency (least sensitive of the continuous quantities "
                           "to the unsettled convention, ~2.4 MHz per cell), then band edges "
                           "and bandwidth (~22-40 MHz per cell, hence reported), then worst "
                           "return loss, and last individual ripple levels and null depths, "
                           "which are not values at all. TOPOLOGY FIRST, AND f0 IS NOT "
                           "EXONERATED: the zero count is the most robust observable, but f0 is "
                           "not thereby safe -- it carries the +12.08 MHz residual this case "
                           "gates, and at -105 MHz per cell of cavity length it is the quantity "
                           "a geometry error moves first. A cell snap is inherently "
                           "non-uniform, since each cavity rounds independently, so every snap "
                           "figure quoted here was MEASURED on the as-snapped geometry and none "
                           "may be re-derived by multiplying a sensitivity coefficient by a "
                           "half cell.",
            "config": {
                "a_m": A_WR90, "freqs_hz": [float(f) for f in FREQS],
                "gated_cells_per_a": GATED_CELLS,
                "coarse_cells_per_a": COARSE_CELLS,
                "b_cells": B_CELLS, "feed_cells": FEED_CELLS,
                "port_cells": PORT_CELLS,
                "cpml_fraction_of_lambda_g": CPML_FRACTION,
                "gated_normalize": "flux",
            },
            "gates": {
                "f0_gate_mhz": GATE_F0_MHZ,
                "f0_measured_envelope_mhz": round(env_f0, 4),
                "f0_envelope_population": [
                    {"config": t, "d_f0_mhz": round(d, 4)} for t, d in residuals],
                "f0_population_excluded": [
                    {"config": t, "max_colpow": cp,
                     "reason": "fails the settling criterion (column power > 1.02)"}
                    for t, cp in excluded],
                "f0_population_criterion": (
                    "every measured leg enters the envelope unless its column "
                    "power exceeds 1.02; membership is a criterion, not a value "
                    "list, so a future failing row cannot be dropped by naming it"),
                "edge_reported_residual_mhz": env_edge,
                "bw_reported_residual_mhz": env_bw,
                "posture": ("gate = round-UP(measured envelope x 1.5) over a "
                            "MULTI-CONFIGURATION population, enforced as EXACT "
                            "equality by the write-fixture self-check. That "
                            "makes the envelope robust rather than resting on "
                            "one datum, but it does NOT make the gate "
                            "independent of the datum: the population spread "
                            "is 0.06 MHz while every member is about 12.08 MHz "
                            "from the oracle, so the envelope is dominated by "
                            "the residual and the gate is 1.5x the measured "
                            "agreement. It is a REGRESSION LOCK with 50% "
                            "headroom, not an independent accuracy bound. What "
                            "gives the agreement meaning is its comparison "
                            "against an external scale (the reference's own "
                            "21.9 MHz f0 spread between two independent "
                            "commercial codes), not the gate. GATED: "
                            "centre frequency f0 and the structural zero COUNT, "
                            "against the oracle on as-realized geometry. "
                            "REPORTED, never gated: individual band edges and "
                            "bandwidth (their comparator-input uncertainty from "
                            "the unsettled iris-thickness convention, ~20 MHz, "
                            "exceeds any defensible gate on them, and d_bw is "
                            "identically d_hi - d_lo so they are one fact), "
                            "worst-case RL, ripple levels, zero depths, "
                            "passband contiguity, the coarse rung and phase"),
            },
            "oracle_nominal_band": nom,
            "oracle_rasterized_band": ras,
            "gated_rfx": row,
            "coarse_diagnostic": coarse,
            "ring_down_witness": ring,
            "b_invariance_witness": binv,
            "feed_clearance_witness": clearance,
            "absorber_depth_witness": absorber,
            "fdfd_formulation_independent": fdfd,
            "electrical_geometry": {
                "rule": ("oracle inputs are READ BACK off the rasterized metal. "
                         "The CAVITY leg is (L_c + 1)*dx -- the distance "
                         "between the bounding zeroed node planes -- and is "
                         "confirmed to 0.04-0.17 cell by the committed "
                         "residual against the measured cavity sensitivity; "
                         "the transverse APERTURE leg d_c*dx is confirmed to "
                         "better than 0.05 cell by an independent refit of 16 "
                         "committed case-18 configurations. The IRIS-THICKNESS "
                         "leg is NOT (t_c - 1)*dx: four FDTD runs at drawn "
                         "t_c = 2/4/6/8 give a flat offset of -0.66/-0.68/"
                         "-0.68/-0.70 cell, i.e. t_elec ~ (t_c - 0.68)*dx, "
                         "matching neither this rule nor case 18's t_c*dx. "
                         "That ~1/3-cell ambiguity is an irreducible "
                         "comparator-input uncertainty here and it is why "
                         "bandwidth and individual band edges are REPORTED "
                         "rather than gated (they move ~40 and ~22-30 MHz per "
                         "cell of it) while f0 and the zero count are gated "
                         "(~2.4 MHz per cell, and an integer). Total electrical "
                         "length is a face-continuity CHECK across region "
                         "types, NOT a uniqueness argument: putting the "
                         "interface at sigma*dx beyond the outermost metal "
                         "node gives total = span - 1 + 2*sigma = the outer "
                         "extent measured at the same sigma, conserved for "
                         "EVERY sigma, and sigma = 0.5 is the drawn pairing "
                         "itself."),
                "iris_thickness_cells": geo_g_e["electrical_thickness_cells"],
                "cavity_cells": geo_g_e["electrical_cavity_cells"],
                "drawn_iris_thickness_cells": int(geo_g["t_cells"]),
                "drawn_cavity_cells": [int(v) for v in geo_g["L_cells"]],
                "compensation": ("drawn counts are chosen so the ELECTRICAL "
                                 "dimensions land on nominal: t_c = "
                                 "round(t/dx) + 1, L_c = round(L/dx) - 1. At "
                                 "a/90 that puts f0 +3.3 MHz from the paper's "
                                 "exact design (inside its own 21.9 MHz "
                                 "CST-vs-HFSS spread) against -101.4 MHz "
                                 "uncompensated. Compensation is NOT a monotone "
                                 "improvement -- it only picks which side of the "
                                 "sub-cell rounding you land on, and at a/60 it "
                                 "moves f0 from -35.8 to +120.2 MHz."),
                "cost_of_using_intended_counts_mhz": 107.5,
                "cost_note": ("measured 2026-07-29: feeding the oracle the DRAWN "
                              "cell counts puts its band at 10.9054-11.2267 GHz "
                              "against 10.7833-11.1337 GHz for the realized "
                              "geometry, a +107.5 MHz f0 error -- five times the "
                              "paper's own 21.9 MHz CST-vs-HFSS spread, and it "
                              "would have had to be absorbed by a ~162 MHz gate "
                              "(46% of the passband) that pins nothing. First "
                              "found as +90.0 MHz on the uncompensated counts; "
                              "compensation changes which pair is confused, not "
                              "the class."),
            },
            "provenance": {
                "generated_by": ("validation/crossval/"
                                 "19_wr90_iris_filter_aghanim.py --write-fixture"),
                "oracle": ("in-script TEn0 mode-matching cascade with arbitrary "
                           "aperture position; the centred limit reproduces the "
                           "merged case-18 single-iris oracle to 1.05e-04 in an "
                           "independent odd-mode formulation (the 1.8e-16 figure "
                           "an earlier revision quoted is same-code tautology), so it "
                           "inherits S1's validation including the PR #480 "
                           "review's formulation-independent FDFD confirmation "
                           "at 5.8e-4"),
                "no_preflight_note": (
                    "compute_waveguide_s_matrix runs its own extractor "
                    "passivity/finiteness self-check (its warnings are part of "
                    "this record) but the functional path runs no "
                    "sim.preflight(); the operating-point guarantees are the "
                    "per-iris raster asserts in raster_assert plus the derived "
                    "CPML depth."),
            },
        }
        art_dir = os.path.join(_SCRIPT_DIR, "_19_iris_filter_results")
        os.makedirs(art_dir, exist_ok=True)
        with open(os.path.join(art_dir, "rfx.json"), "w") as f:
            json.dump(payload, f, indent=1)
        fix_dir = os.path.join(_REPO_ROOT, "tests", "fixtures", "wr90_iris_filter")
        os.makedirs(fix_dir, exist_ok=True)
        with open(os.path.join(fix_dir, "fixture.json"), "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {art_dir}/rfx.json and tests/fixtures/wr90_iris_filter/fixture.json")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
