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
GATED (exit-1), gates = round-UP(measured envelope x 1.5), enforced as EXACT
equality by the --write-fixture self-check:
  * band edges and bandwidth of |S11| <= -10 dB, rfx vs oracle@as-rasterized,
    at the gated mesh a/90.  Edges and BW are where the reference's own two
    solvers agree to 0.4 MHz, so they are the defensible tight observables.
  * structural reflection-zero COUNT inside the passband (a depth-independent
    local-minimum count).  This is the N=4 topology check.
REPORTED, NOT GATED:
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
    uses asymmetric fin placement, which preserves the four-zero topology
    where symmetric even-cell snapping destroys it (measured: 4 zeros vs 2).
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
      # + coarse diagnostic rung, ring-down and b-invariance witnesses,
      # snap decomposition and the paper anchor; regenerates
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

# gates = round-UP(measured envelope x 1.5), MEASURED 2026-07-29 (edge envelope
# 17.08 MHz, BW envelope 9.99 MHz); the --write-fixture self-check demands exact
# equality with that rule, so these cannot be nudged without a re-measurement.
GATE_EDGE_MHZ = 26.0      # band-edge agreement, rfx vs oracle@as-rasterized
GATE_BW_MHZ = 15.0        # bandwidth agreement


# --------------------------------------------------------------------------- #
# Oracle: TEn0 mode-matching cascade with arbitrary aperture position.
# The centred limit reproduces the merged single-iris oracle of case 18 to
# 1.8e-16, so this inherits S1's validation (including the PR #480 review's
# formulation-independent FDFD confirmation at 5.8e-4).
# --------------------------------------------------------------------------- #
def _gamma(n, width, k):
    return np.sqrt(complex((n * np.pi / width) ** 2 - k * k))


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


def _iris(a, d, x0, t, k, n_a):
    n_b = max(6, int(round(n_a * d / a)))
    fwd = _step(a, d, x0, k, n_a, n_b)
    rev = (fwd[3], fwd[2], fwd[1], fwd[0])
    return _star(_star(fwd, _line(d, t, k, n_b)), rev)


def filter_s11_s21(a, apertures, offsets, thicknesses, cavities, freq, n_a=90):
    """(S11, S21) of the N-iris filter; evanescent modes carried across cavities."""
    k = 2 * np.pi * freq / C0
    tot = _iris(a, apertures[0], offsets[0], thicknesses[0], k, n_a)
    for i, L in enumerate(cavities):
        tot = _star(tot, _line(a, L, k, n_a))
        tot = _star(tot, _iris(a, apertures[i + 1], offsets[i + 1],
                               thicknesses[i + 1], k, n_a))
    return tot[0][0, 0], tot[2][0, 0]


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
    w["mode_convergence"] = abs(v90 - v130)
    assert w["mode_convergence"] < 5e-3
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
    (L_c + 1)*dx and each iris (t_c - 1)*dx, and only that pair conserves the
    cascade's total electrical length: 5*(t-1) + 4*(L+1) = span - 1 cells,
    whereas mixing the conventions overshoots by 4 or undershoots by 5 cells.

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
    res = sim.compute_waveguide_s_matrix(normalize="flux", num_periods=num_periods)
    wall = time.time() - t0
    s = np.asarray(res.s_params)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    return dict(cells_per_a=cells, dx_mm=round(geo["dx"] * 1e3, 4),
                b_cells=b_cells, cpml_cells=cpml_c, grid=list(grid.shape),
                glen_cells=glen_c, num_periods=num_periods,
                feed_cells=feed_cells, port_cells=port_cells,
                cpml_fraction=cpml_fraction,
                aperture_nodes=realized, iris_x_nodes=x_runs,
                s11=[round(float(v), 6) for v in s11],
                s21=[round(float(v), 6) for v in s21],
                max_colpow=round(float(np.max(s11 ** 2 + s21 ** 2)), 4),
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
    return dict(lo=lo, hi=hi, f0=(lo + hi) / 2, bw=hi - lo,
                worst_rl_db=float(rl[inb].min()), zeros=zeros)


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
    print(f"\n== GATED rfx a/{GATED_CELLS} vs oracle@as-rasterized "
          f"(edges {GATE_EDGE_MHZ} MHz, BW {GATE_BW_MHZ} MHz) ==")
    row = measure(geo_g, 400.0)
    meas = band_analysis(row["s11"])
    d_lo = (meas["lo"] - ras["lo"]) / 1e6
    d_hi = (meas["hi"] - ras["hi"]) / 1e6
    d_bw = (meas["bw"] - ras["bw"]) / 1e6
    row.update(oracle_s11=ras_curve, band=meas, oracle_band=ras,
               d_lo_mhz=round(d_lo, 2), d_hi_mhz=round(d_hi, 2),
               d_bw_mhz=round(d_bw, 2))
    edge_ok = max(abs(d_lo), abs(d_hi)) <= GATE_EDGE_MHZ
    bw_ok = abs(d_bw) <= GATE_BW_MHZ
    zeros_ok = len(meas["zeros"]) == len(ras["zeros"])
    ok &= edge_ok and bw_ok and zeros_ok
    print(f"  grid {tuple(row['grid'])} cpml={row['cpml_cells']} "
          f"({row['wall_s']:.0f}s): band {meas['lo']/1e9:.4f}-{meas['hi']/1e9:.4f} "
          f"f0={meas['f0']/1e9:.4f} BW={meas['bw']/1e6:.0f} MHz "
          f"worst RL={meas['worst_rl_db']:.1f} dB zeros={len(meas['zeros'])}")
    print(f"  vs oracle@rasterized: edges {d_lo:+.1f}/{d_hi:+.1f} MHz "
          f"({'PASS' if edge_ok else 'FAIL'}), BW {d_bw:+.1f} MHz "
          f"({'PASS' if bw_ok else 'FAIL'}), zeros {len(meas['zeros'])} vs "
          f"{len(ras['zeros'])} ({'PASS' if zeros_ok else 'FAIL'}), "
          f"colpow {row['max_colpow']:.4f}")

    coarse = ring = binv = clearance = absorber = None
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
        for npd in (200.0, 400.0, 800.0):
            # num_periods=400 IS the gated configuration, so reuse that row
            # rather than paying for a bit-identical repeat run.
            r = row if npd == 400.0 else measure(geo_g, npd)
            b = meas if npd == 400.0 else band_analysis(r["s11"])
            ring.append({"num_periods": npd, "f0": b["f0"], "bw": b["bw"],
                         "max_colpow": r["max_colpow"], "wall_s": r["wall_s"]})
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
        for bc in (4, 8):
            # b=B_CELLS at num_periods=400 is the gated row; reuse it.
            r = row if bc == B_CELLS else measure(geo_g, 400.0, b_cells=bc)
            b = meas if bc == B_CELLS else band_analysis(r["s11"])
            binv.append({"b_cells": bc, "f0": b["f0"], "bw": b["bw"],
                         "max_dev_vs_b4": None, "wall_s": r["wall_s"]})
            print(f"  b={bc} cells: f0={b['f0']/1e9:.4f} BW={b['bw']/1e6:.0f} MHz "
                  f"({r['wall_s']:.0f}s)", flush=True)
        if len(binv) == 2:
            binv[1]["max_dev_vs_b4"] = abs(binv[1]["f0"] - binv[0]["f0"])

        # Feed clearance. CPML is exterior to the requested domain, so the
        # absorber never overlaps the irises; what IS thin is the standoff
        # between the absorber interface and the port plane — PORT_CELLS*dx is
        # about 0.08*lambda_g at a/90 where the S1 recipe used roughly one
        # guide wavelength. Deviating from a validated recipe is witnessed,
        # not assumed: re-run the SAME geometry with a generous feed and
        # require the band edges to hold to one frequency bin, since the only
        # thing that changed is standoff.
        print("\n== feed-clearance witness (port standoff from the absorber) ==")
        gen = measure(geo_g, 400.0, feed_cells=FEED_CELLS_WITNESS,
                      port_cells=PORT_CELLS_WITNESS)
        gb = band_analysis(gen["s11"])
        d_lo_c = abs(gb["lo"] - meas["lo"])
        d_hi_c = abs(gb["hi"] - meas["hi"])
        clear_ok = d_lo_c <= bin_hz and d_hi_c <= bin_hz
        ok &= clear_ok
        clearance = {
            "gated": {"feed_cells": FEED_CELLS, "port_cells": PORT_CELLS,
                      "standoff_mm": round(PORT_CELLS * geo_g["dx"] * 1e3, 3),
                      "lo": meas["lo"], "hi": meas["hi"]},
            "generous": {"feed_cells": FEED_CELLS_WITNESS,
                         "port_cells": PORT_CELLS_WITNESS,
                         "standoff_mm": round(
                             PORT_CELLS_WITNESS * geo_g["dx"] * 1e3, 3),
                         "lo": gb["lo"], "hi": gb["hi"],
                         "max_colpow": gen["max_colpow"],
                         "wall_s": gen["wall_s"]},
            "d_lo_mhz": round(d_lo_c / 1e6, 2),
            "d_hi_mhz": round(d_hi_c / 1e6, 2),
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
        deep = measure(geo_g, 400.0, cpml_fraction=CPML_FRACTION_WITNESS)
        db = band_analysis(deep["s11"])
        d_lo_a = abs(db["lo"] - meas["lo"])
        d_hi_a = abs(db["hi"] - meas["hi"])
        absorber_ok = d_lo_a <= bin_hz and d_hi_a <= bin_hz
        ok &= absorber_ok
        absorber = {
            "gated": {"cpml_fraction": CPML_FRACTION,
                      "cpml_cells": row["cpml_cells"],
                      "lo": meas["lo"], "hi": meas["hi"]},
            "deep": {"cpml_fraction": CPML_FRACTION_WITNESS,
                     "cpml_cells": deep["cpml_cells"],
                     "lo": db["lo"], "hi": db["hi"],
                     "max_colpow": deep["max_colpow"], "wall_s": deep["wall_s"]},
            "d_lo_mhz": round(d_lo_a / 1e6, 2),
            "d_hi_mhz": round(d_hi_a / 1e6, 2),
            "passed": bool(absorber_ok),
        }
        print(f"  {CPML_FRACTION}->{CPML_FRACTION_WITNESS} lambda_g "
              f"({row['cpml_cells']}->{deep['cpml_cells']} cells): edges move "
              f"{absorber['d_lo_mhz']:.1f}/{absorber['d_hi_mhz']:.1f} MHz "
              f"(one bin = {bin_hz/1e6:.0f} MHz) -> "
              f"{'PASS' if absorber_ok else 'FAIL — absorber-limited'}")

        env_edge = max(abs(row["d_lo_mhz"]), abs(row["d_hi_mhz"]))
        env_bw = abs(row["d_bw_mhz"])
        print(f"\n  envelopes: edge {env_edge:.2f} MHz (gate {GATE_EDGE_MHZ}), "
              f"BW {env_bw:.2f} MHz (gate {GATE_BW_MHZ})")
        for gate, env, tier in ((GATE_EDGE_MHZ, env_edge, "edge"),
                                (GATE_BW_MHZ, env_bw, "bw")):
            required = np.ceil(max(env, 1e-9) * 1.5)
            if abs(gate - required) > 1e-9:
                print(f"  ENVELOPE/GATE MISMATCH ({tier}): gate {gate} must equal "
                      f"round-up(env x 1.5) = {required}")
                ok = False

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
                           "cascade oracle over 10.40-11.70 GHz on 131 points at 10 MHz. This "
                           "is stage S3 of the waveguide-obstacle campaign, the first RESONANT "
                           "multi-obstacle case in the lane: unlike the single iris of S1, a "
                           "per-face geometry error here is a passband shift rather than a "
                           "magnitude tolerance. GATED: band edges of the -10 dB |S11| passband "
                           "within 26 MHz = round-up(measured envelope 17.1 x 1.5), bandwidth "
                           "within 15 MHz = round-up(10.0 x 1.5), and the structural "
                           "reflection-zero COUNT (a depth-independent local-minimum count), "
                           "all against the oracle evaluated on the AS-REALIZED geometry; the "
                           "write-fixture self-check enforces gate == round-up(envelope x 1.5) "
                           "as EXACT equality. Measured: edges +17.1/+7.1 MHz, BW -10.0 MHz, "
                           "zeros 3 vs 3. The 17.1 MHz edge agreement sits BELOW the "
                           "reference's own 21.9 MHz CST-vs-HFSS spread, i.e. rfx matches the "
                           "analytic cascade on band edges more closely than the paper's two "
                           "commercial solvers match each other - a claim available only "
                           "because the edges are interpolated -10 dB crossings, since on the "
                           "sampled 10 MHz grid 17.1 and 7.1 MHz both quantize away. THE ORACLE "
                           "IS FED THE GEOMETRY THAT WAS BUILT, NOT THE GEOMETRY THAT WAS "
                           "INTENDED, and that distinction is the substance of this case: the "
                           "electrical length of a region is the distance between its bounding "
                           "zeroed node planes, so a cavity drawn with L_c cells of clear space "
                           "is (L_c + 1)*dx and an iris drawn t_c cells thick is (t_c - 1)*dx - "
                           "the transverse S1 rule applied along x, and the only pairing that "
                           "conserves the cascade's total electrical length (span - 1 cells). "
                           "Handing the oracle the drawn cell counts instead biases f0 by "
                           "+107.5 MHz at the shipped geometry (drawn t=9, L=55/61/61/55 fed as "
                           "if electrical gives 10.9054-11.2267 GHz where the realized t=8, "
                           "L=56/62/62/56 gives 10.7833-11.1337), five times the reference's "
                           "own 21.9 MHz solver spread. The envelope-times-1.5 rule does NOT "
                           "catch that class, because the rule bounds SCATTER and this is BIAS: "
                           "it launders the error into a ~162 MHz \"measured\" gate, 46% of the "
                           "passband, that pins nothing. The defect was first found as +90.0 "
                           "MHz on the uncompensated counts; compensation changes which pair "
                           "gets confused, not the class. The lengths are therefore re-derived "
                           "from the committed rasterized node indices in the frozen gate test, "
                           "not asserted in prose. Knowing the convention, the drawn counts are "
                           "COMPENSATED (t_c = round(t/dx) + 1, L_c = round(L/dx) - 1) so the "
                           "electrical dimensions land on nominal; that puts the built filter's "
                           "f0 +3.3 MHz from the paper's exact design, inside the 21.9 MHz "
                           "spread. Compensation is NOT a monotone improvement - it only "
                           "chooses which side of the sub-cell rounding you land on, and at "
                           "a/60 it moves f0 from -35.8 to +120.2 MHz - so it is adopted as a "
                           "statement of intent and its residual is measured, never predicted. "
                           "The built structure is a SNAPPED Aghanim filter: its centre "
                           "frequency is the paper's to within the reference's solver scatter, "
                           "but one of the four structural reflection zeros is lost (4 -> 3) "
                           "and worst in-band return loss degrades 13.8 -> 10.6 dB from the "
                           "snap ALONE (oracle to oracle, with rfx at 10.1 dB), so the RL "
                           "degradation is attributable to rasterization rather than to the "
                           "solver. Say snapped, not equivalent. REPORTED, NEVER GATED: worst "
                           "in-band RL (the reference is not self-consistent there - HFSS "
                           "ripple peaks -19.3/-14.9/-18.4 dB vs CST -24.9/-18.7/-14.2 dB, "
                           "disagreeing on which peak is worst), individual ripple levels, "
                           "every reflection-zero DEPTH (four nominally identical equiripple "
                           "zeros bottom out across a 16 dB spread in the published figure, "
                           "which proves the paper's frequency step sets those depths - zero "
                           "FREQUENCIES are meaningful, depths are not values), the coarse a/60 "
                           "rung (snap f0 +120 MHz, zeros 4 -> 2, rfx-vs-its-own-oracle edges "
                           "+24.5/+15.2 MHz), and phase. SETUP IS GATED SEPARATELY FROM "
                           "PHYSICS, because a resonant band read off an unsettled or "
                           "absorber-limited run is not a measurement: the ring-down must be "
                           "settled at the gated num_periods = 400 (measured: 400 -> 800 moves "
                           "f0 and BW by exactly 0.0 MHz at column power 1.0065 -> 1.0003, "
                           "while num_periods = 200 is non-passive at 1.2070 - truncation shows "
                           "up as non-passivity BEFORE it shows up as a shifted band), and the "
                           "feed-clearance and absorber-depth scans must each hold the edges to "
                           "one bin (measured: a 3.05 -> 15.24 mm port standoff moves the edges "
                           "0.0/0.1 MHz, and deepening the absorber 0.75 -> 1.25 lambda_g, 110 "
                           "-> 183 cells, also 0.0/0.1 MHz). Absorber depth is scanned because "
                           "in S1 (PR #480) it was the envelope-limiting term at 0.5*lambda_g; "
                           "here 0.75*lambda_g at the low band edge is measurably sufficient, "
                           "which is a negative result worth recording rather than a rule "
                           "inherited. Guide height is reduced to 4 cells on a MEASURED "
                           "b-invariance witness (TE10 with y-invariant inductive fins is "
                           "b-independent): b = 4 and b = 8 give identical f0 = 10.9706 GHz and "
                           "BW = 340 MHz on THIS resonant five-iris filter, not merely on the "
                           "single iris where it was first measured. That is the 8x saving "
                           "which makes the case local-CPU affordable, and the witness is "
                           "re-run rather than assumed. The oracle inherits S1's validation: "
                           "its N=1 centred limit reproduces the merged case-18 single-iris "
                           "oracle to 1.8e-16, and that object was confirmed by the PR #480 "
                           "review against a formulation-independent 2-D H-plane FDFD at "
                           "5.8e-4; its own witnesses are unitarity 2.2e-15, exact reciprocity "
                           "and mirror symmetry, mode convergence 8.8e-4 from n_a 90 -> 130, "
                           "and an L -> 0 collapse that turns two 2.00 mm irises 0.2 mm apart "
                           "into one 4.20 mm iris to 0.1%. The frozen gate test's re-typed "
                           "cascade agrees with the producer to 0.0e+00, which makes it a "
                           "REGRESSION LOCK rather than a second opinion, and it is described "
                           "as such; the independent axes (the single-iris reduction, the "
                           "collapse limit, unitarity, mirror symmetry) are re-run in CI "
                           "instead of being trusted from generation time. TOPOLOGY FIRST, AND "
                           "f0 IS NOT EXONERATED: the zero COUNT is the most robust observable "
                           "(an integer, depth-independent), then band edges and BW, then worst "
                           "RL, then ripple levels and null depths (which are not values at "
                           "all); but f0 is 8.1x more sensitive per mm of cavity length than "
                           "bandwidth (-416.1 vs -51.4 MHz/mm, fitted only over the 5 of 6 "
                           "perturbations that preserve the four-zero topology), and plain "
                           "kinematics (-472 MHz/mm) sits within 13.5% of the measured "
                           "coefficient, so a thick iris does not absorb the kinematic shift. "
                           "Those coefficients apply ONLY to topology-preserving, UNIFORM "
                           "perturbations: a cell snap is inherently non-uniform because each "
                           "cavity rounds independently, so every snap budget quoted here was "
                           "MEASURED on the as-snapped geometry and must never be re-derived by "
                           "multiplying a coefficient by a half-cell. FENCED: nothing here "
                           "promotes the lane beyond S1. Multi-iris filters, posts and septa "
                           "remain EXPERIMENTAL per the waveguide-port support matrix; this "
                           "case measures one published design against an analytic oracle on "
                           "one mesh, and certifies neither arbitrary filters nor the a/60 "
                           "rung. Says nothing about phase, group delay, loss, "
                           "higher-order-mode ports, or fabrication tolerance. A digitized "
                           "reference is an anchor, not a solver run: the CST/HFSS scalars come "
                           "from the paper's Fig. 5 (zero frequencies are calibration-invariant "
                           "by construction and bit-identical across three independent "
                           "y-calibrations, worst RL moving <= 0.13 dB), and Setti et al. 2023 "
                           "was evaluated as a second published design and FALSIFIED - its Fig. "
                           "5 geometry closes exactly but implements a ~14.3 GHz filter while "
                           "the VNA trace in the same paper is 9.85 GHz, so measured-hardware "
                           "truth is worthless when the published geometry does not produce the "
                           "published measurement.",
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
                "edge_gate_mhz": GATE_EDGE_MHZ, "edge_measured_envelope_mhz": env_edge,
                "bw_gate_mhz": GATE_BW_MHZ, "bw_measured_envelope_mhz": env_bw,
                "posture": ("gate = round-UP(measured envelope x 1.5), enforced as "
                            "EXACT equality by the write-fixture self-check; band "
                            "edges, bandwidth and the structural zero COUNT are "
                            "gated against the oracle on as-rasterized geometry; "
                            "worst-case RL, ripple levels, zero depths, the coarse "
                            "rung and phase are reported, never gated"),
            },
            "oracle_nominal_band": nom,
            "oracle_rasterized_band": ras,
            "gated_rfx": row,
            "coarse_diagnostic": coarse,
            "ring_down_witness": ring,
            "b_invariance_witness": binv,
            "feed_clearance_witness": clearance,
            "absorber_depth_witness": absorber,
            "electrical_geometry": {
                "rule": ("oracle inputs are READ BACK off the rasterized metal: "
                         "the electrical length of a region is the distance "
                         "between its bounding zeroed node planes, so each "
                         "cavity is (L_c + 1)*dx and each iris (t_c - 1)*dx. "
                         "This is the transverse S1 rule applied along x, and "
                         "it is the only pairing that conserves the cascade's "
                         "total electrical length (span - 1 cells)."),
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
                           "merged case-18 single-iris oracle to 1.8e-16, so it "
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
