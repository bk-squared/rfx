"""WR-90 single symmetric inductive iris vs TEn0 mode-matching (item 3, stage S1).

FIRST calibrated evidence for a PEC obstacle inside the rectangular-waveguide
S-parameter lane (docs/guides/support_matrix.md pins iris/post/septum RF
results as EXPERIMENTAL; the only prior committed iris measurement is a
non-passive-scale reflector-class tripwire). This case establishes the
single-iris extraction envelope BEFORE any multi-iris filter is attempted
(stages S2 Palace / S3 published filter build on it).

Reference: the thick symmetric inductive iris is EXACTLY a cascade of two
symmetric H-plane width-step junctions joined by a length-t width-d guide
section. The oracle solves that cascade by TE_{n0} mode-matching (odd modes,
relative-convergence ratio N_B/N_A ~ d/a for the edge condition) and is
re-derived AGAIN independently in the frozen gate test. Self-witnesses run
before any gate: unitarity |S11|^2+|S21|^2 == 1 (lossless, machine precision),
reciprocity, mode-count convergence, thin-limit vs the Marcuvitz closed form
B/Y0 = (lambda_g/a) cot^2(pi d/2a) (leading-order anchor: 8.6-10.8% with the
inductive sign), d->a identity, deep-constriction limit.

Geometry discipline (grid-exact, comparator-first — the #325/#475 classes):
  * dx = a/30 (coarse) and a/60 (fine): WR-90 a = 22.86 mm is an EXACT
    multiple, so a carries no rasterization ambiguity (ny is the node count,
    a = (ny-1) dx between the PEC wall node planes).
  * iris thickness t = 1.524 mm (exactly 2 coarse / 4 fine cells, and under
    the #931 contract exactly what is realized: walls at both faces, t_c
    cells apart); apertures
    d in {18.288, 12.192, 7.62} mm (24/16/10 coarse cells) — weak / medium /
    strong reflector.
  * fins are drawn PAST the grid walls (the rasterizer clips): the first
    probe revision drew them to the NOMINAL width and left a 1-cell parasitic
    slot at the far wall. run_point asserts, from the realized PEC edge set
    and before any solve, that the iris stands walls at BOTH its drawn faces
    t_c cells apart and that the aperture is ONE contiguous opening d_c cells
    wide, so neither bug class can recur.

WHAT IS GATED vs WHAT IS REPORTED (regenerated under #931 on 2026-09-07;
committed fixture is the numerical source, every axis scanned BEFORE gating)
---------------------------------------------------------------------------
GATED (exit-1 on failure), gates = round-UP(measured envelope x 1.5), and the
--write-fixture self-check demands EXACT equality with that rule:
  * fine rung (dx = a/60), flux extraction, |S11| vs oracle over 8 configs
    (3 apertures x {centred, iris off-centre at 0.42} + 2 extra guide
    lengths): every config within 0.011 -> envelope 0.0106 -> pooled gate 0.02.
    The binding fine gate is per configuration (0.006-0.016); the pooled
    gate is retained as a ceiling.
    No single configuration sets the envelope any more.
  * Richardson witness at EVERY one of those 8 fine/coarse pairs (not just
    the canonical one): 2*S_fine - S_coarse lands on the oracle within
    0.0046 -> gate 0.01. Cross-confirms the oracle AND the first-order
    attribution (fine/coarse gap ratios 0.407-0.440 = textbook first order).
REPORTED, NOT GATED:
  * coarse rung (dx = a/30): 0.008-0.025 abs.
  * raw (normalize=False) record: worse than flux (gaps 0.009-0.025) with a
    pointwise |raw - flux| difference up to 0.0068 at the wide aperture.
  * residual detrended ripple (quadratic detrend of |S11| MINUS the oracle,
    so the oracle's own curvature is not counted — PR #480 R1): fine <=
    0.0076, coarse <= 0.0152, both at the wide aperture with the iris
    off-centre, down from the 0.0706 the review measured on the same basis
    before the absorber fix.
  * phase: NOT claimed (magnitude-only lane posture).
FENCED (never gated): everything beyond ONE symmetric inductive iris —
multi-iris filters, posts, septa, off-centre apertures stay EXPERIMENTAL per
docs/guides/support_matrix.md.
Truncation witness: num_periods 100 -> 200 shifts |S11| by <= 0.00001 at
every gated aperture AND at the asymmetric configuration.

THREE SETUP DEFECTS were found during this campaign; each had corrupted an
earlier revision's numbers, and each is now fenced by an assert or a derived
setting:
 (1) parasitic wall-slot — fins drawn to the NOMINAL guide width leave a
     1-cell gap at the actual grid wall (fins now drawn past the walls, the
     rasterizer clips; contiguous-aperture assert).
 (2) node-plane box corners WERE half-ulp fragile, because the pre-#931
     volume mask was half-open over NODE coordinates: a fine config
     rasterized 3 thickness nodes instead of 4, and an apparent +/-0.07
     "domain sensitivity" was that ulp lottery. The fence then was to put
     every corner half a cell OFF the node planes. The #931 lattice
     ownership contract inverts it: a PEC volume is sampled at cell CENTRES,
     so a node-plane corner selects whole cells and is the well-defined
     position, while the half-cell offset now lands exactly on a centre.
     Corners are back on the node planes and the asserts read the realized
     edge set (realized_pec_edge_masks) instead of a sigma mask.
 (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d,
     which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber
     left the envelope set by CPML reflection rather than discretization
     (PR #480 review, B2/B3). Fins now stand their inner walls at y-nodes
     fin_c and cells - fin_c so the realized aperture equals the nominal d,
     and CPML = 0.75*lambda_g at the 8.2 GHz band edge (60 coarse / 120
     fine).
 (4) [#931] the oracle was fed the drawn t = 1.524 mm while the lattice
     realized (t_c - 1)*dx — 0.762 mm at a/30, 1.143 mm at a/60 — because a
     body's far face was never a wall. Every assert in this case counted
     MASKED PLANES, which agreed with the drawing by construction, so
     nothing measured the deficit. Under the contract realized == drawn and
     the oracle input is correct for the first time; the whole record was
     regenerated on the corrected geometry.

RETRACTED: an earlier revision FENCED normalize=True modal extraction on the
strength of a measured column power 1.112-1.164. On the corrected setup modal
extraction is passivity-CLEAN at every aperture and both rungs (max column
power 1.0200 at d=7.62/a-30, 1.0012 at d=18.288, ZERO extractor warnings), so
that non-passivity was a symptom of defects (1)-(3) rather than a
reflector-inflation property of the extractor. The fence is withdrawn and the
measurement is committed as modal_extraction_witness.

NOTE on preflight: compute_waveguide_s_matrix runs the extractor's own
passivity/finiteness self-check (its warnings are part of this record) but
the functional path here runs NO sim.preflight(); the operating-point
guarantees (grid-exact dims, wall-reaching fins, exact footprint, CPML depth
from the band-edge guide wavelength, transit-scaled record) are enforced by
construction plus the raster asserts in run_point.

Usage:
  python validation/crossval/18_wr90_iris_modematch.py            # gated set (~2.5 h CPU)
  python validation/crossval/18_wr90_iris_modematch.py --write-fixture
      # + coarse tier, modal + raw + truncation witnesses, and the #931
      # one-cell volume witness (iris-thickness sweep t = 1..8 cells at
      # a/30, ~8 min); regenerates
      # validation/crossval/_18_wr90_iris_results/rfx.json AND
      # tests/fixtures/wr90_iris_modematch/fixture.json (~2.8 h CPU)

Exit codes: 0 = all configured gates passed; 1 = oracle self-check, a raster
assert, or a gate failed. Failure prints "SOME CHECKS FAILED".
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

from tests._gate_policy import gate_from_envelope  # noqa: E402

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

sys.path.insert(0, _SCRIPT_DIR)
from _wr90_iris_realized import (  # noqa: E402
    aperture_walls, grid_plane, realized_edge_masks, wall_plane_runs)

C0 = 299792458.0
MU0 = 4e-7 * np.pi
A_WR90 = 22.86e-3
B_WR90 = 10.16e-3
T_IRIS = 1.524e-3
FREQS = np.linspace(8.2e9, 12.4e9, 29)   # 0.15 GHz — resolves the residual ripple (R480 B2)
CPML_DEPTH = 45.7e-3   # 0.75*lambda_g at the 8.2 GHz band edge — R480 measured
# that 0.5*lambda_g removes only ~half the residual ripple (0.0706->0.0366),
# while 0.75*lambda_g collapses it to ~0.009, making the envelope genuinely
# discretization-dominated instead of absorber-limited.
def cpml_layers_for(dx):
    return int(np.ceil(CPML_DEPTH / dx))   # 60 coarse / 120 fine
COARSE_CELLS = 30            # dx = a/30 = 0.762 mm
FINE_CELLS = 60              # dx = a/60 = 0.381 mm
D_APERTURES = [18.288e-3, 12.192e-3, 7.62e-3]   # 24/16/10 coarse cells
D_WORST = 12.192e-3          # worst-GAP aperture (anchors the guide-length scan;
                             # post-footprint-fix the worst RIPPLE moved to d=18.288,
                             # which is why ASYM_CONFIG scans all apertures — R480)
GLEN_SCAN = [(0.16, 0.50), (0.24, 0.50)]        # guide-length axis (worst-gap aperture)
ASYM_CONFIG = (0.20, 0.42)                       # iris-position axis: ALL apertures
# (R480 follow-up: with the footprint fixed the worst ripple moved from the
# medium to the WIDE aperture, so the asymmetric axis is scanned everywhere)
CANONICAL = (0.20, 0.50)

# gates = round-UP(measured envelope x 1.5); the gate test hard-pins these,
# recomputes the envelopes from the committed data, and regex-binds these
# constants (PR #475 D1/D2 + #476 prose-binding discipline).
# #931 RE-DERIVED from VESSL run 369367259159 (pass 1 on the migrated
# builder). The iris now realizes its drawn thickness at both rungs instead of
# (t_c - 1)*dx, so the oracle is fed the geometry that was actually built and
# every measured envelope shrank. Nothing here is tuned: each gate is the
# repo rule applied to the new envelope, and the write-fixture self-check
# re-derives it and demands exact equality.
#   fine       envelope 0.0232 -> 0.0106 : round-up(0.0106 x 1.5 = 0.0159) at
#              quantum 100 = 0.02   (0.04  -> 0.02, TIGHTER by 2x)
#   Richardson envelope 0.0051 -> 0.0046 : round-up(0.0046 x 1.5 = 0.0069) at
#              quantum 100 = 0.01   (0.01  -> 0.01, unchanged)
GATE_FINE_ABS = 0.02    # = round-up(POOLED fine envelope 0.0106 x 1.5)
GATE_RICH_ABS = 0.01    # = round-up(measured Richardson envelope 0.0046 x 1.5)

# --- issue #812 re-gate (G18-A): PER-CONFIGURATION fine gates -------------
# The pooled GATE_FINE_ABS above is set by the WORST of eight configurations
# and then spent at all eight, so the six well-behaved ones carry up to 4x the
# slack their own data earns.  The same repo rule -- gate = round-UP(measured
# envelope x 1.5) -- applied to each configuration's OWN committed envelope,
# at quantum 1000 (precedent: tests/unit/sparams/test_msl_port_integration.py):
#
#   d(mm)   glen  frac   pre-#931 gap   gate   #931 gap   #931 gate
#   18.288  0.20  0.50   0.0122         0.019  0.0079     0.012
#   12.192  0.20  0.50   0.0223         0.034  0.0101     0.016
#    7.620  0.20  0.50   0.0097         0.015  0.0034     0.006
#   18.288  0.20  0.42   0.0145         0.022  0.0102     0.016
#   12.192  0.20  0.42   0.0232         0.035  0.0106     0.016
#    7.620  0.20  0.42   0.0097         0.015  0.0035     0.006
#   12.192  0.16  0.50   0.0222         0.034  0.0100     0.015
#   12.192  0.24  0.50   0.0222         0.034  0.0100     0.015
#
# The #931 column is VESSL run 369367259159 on the corrected thickness, and
# every gate in it is round-up(that row's own gap x 1.5) at quantum 1000 --
# the same rule, re-applied, never widened: all eight moved DOWN.
#
# GATE_FINE_ABS is RETAINED UNCHANGED (nothing is widened); the per-config
# gate is strictly tighter and is the binding one.  Why it matters: the audit
# of issue #812 measured a one-cell aperture error at d = 7.620 moving the
# fine gap 0.0097 -> 0.0265, inside the pooled 0.04 and outside this 0.015.
# Pre-declared, with the sensitivity derivation and the resulting detection
# table, in docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_
# predeclaration.md section 2.3/2.4, in a commit PRECEDING the measurement
# that judges them.
GATE_FINE_ABS_PER_CONFIG = {
    "18.288|0.20|0.50": 0.012,
    "12.192|0.20|0.50": 0.016,
    "7.620|0.20|0.50": 0.006,
    "18.288|0.20|0.42": 0.016,
    "12.192|0.20|0.42": 0.016,
    "7.620|0.20|0.42": 0.006,
    "12.192|0.16|0.50": 0.015,
    "12.192|0.24|0.50": 0.015,
}


def config_key(d_phys_or_mm, glen, frac):
    """Stable key for GATE_FINE_ABS_PER_CONFIG (metres or mm both accepted)."""
    d_mm = d_phys_or_mm * 1e3 if d_phys_or_mm < 1.0 else d_phys_or_mm
    assert 1.0 <= d_mm <= 50.0, ("aperture out of the WR-90 range", d_mm)
    return f"{d_mm:.3f}|{glen:.2f}|{frac:.2f}"


# --- issue #812 re-gate (G18-C): the declared aperture set is a CLAIM ------
# Nothing in this case ever checked that the apertures it ran are the
# apertures it claims: the oracle is evaluated at whatever d_phys run_point
# was handed, so a silently relabelled aperture moves BOTH sides together and
# every residual stays nominal.  The pin is geometric, from a = 22.86 mm and
# the two declared rungs: each aperture must be an exact and EVEN integer
# number of cells at BOTH rungs.  Even, because the symmetric two-fin
# construction can only realise an even d_c: with fin_c = (cells - d_c)//2 the
# two inner fin walls stand at y-nodes fin_c and cells - fin_c, so the REALIZED
# aperture is cells - 2*fin_c, which has the parity of `cells` and therefore
# equals d_c only for even d_c on this even-cell guide.  (Until #931 the same
# conclusion was derived from an OPEN-NODE count, cells-1-2*fin_c == d_c-1;
# open nodes are not a realized dimension and the contract retires them, but
# the parity conclusion is unchanged because both counts differ from the wall
# separation by a constant.)  A one-fine-cell relabel (7.620 -> 8.001 mm) is
# 21 fine cells (odd) and 10.5 coarse cells, so it fails this pin with ZERO
# tolerance.
def assert_declared_aperture(d_phys):
    for cells in (COARSE_CELLS, FINE_CELLS):
        n = d_phys / (A_WR90 / cells)
        n_i = round(n)
        assert abs(n - n_i) < 1e-9, ("aperture not grid-exact", d_phys, cells, n)
        assert n_i % 2 == 0, ("aperture not an EVEN cell count "
                              "(symmetric-fin parity)", d_phys, cells, n_i)
    assert any(abs(d_phys - d) < 1e-12 for d in D_APERTURES), (
        "aperture is not one of the three DECLARED apertures", d_phys)


# --------------------------------------------------------------------------- #
# Mode-matching oracle (thick symmetric inductive iris) + self-witnesses.
# --------------------------------------------------------------------------- #
def _gamma(n, width, k):
    kc = n * np.pi / width
    return np.sqrt(complex(kc * kc - k * k))


def _overlap(a, d, n, m):
    x0 = (a - d) / 2.0
    al = n * np.pi / a
    be = m * np.pi / d

    def I_ss(p, q, L):
        if abs(p - q) < 1e-30:
            return L / 2 - np.sin(2 * p * L) / (4 * p)
        return (np.sin((p - q) * L) / (p - q) - np.sin((p + q) * L) / (p + q)) / 2

    def I_cs(p, q, L):
        if abs(p - q) < 1e-30:
            return (1 - np.cos(2 * q * L)) / (4 * q) if q > 0 else 0.0
        return ((1 - np.cos((q + p) * L)) / (q + p)
                + (1 - np.cos((q - p) * L)) / (q - p)) / 2

    val = np.cos(al * x0) * I_ss(al, be, d) + np.sin(al * x0) * I_cs(al, be, d)
    return np.sqrt(2 / a) * np.sqrt(2 / d) * val


def _step_junction(a, d, k, n_a, n_b):
    Na = np.arange(1, 2 * n_a, 2)
    Nb = np.arange(1, 2 * n_b, 2)
    gA = np.array([_gamma(n, a, k) for n in Na])
    gB = np.array([_gamma(m, d, k) for m in Nb])
    w = k * C0
    YA = gA / (1j * w * MU0)
    YB = gB / (1j * w * MU0)
    C = np.array([[_overlap(a, d, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + C.T @ YAd @ C)
    T_ba = 2 * Minv @ C.T @ YAd
    R_aa = C @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - C.T @ YAd @ C)
    T_ab = C @ (np.eye(n_b) + R_bb)
    sYA, sYB = np.sqrt(YA), np.sqrt(YB)
    S11 = (sYA[:, None] * R_aa) / sYA[None, :]
    S21 = (sYB[:, None] * T_ba) / sYA[None, :]
    S12 = (sYA[:, None] * T_ab) / sYB[None, :]
    S22 = (sYB[:, None] * R_bb) / sYB[None, :]
    return S11, S12, S21, S22


def _redheffer(sa, sb):
    A11, A12, A21, A22 = sa
    B11, B12, B21, B22 = sb
    n = A22.shape[0]
    inv1 = np.linalg.inv(np.eye(n) - A22 @ B11)
    inv2 = np.linalg.inv(np.eye(n) - B11 @ A22)
    return (A11 + A12 @ B11 @ inv1 @ A21,
            A12 @ inv2 @ B12,
            B21 @ inv1 @ A21,
            B22 + B21 @ A22 @ inv2 @ B12)


def iris_smatrix(a, d, t, freq, n_a=40):
    """TE10->TE10 (S11, S21) of the thick symmetric inductive iris."""
    k = 2 * np.pi * freq / C0
    n_b = max(4, int(round(n_a * d / a)))
    s_step = _step_junction(a, d, k, n_a, n_b)
    s_rev = (s_step[3], s_step[2], s_step[1], s_step[0])
    Nb = np.arange(1, 2 * n_b, 2)
    gB = np.array([_gamma(m, d, k) for m in Nb])
    P = np.diag(np.exp(-gB * t))
    z = np.zeros((n_b, n_b), dtype=complex)
    s_tot = _redheffer(_redheffer(s_step, (z, P, P, z)), s_rev)
    return s_tot[0][0, 0], s_tot[2][0, 0]


def marcuvitz_thin_b(a, d, freq):
    k = 2 * np.pi * freq / C0
    beta = np.sqrt(k * k - (np.pi / a) ** 2)
    return (2 * np.pi / beta / a) * (1.0 / np.tan(np.pi * d / (2 * a))) ** 2


def validate_oracle() -> dict:
    """Self-witnesses; raises on failure (refuse to gate on a broken oracle)."""
    w = {}
    f = 10e9
    s11, s21 = iris_smatrix(A_WR90, 12e-3, 2e-3, f)
    w["unitarity_dev"] = abs(abs(s11) ** 2 + abs(s21) ** 2 - 1.0)
    assert w["unitarity_dev"] < 1e-9
    v40 = abs(iris_smatrix(A_WR90, 12e-3, 2e-3, f, n_a=40)[0])
    v80 = abs(iris_smatrix(A_WR90, 12e-3, 2e-3, f, n_a=80)[0])
    w["mode_convergence"] = abs(v40 - v80)
    assert w["mode_convergence"] < 1e-3
    rels = []
    for d in (16e-3, 12e-3, 8e-3):
        s11t, _ = iris_smatrix(A_WR90, d, 1e-9, f)
        b_mm = np.real(-2 * s11t / (1j * (1 + s11t)))
        assert b_mm < 0, "inductive iris must have negative shunt susceptance"
        rels.append(abs(abs(b_mm) / marcuvitz_thin_b(A_WR90, d, f) - 1))
    w["marcuvitz_max_rel"] = max(rels)
    assert w["marcuvitz_max_rel"] < 0.15   # leading-order anchor, ~10% expected
    s11o, s21o = iris_smatrix(A_WR90, A_WR90 - 1e-9, 2e-3, f)
    w["open_limit_s11"] = abs(s11o)
    assert w["open_limit_s11"] < 1e-6
    s11d, _ = iris_smatrix(A_WR90, 4e-3, 6e-3, f)
    w["deep_limit_s11"] = abs(s11d)
    assert w["deep_limit_s11"] > 0.999
    return w


# --------------------------------------------------------------------------- #
# rfx measurement
# --------------------------------------------------------------------------- #
def run_point(d_phys, cells, glen=0.20, iris_frac=0.50, normalize="flux",
              num_periods=100.0, t_cells=None):
    """One 2-port iris run at grid-exact dimensions with realized asserts.

    ``t_cells`` overrides the iris thickness in CELLS (default: T_IRIS at this
    rung).  It exists for the one-cell volume witness — see
    ``thickness_sweep`` — and the realized thickness is t_cells*dx either way,
    because under the #931 contract a PEC volume realizes both of its faces.
    """
    assert_declared_aperture(d_phys)   # issue #812 G18-C
    DX = A_WR90 / cells
    d_c = int(round(d_phys / DX))
    t_c = int(round(T_IRIS / DX)) if t_cells is None else int(t_cells)
    assert t_c >= 1, ("iris thickness must be at least one cell; a "
                      "zero-thickness PEC obstacle is a SHEET declaration "
                      "(add_thin_conductor), not a volume", t_c)
    fin_c = (cells - d_c) // 2
    glen_c = int(round(glen / DX))
    p1 = int(round(0.040 / DX))
    p2 = glen_c - p1
    iris_lo = int(round(glen_c * iris_frac)) - t_c // 2
    sim = Simulation(
        freq_max=float(FREQS[-1]) * 1.1,
        domain=(glen_c * DX, A_WR90, B_WR90), dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=cpml_layers_for(DX))
    big = 1.0   # fins drawn PAST the walls; rasterizer clips (slot-bug fence)
    # Every corner sits ON a node plane.  This INVERTS the recipe this case
    # used until #931, and the reason is the lattice ownership contract: a PEC
    # volume is sampled at cell CENTRES, so a corner on a node plane selects
    # whole cells and is the well-defined position, while a corner half a cell
    # off lands exactly on a centre — the tie the old recipe was invented to
    # avoid, moved.  Under the old half-open NODE mask the opposite was true
    # (a fine-rung domain-scan config rasterized 3 thickness-nodes instead of
    # 4), which is what setup defect (2) below records.
    #
    # Realized, and asserted below from realized_pec_edge_masks: x walls at
    # node planes iris_lo .. iris_lo + t_c, i.e. a thickness of t_c cells =
    # T_IRIS exactly — which is what the oracle has always been fed.  Until
    # #931 the far face was never a wall and the realized thickness was
    # (t_c - 1)*dx, a 50% (a/30) / 25% (a/60) deficit against the oracle input
    # that nothing in this case measured.
    x_lo = iris_lo * DX
    x_hi = (iris_lo + t_c) * DX
    fin_hi_y = fin_c * DX          # y wall at node fin_c
    fin_lo_y = (cells - fin_c) * DX  # y wall at node cells - fin_c
    sim.add(Box((x_lo, -big, -big), (x_hi, fin_hi_y, big)), material="pec")
    sim.add(Box((x_lo, fin_lo_y, -big), (x_hi, big, big)), material="pec")
    for x, dr, nm in ((p1 * DX, "+x", "P1"), (p2 * DX, "-x", "P2")):
        sim.add_waveguide_port(x, mode=(1, 0), mode_type="TE", direction=dr,
                               f0=10.3e9, bandwidth=0.41,
                               waveform="modulated_gaussian",
                               freqs=FREQS, name=nm)
    # REALIZED-geometry asserts (build time, no solve): the operating-point
    # guarantees, read from the ONE edge-set function the #931 contract
    # defines.  They assert realized == drawn, which under the contract is an
    # identity — so a regression shows up as a thickness of t_c - 1 or an
    # aperture off by a cell, not as a silent shift in |S11|.
    grid, edges = realized_edge_masks(sim)
    ny = grid.shape[1]
    assert ny == cells + 1, (ny, cells)          # node convention: a=(ny-1)dx exact
    x_runs = wall_plane_runs(edges, 0)
    assert len(x_runs) == 1, ("iris count", x_runs)
    (x_wall_lo, x_wall_hi), = x_runs
    drawn_x = (grid_plane(grid, 0, iris_lo), grid_plane(grid, 0, iris_lo + t_c))
    assert (x_wall_lo, x_wall_hi) == drawn_x, (
        "realized iris walls != drawn", x_runs, drawn_x)
    # BOTH faces present: this is the check nothing in this case had until
    # #931, and the one the contract exists to make possible.
    y_wall_lo, y_wall_hi = aperture_walls(
        edges, 1, (slice(x_wall_lo, x_wall_lo + 1), slice(None), slice(None)))
    drawn_y = (grid_plane(grid, 1, fin_c), grid_plane(grid, 1, cells - fin_c))
    assert (y_wall_lo, y_wall_hi) == drawn_y, (
        "realized aperture walls != drawn", (y_wall_lo, y_wall_hi), drawn_y)
    assert y_wall_hi - y_wall_lo == d_c, ("realized aperture != drawn",
                                          y_wall_hi - y_wall_lo, d_c)

    t0 = time.time()
    res = sim.compute_waveguide_s_matrix(normalize=normalize,
                                         num_periods=num_periods)
    wall = time.time() - t0
    s = np.asarray(res.s_params)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    return {
        "d_mm": round(d_phys * 1e3, 3), "cells_per_a": cells,
        "dx_mm": round(DX * 1e3, 4), "glen_m": glen, "iris_frac": iris_frac,
        "t_mm": round(t_c * DX * 1e3, 4),
        "normalize": str(normalize), "num_periods": num_periods,
        # RENAMED for #931 (was aperture_cells / thickness_cells): the
        # committed fields were an OPEN-NODE count (d_c - 1) and a
        # MASKED-PLANE count (t_c), neither of which is a realized dimension,
        # and the aperture one changes VALUE under the contract.  These are
        # the realized dimensions in cells, plus the wall-plane indices they
        # were measured between.  Renamed rather than reused, because a
        # same-named key with a new meaning is read wrong exactly once.
        "realized_aperture_cells": int(y_wall_hi - y_wall_lo),
        "realized_thickness_cells": int(x_wall_hi - x_wall_lo),
        "iris_wall_nodes": [int(x_wall_lo), int(x_wall_hi)],
        "aperture_wall_nodes": [int(y_wall_lo), int(y_wall_hi)],
        "s11": [round(float(v), 5) for v in s11],
        "s21": [round(float(v), 5) for v in s21],
        "max_colpow": round(float(np.max(s11 ** 2 + s21 ** 2)), 4),
        "wall_s": round(wall, 1),
    }


def oracle_s11(d_phys, t_phys=T_IRIS):
    return [round(float(abs(iris_smatrix(A_WR90, d_phys, t_phys, f)[0])), 5)
            for f in FREQS]


# --------------------------------------------------------------------------- #
# #931 one-cell volume witness (design note 20260906 section 5).
# --------------------------------------------------------------------------- #
THICK_SWEEP_CELLS = [1, 2, 3, 4, 5, 6, 8]
THICK_SWEEP_D = D_WORST          # 12.192 mm, the worst-GAP aperture
THICK_SWEEP_RUNG = COARSE_CELLS  # a/30, ~66 s per run


def thickness_sweep(oracle_cache=None):
    """rfx vs the mode-matching oracle across iris thickness, INCLUDING t = 1 cell.

    WHY THIS EXISTS.  Under the #931 lattice ownership contract a PEC volume
    one cell thick is a filled slab with a tangential wall at BOTH of its
    faces, at every thickness, on every axis, with no flag.  Before #931 the
    far face was never a wall, so a one-cell body stood ONE wall and a
    ``two_plane`` flag existed to put the second one back for t = 1 only.
    Nothing independent ever said which of those was right AT ONE CELL: the
    thin-limit was checked against Marcuvitz, which is a t -> 0 statement, not
    a t = dx one.

    This is that independent witness.  The oracle is a TEn0 mode-matching
    cascade of two width-step junctions joined by a length-t guide section;
    it takes the physical t and knows nothing about the lattice.  If the
    two-face rule is right at one cell, the t = 1 residual sits on the same
    curve as t = 2..8; if a one-cell slab were realizing something other than
    a dx-thick iris, t = 1 would be the outlier.  Run at the coarse rung
    (a/30, dx = 0.762 mm) at the worst-gap aperture, seven thicknesses,
    roughly 66 s each.

    THE CRITERION, and why it is not the one this function shipped with.
    The witness was first stated as "the t = 1 residual lies inside the range
    spanned by t = 2..8".  VESSL run 369367259159 measured the residual to be
    MONOTONE DECREASING in t (0.0312, 0.0246, 0.0207, 0.0179, 0.0157, 0.0139,
    0.0109 for t = 1,2,3,4,5,6,8), and for any monotone family the t = 1 point
    is necessarily the extremum, hence necessarily outside the range its own
    t >= 2 rungs span.  That criterion therefore fails for EVERY outcome: a
    t = 1 residual of 0.0000 — a perfect one-cell iris — fails it too.  A test
    that cannot pass carries no information about the physics, so it is
    retired for vacuity, not because of the answer it gave.  Its verdict is
    kept in the record (``monotone_range_criterion``) so the retirement is
    auditable.

    THE REPLACEMENT is an identification test with no tunable constant.  The
    question the contract actually poses at one cell is *which* thickness the
    lattice built, and the lattice-blind oracle can be asked directly: for
    each swept rung, evaluate the oracle at t-1, t and t+1 cells and take the
    argmin of the residual.  The witness passes when every rung identifies its
    OWN drawn thickness.  It can fail three ways per rung, and at t = 1 the
    t-1 alternative is exactly the pre-#931 realization — one wall, a
    zero-thickness screen — so this is a direct discriminator between the two
    rules at the one place they disagree.  The margins are reported with the
    verdict; they are not a gate, because argmin needs no threshold.
    """
    dx = A_WR90 / THICK_SWEEP_RUNG
    rows = []
    for t_c in THICK_SWEEP_CELLS:
        t_phys = t_c * dx
        orc = (oracle_cache or {}).get(t_c) or oracle_s11(THICK_SWEEP_D, t_phys)
        r = run_point(THICK_SWEEP_D, THICK_SWEEP_RUNG, t_cells=t_c)
        gap = max(_gaps(r, orc))
        # identification: which realized thickness does this trace match?
        # t-1 at t = 1 is the pre-#931 realization (one wall, zero thickness).
        ident = {}
        for n in (t_c - 1, t_c, t_c + 1):
            o_n = (oracle_cache or {}).get(n) or oracle_s11(THICK_SWEEP_D, n * dx)
            ident[n] = round(max(abs(a_ - b_) for a_, b_ in zip(r["s11"], o_n)), 4)
        best = min(ident, key=lambda n: ident[n])
        runner_up = min(v for n, v in ident.items() if n != best)
        rows.append({"t_cells": t_c, "t_mm": round(t_phys * 1e3, 4),
                     "identification": {
                         "gap_at_t_minus_1": ident[t_c - 1],
                         "gap_at_t": ident[t_c],
                         "gap_at_t_plus_1": ident[t_c + 1],
                         "argmin_t_cells": int(best),
                         "identified_own_thickness": bool(best == t_c),
                         "margin_vs_runner_up_x": round(
                             runner_up / max(ident[best], 1e-12), 3)},
                     "realized_aperture_cells": r["realized_aperture_cells"],
                     "realized_thickness_cells": r["realized_thickness_cells"],
                     "iris_wall_nodes": r["iris_wall_nodes"],
                     "max_gap_abs": round(gap, 4),
                     "max_colpow": r["max_colpow"],
                     "s11": r["s11"], "oracle_s11": orc,
                     "wall_s": r["wall_s"]})
    return rows


def _gaps(row, orc):
    return [abs(a - b) for a, b in zip(row["s11"], orc)]


def main(argv):
    write_fixture = "--write-fixture" in argv
    ok = True

    witnesses = validate_oracle()
    print("[oracle] mode-matching self-witnesses PASS:",
          {k: float(f"{v:.3e}") for k, v in witnesses.items()})

    oracles = {str(d): oracle_s11(d) for d in D_APERTURES}

    # --- GATED fine rung: all apertures at canonical + worst-aperture scan ---
    print(f"\n== GATED fine rung dx=a/{FINE_CELLS} flux (PER-CONFIG gates, "
          f"pooled ceiling {GATE_FINE_ABS} abs) ==")
    fine_rows = []
    fine_configs = ([(d, CANONICAL) for d in D_APERTURES]
                    + [(d, ASYM_CONFIG) for d in D_APERTURES]
                    + [(D_WORST, c) for c in GLEN_SCAN])
    for d, (glen, frac) in fine_configs:
        r = run_point(d, FINE_CELLS, glen=glen, iris_frac=frac)
        r["oracle_s11"] = oracles[str(d)]
        gap = max(_gaps(r, oracles[str(d)]))
        r["max_gap_abs"] = round(gap, 4)
        key = config_key(d, glen, frac)
        cfg_gate = GATE_FINE_ABS_PER_CONFIG[key]
        r["fine_gate_abs"] = cfg_gate
        fine_rows.append(r)
        # BOTH: the pre-#812 pooled ceiling (never widened) and the tighter
        # per-configuration gate this case is now judged on.
        passed = gap <= cfg_gate and gap <= GATE_FINE_ABS
        ok &= passed
        print(f"  d={d*1e3:6.2f} glen={glen:.2f} frac={frac:.2f}: "
              f"max|dS11|={gap:.4f} gate {cfg_gate:.3f} "
              f"colpow={r['max_colpow']:.3f} "
              f"({r['wall_s']:.0f}s) {'PASS' if passed else 'FAIL'}", flush=True)

    # --- coarse rung + Richardson gate, DOMAIN-SCANNED (R480 B1: the
    # Richardson witness is evaluated at EVERY committed fine/coarse pair,
    # not just the canonical configuration) ------------------------------
    print(f"\n== coarse rung dx=a/{COARSE_CELLS} + Richardson over ALL pairs "
          f"(gate {GATE_RICH_ABS} abs) ==")
    coarse_rows = []
    for d, (glen, frac) in fine_configs:
        r = run_point(d, COARSE_CELLS, glen=glen, iris_frac=frac)
        r["oracle_s11"] = oracles[str(d)]
        r["max_gap_abs"] = round(max(_gaps(r, oracles[str(d)])), 4)
        fine_mate = next(fr for fr in fine_rows
                         if (fr["d_mm"], fr["glen_m"], fr["iris_frac"])
                         == (r["d_mm"], glen, frac))
        rich = [2 * f_ - c_ for f_, c_ in zip(fine_mate["s11"], r["s11"])]
        rich_dev = max(abs(a - b) for a, b in zip(rich, oracles[str(d)]))
        r["richardson_dev_abs"] = round(rich_dev, 4)
        coarse_rows.append(r)
        passed = rich_dev <= GATE_RICH_ABS
        ok &= passed
        print(f"  d={d*1e3:6.2f} glen={glen:.2f} frac={frac:.2f}: "
              f"coarse max|dS11|={r['max_gap_abs']:.3f} "
              f"richardson dev={rich_dev:.3f} ({r['wall_s']:.0f}s) "
              f"{'PASS' if passed else 'FAIL'}", flush=True)

    trunc = []
    raw_rows = []
    modal_witness = None
    if write_fixture:
        print("\n== modal-extraction witness (normalize=True; both rungs, "
              "3 apertures) ==")
        import warnings as _warnings
        modal_witness = {
            "rows": [],
            "note": "passivity-CLEAN at every aperture and both rungs on the "
                    "corrected setup — the earlier 1.112-1.164 fence evidence "
                    "did not survive the footprint/absorber fixes, so the "
                    "modal fence is RETRACTED (PR #480)",
        }
        fence_set = ([(7.62e-3, COARSE_CELLS), (7.62e-3, FINE_CELLS),
                      (12.192e-3, COARSE_CELLS), (18.288e-3, COARSE_CELLS)])
        for d_f, cells_f in fence_set:
            with _warnings.catch_warnings(record=True) as _wrec:
                _warnings.simplefilter("always")
                rm = run_point(d_f, cells_f, normalize=True)
            fence_warns = [str(w.message) for w in _wrec
                           if "passivity" in str(w.message).lower()]
            # accuracy rides with the retraction (PR #480 R2): a fence removed
            # on passivity grounds alone would leave "no longer fenced"
            # resting on an unstated accuracy inference.
            rm["oracle_s11"] = oracles[str(d_f)]
            rm["max_gap_abs"] = round(max(_gaps(rm, oracles[str(d_f)])), 4)
            modal_witness["rows"].append(
                {"d_mm": round(d_f * 1e3, 3), "cells_per_a": cells_f,
                 "max_colpow": rm["max_colpow"],
                 "extractor_warnings": fence_warns,
                 "s11": rm["s11"], "max_gap_abs": rm["max_gap_abs"]})
            print(f"  d={d_f*1e3:6.2f} a/{cells_f}: colpow "
                  f"{rm['max_colpow']:.4f} gap {rm['max_gap_abs']:.4f} "
                  f"({len(fence_warns)} warnings)", flush=True)

        print("\n== raw-extraction cross-record (normalize=False) ==")
        for d in D_APERTURES:
            r = run_point(d, COARSE_CELLS, normalize=False)
            r["oracle_s11"] = oracles[str(d)]
            r["max_gap_abs"] = round(max(_gaps(r, oracles[str(d)])), 4)
            raw_rows.append(r)
            print(f"  d={d*1e3:6.2f}: raw max|dS11|={r['max_gap_abs']:.3f} "
                  f"colpow={r['max_colpow']:.3f}", flush=True)

        print("\n== truncation witness (gated apertures + asymmetric config, "
              "1x/2x periods) ==")
        trunc_configs = ([(d, CANONICAL) for d in D_APERTURES]
                         + [(D_WORST, (0.20, 0.42))])   # R480 B2: test the config
        for d, (glen, frac) in trunc_configs:
            r1 = run_point(d, COARSE_CELLS, glen=glen, iris_frac=frac,
                           num_periods=100.0)
            r2 = run_point(d, COARSE_CELLS, glen=glen, iris_frac=frac,
                           num_periods=200.0)
            shift = max(abs(a - b) for a, b in zip(r1["s11"], r2["s11"]))
            trunc.append({"d_mm": round(d * 1e3, 3), "glen_m": glen,
                          "iris_frac": frac, "shift_abs": round(shift, 5)})
            print(f"  d={d*1e3:6.2f} frac={frac:.2f}: 1x->2x shift {shift:.5f}",
                  flush=True)

        env_fine = max(r["max_gap_abs"] for r in fine_rows)
        env_rich = max(r["richardson_dev_abs"] for r in coarse_rows)
        ratios = []
        for fr in fine_rows:
            for cr in coarse_rows:
                if (fr["d_mm"], fr["glen_m"], fr["iris_frac"]) == (
                        cr["d_mm"], cr["glen_m"], cr["iris_frac"]):
                    ratios.append(round(fr["max_gap_abs"] / cr["max_gap_abs"], 3))
        print(f"\n  envelopes: fine {env_fine:.3f} (gate {GATE_FINE_ABS}), "
              f"richardson {env_rich:.3f} (gate {GATE_RICH_ABS}); "
              f"first-order ratios {ratios}")
        for gate, env, tier in ((GATE_FINE_ABS, env_fine, "fine"),
                                (GATE_RICH_ABS, env_rich, "richardson")):
            required = gate_from_envelope(env, quantum=100)
            if abs(gate - required) > 1e-9:   # EXACT ceil(x1.5) — R480
                print(f"  ENVELOPE/GATE MISMATCH ({tier}): gate {gate} "
                      f"must equal round-up(env x 1.5) = {required}")
                ok = False
        # issue #812 G18-A: the SAME exact-equality demand, per configuration.
        for fr in fine_rows:
            key = config_key(fr["d_mm"], fr["glen_m"], fr["iris_frac"])
            required = gate_from_envelope(fr["max_gap_abs"], quantum=1000)
            if abs(GATE_FINE_ABS_PER_CONFIG[key] - required) > 1e-9:
                print(f"  ENVELOPE/GATE MISMATCH (fine {key}): gate "
                      f"{GATE_FINE_ABS_PER_CONFIG[key]} must equal "
                      f"round-up(env x 1.5) = {required}")
                ok = False
        if set(GATE_FINE_ABS_PER_CONFIG) != {
                config_key(fr["d_mm"], fr["glen_m"], fr["iris_frac"])
                for fr in fine_rows}:
            print("  PER-CONFIG GATE SET does not match the measured configs")
            ok = False

        # issue #812 G18-B: the one-cell aperture DETECTION table, recomputed
        # from the rows and the oracle (no FDTD).  The defect modelled is the
        # audit's: aperture one cell off AT EACH RUNG (dx-proportional), which
        # is what the campaign's own setup defect (3) was, at half its size.
        one_cell = []
        for fr in fine_rows:
            # resolve back to the DECLARED aperture object rather than
            # reconstructing it from the mm field: 18.288 * 1e-3 is not the
            # same float as 18.288e-3, so str() would miss the oracle cache.
            d = next(x for x in D_APERTURES if abs(x * 1e3 - fr["d_mm"]) < 1e-9)
            key = config_key(fr["d_mm"], fr["glen_m"], fr["iris_frac"])
            base = np.asarray(oracles[str(d)])
            cr = next(c for c in coarse_rows
                      if (c["d_mm"], c["glen_m"], c["iris_frac"])
                      == (fr["d_mm"], fr["glen_m"], fr["iris_frac"]))
            for sgn in (+1, -1):
                shift_f = np.asarray(oracle_s11(d + sgn * A_WR90 / FINE_CELLS)) - base
                shift_c = np.asarray(oracle_s11(d + sgn * A_WR90 / COARSE_CELLS)) - base
                f_def = np.asarray(fr["s11"]) + shift_f
                c_def = np.asarray(cr["s11"]) + shift_c
                gap = float(np.max(np.abs(f_def - base)))
                rich = float(np.max(np.abs(2 * f_def - c_def - base)))
                one_cell.append({
                    "config": key, "sign": sgn,
                    "fine_gap_abs": round(gap, 4),
                    "fine_gate_abs": GATE_FINE_ABS_PER_CONFIG[key],
                    "detected_by_fine_gate": bool(
                        gap > GATE_FINE_ABS_PER_CONFIG[key]),
                    "richardson_dev_abs": round(rich, 4),
                    "detected_by_richardson_gate": bool(rich > GATE_RICH_ABS),
                })
        n_pos = sum(o["detected_by_fine_gate"] for o in one_cell if o["sign"] > 0)
        n_neg = sum(o["detected_by_fine_gate"] for o in one_cell if o["sign"] < 0)
        print(f"\n  one-cell aperture detection: +1 cell {n_pos}/8, "
              f"-1 cell {n_neg}/8; Richardson "
              f"{sum(o['detected_by_richardson_gate'] for o in one_cell)}/16 "
              f"(dx-proportional errors cancel in 2*fine - coarse BY "
              f"CONSTRUCTION — see the module docstring)")

        # #931 one-cell volume witness (design note section 5). Runs last
        # because it is the cheapest block and its verdict is a comparison
        # against the rest of the sweep, not against a gate constant.
        print("\n== #931 one-cell volume witness: iris-thickness sweep "
              f"(a/{THICK_SWEEP_RUNG}, d = {THICK_SWEEP_D*1e3:.3f} mm) ==")
        thick_rows = thickness_sweep()
        for r in thick_rows:
            print(f"  t={r['t_cells']} cell(s) ({r['t_mm']:.3f} mm), walls "
                  f"{r['iris_wall_nodes']}: max|dS11|={r['max_gap_abs']:.4f} "
                  f"colpow={r['max_colpow']:.3f} ({r['wall_s']:.0f}s)",
                  flush=True)
        multi = [r["max_gap_abs"] for r in thick_rows if r["t_cells"] >= 2]
        one = next(r["max_gap_abs"] for r in thick_rows if r["t_cells"] == 1)
        # RETIRED criterion, still measured and recorded: see the
        # thickness_sweep docstring for why it can never pass.
        one_cell_on_curve = min(multi) <= one <= max(multi)
        print(f"  [retired, vacuous] t=1 residual {one:.4f} vs the t=2..8 "
              f"range [{min(multi):.4f}, {max(multi):.4f}]: "
              f"{'ON the curve' if one_cell_on_curve else 'OUTLIER'}")
        for r in thick_rows:
            idn = r["identification"]
            print(f"  t={r['t_cells']}: oracle gaps at t-1/t/t+1 = "
                  f"{idn['gap_at_t_minus_1']:.4f}/{idn['gap_at_t']:.4f}/"
                  f"{idn['gap_at_t_plus_1']:.4f} -> identifies t="
                  f"{idn['argmin_t_cells']} "
                  f"({idn['margin_vs_runner_up_x']:.2f}x) "
                  f"{'OK' if idn['identified_own_thickness'] else 'MISIDENTIFIED'}")
        identified = all(r["identification"]["identified_own_thickness"]
                         for r in thick_rows)
        one_ident = next(r["identification"] for r in thick_rows
                         if r["t_cells"] == 1)
        print(f"  GATED: every rung identifies its own thickness -> "
              f"{'PASS' if identified else 'FAIL'}; at t = 1 the pre-#931 "
              f"one-wall alternative is "
              f"{one_ident['gap_at_t_minus_1'] / max(one_ident['gap_at_t'], 1e-12):.2f}x "
              f"worse than the two-wall one")
        ok &= identified

        payload = {
            "schema": "rfx.wr90_iris_modematch",
            "schema_version": 2,
            "campaign": (
                "cross-solver validation campaign, item 3 stage S1: single "
                "symmetric inductive iris in WR-90 vs TEn0 mode-matching — "
                "the calibrated prerequisite for any multi-iris filter case"
            ),
            "claim_scope": (
                "One symmetric inductive PEC iris (t = 1.524 mm = exactly 2 "
                "coarse / 4 fine cells; apertures 18.288/12.192/7.62 mm, "
                "grid-exact) in WR-90 over 8.2-12.4 GHz on 29 frequency "
                "points, flux-normalized |S11| vs a twice-implemented TEn0 "
                "mode-matching cascade oracle (self-witnesses: unitarity "
                "1.1e-16, mode convergence 4.3e-5, Marcuvitz cot^2 "
                "thin-limit anchor 10.8% with the inductive sign, d->a and "
                "deep-constriction limits; the PR #480 review reproduced "
                "the oracle with a formulation-independent 2-D H-plane "
                "FDFD to 6e-4 and measured rfx's same-geometry agreement "
                "at <= 0.02 — attributed, not imported). GATED: fine rung "
                "dx = a/60 within 0.02 abs = round-up(measured envelope "
                "0.0106 x 1.5) over 8 configs (3 apertures x {centred, "
                "iris off-centre at 0.42 of the guide} + 2 extra guide "
                "lengths; every config lands within 0.011, so no single "
                "configuration sets the envelope), and the Richardson "
                "extrapolation 2*S(a/60) - S(a/30) on the oracle within "
                "0.01 abs (envelope 0.0046) at EVERY one of those 8 pairs, "
                "which cross-confirms the oracle and the first-order "
                "attribution (gap ratios 0.407-0.440 = textbook first "
                "order). REPORTED, NOT GATED: the coarse rung dx = a/30 "
                "(0.008-0.025 abs); the raw normalize=False record, which "
                "is WORSE than flux (gaps 0.009-0.025) with a pointwise "
                "|raw - flux| difference up to 0.0068 at the wide "
                "aperture; "
                "residual detrended ripple, i.e. a quadratic detrend of "
                "|S11| MINUS the oracle so the oracle's own curvature is not "
                "counted (fine <= 0.0076, coarse <= 0.0152, both at the wide "
                "aperture with the iris off-centre, down from the 0.0706 the "
                "PR #480 review measured on the same basis before the "
                "absorber fix); "
                "and phase (magnitude-only lane posture). FENCED, never "
                "gated: everything beyond ONE symmetric inductive iris — "
                "multi-iris filters, posts, septa and off-centre apertures "
                "stay EXPERIMENTAL per docs/guides/support_matrix.md. "
                "THREE SETUP DEFECTS were found during this campaign, each "
                "having corrupted an earlier revision's numbers and each "
                "now fenced by an assert or a derived setting: (1) a "
                "parasitic wall-slot (fins drawn to the NOMINAL guide "
                "width leave a 1-cell gap at the actual grid wall); (2) "
                "node-plane box corners were half-ulp fragile under the "
                "pre-#931 half-open NODE mask — one fine config rasterized "
                "3 thickness nodes instead of 4, and an apparent +/-0.07 "
                "'domain sensitivity' was that ulp lottery — so every "
                "corner was moved half a cell OFF the node planes. The "
                "#931 lattice ownership contract INVERTS that: a PEC volume "
                "is sampled at cell CENTRES, so a node-plane corner selects "
                "whole cells and is the well-defined position while a "
                "half-cell offset lands exactly on a centre. The corners "
                "are back on the node planes and the footprint asserts read "
                "the realized edge set rather than a sigma mask; (3) the "
                "fin footprint made the ELECTRICAL aperture d + 2*dx "
                "instead of d, which alone inflated the envelope 4-6x, and "
                "a 0.5*lambda_g absorber left the envelope set by CPML "
                "reflection rather than discretization (PR #480 review "
                "B2/B3; CPML is now 0.75*lambda_g at the band edge = 60 "
                "coarse / 120 fine). THE #931 THICKNESS CORRECTION: until "
                "the contract landed, this case fed its oracle the drawn "
                "t = 1.524 mm while the lattice realized (t_c - 1)*dx — "
                "0.762 mm at a/30 and 1.143 mm at a/60, a 50% / 25% "
                "thickness deficit — because a body's far face was never a "
                "wall. Nothing in this case measured that: every assert "
                "counted MASKED PLANES, a quantity that agreed with the "
                "drawing by construction. Under the contract the realized "
                "thickness is the drawn thickness and the oracle input is "
                "correct for the first time; the whole record was "
                "regenerated on the corrected geometry and every gate "
                "re-derived from the new envelopes. The case also gains the "
                "contract's one-cell volume witness (one_cell_volume_witness): "
                "an iris-thickness sweep t = 1..8 cells against the "
                "lattice-blind mode-matching oracle, so that a one-cell "
                "PEC body standing two walls has an independent check "
                "rather than a thin-limit anchor that only speaks about "
                "t -> 0. Each rung is asked to IDENTIFY its own thickness "
                "-- the oracle at t-1, t and t+1 cells, argmin on t -- and "
                "all seven do; at t = 1 the pre-#931 one-wall alternative "
                "(a zero-thickness screen) is 4.32x worse than the two-wall "
                "one, so the contract's rule at one cell is measured rather "
                "than assumed. The witness's first-stated criterion (t = 1 "
                "inside the t = 2..8 range) is RETIRED as vacuous and its "
                "verdict kept: the residual is monotone in t, so t = 1 is "
                "the extremum whatever the physics does, and a perfect "
                "0.0000 would fail it too. "
                "RETRACTED: an earlier revision fenced normalize=True "
                "modal extraction on the strength of a measured column "
                "power 1.112-1.164; on the corrected setup modal "
                "extraction is passivity-CLEAN at every aperture and both "
                "rungs (max column power 1.0200 at d = 7.62 mm / a-30, "
                "1.0012 at d = 18.288 mm, ZERO extractor warnings), so "
                "that non-passivity was a symptom of defects (1)-(3) and "
                "not a reflector-inflation property of the extractor — the "
                "fence is withdrawn and the measurement is committed as "
                "modal_extraction_witness, which also records modal "
                "ACCURACY so the retraction does not rest on passivity "
                "alone: modal |S11| gaps come out comparable to flux and "
                "consistently a little worse, which is why flux still "
                "carries the gate. Palace WavePort corroboration (stage S2) "
                "and a published multi-iris filter (stage S3) are follow-on "
                "stages, not claimed here. "
                "APERTURE RESOLUTION (issue #812 re-gate 2026-09-01, RE-MEASURED under #931 2026-09-07): the fine gate is per-CONFIGURATION -- gate = round-up(that configuration's own committed envelope x 1.5) at quantum 1000, giving 0.012/0.016/0.006/0.016/0.016/0.006/0.015/0.015 for the eight configs -- because the pooled gate is set by the worst configuration and then spent at all eight. All eight moved DOWN when the thickness deficit closed (0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034 before it), which is a re-derivation of the same rule on a better geometry, not a re-tuning. Measured against those gates, a one-cell aperture error at each rung (the smallest the grid-snapped geometry can express, and the campaign's own setup defect (3) at half its size) is now detected in BOTH signs at every one of the eight configurations, at worst 1.623x the gate for an over-aperture and 2.608x for an under-aperture. Those counts, the margins and the per-configuration oracle distances are COMMITTED rather than restated here: validation/crossval/_18_wr90_iris_results/aperture_resolution.json, keys summary.over_aperture_detected, summary.over_aperture_min_margin_x, summary.under_aperture_detected, summary.under_aperture_detected_configs, summary.under_aperture_min_margin_x and summary.under_aperture_max_margin_x, with the per-configuration rows under pairs[*] (pairs[2] is d = 7.620 mm centred); each one is re-derived from the committed traces by an INDEPENDENT oracle in tests/crossval/test_wr90_iris_modematch_gates.py. WHAT #931 CHANGED HERE, and it is the whole paragraph: before the contract, the committed fine trace's NEAREST oracle over the declared offset grid sat at d PLUS half a fine cell at all eight configurations, an apparent effective aperture WIDER than nominal; a one-cell under-aperture therefore moved the geometry TOWARD the trace, scored BETTER than the undefected row at both d = 7.620 configurations, and was detected at only two of the eight. That half-cell offset was not an aperture property at all -- it was the thickness deficit ((t_c - 1)*dx instead of t) reading out on the aperture axis, the two being the only free dimensions of a symmetric iris. With the realized thickness equal to the drawn one, the nearest oracle sits at the DECLARED d at all eight configurations (summary.nearest_offset_fine_cells_values == [0.0]), no defect scores better than the undefected row (summary.under_aperture_scores_better_configs == []), and the asymmetry between the two signs is gone. CORRECTION HISTORY (issue #812 round 2): an earlier revision of this paragraph asserted that the committed fine trace sat closer to the oracle one fine cell NARROW, quoting the under-aperture DEFECT metric as if it were that distance; that claim was mis-sourced and sign-inverted, and aperture_resolution.json is the only source for this class. The Richardson witness is blind to this whole class in both signs at all eight configurations BY CONSTRUCTION: an aperture error of one cell at each rung is proportional to dx, which is exactly what 2*S(a/60) - S(a/30) is built to remove, so no tightening of its 0.01 gate can catch it and none is attempted. The calibration this case supplies to any downstream multi-iris filter is aperture-resolved to one fine cell in both signs. The three declared apertures are pinned as claims (G18-C): each must be an exact and EVEN integer cell count at BOTH rungs, a geometric condition no one-fine-cell relabel can satisfy."
            ),
            "config": {
                "a_m": A_WR90, "b_m": B_WR90, "t_m": T_IRIS,
                "freqs_hz": [float(f) for f in FREQS],
                "cpml_layers_rule": "ceil(0.75*lambda_g(8.2 GHz)/dx) = 60 coarse / "
                                     "120 fine — PR #480 B2 measured that 0.5*lambda_g "
                                     "left the envelope absorber-limited (ripple 0.0366 "
                                     "vs 0.0093 at 0.75)",
                "coarse_cells_per_a": COARSE_CELLS,
                "fine_cells_per_a": FINE_CELLS,
                "canonical_glen_m": CANONICAL[0],
                "canonical_iris_frac": CANONICAL[1],
                "gated_normalize": "flux",
            },
            "gates": {
                "fine_gate_abs": GATE_FINE_ABS,
                "fine_gate_abs_per_config": dict(GATE_FINE_ABS_PER_CONFIG),
                "fine_measured_envelope_abs": round(env_fine, 4),
                "richardson_gate_abs": GATE_RICH_ABS,
                "richardson_measured_envelope_abs": round(env_rich, 4),
                "first_order_ratios": ratios,
                "posture": "gate = round-UP(measured envelope x 1.5), "
                           "enforced as EXACT equality by the write-fixture "
                           "self-check (PR #475 convention, PR #480 "
                           "tightening); coarse rung, raw extraction, ripple "
                           "and phase are reported, never gated; modal "
                           "extraction is no longer fenced (retracted, see "
                           "provenance) but structures beyond one symmetric "
                           "inductive iris remain fenced, never gated"
                           "; issue #812 re-gate: the BINDING fine gate is now per-configuration, gate = round-UP(that configuration's own envelope x 1.5) at quantum 1000, with the pooled 0.02 kept as a ceiling and the one-cell aperture detection table gated as its own claim; #931: pooled 0.04 -> 0.02 and all eight per-config gates re-derived DOWN from the corrected-thickness envelopes (VESSL 369367259159), never widened",
            },
            "gated_fine": fine_rows,
            "one_cell_volume_witness": {
                "note": ("#931 lattice ownership contract, design note "
                         "20260906 section 5: a PEC volume one cell thick is "
                         "a filled slab with a tangential wall at BOTH faces, "
                         "at every thickness, with no flag. Before #931 the "
                         "far face was never a wall and a two_plane flag put "
                         "it back for t = 1 only; nothing independent said "
                         "which was right AT ONE CELL, because the thin-limit "
                         "anchor is a t -> 0 statement rather than a t = dx "
                         "one. Here the mode-matching oracle — which takes the "
                         "physical t and knows nothing about the lattice — is "
                         "run against rfx at t = 1..8 cells on the coarse "
                         "rung at the worst-gap aperture. GATE: every swept "
                         "rung must IDENTIFY its own drawn thickness — the "
                         "oracle is evaluated at t-1, t and t+1 cells and the "
                         "residual argmin must land on t. At t = 1 the t-1 "
                         "alternative is precisely the pre-#931 realization "
                         "(one wall, a zero-thickness screen), so the rule in "
                         "dispute is decided by a measurement rather than by "
                         "a convention. RETIRED, and recorded rather than "
                         "deleted (monotone_range_criterion): the original "
                         "statement — the t = 1 residual lies inside the "
                         "range t = 2..8 spans — turned out to be vacuous "
                         "once the residual was measured to be monotone "
                         "decreasing in t, because then t = 1 is the "
                         "extremum for every possible outcome, a perfect "
                         "0.0000 included. It was retired for having no "
                         "power in either direction, not for its verdict."),
                "aperture_mm": round(THICK_SWEEP_D * 1e3, 3),
                "cells_per_a": THICK_SWEEP_RUNG,
                "rows": thick_rows,
                "one_cell_gap_abs": one,
                "multi_cell_gap_range_abs": [min(multi), max(multi)],
                "identified_every_thickness": bool(identified),
                "one_cell_two_wall_vs_one_wall_x": round(
                    one_ident["gap_at_t_minus_1"]
                    / max(one_ident["gap_at_t"], 1e-12), 3),
                "monotone_range_criterion": {
                    "verdict": bool(one_cell_on_curve),
                    "status": "RETIRED — vacuous for a monotone residual",
                },
                "passed": bool(identified),
            },
            "one_cell_aperture_detection_witness": one_cell,
            "modal_extraction_witness": modal_witness,
            "coarse_diagnostic": coarse_rows,
            "raw_extraction_record": raw_rows,
            "truncation_witness": trunc,
            "provenance": {
                "generated_by": "validation/crossval/18_wr90_iris_modematch.py --write-fixture",
                "oracle": "in-script TEn0 mode-matching cascade with "
                          "unitarity/convergence/Marcuvitz/limit witnesses "
                          "(re-run and printed above); re-derived again "
                          "independently in the frozen gate test",
                "no_preflight_note": (
                    "compute_waveguide_s_matrix runs its own extractor "
                    "passivity self-check (warnings are part of this "
                    "record) but no sim.preflight(); operating-point "
                    "guarantees are the realized-geometry asserts in "
                    "run_point, which read realized_pec_edge_masks through "
                    "validation/crossval/_wr90_iris_realized.py."
                ),
                "modal_fence_retraction_2026_07_28": (
                    "An earlier revision of this case FENCED normalize=True "
                    "modal extraction, citing measured max column power "
                    "1.112 (later 1.15374 at driven port 0 coarse / "
                    "1.16407 at driven port 1 fine, with a second "
                    "per-frequency advisory referencing issue #337). Those "
                    "runs carried the d + 2*dx electrical aperture and a "
                    "0.5*lambda_g absorber. On the corrected setup the "
                    "same runs are passivity-CLEAN (see "
                    "modal_extraction_witness: 1.0200 / 1.0099 / 1.0150 / "
                    "1.0012, zero extractor warnings), so the fence is "
                    "RETRACTED: the non-passivity was a setup symptom, not "
                    "an extractor property. Recorded so the withdrawn "
                    "claim stays auditable."
                ),
            },
        }
        art_dir = os.path.join(_SCRIPT_DIR, "_18_wr90_iris_results")
        os.makedirs(art_dir, exist_ok=True)
        with open(os.path.join(art_dir, "rfx.json"), "w") as f:
            json.dump(payload, f, indent=1)
        fix_dir = os.path.join(_REPO_ROOT, "tests", "fixtures", "wr90_iris_modematch")
        os.makedirs(fix_dir, exist_ok=True)
        with open(os.path.join(fix_dir, "fixture.json"), "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {art_dir}/rfx.json and tests/fixtures/wr90_iris_modematch/fixture.json")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
