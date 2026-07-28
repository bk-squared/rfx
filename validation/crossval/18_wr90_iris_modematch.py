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
  * iris thickness t = 1.524 mm (exactly 2 coarse / 4 fine cells); apertures
    d in {18.288, 12.192, 7.62} mm (24/16/10 coarse cells) — weak / medium /
    strong reflector.
  * fins are drawn PAST the grid walls (the rasterizer clips): the first
    probe revision drew them to the NOMINAL width and left a 1-cell parasitic
    slot at the far wall. run_point asserts the rasterized iris is 2 cells
    thick with one CONTIGUOUS aperture, so that bug class cannot recur.

WHAT IS GATED vs WHAT IS REPORTED (measured 2026-07-28 on the corrected
setup; every axis scanned BEFORE gating)
---------------------------------------------------------------------------
GATED (exit-1 on failure), gates = round-UP(measured envelope x 1.5), and the
--write-fixture self-check demands EXACT equality with that rule:
  * fine rung (dx = a/60), flux extraction, |S11| vs oracle over 8 configs
    (3 apertures x {centred, iris off-centre at 0.42} + 2 extra guide
    lengths): every config within 0.023 -> envelope 0.0232 -> gate 0.04.
    No single configuration sets the envelope any more.
  * Richardson witness at EVERY one of those 8 fine/coarse pairs (not just
    the canonical one): 2*S_fine - S_coarse lands on the oracle within
    0.0051 -> gate 0.01. Cross-confirms the oracle AND the first-order
    attribution (fine/coarse gap ratios 0.527-0.604 = textbook first order).
REPORTED, NOT GATED:
  * coarse rung (dx = a/30): 0.018-0.043 abs.
  * raw (normalize=False) record: worse than flux (gaps 0.021-0.054) with a
    pointwise |raw - flux| difference up to 0.033 at the wide aperture.
  * residual detrended ripple: fine <= 0.0116, coarse <= 0.0166, both at the
    wide aperture with the iris off-centre (down from 0.0706 before the
    absorber fix).
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
 (2) node-plane box corners are half-ulp fragile because the volume mask is
     half-open over NODE coordinates: a fine config rasterized 3 thickness
     nodes instead of 4, and an apparent +/-0.07 "domain sensitivity" was
     that ulp lottery. All interior corners now sit half a cell off the node
     planes; the footprint asserts are exact.
 (3) the fin footprint made the ELECTRICAL aperture d + 2*dx instead of d,
     which alone inflated the envelope 4-6x, and a 0.5*lambda_g absorber
     left the envelope set by CPML reflection rather than discretization
     (PR #480 review, B2/B3). Fins now cover nodes 0..fin_c so the
     electrical aperture equals the nominal d, and CPML = 0.75*lambda_g at
     the 8.2 GHz band edge (60 coarse / 120 fine).

RETRACTED: an earlier revision FENCED normalize=True modal extraction on the
strength of a measured column power 1.112-1.164. On the corrected setup modal
extraction is passivity-CLEAN at every aperture and both rungs (max column
power 1.0207 at d=7.62/a-30, 1.0013 at d=18.288, ZERO extractor warnings), so
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
      # + coarse tier, modal + raw + truncation witnesses; regenerates
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
GATE_FINE_ABS = 0.04    # = round-up(measured fine envelope 0.0232 x 1.5)
GATE_RICH_ABS = 0.01    # = round-up(measured Richardson envelope 0.0051 x 1.5)


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
              num_periods=100.0):
    """One 2-port iris run at grid-exact dimensions with raster asserts."""
    DX = A_WR90 / cells
    d_c = int(round(d_phys / DX))
    t_c = int(round(T_IRIS / DX))
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
    # The volume mask is HALF-OPEN [lo, hi) over NODE coordinates i*DX, so a
    # box corner landing exactly on a node is half-ulp fragile (a fine-rung
    # domain-scan config rasterized 3 thickness-nodes instead of 4 — caught
    # by the assert below). All interior corners therefore sit HALF A CELL
    # off the node planes: x spans nodes [iris_lo, iris_lo + t_c) and the
    # fin edges exclude the first aperture node deterministically, making
    # the rasterized footprint exact regardless of float rounding.
    x_lo = (iris_lo - 0.5) * DX
    x_hi = (iris_lo + t_c - 0.5) * DX
    fin_hi_y = (fin_c + 0.5) * DX   # metal on nodes 0..fin_c: aperture == d (R480 B3)
    sim.add(Box((x_lo, -big, -big), (x_hi, fin_hi_y, big)), material="pec")
    sim.add(Box((x_lo, A_WR90 - fin_hi_y, -big), (x_hi, big, big)), material="pec")
    for x, dr, nm in ((p1 * DX, "+x", "P1"), (p2 * DX, "-x", "P2")):
        sim.add_waveguide_port(x, mode=(1, 0), mode_type="TE", direction=dr,
                               f0=10.3e9, bandwidth=0.41,
                               waveform="modulated_gaussian",
                               freqs=FREQS, name=nm)
    # raster asserts (operating-point guarantees, hand-ported)
    grid = sim._build_grid()
    ny = grid.shape[1]
    assert ny == cells + 1, (ny, cells)          # node convention: a=(ny-1)dx exact
    sig = np.asarray(rasterize(grid, [(e.shape, 1.0, 1e7)
                                      for e in sim._geometry])[1])
    xc = np.where(sig.max(axis=(1, 2)) > 1e6)[0]
    assert len(xc) == t_c, ("iris thickness cells", len(xc), t_c)
    open_y = np.where(sig[xc[0]].max(axis=1) < 1e6)[0]
    assert bool(np.all(np.diff(open_y) == 1)), "aperture not contiguous (slot bug)"
    # deterministic with the half-cell-offset fin edges: d_c + 1 open E-node
    # columns between the innermost conductor node planes (the half-cell
    # effective-aperture ambiguity is part of the measured first-order gap)
    assert len(open_y) == d_c - 1, ("aperture nodes", len(open_y), d_c - 1)

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
        "normalize": str(normalize), "num_periods": num_periods,
        "aperture_cells": int(len(open_y)), "thickness_cells": int(len(xc)),
        "s11": [round(float(v), 5) for v in s11],
        "s21": [round(float(v), 5) for v in s21],
        "max_colpow": round(float(np.max(s11 ** 2 + s21 ** 2)), 4),
        "wall_s": round(wall, 1),
    }


def oracle_s11(d_phys):
    return [round(float(abs(iris_smatrix(A_WR90, d_phys, T_IRIS, f)[0])), 5)
            for f in FREQS]


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
    print(f"\n== GATED fine rung dx=a/{FINE_CELLS} flux (gate {GATE_FINE_ABS} abs) ==")
    fine_rows = []
    fine_configs = ([(d, CANONICAL) for d in D_APERTURES]
                    + [(d, ASYM_CONFIG) for d in D_APERTURES]
                    + [(D_WORST, c) for c in GLEN_SCAN])
    for d, (glen, frac) in fine_configs:
        r = run_point(d, FINE_CELLS, glen=glen, iris_frac=frac)
        r["oracle_s11"] = oracles[str(d)]
        gap = max(_gaps(r, oracles[str(d)]))
        r["max_gap_abs"] = round(gap, 4)
        fine_rows.append(r)
        passed = gap <= GATE_FINE_ABS
        ok &= passed
        print(f"  d={d*1e3:6.2f} glen={glen:.2f} frac={frac:.2f}: "
              f"max|dS11|={gap:.3f} colpow={r['max_colpow']:.3f} "
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
            modal_witness["rows"].append(
                {"d_mm": round(d_f * 1e3, 3), "cells_per_a": cells_f,
                 "max_colpow": rm["max_colpow"],
                 "extractor_warnings": fence_warns})
            print(f"  d={d_f*1e3:6.2f} a/{cells_f}: colpow "
                  f"{rm['max_colpow']:.4f} ({len(fence_warns)} warnings)",
                  flush=True)

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
            required = np.ceil(env * 1.5 * 100) / 100
            if abs(gate - required) > 1e-9:   # EXACT ceil(x1.5) — R480
                print(f"  ENVELOPE/GATE MISMATCH ({tier}): gate {gate} "
                      f"must equal round-up(env x 1.5) = {required}")
                ok = False

        payload = {
            "schema": "rfx.wr90_iris_modematch",
            "schema_version": 1,
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
                "dx = a/60 within 0.04 abs = round-up(measured envelope "
                "0.0232 x 1.5) over 8 configs (3 apertures x {centred, "
                "iris off-centre at 0.42 of the guide} + 2 extra guide "
                "lengths; every config lands within 0.023, so no single "
                "configuration sets the envelope), and the Richardson "
                "extrapolation 2*S(a/60) - S(a/30) on the oracle within "
                "0.01 abs (envelope 0.0051) at EVERY one of those 8 pairs, "
                "which cross-confirms the oracle and the first-order "
                "attribution (gap ratios 0.527-0.604 = textbook first "
                "order). REPORTED, NOT GATED: the coarse rung dx = a/30 "
                "(0.018-0.043 abs); the raw normalize=False record, which "
                "is WORSE than flux (gaps 0.021-0.054) with a pointwise "
                "|raw - flux| difference up to 0.033 at the wide aperture; "
                "residual detrended ripple (fine <= 0.0116, coarse <= "
                "0.0166, both at the wide aperture with the iris "
                "off-centre, down from 0.0706 before the absorber fix); "
                "and phase (magnitude-only lane posture). FENCED, never "
                "gated: everything beyond ONE symmetric inductive iris — "
                "multi-iris filters, posts, septa and off-centre apertures "
                "stay EXPERIMENTAL per docs/guides/support_matrix.md. "
                "THREE SETUP DEFECTS were found during this campaign, each "
                "having corrupted an earlier revision's numbers and each "
                "now fenced by an assert or a derived setting: (1) a "
                "parasitic wall-slot (fins drawn to the NOMINAL guide "
                "width leave a 1-cell gap at the actual grid wall); (2) "
                "node-plane box corners are half-ulp fragile because the "
                "volume mask is half-open over NODE coordinates — one fine "
                "config rasterized 3 thickness nodes instead of 4, and an "
                "apparent +/-0.07 'domain sensitivity' was that ulp "
                "lottery; (3) the fin footprint made the ELECTRICAL "
                "aperture d + 2*dx instead of d, which alone inflated the "
                "envelope 4-6x, and a 0.5*lambda_g absorber left the "
                "envelope set by CPML reflection rather than "
                "discretization (PR #480 review B2/B3; CPML is now "
                "0.75*lambda_g at the band edge = 60 coarse / 120 fine). "
                "RETRACTED: an earlier revision fenced normalize=True "
                "modal extraction on the strength of a measured column "
                "power 1.112-1.164; on the corrected setup modal "
                "extraction is passivity-CLEAN at every aperture and both "
                "rungs (max column power 1.0207 at d = 7.62 mm / a-30, "
                "1.0013 at d = 18.288 mm, ZERO extractor warnings), so "
                "that non-passivity was a symptom of defects (1)-(3) and "
                "not a reflector-inflation property of the extractor — the "
                "fence is withdrawn and the measurement is committed as "
                "modal_extraction_witness. Palace WavePort corroboration "
                "(stage S2) and a published multi-iris filter (stage S3) "
                "are follow-on stages, not claimed here."
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
                           "inductive iris remain fenced, never gated",
            },
            "gated_fine": fine_rows,
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
                    "guarantees are the raster asserts in run_point."
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
                    "modal_extraction_witness: 1.0207 / 1.0101 / 1.0144 / "
                    "1.0013, zero extractor warnings), so the fence is "
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
