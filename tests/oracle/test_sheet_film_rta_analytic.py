"""Resistive-sheet R/T/A vs the exact conductive-slab analytic solution (#711).

WHAT THIS PINS
--------------
``add_thin_conductor``'s lossy path folds a film into one cell of volumetric
conductivity (sigma_eff = sigma_bulk * t / d_norm).  That is the model of a
PENETRABLE resistive film — it transmits.  The Leontovich surface resistance
the ``surface_impedance_f0`` path computes is the boundary impedance of an
OPAQUE thick conductor — it does not.  Issue #711 asks which contract the
implementation carries; this test measures it:

* a film with sheet resistance Rs = eta0/2 has (thin-film limit)
  R = 0.25, T = 0.25, A = 0.50 — the maximally discriminating point;
* the gate compares against the EXACT transfer-matrix R/T of a slab of
  thickness dx and complex permittivity eps = 1 + i*sigma_eff/(w*eps0), so
  the one-cell realization is judged against its own physics with no
  thin-film approximation error;
* the PEC endpoint (sigma_bulk >= 1e6 -> PEC sheet) must be OPAQUE in this
  orientation: the sheet is an x-normal plane whose footprint spans the
  whole cross-section, so both components tangential to it (Ez and Ey) are
  PEC on that plane and T must collapse. On this 2-D TMz lane the z axis
  is length 1, so the Ez edge's backward neighbour along z is itself
  (#689 wrap) and every footprint node is a zeroed Ez.

REALIZATION UNDER THE LATTICE OWNERSHIP CONTRACT (#931). This module hands
``add_thin_conductor`` a Box ONE CELL thick, where two sibling oracle
fixtures hand it a zero-extent Box, and nothing said which spelling was
normative. §1.3 settles it without a new rule: a sheet lands on the node
plane nearest its mid-plane, a tie resolving to the LOWER plane, and a
Box from node k to node k+1 has its mid-plane exactly on that tie. Both
spellings therefore land on node k. Measured on this grid (no solve): the
sheet realizes at x-node 110 (x = 45 mm + the 20-layer CPML pad) with 61
Ez edges and 60 Ey edges, which is what the pre-#931 rule realized for
the same declaration — so nothing here was re-measured, and the R/T
numbers below are the same measurement they always were.

Measurement follows the committed R/T recipe verbatim (TFSF plane wave +
flux monitors + two-run reference subtraction; never FFT-of-probe).
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.probes.probes import flux_spectrum

ETA0 = 376.730
EPS0 = 8.8541878128e-12
C0 = 299792458.0

F0 = 10e9
BW = 0.5
DX = 0.5e-3                 # lambda/60 at 10 GHz
DOM_X = 90e-3
DOM_Y = 10e-3
SHEET_X = 45e-3             # one-cell slab [SHEET_X, SHEET_X+DX)
REFL_X = 25e-3
TRANS_X = 65e-3
FREQS = np.linspace(8e9, 12e9, 9)

# film target: Rs = eta0/2 -> sigma_bulk * t = 2/eta0
T_FILM = 35e-6
SIGMA_FILM = (2.0 / ETA0) / T_FILM          # ~151.7 S/m << 1e6 -> lossy path
SIGMA_EFF = SIGMA_FILM * T_FILM / DX        # what one cell carries


def slab_rt_exact(f, sigma_eff, d):
    """Exact normal-incidence R,T of a slab: eps = 1 + i*sigma/(w*eps0)."""
    w = 2 * np.pi * f
    n = np.sqrt(1 + 1j * sigma_eff / (w * EPS0))
    k1 = n * w / C0
    r01 = (1 - n) / (1 + n)
    t01, t10 = 2 / (1 + n), 2 * n / (1 + n)
    ph = np.exp(1j * k1 * d)
    # standard Airy summation, r10 = -r01:
    r = r01 + (t01 * t10 * (-r01) * ph**2) / (1 - (-r01)**2 * ph**2)
    t = (t01 * t10 * ph) / (1 - (-r01)**2 * ph**2)
    return np.abs(r)**2, np.abs(t)**2


def _build(kind: str) -> Simulation:
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DX), dx=DX,
                     boundary="cpml", cpml_layers=20, mode="2d_tmz")
    if kind == "film":
        sim.add_thin_conductor(Box((SHEET_X, -1, -1), (SHEET_X + DX, 1, 1)),
                               sigma_bulk=SIGMA_FILM, thickness=T_FILM)
    elif kind == "pec":
        sim.add_thin_conductor(Box((SHEET_X, -1, -1), (SHEET_X + DX, 1, 1)),
                               sigma_bulk=5.8e7)
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x")
    sim.add_flux_monitor(axis="x", coordinate=REFL_X, freqs=FREQS, name="refl")
    sim.add_flux_monitor(axis="x", coordinate=TRANS_X, freqs=FREQS, name="trans")
    return sim


def _run(sim):
    return sim.run(n_steps=40000, until_decay=1e-6,
                   decay_monitor_component="ez",
                   decay_monitor_position=(TRANS_X, DOM_Y / 2, 0))


def test_the_pec_sheet_realizes_one_plane_at_the_declared_x():
    """Build-time (no solve): where the opaque endpoint actually is.

    The T < 1e-3 gate below only means "opaque at the declared plane" if
    the plane is where the declaration put it. This reads it back off the
    single owner (#931 §1.7) instead of trusting the Box coordinates.
    """
    from tests._realized_geometry import assert_sheet_planes, realized

    sim = _build("pec")
    assert_sheet_planes(sim, 0, expected_m=[SHEET_X], what="PEC film")
    rz = realized(sim)
    assert rz.pec_mask is None or not bool(np.asarray(rz.pec_mask).any()), (
        "a sheet owns no cell")
    mx, my, mz = (np.asarray(m) for m in rz.edge_masks)
    # x-normal sheet: the two tangential components are PEC on the plane,
    # the normal one (Ex) stays live so charge can sit on the film.
    assert mx.sum() == 0, mx.sum()
    assert mz.sum() == rz.grid.shape[1], (mz.sum(), rz.grid.shape)
    assert my.sum() == rz.grid.shape[1] - 1, my.sum()


@pytest.fixture(scope="module")
def rta():
    res_ref = _run(_build("ref"))
    ref_refl = res_ref.flux_monitors["refl"]
    ref_trans = np.asarray(flux_spectrum(res_ref.flux_monitors["trans"]))

    out = {}
    for kind in ("film", "pec"):
        res = _run(_build(kind))
        fm = res.flux_monitors["refl"]
        scat = fm._replace(e1_dft=fm.e1_dft - ref_refl.e1_dft,
                           e2_dft=fm.e2_dft - ref_refl.e2_dft,
                           h1_dft=fm.h1_dft - ref_refl.h1_dft,
                           h2_dft=fm.h2_dft - ref_refl.h2_dft)
        R = -np.asarray(flux_spectrum(scat)) / ref_trans
        T = np.asarray(flux_spectrum(res.flux_monitors["trans"])) / ref_trans
        out[kind] = (np.asarray(R, float), np.asarray(T, float))
    return out


def test_film_matches_exact_slab(rta):
    R, T = rta["film"]
    R_an, T_an = slab_rt_exact(FREQS, SIGMA_EFF, DX)
    print(f"\n[SHEET-RTA] film measured  R={R.round(4).tolist()}")
    print(f"[SHEET-RTA] film analytic  R={R_an.round(4).tolist()}")
    print(f"[SHEET-RTA] film measured  T={T.round(4).tolist()}")
    print(f"[SHEET-RTA] film analytic  T={T_an.round(4).tolist()}")
    # Tolerances = the measurement chain's own committed floor, not this
    # fixture's wish: the validated Fresnel crossval (04_multilayer_fresnel)
    # gates this same TFSF+flux chain at mean error < 0.05 and per-bin
    # |R+T-1| <= 0.06. Band means here land within 0.01 of analytic; the
    # per-bin ripple is the chain's standing-artifact floor (settling-
    # invariant: identical to 4 decimals at 8k and 40k steps).
    assert abs(R.mean() - R_an.mean()) < 0.02, "film band-mean R off the exact slab solution"
    assert abs(T.mean() - T_an.mean()) < 0.02, "film band-mean T off the exact slab solution"
    assert np.max(np.abs(R - R_an)) < 0.06, "film per-bin R outside the chain floor"
    assert np.max(np.abs(T - T_an)) < 0.06, "film per-bin T outside the chain floor"
    A = 1 - R - T
    assert np.all(A > 0.3), f"film should absorb ~half the power, got A={A.round(3)}"


def test_film_is_penetrable_not_opaque(rta):
    """The #711 discriminator: a Leontovich OPAQUE boundary at this Rs would
    transmit ~nothing; the penetrable film transmits ~25%."""
    _, T = rta["film"]
    assert np.all(T > 0.10), (
        f"transmission collapsed (T={T.round(4)}): the operator behaves as an "
        "opaque boundary, not the penetrable film its sigma-folding documents")


def test_pec_endpoint_is_opaque(rta):
    R, T = rta["pec"]
    print(f"\n[SHEET-RTA] pec R={R.round(4).tolist()} T={T.round(6).tolist()}")
    assert np.all(np.abs(T) < 1e-3), f"PEC sheet leaked: T={T.round(6)}"
    assert abs(R.mean() - 1.0) < 0.03, f"PEC band-mean R != 1: {R.mean():.4f}"
    assert np.max(np.abs(R - 1.0)) < 0.15, f"PEC per-bin R outside the chain ripple floor: {R.round(4)}"
