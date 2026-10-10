"""Penetrable resistive-sheet R/T/A against the ideal shunt sheet (#711).

A film with Rs = eta0/2 has R=0.25, T=0.25, A=0.50. The one-plane
Yee realization has second-order residuals: y'=y*cos(w*dt/2)/cos(k*dx/2).
The 1e-3 per-bin bars are four times the measured worst R/T residual.
The one-cell Box lands on its lower node plane under lattice ownership
contract #931. PEC opacity and film penetrability remain separate checks.
Measurement uses TFSF, flux monitors and two-run reference subtraction.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.probes.probes import flux_spectrum

ETA0 = 376.730

F0 = 10e9
BW = 0.5
DX = 0.5e-3                 # lambda/60 at 10 GHz
DOM_X = 90e-3
DOM_Y = 10e-3
SHEET_X = 45e-3             # one-cell Box, film on its lower node plane
REFL_X = 25e-3
TRANS_X = 65e-3
FREQS = np.linspace(8e9, 12e9, 9)

# film target: Rs = eta0/2 -> sigma_bulk * t = 2/eta0
T_FILM = 35e-6
SIGMA_FILM = (2.0 / ETA0) / T_FILM          # ~151.7 S/m << 1e6 -> lossy path


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


def test_film_matches_the_ideal_sheet(rta):
    R, T = rta["film"]
    A = 1 - R - T
    for name, measured, ideal in (("R", R, 0.25), ("T", T, 0.25), ("A", A, 0.50)):
        print(f"[SHEET-RTA] {name}={measured.tolist()}, ideal={ideal}, "
              f"max_error={np.max(np.abs(measured - ideal)):.12g}")
        assert np.all(np.abs(measured - ideal) < 1e-3), (
            f"film per-bin {name} off the ideal sheet: {measured}")


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
