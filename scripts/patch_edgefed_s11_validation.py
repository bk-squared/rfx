"""Issue #80 acceptance check — edge-fed patch |S11| passivity (historical harness).

Runs the GitHub issue #80 reproduction (edge-fed Hammerstad patch on
RO4003C, 50 ohm microstrip feed) through ``compute_msl_s_matrix`` and dumps
the full |S11|(f) trace. The exit code gates PASSIVITY only (max|S11| <=
1.05 — the issue #80 defect was |S11| > 1).

HISTORY. The original acceptance criterion 1 ("|S11| dip at the analytic
Balanis 9.21 +/- 0.20 GHz"; pre-fix the dip sat at 10.11 GHz) is retired
twice over and is no longer gated here:

  * the |S11| dip of a directly edge-fed patch is the OFF-RESONANCE match
    point, not the resonance — reading the dip as the resonance is a
    category error (issue #118);
  * 9.21 GHz is Balanis on the DESIGN dimensions, realized on no mesh, and
    the 9.32 GHz fed resonance it seemed to confirm was two errors
    cancelling — the #702 sheet-node material fix moved the fed TM010 to
    8.16 GHz on the harminv-gate board (issue #782).

WHICH BOARD THIS IS (#782 one-mesh anchor rule, #931 redraw). There are two
committed gates and they sit on two DIFFERENT realized boards on purpose:

  * tests/locks/test_patch_edgefed_resonance_harminv.py — "Board H",
    dx = H_SUB/4 = 196.75 um, ground plane at round(4 mm / dx) * dx =
    3.935 mm, patch raster 43 x 51 cells;
  * tests/locks/test_patch_edgefed_s11_passivity.py — "Board S",
    dx = 0.197 mm exactly, ground plane 3.940 mm, patch raster 44 x 51.

This script is on BOARD H: same dx, same ground plane, same W / L / W_MSL /
PORT_MARGIN / L_MSL / DOM_X / DOM_Y / DOM_Z. Its trace is comparable with the
harminv gate's numbers and NOT with the passivity gate's — Board S's band
(7.4, 8.2 GHz), its Re(Zin) floor and its 44-cell raster were measured on a
board this script does not build. Mixing a constant from one into a reading
from the other describes a board that exists on no mesh, which is the ~2-point
error class issue #782 documents. The passivity check below is the generic
max|S11| <= 1.05 defect gate from issue #80, which is a physics bound rather
than a per-board pin, so it applies to either.

(The passivity gate module still carries a comment calling its geometry a
mirror of this script. That was true before #931 redrew the boards; it is
Board S's file to correct.)

S11 = gamma/alpha is a pure voltage-wave amplitude ratio (it does NOT
use Z0), so the Fix-C N-probe voltage decomposition is what this tests.
The separate Z0-extraction error (contaminated I1, ~74 vs ~54 ohm) does
not enter S11 and is tracked as a distinct follow-up.

Exit 0 = PASS (max|S11| <= 1.05), exit 1 = FAIL.
"""
from __future__ import annotations

import sys

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
L_MSL = 8.0e-3
PORT_MARGIN = 5.0e-3
# ON-LATTICE BOARD (#931 §1.3). A sheet lands on the node plane nearest its
# declared plane, so a foil meant to lie on a dielectric interface needs that
# interface ON a node line — otherwise it snaps half a cell into the laminate
# and the cavity carries the wrong medium in series (preflight says so). The
# cell size is therefore h_sub / 4 rather than a round 0.197 mm: 0.19675 mm,
# a 0.13 % mesh change, and both board faces are exact nodes. The old
# 0.197 mm mesh put the top face 0.3 cell off the node line.
N_SUB_CELLS = 4
DX = H_SUB / N_SUB_CELLS
N_AIR_BELOW_CELLS = 20         # ~3.9 mm of air under the board, on-lattice
DOM_X = 29.747e-3
DOM_Y = 18.130e-3
DOM_Z = 12.787e-3
Y_C = DOM_Y / 2.0

# Stack z coordinates, named once. The ground, the feed trace and the patch
# are etched copper on a 0.787 mm RO4003C board, so under the lattice
# ownership contract (#931 §1.3) each is a SHEET — a footprint on ONE node
# plane, zero thickness, owning no cell — declared at the board face it is
# etched on. Before the contract all three were 1-cell PEC Boxes and the
# script parked them AROUND the substrate rather than on it: the ground one
# cell below the dielectric and the trace/patch one cell above it. The old
# rule put a wall only on a masked cell's lower face, so those offsets were
# the compensation that landed the walls near the board — at the cost of a
# vacuum cell in series on each side of a 4-cell substrate.
Z_GND = N_AIR_BELOW_CELLS * DX  # board bottom face = ground foil plane
Z_SUB_LO = Z_GND
Z_SUB_HI = Z_SUB_LO + H_SUB    # board top face = trace / patch foil plane
Z_TRACE = Z_SUB_HI


def main() -> int:
    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    # Ground foil, on the board's bottom face.
    sim.add_thin_conductor(Box((0, 0, Z_GND), (DOM_X, DOM_Y, Z_GND)))
    # RO4003C substrate.
    sim.add(Box((0, 0, Z_SUB_LO), (DOM_X, DOM_Y, Z_SUB_HI)),
            material="ro4003c")
    # 50 ohm microstrip feed trace, on the board's top face.
    sim.add_thin_conductor(
        Box((0, Y_C - W_MSL / 2, Z_TRACE),
            (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2, Z_TRACE)))
    # Edge-fed patch, abutting the feed trace on the same face. Sheet
    # footprints on one plane are UNIONED before the edge rule, so the shared
    # edge between trace and patch is metal, not a slit (#931 §1.3).
    sim.add_thin_conductor(
        Box((PORT_MARGIN + L_MSL, Y_C - W / 2, Z_TRACE),
            (PORT_MARGIN + L_MSL + L, Y_C + W / 2, Z_TRACE)))
    # Wider, higher-centre source than the default
    # GaussianPulse(f0=freq_max/2=7.5GHz, bw=0.8) — that default rolls off
    # ~exp(-6.25) ≈ 0.002 at 15 GHz, starving the upper part of the
    # frequency sweep of signal. The previous long-window run
    # (369367239037) had max|S11|=1.527 at 11.96 GHz — exactly the
    # low-SNR tail. f0=8.5 GHz, bw=1.6 puts the spectral peak near
    # ~10 GHz and gives ~14 GHz 1/e width, covering the full 1.5-15 GHz
    # sweep with usable SNR (~77% of peak amplitude at 15 GHz vs 0.2%).
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, Z_SUB_LO),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )

    # Preflight (user directive 2026-05-20: never ignore preflight). What this
    # fixture draws on the on-lattice mesh: the lossless-dielectric infinite-Q
    # advisory (RO4003C is modelled with sigma = 0 here — real, and it is why
    # the gate below is passivity, not Q). The sheets-dropped notice that used
    # to accompany it is GONE and was never about this geometry: preflight
    # assembled without a PEC-sheet collector (#931 §6), and it passes one now
    # (rfx/api/_preflight.py::_assemble_realized, merged ab56f5b2), so the
    # advisory it stood in for — where each foil actually realized — is the
    # sheet_plane_realized INFO instead.
    #
    # Three advisories the old drawing produced are gone, and for a reason
    # worth recording: the off-lattice conductor faces, the buried-sheet
    # warning and the sheet-cavity electrical-thickness finding all came from
    # a board whose faces were 0.3 cell off the node line and whose metal sat
    # a cell away from the laminate. The board is on the lattice now and the
    # realized cavity IS the declared 787 um. Nothing was suppressed.
    # Build-time realization check (no solve): the three declared foils must
    # BE the realized tangential-wall planes along z, and the board between
    # them the declared H_SUB. Read from the contract's own realization
    # (#931 §1.7) over the arrays the assembly hands the stepper.
    from rfx import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    _grid = sim._build_grid()
    _sheets: list = []
    _m, _d, _l, _pec, _a, _b, _c = sim._assemble_materials(
        _grid, pec_sheets=_sheets)
    _z = np.asarray(coords_from_uniform_grid(_grid).z, dtype=float)
    _walls = realized_wall_planes(
        realized_pec_edge_masks(_pec, sheets=_sheets,
                                periodic=sim._periodic_flags()), 2)
    _k_gnd = int(np.argmin(np.abs(_z - Z_GND)))
    _k_trc = int(np.argmin(np.abs(_z - Z_TRACE)))
    print(f"realized z wall planes {_walls} = "
          f"{[round(float(_z[k]) * 1e6, 1) for k in _walls]} um "
          f"(declared ground {Z_GND * 1e6:.1f} um -> node {_k_gnd}, "
          f"trace {Z_TRACE * 1e6:.1f} um -> node {_k_trc}); "
          f"realized board {float(_z[_k_trc] - _z[_k_gnd]) * 1e6:.1f} um "
          f"against a declared {H_SUB * 1e6:.1f} um", flush=True)
    if _walls != sorted({_k_gnd, _k_trc}):
        raise SystemExit(
            f"realized z wall planes {_walls} != declared "
            f"{sorted({_k_gnd, _k_trc})} — the foils did not land on the "
            "board faces")
    # H_SUB is 3.995 cells at this dx, so the top face snaps to the nearest
    # node and the realized board is one part in 800 thicker than declared.
    # That is the off-lattice residual the contract reports instead of
    # absorbing; anything larger than half a cell is a different board.
    if abs(float(_z[_k_trc] - _z[_k_gnd]) - H_SUB) > 0.5 * DX:
        raise SystemExit("realized board thickness is more than half a cell "
                         "from the declared H_SUB")

    print("=== sim.preflight() ===", flush=True)
    sim.preflight()

    # num_periods 200: long-window diagnostic for the truncation
    # hypothesis (issue #80 stage S1 post-mortem). At the patch's
    # Q~30–50 around 9 GHz, 25 periods (~3.3 ns) leaves significant
    # ring-down energy in the DFT window — V (Ez) and I (Hy/Hz) leak
    # differently and corrupt the V·I-split denominator a=(V+Z0·I)/2.
    # 200 periods (~27 ns) is comfortably >60 dB down. If |S11| becomes
    # bounded and smooth, truncation was the upstream cause; if not, keep
    # diagnosing. (The 2026-05 note here expected the dip near 9.21 GHz —
    # retired: the dip is the match point (#118), and 9.21 GHz predates the
    # #702 sheet-node material fix (#782).)
    res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=200.0)

    freqs = np.asarray(res.freqs, dtype=float)
    s11 = np.abs(np.asarray(res.S)[0, 0, :])
    z0 = np.asarray(res.Z0)[0, :]

    i_dip = int(np.argmin(s11))
    f_dip = freqs[i_dip] / 1e9
    s11_dip_db = 20.0 * np.log10(max(float(s11[i_dip]), 1e-12))
    s11_max = float(np.max(s11))

    print("=== issue #80 acceptance — patch S11 (stage S1: V·I split) ===")
    print(f"PATCH-EDGEFED: S11 minimum = {s11_dip_db:.1f} dB at {f_dip:.3f} GHz")
    print("PATCH-EDGEFED: dip is reported, NOT gated — it is the off-resonance "
          "match point (issue #118); the retired 9.21 GHz Balanis target "
          "predates #702 (issue #782)")
    print(f"PATCH-EDGEFED: max|S11| = {s11_max:.3f} (headline — must be <= 1 for "
          f"a passive patch; pre-S1 Fix-C blew up to ~8.6)")
    print(f"PATCH-EDGEFED: Z0[0] median Re = {np.median(z0.real):.2f} ohm")
    # full |S11|(f) trace for the log
    for f, a in zip(freqs / 1e9, s11):
        print(f"PATCH-EDGEFED-TRACE: {f:7.3f} GHz  |S11|={a:.5f}")

    ok_passive = s11_max <= 1.0 + 0.05
    print(f"PATCH-EDGEFED: dip at {f_dip:.3f} GHz (reported, not gated — "
          "off-resonance match point, issue #118)")
    print(f"PATCH-EDGEFED: ACCEPTANCE (|S11| <= 1.05) "
          f"{'PASS' if ok_passive else 'FAIL'} (max|S11| = {s11_max:.3f})")
    return 0 if ok_passive else 1


if __name__ == "__main__":
    sys.exit(main())
