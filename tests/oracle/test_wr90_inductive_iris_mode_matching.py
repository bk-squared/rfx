"""A thick symmetric inductive iris in WR-90, against TE_n0 mode matching.

The structure is a PEC septum of thickness 1.524 mm standing across a WR-90
guide (a = 22.86 mm) from both side walls, leaving a centred slot of width d.
Below the aperture's own cutoff the slot is inductive: the iris behaves as a
shunt inductance across the guide, so |S11| rises as d narrows and falls with
frequency across 8.2-12.4 GHz.  The reference is semi-analytic and lives in
this repository — a TE_{n0} mode-matching cascade of two H-plane width steps
joined by a length-t guide section (``tests/_wr90_iris_mode_matching.py``) —
which makes this an oracle comparison, not a cross-solver one.  It is the
only place in the suite where a PEC obstacle INSIDE a waveguide is held
against an analytic reference.

The tests come in two tiers.

The fast tier never solves.  It re-runs the oracle's own self-witnesses
(unitarity on a lossless obstacle, reciprocity, convergence in the mode
count, the thin-iris limit against the Marcuvitz closed form, the d -> a
no-obstacle identity, the deep-constriction limit).  Those are the oracle's
independent anchors.  The cross-validation gate this test replaces also
compared the oracle with a "second, independently typed" copy of the same
formulation; the two agreed to exactly 0.0 at all 87 points — the same
expressions in the same order — so that comparison checked nothing and is not
carried over.

The second tier is one FDTD run (about 15 s, so it runs on every PR): the 12.192 mm aperture on the coarse mesh
(dx = a/30 = 0.762 mm, an exact divisor of a, so nothing about this geometry
is rounded), flux-normalized |S11| at 29 frequencies against the oracle.

BEFORE that solve the test reads back what the lattice actually built —
through ``rfx.boundaries.pec.realized_pec_edge_masks``, the product's own
realization, reached by the shared test-side reader
``tests/_realized_geometry.py`` — and refuses to run unless the iris stands
tangential walls at BOTH drawn faces exactly t_c cells apart and the aperture
between the two fins is ONE contiguous opening exactly d_c cells wide.  Both
refusals are defects this structure has actually had: fins drawn only to the
NOMINAL guide width left a one-cell parasitic slot at the real wall, and
before the lattice ownership contract a body's far face was never a wall, so
the realized iris was one cell thinner than the one the oracle was handed.

The gate is the repository's shared rule applied to the gap measured at this
one configuration: ``gate_from_envelope(measured, quantum=100)``.  It pins a
measured envelope on the coarse mesh; it is not an accuracy claim, and the
per-frequency curve is printed with every run so the number is never read
without it.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from tests._gate_policy import gate_from_envelope
from tests._realized_geometry import node_index, realized
from tests._wr90_iris_mode_matching import (A_WR90, C0, MU0, _gamma, _redheffer,
                                            _step_junction, iris_smatrix,
                                            marcuvitz_thin_b, validate_oracle)

B_WR90 = 10.16e-3
T_IRIS = 1.524e-3
#: 0.15 GHz spacing resolves the residual ripple of the extraction.
FREQS = np.linspace(8.2e9, 12.4e9, 29)
#: 0.75 * lambda_g at the 8.2 GHz band edge: at half that depth the residual
#: ripple is absorber-limited rather than discretization-limited.
CPML_DEPTH = 45.7e-3
#: dx = a/30 = 0.762 mm. a is an exact multiple, so the guide width, the
#: aperture (16 cells) and the iris thickness (2 cells) are all grid-exact.
COARSE_CELLS = 30
D_APERTURES = (18.288e-3, 12.192e-3, 7.62e-3)   # 24 / 16 / 10 coarse cells
D_ARM = 12.192e-3            # the FDTD arm's aperture
GLEN_M = 0.20                # guide length
IRIS_FRAC = 0.50             # iris at mid-guide
NUM_PERIODS = 100.0

#: Measured on this configuration (d = 12.192 mm, dx = a/30, flux
#: normalization, 29 frequencies) — see the per-frequency table the FDTD arm
#: prints.  The gate is the shared repository rule applied to it; changing
#: either needs a re-measurement, not an edit.
MEASURED_MAX_GAP_ABS = 0.02456
GATE_ABS = gate_from_envelope(MEASURED_MAX_GAP_ABS, quantum=100)


def _all_blocks(a, d, t, f, n_a=40):
    """All four generalized S blocks of the iris, assembled from the helper's own
    junction and cascade functions (``iris_smatrix`` returns only the TE10 S11 and
    S21)."""
    k = 2 * np.pi * f / C0
    n_b = max(4, int(round(n_a * d / a)))
    s_step = _step_junction(a, d, k, n_a, n_b)
    s_rev = (s_step[3], s_step[2], s_step[1], s_step[0])
    g_b = np.array([_gamma(m, d, k) for m in np.arange(1, 2 * n_b, 2)])
    prop = np.diag(np.exp(-g_b * t))
    zero = np.zeros((n_b, n_b), dtype=complex)
    return _redheffer(_redheffer(s_step, (zero, prop, prop, zero)), s_rev)


def oracle_s11(d_phys, t_phys=T_IRIS, freqs=FREQS):
    """|S11|(f) of the iris from the first implementation."""
    return np.array([abs(iris_smatrix(A_WR90, d_phys, t_phys, float(f))[0])
                     for f in freqs])


# --------------------------------------------------------------------------- #
# FAST TIER — the oracle's own witnesses, no solve.
# --------------------------------------------------------------------------- #
def test_oracle_self_witnesses():
    """Unitarity, mode-count convergence, Marcuvitz, and the two limits.

    Each number is asserted here against its own bound as well as inside
    ``validate_oracle``, and printed, because a witness nobody reads is a
    witness nobody can contradict.
    """
    w = validate_oracle()
    print("\n[oracle self-witnesses]",
          {k: float(f"{v:.3e}") for k, v in w.items()})
    assert w["unitarity_dev"] < 1e-9            # lossless: |S11|^2 + |S21|^2 = 1
    assert w["mode_convergence"] < 1e-3         # n_a 40 -> 80
    assert w["marcuvitz_max_rel"] < 0.15        # leading-order closed form
    assert w["open_limit_s11"] < 1e-6           # d -> a: no obstacle
    assert w["deep_limit_s11"] > 0.999          # d = a/5.7, t = 6 mm: a short


def test_oracle_is_reciprocal_and_symmetric():
    """S12 = S21 and S22 = S11 on a geometrically symmetric obstacle.

    The two transmission blocks come from different matrix products (T_ab
    through the reverse junction, T_ba through the forward one), so their
    agreement is a check on the junction algebra and not an identity.
    """
    worst_rec = 0.0
    worst_sym = 0.0
    for d in D_APERTURES:
        for f in (FREQS[0], FREQS[len(FREQS) // 2], FREQS[-1]):
            s11, s12, s21, s22 = _all_blocks(A_WR90, d, T_IRIS, float(f))
            worst_rec = max(worst_rec, abs(s12[0, 0] - s21[0, 0]))
            worst_sym = max(worst_sym, abs(s22[0, 0] - s11[0, 0]))
    print(f"[oracle] max |S12 - S21| = {worst_rec:.3e}, "
          f"max |S22 - S11| = {worst_sym:.3e}")
    assert worst_rec < 1e-12
    assert worst_sym < 1e-12


def test_thin_iris_limit_follows_the_marcuvitz_closed_form():
    """B/Y0 = (lambda_g/a) cot^2(pi d / 2a), inductive sign, at t -> 0.

    Marcuvitz is a leading-order result for a zero-thickness iris, so the
    witness is the agreement being of the expected ~10 % size AND the
    susceptance being negative (inductive) at every aperture — a capacitive
    sign would be a different obstacle.
    """
    f = 10e9
    rows = []
    for d in (16e-3, 12e-3, 8e-3):
        s11t, _ = iris_smatrix(A_WR90, d, 1e-9, f)
        b_mm = float(np.real(-2 * s11t / (1j * (1 + s11t))))
        b_closed = -float(marcuvitz_thin_b(A_WR90, d, f))
        rows.append((d * 1e3, b_mm, b_closed, abs(b_mm / b_closed - 1)))
        assert b_mm < 0, ("an inductive iris must have a negative shunt "
                          "susceptance", d, b_mm)
    print("[marcuvitz] d(mm), B/Y0 mode-matching, B/Y0 closed form, rel:")
    for d_mm, b_mm, b_closed, rel in rows:
        print(f"  {d_mm:7.3f}  {b_mm:10.4f}  {b_closed:10.4f}  {rel:.4f}")
    assert max(r[3] for r in rows) < 0.15


# --------------------------------------------------------------------------- #
# The realized iris — read back from the product, before any solve.
# --------------------------------------------------------------------------- #
def _wall_runs(planes):
    """Contiguous runs of realized wall planes, as ``(first, last)`` pairs.

    A run's length in CELLS is ``last - first``: the distance between the two
    bounding walls of one body, which under the lattice ownership contract is
    the drawn extent.
    """
    planes = np.asarray(sorted(int(p) for p in planes), dtype=int)
    if planes.size == 0:
        return []
    cuts = np.where(np.diff(planes) != 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(planes, cuts)]


def _build_iris(d_phys, cells, *, glen=GLEN_M, iris_frac=IRIS_FRAC,
                t_cells=None):
    """One two-port iris run, drawn in physical absolute coordinates."""
    dx = A_WR90 / cells
    d_c = int(round(d_phys / dx))
    t_c = int(round(T_IRIS / dx)) if t_cells is None else int(t_cells)
    fin_c = (cells - d_c) // 2
    glen_c = int(round(glen / dx))
    p1 = int(round(0.040 / dx))
    p2 = glen_c - p1
    iris_lo = int(round(glen_c * iris_frac)) - t_c // 2
    sim = Simulation(
        freq_max=float(FREQS[-1]) * 1.1,
        domain=(glen_c * dx, A_WR90, B_WR90), dx=dx,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=int(np.ceil(CPML_DEPTH / dx)))
    big = 1.0   # fins are drawn PAST the guide walls; the rasterizer clips
    # Every face sits on a node plane: a PEC volume is sampled at cell
    # centres, so a node-plane face selects whole cells.
    x_lo, x_hi = iris_lo * dx, (iris_lo + t_c) * dx
    fin_hi_y = fin_c * dx                 # lower fin's inner wall
    fin_lo_y = (cells - fin_c) * dx       # upper fin's inner wall
    sim.add(Box((x_lo, -big, -big), (x_hi, fin_hi_y, big)), material="pec")
    sim.add(Box((x_lo, fin_lo_y, -big), (x_hi, big, big)), material="pec")
    for x, direction, name in ((p1 * dx, "+x", "P1"), (p2 * dx, "-x", "P2")):
        sim.add_waveguide_port(x, mode=(1, 0), mode_type="TE",
                               direction=direction, f0=10.3e9, bandwidth=0.41,
                               waveform="modulated_gaussian", freqs=FREQS,
                               name=name)
    return sim, {"dx": dx, "d_c": d_c, "t_c": t_c, "fin_c": fin_c,
                 "cells": cells, "x_lo": x_lo, "x_hi": x_hi,
                 "fin_hi_y": fin_hi_y, "fin_lo_y": fin_lo_y}


def assert_realized_iris(sim, geo):
    """Refuse to solve unless the lattice built the declared iris.

    Reads the realized PEC edge set (the product's own
    ``realized_pec_edge_masks``, through the shared test-side reader) and
    checks three things, each of which has failed on this structure before:
    the iris is ONE body standing walls at BOTH drawn faces, those walls are
    ``t_c`` cells apart, and the aperture between the fins is ONE contiguous
    opening ``d_c`` cells wide.
    """
    rz = realized(sim)
    grid = rz.grid
    assert grid.shape[1] == geo["cells"] + 1, (grid.shape[1], geo["cells"])

    x_runs = _wall_runs(rz.wall_planes(0))
    assert len(x_runs) == 1, ("the guide holds one iris; the lattice realized "
                              f"{len(x_runs)} bodies along x: {x_runs}")
    (x_wall_lo, x_wall_hi), = x_runs
    drawn_x = (node_index(grid, 0, geo["x_lo"]), node_index(grid, 0, geo["x_hi"]))
    assert (x_wall_lo, x_wall_hi) == drawn_x, (
        "realized iris walls are not the drawn faces", x_runs, drawn_x)
    assert x_wall_hi - x_wall_lo == geo["t_c"], (
        "realized iris thickness in cells != drawn",
        x_wall_hi - x_wall_lo, geo["t_c"])

    region = (slice(x_wall_lo, x_wall_lo + 1), slice(None), slice(None))
    y_runs = _wall_runs(rz.wall_planes(1, region=region))
    assert len(y_runs) == 2, (
        f"expected exactly two fins across the guide, got {len(y_runs)}: "
        f"{y_runs} — an aperture that is not one contiguous opening is the "
        "parasitic wall-slot defect")
    y_wall_lo, y_wall_hi = y_runs[0][1], y_runs[1][0]
    drawn_y = (node_index(grid, 1, geo["fin_hi_y"]),
               node_index(grid, 1, geo["fin_lo_y"]))
    assert (y_wall_lo, y_wall_hi) == drawn_y, (
        "realized aperture walls are not the drawn fin faces",
        (y_wall_lo, y_wall_hi), drawn_y)
    assert y_wall_hi - y_wall_lo == geo["d_c"], (
        "realized aperture in cells != drawn", y_wall_hi - y_wall_lo,
        geo["d_c"])
    return {"aperture_cells": int(y_wall_hi - y_wall_lo),
            "thickness_cells": int(x_wall_hi - x_wall_lo),
            "iris_wall_planes": [int(x_wall_lo), int(x_wall_hi)],
            "aperture_wall_planes": [int(y_wall_lo), int(y_wall_hi)]}


# --------------------------------------------------------------------------- #
# SLOW TIER — one FDTD run against the oracle.
# --------------------------------------------------------------------------- #
def test_a_split_aperture_is_refused_before_any_solve():
    """Negative arm of the realized-iris check: a third PEC strip drawn across
    the middle of the aperture (the parasitic wall-slot class - the aperture is
    no longer one contiguous opening) must be refused before a single step is
    taken. No FDTD run; only the assembled PEC edge set is read."""
    sim, geo = _build_iris(D_ARM, COARSE_CELLS)
    dx = A_WR90 / COARSE_CELLS
    y_mid = (COARSE_CELLS // 2) * dx
    sim.add(Box((geo["x_lo"], y_mid - dx, -1.0), (geo["x_hi"], y_mid + dx, 1.0)),
            material="pec")
    with pytest.raises(AssertionError, match="not one contiguous opening"):
        assert_realized_iris(sim, geo)


def test_coarse_mesh_s11_tracks_the_mode_matching_oracle():
    """|S11|(f) of the 12.192 mm iris on dx = a/30, against the oracle."""
    sim, geo = _build_iris(D_ARM, COARSE_CELLS)
    print(f"\n[geometry] dx = {geo['dx']*1e3:.4f} mm, aperture {geo['d_c']} "
          f"cells, thickness {geo['t_c']} cells, fins {geo['fin_c']} cells")
    print("[realized]", assert_realized_iris(sim, geo))

    report = sim.preflight()
    print(f"[preflight] {len(report)} finding(s)")
    for issue in report:
        print(f"  {issue}")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_waveguide_s_matrix(normalize="flux",
                                             num_periods=NUM_PERIODS)
    wall = time.time() - t0
    s = np.asarray(res.s_params)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    orc = oracle_s11(D_ARM)
    gap = np.abs(s11 - orc)
    gap_db = 20.0 * np.log10(s11 / orc)

    # The extractor's own passivity / finiteness self-check is part of this
    # record; the x64 dtype-truncation notices this build emits on every
    # astype are counted but not reprinted 27 times.
    notable = [w for w in caught if "jax_enable_x64" not in str(w.message)]
    print(f"[run] {wall:.1f} s, max column power "
          f"{float(np.max(s11**2 + s21**2)):.4f}, "
          f"{len(caught)} warning(s), {len(notable)} not dtype notices")
    for w in notable:
        print(f"  warning: {w.message}")
    print("  f(GHz)   |S11|_rfx  |S11|_oracle   gap      gap(dB)")
    for f, a, b, g, gd in zip(FREQS, s11, orc, gap, gap_db):
        print(f"  {f/1e9:6.2f}   {a:9.5f}  {b:11.5f}  {g:8.5f}  {gd:8.3f}")
    print(f"[gap] max {float(gap.max()):.5f} at "
          f"{float(FREQS[int(np.argmax(gap))])/1e9:.2f} GHz, "
          f"mean {float(gap.mean()):.5f}; max |gap(dB)| "
          f"{float(np.max(np.abs(gap_db))):.3f} dB; gate {GATE_ABS}")

    # Passivity first: an |S| above one on a lossless obstacle is an
    # extraction fault, and no accuracy statement survives it.
    assert float(np.max(s11 ** 2 + s21 ** 2)) <= 1.02, "column power > 1.02"
    assert float(gap.max()) <= GATE_ABS, (
        f"max |S11 - oracle| = {gap.max():.5f} exceeds the gate {GATE_ABS} "
        f"derived from the measured envelope {MEASURED_MAX_GAP_ABS}")
