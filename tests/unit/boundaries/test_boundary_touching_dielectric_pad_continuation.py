"""A dielectric that ends at the absorber seam must be solved WITH its pad.

The CPML pad material extension exists "so that guided modes in dielectric
waveguides see an impedance-matched absorber". On the staircase lane
``_assemble_materials`` delivers that by replicating the interior-edge slice of
the material arrays outward. The SMOOTHED lane rebuilds the update permittivity
from the declared geometry, and for the life of the feature it rebuilt it
WITHOUT any pad step: a guide touching the domain edge was solved with vacuum
in its own absorber and a Kottke half-cell at the seam, i.e. terminated by an
end facet the guided mode reflects off.

Why the feature's existing 11 tests did not see it, measured rather than
assumed: nine assert on the array ``_assemble_materials`` RETURNS, which is not
the array the run solves, and the two that run the solver both pass
``subpixel_smoothing=False``. Flipping those two to ``True`` does not reach it
either -- both fixtures carry Lorentz poles and the anisotropic branch is gated
behind ``debye is None and lorentz is None``, so the smoothed array is never
consulted. The gate has to be a STATIC dielectric, solved, with the array the
update reads dumped out of the run.

So this file measures four things on one boundary-touching slab guide, under
both absorber families:

1. the pad column of the array the E update consumes carries the guide's
   permittivity, with no half-cell stranded at the seam (the shape a naive
   array replication produces, and the shape sourcing one column inward
   produces, are both asserted against);
2. the run stays finite over a long record -- the continuation puts a
   dielectric inside a CPML pad, which is the configuration #627b found
   divergent for a high-Q pole and which diverged here too until the psi
   coefficient and the Yee update were made to read the same epsilon;
3. the ring-down settling witness resolves;
4. the guided mode's round trip, ``|B/A|`` from the same two-wave estimator
   crossval 03 uses, is small -- 0.53 with the facet, <= 0.03 without it.

The rig is crossval 01's guide (eps_r = 12 slab, one lattice constant wide,
spanning the full x extent so BOTH x faces touch) at a shorter record.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

import rfx
from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec

_REPO = pathlib.Path(__file__).resolve().parents[3]

C0 = 2.998e8

# --- the guide, from crossval 01's Run 1 rig ---------------------------------
A = 1.0e-6
EPS_WG = 12.0
W_WG = 1.0 * A
DX = A / 10

#: Absorber depth for the round-trip bar. Forty, not twenty, and the reason is
#: measured on this rig rather than chosen: with the continuation in place
#: ``|B/A|`` under CPML reads 0.0509 / 0.0061 / 0.0002 at 20 / 40 / 60 cells
#: (UPML 0.0291 at 20). What remains at 20 is the absorber's OWN residual
#: reflection for an eps_r = 12 guided mode -- it falls by ~8x per depth
#: doubling, and it does not move with transverse clearance (sy 8a vs 16a:
#: 0.0510 vs 0.0509), source position, fit window or record length (2x steps:
#: 0.0509). A seam facet behaves the opposite way: #831 measured 0.53 / 0.59 /
#: 0.62 over the same depths, WORSE as the absorber deepens, because the first
#: pad cell's conductivity falls as N^-3 and loads the facet less. Reading the
#: 0.03 bar at a depth where the absorber is not the limit is what makes it a
#: measurement of the seam. The depth trend itself is asserted separately.
CPML_LAYERS = 40

#: The shallower depth the trend test pairs with ``CPML_LAYERS``.
SHALLOW_LAYERS = 20

FCEN = 0.15 * C0 / A
FWIDTH = 0.1 * C0 / A

SX = 16.0 * A
SY = 8.0 * A
WG_Y = SY / 2
SRC_X = 3.0 * A

#: Fit window for the two-wave estimator: inside the guide, clear of the
#: source's near field and of both seams, so what it reads is the travelling
#: pair and not a source artefact.
FIT_LO, FIT_HI = 6.0 * A, 13.0 * A

#: 350 a/c0 of record. Long enough that the pulse has crossed the guide, been
#: absorbed, and any round trip off a seam has come back and been recorded --
#: the standing wave a facet makes is a steady-state pattern, not a transient.
RECORD_T = 350.0 * A / C0


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _n_steps() -> int:
    dt = DX / (C0 * np.sqrt(2)) * 0.99
    return int(RECORD_T / dt) + 200


def _build(boundary: str, layers: int = CPML_LAYERS) -> Simulation:
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(SX, SY, DX), dx=DX,
                     boundary=BoundarySpec.uniform(boundary),
                     cpml_layers=layers, mode="2d_tmz")
    sim.add_material("wg", eps_r=EPS_WG)
    # Spans the FULL x extent: both x faces sit exactly on the interior/pad
    # seam. Nothing is declared inside the absorber, so the geometry-in-pad
    # advisory does not fire and this is the configuration a user writes.
    sim.add(Box((0, WG_Y - W_WG / 2, 0), (SX, WG_Y + W_WG / 2, DX)),
            material="wg")
    for i in range(10):
        y = WG_Y - W_WG / 2 + (i + 0.5) * W_WG / 10
        sim.add_source(position=(SRC_X, y, 0), component="ez",
                       waveform=GaussianPulse(f0=FCEN,
                                              bandwidth=FWIDTH / FCEN,
                                              amplitude=1.0 / 10))
    # A point probe so the ring-down settling witness has a record to score;
    # flux monitors and DFT planes do not feed it.
    sim.add_probe(position=((FIT_LO + FIT_HI) / 2, WG_Y, 0), component="ez")
    return sim


def _run_capturing_update_eps(sim, n_steps, monkeypatch, **run_kw):
    """Run, and hand back the exact permittivity array the E update consumed.

    Captured by wrapping ``rfx.geometry.smoothing.compute_smoothed_eps`` on the
    module the runner imports it from at CALL time -- not a re-export -- so
    what comes back is the solver's own array rather than a rebuild that could
    silently diverge from it.
    """
    from rfx.geometry import smoothing as _smoothing

    captured = {}
    real = _smoothing.compute_smoothed_eps

    def _spy(grid, shapes, background_eps=1.0):
        out = real(grid, shapes, background_eps=background_eps)
        captured["aniso_eps"] = out
        captured["shapes"] = shapes
        return out

    monkeypatch.setattr(_smoothing, "compute_smoothed_eps", _spy)
    result = sim.run(n_steps=n_steps, subpixel_smoothing=True, **run_kw)
    assert "aniso_eps" in captured, (
        "the run never called compute_smoothed_eps -- this fixture is not on "
        "the smoothed lane and proves nothing about it")
    return result, captured


def _centre_row(aniso_eps, grid) -> np.ndarray:
    """Ez permittivity along the guide centre line, pads included."""
    _, _, eps_ez = aniso_eps
    jy = int(round(WG_Y / DX)) + int(grid.pad_y_lo)
    return np.asarray(eps_ez)[:, jy, 0].astype(float)


@pytest.mark.parametrize("boundary", ["cpml", "upml"])
def test_solved_pad_column_carries_the_guide_material(boundary, monkeypatch):
    """The pad is the guide, and the seam cell is not a half-cell.

    Three wrong answers are named explicitly, because each is a plausible
    implementation and only one of them is visible in a headline count:
    vacuum (no continuation at all), the Kottke half-cell 6.5 (replicating the
    smoothed array's interior-edge column), and a half-cell stranded one column
    inside the absorber (sourcing the replication one column inward, which
    gets the pad right and leaves a one-cell film between the guide and its
    own matched pad).
    """
    sim = _build(boundary)
    result, cap = _run_capturing_update_eps(sim, 10, monkeypatch,
                                            skip_preflight=True)
    grid = result.grid
    row = _centre_row(cap["aniso_eps"], grid)
    plx, phx = int(grid.pad_x_lo), int(grid.pad_x_hi)
    nx = row.size
    assert plx > 0 and phx > 0, "this fixture needs absorber pads on both x faces"

    half_cell = 0.5 * (EPS_WG + 1.0)
    for label, cells in (("lo pad", row[:plx]), ("hi pad", row[nx - phx:])):
        assert np.allclose(cells, EPS_WG, rtol=1e-6), (
            f"{boundary}: the {label} the E update reads is {cells[:4]}, not "
            f"the guide's eps_r = {EPS_WG}. Vacuum there terminates the guide "
            f"with a facet at the seam; {half_cell} there is the Kottke "
            "half-cell replicated outward")

    seam_lo = row[plx - 1:plx + 2]
    seam_hi = row[nx - phx - 2:nx - phx + 1]
    for label, seam in (("lo", seam_lo), ("hi", seam_hi)):
        assert np.allclose(seam, EPS_WG, rtol=1e-6), (
            f"{boundary}: the {label} seam reads {seam}; a cell at "
            f"{half_cell} here is a one-cell film between the guide and its "
            "own absorber (the #655 failure mode)")

    # The continuation must not flood the cladding: it continues the declared
    # shape, it does not change what the shape is.
    jy_clad = int(grid.pad_y_lo) + 1
    _, _, eps_ez = cap["aniso_eps"]
    clad = np.asarray(eps_ez)[:, jy_clad, 0].astype(float)
    assert np.allclose(clad, 1.0, rtol=1e-6), (
        f"{boundary}: a cladding row reads {clad.min()}..{clad.max()}, so the "
        "continuation widened the guide instead of extending it")


def _measure_round_trip(boundary, layers, monkeypatch) -> dict:
    """One long record: stability, settling witness, and the two-wave fit."""
    cmp_mod = _load("slab_te_dispersion",
                    "validation/crossval/comparators/slab_te_dispersion.py")

    freqs = np.linspace(0.10 * C0 / A, 0.20 * C0 / A, 40)
    sim = _build(boundary, layers)
    sim.add_dft_plane_probe(axis="y", coordinate=WG_Y, component="ez",
                            freqs=freqs, name="guide_axis_ez")
    n_steps = _n_steps()
    assert n_steps >= 5000, f"record is only {n_steps} steps"
    result, _ = _run_capturing_update_eps(sim, n_steps, monkeypatch,
                                          skip_preflight=True)

    ts = np.asarray(result.time_series)
    line = np.asarray(result.dft_planes["guide_axis_ez"].accumulator)[:, :, 0]
    x_full = (np.arange(line.shape[1]) - result.grid.pad_x_lo) * DX
    win = (x_full >= FIT_LO - 1e-12) & (x_full <= FIT_HI + 1e-12)
    fits = cmp_mod.measure_neff_two_wave(
        line[:, win], x_full[win], freqs,
        c0=C0, eps_core=EPS_WG, eps_clad=1.0)
    fit = fits[int(np.argmin(np.abs(freqs - FCEN)))]
    return dict(n_steps=n_steps,
                n_nonfinite=int(np.count_nonzero(~np.isfinite(ts))),
                peak=float(np.max(np.abs(ts))),
                settling=(None if result.settling_db is None
                          else float(result.settling_db)),
                b_over_a=float(fit.b_over_a),
                rel_residual=float(fit.rel_residual),
                n_eff=float(fit.n_eff))


@pytest.mark.parametrize("boundary", ["cpml", "upml"])
def test_boundary_touching_guide_is_stable_and_has_no_round_trip(
        boundary, monkeypatch):
    """Long record: finite, settled, and no standing wave off the seam.

    Stability is asserted first and is not a formality. A dielectric carried
    into a CPML pad is the configuration ``extend_cpml_pad_materials``'s
    docstring records as divergent for a high-Q pole (#627b), and the same
    continuation on a static guide diverged at step ~400 of 25000 while the
    psi correction and the Yee update were integrating different
    permittivities. A permittivity that is right in a simulation that blows up
    is worse than the facet.
    """
    m = _measure_round_trip(boundary, CPML_LAYERS, monkeypatch)

    assert m["n_nonfinite"] == 0, (
        f"{boundary}: {m['n_nonfinite']} non-finite samples in a "
        f"{m['n_steps']}-step record -- the run diverged")
    assert m["peak"] < 1e3, (
        f"{boundary}: peak |Ez| {m['peak']:.3e} on a unit-amplitude source; "
        "finite is not the same as bounded")

    assert m["settling"] is not None, "no settling witness -- add a point probe"
    assert m["settling"] <= -40.0, (
        f"{boundary}: settling witness {m['settling']:.1f} dB > -40 dB, so "
        "the record ends while the guide is still ringing and every DFT "
        "number below is read off an unsettled record")

    assert m["rel_residual"] < 0.2, (
        f"{boundary}: two-wave residual {m['rel_residual']:.3f} -- the "
        "estimator's own premise failed, so its |B/A| is not readable")
    assert m["b_over_a"] <= 0.03, (
        f"{boundary}: |B/A| = {m['b_over_a']:.4f} at the carrier bin "
        f"(n_eff {m['n_eff']:.4f}, residual {m['rel_residual']:.4f}). The "
        "guide is uniform and both ends are absorbers, so a round trip this "
        "size is a reflection off the interior/pad seam, not physics")


def test_round_trip_falls_with_absorber_depth(monkeypatch):
    """The #831 signature, inverted.

    This is the discriminator, not the bar above: an absorber's own residual
    reflection falls as it deepens, and a facet at the interior/pad seam
    RISES, because the first pad cell's conductivity falls as N^-3 and loads
    the facet less. #831 measured 0.53 / 0.59 / 0.62 at 20 / 40 / 60 cells on
    a guide that should have shown the opposite. If a future change puts the
    facet back, this test reds even if someone has moved the bar above.
    """
    shallow = _measure_round_trip("cpml", SHALLOW_LAYERS, monkeypatch)
    deep = _measure_round_trip("cpml", CPML_LAYERS, monkeypatch)
    assert shallow["b_over_a"] < 0.2, (
        f"|B/A| = {shallow['b_over_a']:.4f} at {SHALLOW_LAYERS} cells is the "
        "order the vacuum facet produced (0.53), not the order an absorber's "
        "residual reflection has")
    assert deep["b_over_a"] < shallow["b_over_a"], (
        f"|B/A| grew with absorber depth: {shallow['b_over_a']:.4f} at "
        f"{SHALLOW_LAYERS} cells -> {deep['b_over_a']:.4f} at {CPML_LAYERS}. "
        "That is the #831 signature of a reflector at the seam whose loading "
        "weakens as the pad deepens, not of the absorber itself")


def test_continuation_is_an_identity_for_an_interior_dielectric():
    """A shape clear of every pad comes back as the SAME object.

    This is the bit-identity guarantee stated as a unit rather than inferred
    from a field hash: the smoothing is handed the identical list, so it
    cannot produce a different array.
    """
    from rfx.geometry.smoothing import smoothed_shape_pairs

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(SX, SY, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=CPML_LAYERS, mode="2d_tmz")
    sim.add_material("wg", eps_r=EPS_WG)
    interior = Box((4 * A, WG_Y - W_WG / 2, 0), (12 * A, WG_Y + W_WG / 2, DX))
    sim.add(interior, material="wg")
    grid = sim._build_grid()
    pairs, unextendable = smoothed_shape_pairs(sim, grid)
    assert unextendable == []
    assert pairs[0][0] is interior, (
        "an interior dielectric was rewritten by the pad continuation")


@pytest.mark.parametrize("gap_cells", [0.3, 1.0])
def test_a_face_drawn_inside_the_boundary_is_not_moved_to_it(gap_cells):
    """The reach rule is "the declared face reaches the boundary", not "nearly".

    A face drawn 0.3 cells inside the domain edge IS 0.3 cells inside it, and
    resolving that is what subpixel smoothing is for. A half-cell tolerance
    here would quietly move such a structure onto the boundary and continue it,
    which is the class of silent geometry change the Box docstring, #802 and
    #325 all exist about. Only the numerical slack that keeps one f64 ulp from
    deciding the question is allowed.

    The consequence is a one-cell window on the hi face where this lane and the
    staircase lane disagree — the staircase rule is a cell-centre test and this
    one is sub-cell, so no tolerance makes them agree everywhere. Pinned here
    so the disagreement is a recorded choice rather than a discovery.
    """
    from rfx.geometry.smoothing import smoothed_shape_pairs

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(SX, SY, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=CPML_LAYERS, mode="2d_tmz")
    sim.add_material("wg", eps_r=EPS_WG)
    inset = Box((gap_cells * DX, WG_Y - W_WG / 2, 0),
                (SX - gap_cells * DX, WG_Y + W_WG / 2, DX))
    sim.add(inset, material="wg")
    grid = sim._build_grid()
    pairs, _ = smoothed_shape_pairs(sim, grid)
    assert pairs[0][0] is inset, (
        f"a face {gap_cells} cells inside the boundary was moved to it")


def test_a_face_on_the_boundary_survives_the_float_route():
    """``corner_hi = SX`` and the node builder's last interior node are the
    same point by two different arithmetic routes, and the reach test must not
    be decided by which one rounds last. This is the knife-edge the Box
    docstring names; the rule carries 1e-6 cells of slack for it and no more.
    """
    from rfx.geometry.smoothing import smoothed_shape_pairs

    sim = _build("cpml")
    grid = sim._build_grid()
    pairs, _ = smoothed_shape_pairs(sim, grid)
    box = pairs[0][0]
    assert box is not sim._geometry[0].shape, (
        "the boundary-touching guide was not continued — the reach test lost "
        "to a float route")
    assert box.corner_lo[0] < 0.0 and box.corner_hi[0] > SX


def test_continuation_is_an_identity_without_absorbing_pads():
    """PEC walls have no pad to continue into, so nothing is rewritten."""
    from rfx.geometry.smoothing import smoothed_shape_pairs

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(SX, SY, DX), dx=DX,
                     boundary="pec", mode="2d_tmz")
    sim.add_material("wg", eps_r=EPS_WG)
    touching = Box((0, WG_Y - W_WG / 2, 0), (SX, WG_Y + W_WG / 2, DX))
    sim.add(touching, material="wg")
    grid = sim._build_grid()
    pairs, unextendable = smoothed_shape_pairs(sim, grid)
    assert unextendable == []
    assert pairs[0][0] is touching


def test_dispersive_material_is_not_continued_into_the_pad():
    """A pole-carrying shape keeps the facet, on purpose.

    ``extend_cpml_pad_materials`` does not extend dispersion-pole masks
    (#627b: a high-Q pole in the pad turns a stable edge-touching run into a
    divergent one, with no NaN to catch it), and #808 measured that promoting
    such a column's STATICS alone moves a committed Debye recovery past its
    gate -- the promoted material, eps_inf without its poles, matches no
    declared model. The smoothed lane inherits both rules rather than
    inventing a third answer.
    """
    from rfx.geometry.smoothing import smoothed_shape_pairs

    sim = Simulation(freq_max=0.25 * C0 / A, domain=(SX, SY, DX), dx=DX,
                     boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=CPML_LAYERS, mode="2d_tmz")
    w0 = 2.0 * np.pi * FCEN
    sim.add_material("lossy", eps_r=4.0,
                     lorentz_poles=[rfx.lorentz_pole(delta_eps=1.0,
                                                     omega_0=w0,
                                                     delta=w0 / 60)])
    touching = Box((0, WG_Y - W_WG / 2, 0), (SX, WG_Y + W_WG / 2, DX))
    sim.add(touching, material="lossy")
    grid = sim._build_grid()
    pairs, _ = smoothed_shape_pairs(sim, grid)
    assert pairs[0][0] is touching, (
        "a Lorentz-pole material was continued into the absorber pad; "
        "see #627b in extend_cpml_pad_materials' docstring")
