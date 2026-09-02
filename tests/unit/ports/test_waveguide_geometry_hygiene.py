"""Geometry / absorber hygiene for waveguide S-parameter setups.

Two independent hygiene defects found while reviewing PR #480 (WR-90 single
inductive iris), both filed as general issues because the iris lane only
exposed them:

* **#493** — ``Box``'s volume branch is half-open ``[lo, hi)`` over NODE
  coordinates, so a PEC obstacle drawn to its nominal physical dimension
  rasterizes short at its ``hi`` face. The tests below pin that arithmetic
  so the convention is executable documentation rather than folklore.
  Since the exact-coordinate fix (#802) production nodes ARE the float64
  construction, drawing to nominal costs exactly ONE cell at every
  aperture, and the old per-aperture one-or-two-cell scatter is history:
  it was float32 double-rounding deciding whether the hi fin's lo corner
  captured its node. They still derive node coordinates from a real
  ``Grid`` so they cannot drift from production. They are
  CHARACTERIZATION tests: the convention is deliberate and other paths
  depend on it, so a change here is a deliberate behaviour change and must
  be reviewed as one, not silenced.

* **#494** — ``compute_waveguide_s_matrix``'s own docstring requires an
  absorber ``>= ~0.5 * lambda_g`` but nothing checked it on the plain
  two-port path, so a 0.30-``lambda_g`` stack shipped in a gated revision
  and the absorber, not discretization, set the accuracy envelope. Each
  advisory test comes in a firing / non-firing pair.
"""

from __future__ import annotations

import warnings

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.api._sparams import _warn_thin_absorber_vs_guide_wavelength
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box, _grid_coords


ADVISORY_KEY = "far-port discipline"


# --------------------------------------------------------------------------- #
# #493 — half-open node rasterization of PEC obstacles
# --------------------------------------------------------------------------- #
A_WR90 = 22.86e-3
B_WR90 = 10.16e-3
_ZERO = np.array([0.0])
_COORD_CACHE: dict = {}


def _real_node_coords(cells: int):
    """y-node coordinates from a REAL ``Grid`` — the production path.

    Since the exact-coordinate fix (#802) production nodes are host
    float64 ``(i - pad) * dx`` — bit-identical to the f64 construction and
    independent of ``jax_enable_x64`` (pinned by the equality test below).
    Historically ``_grid_coords`` computed them in float32, double-rounded
    ``f32(f32(i) * f32(dx))``, disagreeing with an f64 construction on 30
    of 31 nodes by up to 1.12e-9 m — enough to flip a whole cell of metal
    per aperture. Deriving coordinates from a real grid is still what
    keeps these tests from drifting away from the code they document.
    """
    if cells not in _COORD_CACHE:
        dx = A_WR90 / cells
        sim = Simulation(
            freq_max=14e9, domain=(0.05, A_WR90, B_WR90), dx=dx,
            boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                  y=Boundary(lo="pec", hi="pec"),
                                  z=Boundary(lo="pec", hi="pec")),
            cpml_layers=10)
        y = np.asarray(_grid_coords(sim._build_grid())[1])
        _COORD_CACHE[cells] = (y, dx)
    return _COORD_CACHE[cells]


def _occupied(hi_face: float, coords: np.ndarray) -> np.ndarray:
    """Indices of nodes occupied by a box spanning up to ``hi_face`` in y."""
    m = Box((-1.0, -1.0, -1.0), (1.0, float(hi_face), 1.0)).mask_on_coords(
        _ZERO, coords, _ZERO)
    return np.nonzero(np.asarray(m)[0, :, 0])[0]


def _fin_pair(hi_face: float, coords: np.ndarray, dx: float):
    """Electrical aperture between two facing fins.

    Returns ``(aperture_cells, free_node_indices)``. The aperture is the span
    between the innermost OCCUPIED node planes, i.e. between the innermost
    planes where tangential E is zeroed. That is the same convention under
    which the guide itself measures ``a``: with PEC on the outermost node
    planes 0 and ``cells``, the width is ``cells * dx`` = 22.86 mm exactly,
    and the free-node count is ``width/dx - 1``.
    """
    lo_fin = Box((-1.0, -1.0, -1.0), (1.0, float(hi_face), 1.0))
    hi_fin = Box((-1.0, A_WR90 - float(hi_face), -1.0), (1.0, 1.0, 1.0))
    metal = (np.asarray(lo_fin.mask_on_coords(_ZERO, coords, _ZERO))[0, :, 0]
             | np.asarray(hi_fin.mask_on_coords(_ZERO, coords, _ZERO))[0, :, 0])
    free = np.nonzero(~metal)[0]
    return int((free[-1] + 1) - (free[0] - 1)), free


def _iris(cells: int, d_phys: float):
    y, dx = _real_node_coords(cells)
    d_c = int(round(d_phys / dx))
    return y, dx, d_c, (cells - d_c) // 2


@pytest.mark.parametrize("cells", [30, 60])
def test_guide_width_fixes_the_zeroed_plane_convention(cells):
    """The convention under which the guide itself measures ``a`` exactly.

    Anchors every aperture number below: distance between the bounding
    zeroed node planes, NOT the span of open nodes (which would call WR-90
    22.098 mm at a/30).
    """
    y, dx = _real_node_coords(cells)
    assert len(y) == cells + 1
    assert cells * dx == pytest.approx(A_WR90, rel=1e-9)
    assert (cells - 1) == (cells * dx) / dx - 1


@pytest.mark.parametrize("cells", [30, 60])
def test_box_volume_branch_excludes_the_hi_node_plane(cells):
    """Half-open ``[lo, hi)``: the ``hi`` face contributes no cell.

    ``hi`` is taken from the realized node value so the comparison is an
    exact equality and the result cannot depend on rounding — the rule
    itself, isolated from the float32 effects tested separately below.
    """
    y, _ = _real_node_coords(cells)
    for k in (8, 12):
        occ = _occupied(float(y[k]), y)
        assert occ[-1] == k - 1, f"node {k} must be excluded"
        assert len(occ) == k


def test_production_node_coords_equal_the_f64_construction():
    """Inverted at the exact-coordinate fix (#802) — deliberately.

    This test used to pin the OPPOSITE claim (production f32 nodes differ
    from an f64 ``arange`` on almost every node, ~1e-9 m, enough to move
    whole cells of metal). That was the defect, not a contract: realized
    geometry depended on ``jax_enable_x64`` and on which of three
    coordinate constructions a lane used. Production nodes are now host
    float64 and bit-identical to the f64 construction, so the equality IS
    the contract.
    """
    for cells in (30, 60):
        y, dx = _real_node_coords(cells)
        y_f64 = np.arange(len(y)) * dx
        assert y.dtype == np.float64
        assert np.array_equal(y, y_f64), (
            "production node coordinates must equal the exact f64 "
            "construction (#802)")


# Measured on the production coordinate path. Re-pinned at the
# exact-coordinate fix (#802): the old table carried 2 at 12.192 mm (both
# mesh rungs) — a float32 double-rounding artifact that made the hi fin's
# lo corner lose its node. With exact float64 nodes the nominal drawing
# costs exactly ONE cell everywhere (the half-open convention's own
# hi-face shortfall), and the excess IS predictable now. Still pinned as a
# characterization table so any change in rounding behaviour is visible.
_NOMINAL_EXCESS = {
    (30, 7.620): 1, (30, 12.192): 1, (30, 18.288): 1,
    (60, 7.620): 1, (60, 12.192): 1, (60, 18.288): 1,
}


@pytest.mark.parametrize("cells,d_mm", sorted(_NOMINAL_EXCESS))
def test_fins_drawn_to_nominal_aperture_are_one_cell_too_wide(
        cells, d_mm):
    """#493's mechanism, with the measured per-config value.

    Drawing to the nominal opening never yields the nominal ELECTRICAL
    opening: exactly one cell from the half-open convention itself. (The
    former second cell — the hi fin's lo corner failing to capture its
    node — was float32 rounding, gone at the exact-coordinate fix #802.)
    """
    d_phys = d_mm * 1e-3
    y, dx, d_c, fin_c = _iris(cells, d_phys)
    aperture_cells, _ = _fin_pair(fin_c * dx, y, dx)

    excess = aperture_cells - d_c
    assert excess == _NOMINAL_EXCESS[(cells, d_mm)]
    assert excess == 1, "the drawn-to-nominal defect is exactly 1 cell (#802)"
    assert aperture_cells * dx > d_phys, "must never be the nominal opening"


@pytest.mark.parametrize("cells,d_mm", sorted(_NOMINAL_EXCESS))
def test_which_interior_face_retreats_decides_the_symmetry(cells, d_mm):
    """The two interior faces of a facing pair are different corner types.

    The lo fin's interior face is a ``hi`` corner, which half-openness ALWAYS
    drops — so that fin always retreats one cell. The hi fin's interior face
    is a ``lo`` corner, which ``coords >= lo`` KEEPS, unless float32 rounding
    puts the node just below it. Hence excess 1 cell means only the lo fin
    retreated and the opening is asymmetric (centre ``dx/2`` low), while
    excess 2 cells means both retreated, the retreats cancel, and the opening
    is symmetric.

    Pinned because an unconditional "the obstacle is asymmetric" claim would
    send the next reader hunting a bug that is not there at the apertures
    where it is centred (PR #495 review, item 2).
    """
    d_phys = d_mm * 1e-3
    y, dx, d_c, fin_c = _iris(cells, d_phys)
    lo_metal = _occupied(fin_c * dx, y)
    hi_fin = Box((-1.0, A_WR90 - fin_c * dx, -1.0), (1.0, 1.0, 1.0))
    hi_metal = np.nonzero(
        np.asarray(hi_fin.mask_on_coords(_ZERO, y, _ZERO))[0, :, 0])[0]
    aperture_cells, free = _fin_pair(fin_c * dx, y, dx)
    excess = aperture_cells - d_c

    # The lo fin's interior face is a hi corner: it always loses exactly one.
    assert lo_metal[-1] == fin_c - 1
    # The hi fin's interior face is a lo corner: it loses one only under (2).
    hi_retreat = int(hi_metal[0] - (cells - fin_c))
    assert hi_retreat == excess - 1, (hi_retreat, excess)

    centre_offset = 0.5 * (free[0] + free[-1]) - cells / 2
    if excess == 1:
        assert centre_offset == pytest.approx(-0.5), "one retreat => asymmetric"
    else:
        assert excess == 2
        assert centre_offset == pytest.approx(0.0), "two retreats => symmetric"


@pytest.mark.parametrize("cells,d_mm", sorted(_NOMINAL_EXCESS))
def test_midpoint_recipe_is_exact_at_even_parity(cells, d_mm):
    """The documented recipe, and the non-firing control for the test above.

    Interior faces on cell midpoints sit half a cell from either edge, so the
    footprint is rounding-independent. Every case-18 configuration has
    ``(cells - d_c)`` even and lands on the nominal aperture exactly.
    """
    d_phys = d_mm * 1e-3
    y, dx, d_c, fin_c = _iris(cells, d_phys)
    assert (cells - d_c) % 2 == 0, "case-18 configs are even-parity by design"

    aperture_cells, free = _fin_pair((fin_c + 0.5) * dx, y, dx)

    assert aperture_cells == d_c
    assert aperture_cells * dx == pytest.approx(d_phys, rel=1e-6)
    assert len(free) == d_c - 1, "free-node count == aperture/dx - 1"
    assert 0.5 * (free[0] + free[-1]) == pytest.approx(cells / 2), "centred"


@pytest.mark.parametrize("cells", [30, 60])
def test_midpoint_recipe_costs_a_cell_at_odd_parity(cells):
    """A representability limit, not a rasterization defect.

    ``fin_c = (cells - d_c)//2`` truncates, so when ``(cells - d_c)`` is odd
    a SYMMETRIC iris of that aperture cannot be placed on the node grid at
    all and the opening is one cell wide regardless of how it is drawn. Keep
    ``(cells - d_c)`` even, i.e. the fin depth an exact number of cells.
    """
    y, dx = _real_node_coords(cells)
    d_c = 3
    assert (cells - d_c) % 2 == 1
    fin_c = (cells - d_c) // 2

    aperture_cells, _ = _fin_pair((fin_c + 0.5) * dx, y, dx)

    assert aperture_cells == d_c + 1


@pytest.mark.parametrize("cells,d_mm", sorted(_NOMINAL_EXCESS))
def test_half_cell_inward_offset_opens_two_cells_too_wide(cells, d_mm):
    """Offsetting interior faces the WRONG way gives d + 2*dx everywhere.

    Unlike the nominal drawing this one is uniform across the table: the
    corner sits half a cell from either edge, so it is rounding-independent
    and the two cells are structural.
    """
    d_phys = d_mm * 1e-3
    y, dx, d_c, fin_c = _iris(cells, d_phys)

    aperture_cells, _ = _fin_pair((fin_c - 0.5) * dx, y, dx)

    assert aperture_cells == d_c + 2


def test_node_plane_corner_is_a_single_float32_ulp_knife_edge():
    """Why the recipe says "cell midpoint" and not "on the node plane".

    A corner one ULP below a node plane loses that node — half-open
    ``[lo, hi)`` is exact, in whatever precision the caller's corner
    arithmetic ran. Since #802 the comparison itself is float64, so the
    knife edge is narrower (one f64 ulp, not the old f32 one) but not
    gone: a corner computed as ``a - n*dx`` can still rasterize
    differently from an algebraically identical ``m*dx``. Midpoint corners
    sit half a cell from either edge and are immune. (The perturbations
    below use f32 ULPs — generous by 2^29 against an f64 comparison, and
    the footprint still flips, which is the point.)
    """
    y, dx = _real_node_coords(30)
    node = np.float32(y[8])
    ulp = float(np.nextafter(node, np.float32(1)) - node)
    assert 0 < ulp < 1e-9

    below = _occupied(float(np.nextafter(node, np.float32(0))), y)
    above = _occupied(float(np.nextafter(node, np.float32(1))), y)

    assert below[-1] == 7
    assert above[-1] == 8, "one float32 ULP must flip the footprint by a cell"
    mid = float(y[8]) + 0.5 * dx
    assert _occupied(mid, y)[-1] == _occupied(mid + 8 * ulp, y)[-1]


def test_drawn_vs_realized_gap_is_ambiguous_between_correct_and_defective():
    """Why #493 ships as documentation and not as a rasterized-vs-drawn check.

    Issue #493 floated an advisory that fires when a PEC volume's rasterized
    opening differs from its drawn opening by >= 1 cell. The reading is
    AMBIGUOUS: at d = 7.620 mm the defective nominal drawing and the correct
    midpoint recipe both read +1 cell, so +1 cell cannot support a defect
    conclusion — and +1 cell is the common case. The defect is
    ``realized != INTENDED``, and the intended dimension is never
    communicated to the simulator, so it is not recoverable from geometry.
    """
    def drawn_vs_realized(cells, d_mm, offset):
        d_phys = d_mm * 1e-3
        y, dx, d_c, fin_c = _iris(cells, d_phys)
        hi_face = (fin_c + offset) * dx
        aperture_cells, _ = _fin_pair(hi_face, y, dx)
        drawn = (A_WR90 - hi_face) - hi_face
        return (aperture_cells * dx - drawn) / dx

    # The collision that kills the predicate: same reading, opposite verdicts.
    assert drawn_vs_realized(30, 7.620, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert drawn_vs_realized(30, 7.620, 0.5) == pytest.approx(1.0, abs=1e-6)
    # Since the exact-coordinate fix (#802) the collision is TOTAL: the
    # defective nominal drawing and the correct midpoint recipe read +1
    # cell at every aperture (the old +2 at 12.192 mm was float32
    # rounding), so the predicate cannot separate them anywhere.
    assert drawn_vs_realized(30, 12.192, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert drawn_vs_realized(30, 12.192, 0.5) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# #494 — thin-absorber advisory on the plain two-port path
# --------------------------------------------------------------------------- #
_FREQS = np.linspace(4.5e9, 8.0e9, 4)
_F0 = float(_FREQS.mean())


def _two_port(cpml_layers, *, boundary="cpml", freqs=_FREQS, dx=0.004):
    sim = Simulation(
        freq_max=float(freqs[-1]),
        domain=(0.12, 0.04, 0.02),
        dx=dx,
        boundary=boundary,
        cpml_layers=cpml_layers,
    )
    for x, direction in ((0.02, "+x"), (0.10, "-x")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(freqs), f0=float(np.mean(freqs)), bandwidth=0.6,
        )
    return sim


def _advisories(sim, freqs=_FREQS):
    """Run the predicate on real port configs without paying for FDTD."""
    grid = sim._build_grid()
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(freqs), 2000)
            for e in sim._waveguide_ports]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_thin_absorber_vs_guide_wavelength(
            grid, cfgs, freqs, sim._cpml_layers, sim._boundary_spec)
    return [str(w.message) for w in caught if ADVISORY_KEY in str(w.message)]


def test_thin_absorber_advisory_fires_end_to_end_on_the_two_port_path():
    """FIRING, end-to-end: the advisory must reach the functional path.

    The functional entry points run no ``sim.preflight()``, which is why the
    check lives in the method. A predicate unit test alone would not prove
    it is wired, so this drives the real extraction.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _two_port(10).compute_waveguide_s_matrix(num_periods=6.0)
    hits = [str(w.message) for w in caught if ADVISORY_KEY in str(w.message)]

    assert len(hits) == 1, [str(w.message) for w in caught]
    msg = hits[0]
    # Both numbers must be quoted: what you have and what is required.
    assert "40.0 mm" in msg, msg          # 10 cells * 4 mm
    assert "against a required" in msg
    assert "lowest measured frequency 4.500 GHz" in msg
    assert "cpml_layers" in msg           # actionable remedy


def test_thin_absorber_advisory_silent_end_to_end_when_absorber_is_thick():
    """NON-FIRING control: same band and geometry, adequate absorber.

    16 cells * 4 mm = 64 mm against a 50.8 mm requirement at 4.5 GHz.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _two_port(16).compute_waveguide_s_matrix(num_periods=6.0)

    assert [str(w.message) for w in caught if ADVISORY_KEY in str(w.message)] == []


def test_thin_absorber_advisory_quotes_the_lowest_frequency_not_band_centre():
    """The requirement is evaluated where lambda_g is longest.

    Band centre would understate it — that is what the sibling junction
    advisory does, and it is why a 0.30-lambda_g stack passed unnoticed.
    """
    hits = _advisories(_two_port(10))
    assert len(hits) == 1
    assert "4.500 GHz" in hits[0]
    assert "6.250 GHz" not in hits[0]     # band centre


def test_thin_absorber_advisory_fires_once_per_axis_not_once_per_port():
    """Both ports share one propagation axis and one absorber."""
    sim = _two_port(10)
    assert len(sim._waveguide_ports) == 2
    assert len(_advisories(sim)) == 1


def test_thin_absorber_advisory_silent_on_pec_closed_propagation_axis():
    """NON-FIRING control: no absorber on the propagation axis to under-drain.

    The absorber must be somewhere (``add_waveguide_port`` requires an
    absorbing boundary), so this closes the x propagation axis with PEC and
    leaves the transverse faces absorbing. The advisory is scoped to the
    propagation axis, so it must stay silent even though the scalar
    ``cpml_layers`` is far below 0.5 lambda_g.
    """
    sim = _two_port(
        10,
        boundary=BoundarySpec(x=Boundary(lo="pec", hi="pec"),
                              y="cpml", z="cpml"),
    )
    assert _advisories(sim) == []


def test_thin_absorber_advisory_silent_when_band_starts_below_cutoff():
    """NON-FIRING control: lambda_g is undefined below cutoff.

    A band that starts at or below cutoff has a more fundamental problem,
    already named by the ``port_freqs_below_cutoff`` preflight check;
    piling an absorber advisory on top would be noise. Documented
    consequence: such a band gets no absorber advisory at all.
    """
    # TE10 cutoff of the 40 mm guide is ~3.4-3.75 GHz; start well below it.
    freqs = np.linspace(1.0e9, 8.0e9, 4)
    assert _advisories(_two_port(10, freqs=freqs), freqs=freqs) == []


def test_thin_absorber_advisory_honours_per_face_thickness_overrides():
    """A per-face override, not the scalar, sets the reported thickness.

    Firing/non-firing in one setup: the thin face is reported and the thick
    face is not. Note the scalar is the allocation BUDGET — a per-face
    override may only reduce thickness below it (``rfx/grid.py`` rejects
    ``face_layers > cpml_layers``), so the budget here is the thick value.
    """
    sim = _two_port(
        40,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=40),
            y="cpml", z="cpml"),
    )
    hits = _advisories(sim)
    assert len(hits) == 1
    assert "x-lo 4 cells" in hits[0], hits[0]
    assert "x-hi" not in hits[0], hits[0]


# --------------------------------------------------------------------------- #
# #494 advisory — coverage added after an independent mutation battery
#
# The original fixture exercised the advisory at ONE point in its option space
# (normalize=False, empty geometry, one x-propagating single-mode port pair,
# both absorbing faces equally thin), so single-value substitutions at five
# separate branch points all survived. The worst of them: gating the call site
# on ``if not self._geometry`` or ``if normalize != "flux"`` left all 40 tests
# green, and BOTH of those conditions are the actual settings of
# ``validation/crossval/18_wr90_iris_modematch.py`` — the script that motivated
# issue #494. A one-line regression restoring precisely the #494 blind spot on
# precisely the motivating script was invisible.
# --------------------------------------------------------------------------- #
def _two_port_with_obstacle(cpml_layers, *, freqs=_FREQS, dx=0.004):
    """A two-port sim that is NOT empty — mirrors a real crossval setup."""
    sim = _two_port(cpml_layers, freqs=freqs, dx=dx)
    # A PEC obstacle, as every real waveguide device has.
    sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
    sim.add(Box((0.058, 0.0, 0.0), (0.062, 0.012, 0.02)), material="pec_like")
    return sim


@pytest.mark.parametrize("normalize", [False, "flux"])
def test_advisory_fires_with_geometry_present_and_under_flux(normalize):
    """The option combination the motivating crossval script actually uses.

    ``18_wr90_iris_modematch.py`` runs `normalize="flux"` with PEC boxes
    registered. Gating the advisory on either condition must not hide it.
    """
    sim = _two_port_with_obstacle(10)
    assert sim._geometry, "fixture must be non-empty to bind the geometry gate"
    with pytest.warns(UserWarning, match=ADVISORY_KEY):
        sim.compute_waveguide_s_matrix(normalize=normalize, num_periods=6.0)


def test_advisory_message_recomputes_every_number_it_quotes():
    """Bind the quoted numbers, not merely the strings around them.

    Six of the seven numbers the advisory prints were unasserted, including its
    only actionable output ("Raise cpml_layers to at least N"): a mutant that
    advised raising it to 1 instead of 13 passed. Each value here is derived
    independently from the port geometry rather than copied from the message.
    """
    sim = _two_port(10)
    grid = sim._build_grid()
    dx = float(grid.dx)
    cfg = sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, jnp.asarray(_FREQS), 2000)
    fc = float(cfg.f_cutoff)
    f_lo = float(np.min(_FREQS))
    c0 = 299792458.0
    lambda_g = (c0 / f_lo) / np.sqrt(1.0 - (fc / f_lo) ** 2)
    required = 0.5 * lambda_g

    hits = _advisories(sim)
    assert len(hits) == 1
    msg = hits[0]

    assert f"{10 * dx * 1e3:.1f} mm" in msg              # what you have
    assert f"against a required {required * 1e3:.1f} mm" in msg
    assert f"lambda_g = {lambda_g * 1e3:.1f} mm" in msg
    assert f"mode cutoff {fc / 1e9:.3f} GHz" in msg
    assert f"({10 * dx / lambda_g:.2f} lambda_g)" in msg
    assert (f"Raise cpml_layers to at least {int(np.ceil(required / dx))} "
            f"(0.75 lambda_g needs "
            f"{int(np.ceil(0.75 * lambda_g / dx))})") in msg
    # The quoted threshold must be the one actually enforced, not a second
    # copy of the constant that can drift from it.
    from rfx.api._sparams import _FAR_PORT_LAMBDA_G_FRACTION
    assert f"documented {_FAR_PORT_LAMBDA_G_FRACTION:g} guide-wavelength" in msg
    # The #494 ripple ladder is a claims-bearing measurement quoted as fact, so
    # it is pinned too: prose numbers were freely corruptible otherwise.
    assert ("residual |S11| ripple was 0.0706 at 0.30 lambda_g, 0.0366 at 0.50, "
            "and 0.0093 at 0.75") in msg


def _z_two_port(cpml_layers, *, freqs=_FREQS, dx=0.004):
    """Ports propagating along z, so the axis cannot be hardcoded to 'x'."""
    sim = Simulation(
        freq_max=float(freqs[-1]), domain=(0.04, 0.02, 0.12), dx=dx,
        boundary=BoundarySpec(x="pec", y="pec",
                              z=Boundary(lo="cpml", hi="cpml")),
        cpml_layers=cpml_layers,
    )
    for z, direction in ((0.02, "+z"), (0.10, "-z")):
        sim.add_waveguide_port(
            z, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(freqs), f0=float(np.mean(freqs)), bandwidth=0.6,
        )
    return sim


def test_advisory_names_the_actual_propagation_axis_not_always_x():
    """Every original fixture propagated along x, so `axis = "x"` survived.

    Also binds the PEC-closed fence in the direction a hardcoded axis would
    NOT satisfy: here x/y are PEC and only z absorbs.
    """
    hits = _advisories(_z_two_port(10))
    assert len(hits) == 1
    assert "on the z propagation axis" in hits[0], hits[0]
    assert "z-lo 10 cells" in hits[0] and "z-hi 10 cells" in hits[0]
    assert "x-" not in hits[0] and "y-" not in hits[0]


def test_advisory_dedupes_per_axis_and_still_reports_both_axes():
    """Binds the ENUMERATION, not just the deduped count.

    `len(hits) == 1` on a same-axis two-port is satisfied both by working
    dedupe and by looking at only the first port. Pairing it with a two-axis
    sim that must yield TWO warnings distinguishes them.
    """
    # Four ports on two absorbing axes. The y-extent sets the x-normal ports'
    # cutoff (2.932 GHz) and the x-extent sets the z-normal ports' (1.208 GHz),
    # so both bands are above cutoff and cpml=6 (24.0 mm) is thin against both
    # requirements (43.9 mm and 34.6 mm) — i.e. all four ports have something
    # to report, and correct dedupe collapses them to one per axis.
    dx = 0.004
    sim = Simulation(
        freq_max=float(_FREQS[-1]), domain=(0.12, 0.05, 0.12), dx=dx,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"), y="pec",
                              z=Boundary(lo="cpml", hi="cpml")),
        cpml_layers=6,
    )
    for pos, direction in ((0.02, "+x"), (0.10, "-x"),
                           (0.02, "+z"), (0.10, "-z")):
        sim.add_waveguide_port(
            pos, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(_FREQS), f0=float(np.mean(_FREQS)), bandwidth=0.6)
    assert len(sim._waveguide_ports) == 4

    hits = _advisories(sim)
    assert len(hits) == 2, hits
    axes = sorted(h.split("on the ")[1].split(" propagation")[0] for h in hits)
    assert axes == ["x", "z"], axes


def test_advisory_dedupe_key_keeps_axes_apart_when_their_cutoffs_coincide():
    """Binds the AXIS component of the dedupe key specifically.

    On a cube domain the x-normal and z-normal ports have the SAME cutoff, so a
    key of `(cutoff,)` alone would collapse two physically distinct axes into
    one warning. `(axis, cutoff)` must keep them separate.
    """
    sim = Simulation(
        freq_max=float(_FREQS[-1]), domain=(0.12, 0.12, 0.12), dx=0.004,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"), y="pec",
                              z=Boundary(lo="cpml", hi="cpml")),
        cpml_layers=6,
    )
    for pos, direction in ((0.02, "+x"), (0.02, "+z")):
        sim.add_waveguide_port(
            pos, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(_FREQS), f0=float(np.mean(_FREQS)), bandwidth=0.6)

    grid = sim._build_grid()
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(_FREQS), 2000)
            for e in sim._waveguide_ports]
    cutoffs = {round(float(c.f_cutoff), 3) for c in cfgs}
    assert len(cutoffs) == 1, ("fixture must share one cutoff", cutoffs)

    hits = _advisories(sim)
    assert len(hits) == 2, hits
    assert sorted(h.split("on the ")[1].split(" propagation")[0]
                  for h in hits) == ["x", "z"]


def test_advisory_reports_a_thin_hi_face_behind_a_thick_lo_face():
    """Mirror of the per-face test — the original only made the LO face thin.

    `for side in ("lo", "hi")` -> `("lo",)` survived because no fixture had an
    independently thin hi face.
    """
    sim = _two_port(
        40,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=40, hi_thickness=4),
            y="cpml", z="cpml"),
    )
    hits = _advisories(sim)
    assert len(hits) == 1
    assert "x-hi 4 cells" in hits[0], hits[0]
    assert "x-lo" not in hits[0], hits[0]


def test_advisory_reports_both_faces_when_both_are_thin():
    """`thin[:1]` (report only the first thin face) survived otherwise."""
    hits = _advisories(_two_port(10))
    assert len(hits) == 1
    assert "x-lo 10 cells" in hits[0] and "x-hi 10 cells" in hits[0], hits[0]


def test_advisory_uses_the_lowest_cutoff_mode_of_a_multimode_port():
    """Fence (c) had zero coverage: every port was single-mode.

    With n_modes=2 the config is a LIST, so `min`, `max` and `modes[-1]` become
    distinguishable. The documented behaviour is the lowest-cutoff mode (TE10),
    which is the least demanding.
    """
    dx = 0.004
    sim = Simulation(
        freq_max=float(_FREQS[-1]), domain=(0.12, 0.04, 0.02), dx=dx,
        boundary="cpml", cpml_layers=10,
    )
    for pos, direction in ((0.02, "+x"), (0.10, "-x")):
        sim.add_waveguide_port(
            pos, direction=direction, mode=(1, 0), mode_type="TE", n_modes=2,
            freqs=jnp.asarray(_FREQS), f0=float(np.mean(_FREQS)), bandwidth=0.6)

    grid = sim._build_grid()
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(_FREQS), 2000)
            for e in sim._waveguide_ports]
    assert isinstance(cfgs[0], list) and len(cfgs[0]) == 2, "need a multimode cfg"
    cutoffs = sorted(float(c.f_cutoff) for c in cfgs[0])
    assert cutoffs[0] < cutoffs[1]

    hits = _advisories(sim)
    assert len(hits) == 1
    assert f"mode cutoff {cutoffs[0] / 1e9:.3f} GHz" in hits[0], (hits[0], cutoffs)
    assert f"mode cutoff {cutoffs[1] / 1e9:.3f} GHz" not in hits[0]


def test_advisory_is_a_userwarning_so_the_default_filter_shows_it():
    """The category was unbound: UserWarning -> DeprecationWarning survived.

    A DeprecationWarning raised into library-caller code is suppressed under
    Python's default filters, which would silently mute an advisory that exists
    precisely because the functional entry points run no preflight.
    """
    sim = _two_port(10)
    grid = sim._build_grid()
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(_FREQS), 2000)
            for e in sim._waveguide_ports]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_thin_absorber_vs_guide_wavelength(
            grid, cfgs, _FREQS, sim._cpml_layers, sim._boundary_spec)
    hits = [w for w in caught if ADVISORY_KEY in str(w.message)]
    assert len(hits) == 1
    assert hits[0].category is UserWarning, hits[0].category


# --------------------------------------------------------------------------- #
# #493 characterization — independent oracle for the pinned excess table
#
# `_NOMINAL_EXCESS` is a table of measured constants, so a coherent author could
# mutate the rasterization rule AND re-pin the table in one commit and stay
# green. Verified: `(coords >= lo)` -> `(coords > lo)` plus five numeric edits
# passed 40/40, which silently falsified the ambiguity docstring and left the
# asymmetric branch of the symmetry test dead in every parametrization.
# --------------------------------------------------------------------------- #
def _predicted_excess(cells: int, d_phys: float) -> int:
    """Excess in cells, derived from the mechanism rather than measured.

    The lo fin's interior face is a ``hi`` corner, which half-openness always
    drops: one cell, unconditionally. The hi fin's interior face is a ``lo``
    corner, kept unless float32 rounding puts its node strictly below it.
    """
    y, dx = _real_node_coords(cells)
    d_c = int(round(d_phys / dx))
    fin_c = (cells - d_c) // 2
    # Float64 since #802 — the same precision the mask comparison runs at.
    hi_corner = np.float64(A_WR90 - fin_c * dx)
    node = np.float64(y[cells - fin_c])
    return 1 + (1 if node < hi_corner else 0)


@pytest.mark.parametrize("cells,d_mm", sorted(_NOMINAL_EXCESS))
def test_pinned_excess_matches_an_independently_derived_prediction(cells, d_mm):
    """The table and the mechanism must agree, so neither can be re-pinned alone."""
    predicted = _predicted_excess(cells, d_mm * 1e-3)
    assert predicted == _NOMINAL_EXCESS[(cells, d_mm)], (
        cells, d_mm, predicted, _NOMINAL_EXCESS[(cells, d_mm)])


def test_excess_table_is_uniformly_one_cell_since_exact_coordinates():
    """Re-pinned at #802 (was: table must exercise both 1 and 2).

    The two-cell case existed only through float32 rounding, so with exact
    coordinates it is unreachable from a nominal drawing and every pinned
    excess is 1. Consequence for the symmetry test above: its
    ``excess == 2`` branch is intentionally dead code retained as
    documentation of the mechanism — the two corner types still differ,
    but only the ``hi``-corner retreat occurs. If a value other than 1
    ever appears here, rounding behaviour changed: investigate before
    re-pinning (this table caught #802's class in the first place).
    """
    values = set(_NOMINAL_EXCESS.values())
    assert values == {1}, values


def test_advisory_dedupe_key_keeps_cutoffs_apart_on_one_axis():
    """Binds the CUTOFF component of the dedupe key.

    Two ports on the SAME axis with different modes (TE10 and TE20) have
    different cutoffs and therefore different `lambda_g` requirements, so each
    deserves its own line. A key of `(axis,)` alone would collapse them. The
    guide is widened so BOTH modes propagate at the lowest measured frequency,
    otherwise the below-cutoff fence would hide the TE20 port.
    """
    sim = Simulation(
        freq_max=float(_FREQS[-1]), domain=(0.12, 0.08, 0.02), dx=0.004,
        boundary="cpml", cpml_layers=6,
    )
    for pos, direction, mode in ((0.02, "+x", (1, 0)), (0.10, "-x", (2, 0))):
        sim.add_waveguide_port(
            pos, direction=direction, mode=mode, mode_type="TE",
            freqs=jnp.asarray(_FREQS), f0=float(np.mean(_FREQS)), bandwidth=0.6)

    grid = sim._build_grid()
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(_FREQS), 2000)
            for e in sim._waveguide_ports]
    cutoffs = sorted(round(float(c.f_cutoff), 3) for c in cfgs)
    assert len(set(cutoffs)) == 2, ("fixture needs two cutoffs", cutoffs)
    assert float(np.min(_FREQS)) > cutoffs[-1], "both modes must propagate"

    hits = _advisories(sim)
    assert len(hits) == 2, hits
    for fc in cutoffs:
        assert any(f"mode cutoff {fc / 1e9:.3f} GHz" in h for h in hits), (fc, hits)


def test_advisory_also_fires_on_the_port_reference_sims_junction_path():
    """The junction path must not lose the LOWEST-frequency check.

    That path already has a sibling advisory
    (``_warn_junction_cpml_thickness``), but it evaluates at BAND CENTRE — the
    very weakness issue #494 was filed about, since `lambda_g` is longest at
    the band edge. Gating this advisory on ``port_reference_sims is None``
    therefore silently drops the band-edge check exactly where junction
    geometry needs it most, and previously left the suite green.
    """
    def build():
        sim = _two_port(10)
        return sim

    device = build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        device.compute_waveguide_s_matrix(
            normalize="flux", num_periods=6.0,
            port_reference_sims=[build(), build()])
    hits = [str(w.message) for w in caught if ADVISORY_KEY in str(w.message)]

    assert len(hits) == 1, [str(w.message) for w in caught]
    # It is the band-EDGE check, distinct from the sibling's band-centre one.
    assert "lowest measured frequency 4.500 GHz" in hits[0], hits[0]


# --------------------------------------------------------------------------- #
# #576 review F3 — the advisory has to reach the NON-UNIFORM lane too
# --------------------------------------------------------------------------- #
def _two_port_nu(cpml_layers, *, ratio=2.0, freqs=_FREQS, dx=0.004):
    """The same two-port sim on a transversely graded mesh.

    `compute_waveguide_s_matrix` dispatches a `dy_profile` mesh to a dedicated
    extractor and returns from that branch ~110 lines ABOVE where the absorber
    advisory used to sit, so the check was unreachable from here: the guard
    existed, its text named the exact remedy, and no NU caller ever asked it.
    Both NU fixture producers consequently shipped 0.33 and 0.099 lambda_g
    stacks in silence (#496 / #576).
    """
    total = 0.04
    n = int(round(total / dx))
    x = np.linspace(-1.0, 1.0, n)
    w = 1.0 + (ratio - 1.0) * np.abs(x)
    sim = Simulation(
        freq_max=float(freqs[-1]),
        domain=(0.12, total, 0.02),
        dx=dx,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=cpml_layers,
        dy_profile=w / w.sum() * total,
    )
    for x_pos, direction in ((0.02, "+x"), (0.10, "-x")):
        sim.add_waveguide_port(
            x_pos, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(freqs), f0=float(np.mean(freqs)), bandwidth=0.6,
        )
    return sim


def _nu_advisory_hits(sim):
    """Drive the NU extraction far enough to reach the advisory, keep its hits.

    `num_periods` is deliberately tiny: the advisory is emitted before the FDTD
    runs, so this needs the dispatch, not a converged solve. Any solver error
    after the advisory is irrelevant here and is swallowed on purpose.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sim.compute_waveguide_s_matrix(num_periods=1, normalize="flux")
        except Exception:
            pass
    return [str(w.message) for w in caught if ADVISORY_KEY in str(w.message)]


def test_thin_absorber_advisory_fires_on_the_nonuniform_lane():
    """FIRING on the NU lane — the wiring #576's review found missing."""
    hits = _nu_advisory_hits(_two_port_nu(10))
    assert len(hits) == 1, hits
    msg = hits[0]
    assert "40.0 mm" in msg, msg           # 10 cells * 4 mm
    assert "against a required" in msg
    assert "lowest measured frequency 4.500 GHz" in msg
    assert "cpml_layers" in msg


def test_thin_absorber_advisory_silent_on_the_nonuniform_lane_when_thick():
    """NON-FIRING control: same graded mesh and band, adequate absorber.

    Without this the firing test above could be satisfied by an advisory that
    warns unconditionally on any NU mesh.
    """
    assert _nu_advisory_hits(_two_port_nu(40)) == []


def test_nonuniform_advisory_plumbing_failure_is_reported_not_swallowed():
    """A broken advisory must SAY so rather than read as a clean bill of health.

    The first version of this wiring wrapped the whole block in
    `except Exception: pass`, and two separate plumbing bugs (an unresolved
    `freqs`, then a `None` `n_steps`) presented exactly like a passing absorber.
    That silence is the failure mode #576's F3 is about, so the block reports
    instead — asserted here by breaking the config builder on purpose.
    """
    import rfx.runners.nonuniform as _nu

    original = _nu._build_waveguide_port_config_nu
    _nu._build_waveguide_port_config_nu = (
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("planted")))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _two_port_nu(10).compute_waveguide_s_matrix(
                    num_periods=1, normalize="flux")
            except Exception:
                pass
        msgs = [str(w.message) for w in caught]
    finally:
        _nu._build_waveguide_port_config_nu = original

    reported = [m for m in msgs if "could not evaluate the far-port" in m]
    assert reported, msgs
    assert "planted" in reported[0]
    assert "UNCHECKED" in reported[0]
