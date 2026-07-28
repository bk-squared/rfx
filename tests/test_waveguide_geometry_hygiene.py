"""Geometry / absorber hygiene for waveguide S-parameter setups.

Two independent hygiene defects found while reviewing PR #480 (WR-90 single
inductive iris), both filed as general issues because the iris lane only
exposed them:

* **#493** — ``Box``'s volume branch is half-open ``[lo, hi)`` over NODE
  coordinates, so a PEC obstacle drawn to its nominal physical dimension
  rasterizes one cell short at its ``hi`` face. The tests below pin that
  arithmetic (including the float32 knife edge at a node plane) so the
  convention is executable documentation rather than folklore. They are
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
from rfx.geometry.csg import Box


ADVISORY_KEY = "far-port discipline"


# --------------------------------------------------------------------------- #
# #493 — half-open node rasterization of PEC obstacles
# --------------------------------------------------------------------------- #
A_WR90 = 22.86e-3
_ZERO = np.array([0.0])


def _occupied(hi_face: float, coords: np.ndarray) -> np.ndarray:
    """Indices of nodes occupied by a box spanning up to ``hi_face`` in y."""
    m = Box((-1.0, -1.0, -1.0), (1.0, float(hi_face), 1.0)).mask_on_coords(
        _ZERO, coords, _ZERO)
    return np.nonzero(np.asarray(m)[0, :, 0])[0]


def _fin_pair_aperture(hi_face: float, coords: np.ndarray, dx: float):
    """Electrical aperture between two facing fins, in metres.

    Returns ``(aperture_m, free_node_indices)``. The aperture is the span
    between the innermost OCCUPIED (zeroed-tangential-E) node planes.
    """
    lo_fin = Box((-1.0, -1.0, -1.0), (1.0, float(hi_face), 1.0))
    hi_fin = Box((-1.0, A_WR90 - float(hi_face), -1.0), (1.0, 1.0, 1.0))
    metal = (np.asarray(lo_fin.mask_on_coords(_ZERO, coords, _ZERO))[0, :, 0]
             | np.asarray(hi_fin.mask_on_coords(_ZERO, coords, _ZERO))[0, :, 0])
    free = np.nonzero(~metal)[0]
    return ((free[-1] + 1) - (free[0] - 1)) * dx, free


@pytest.mark.parametrize("cells", [30, 60])
def test_box_volume_branch_is_half_open_and_asymmetric_over_nodes(cells):
    """A box between two node planes occupies nodes ``i..k-1``, not ``i..k``.

    The drawn extent realizes as ``extent - dx`` between the outermost
    occupied planes, and the shortfall is entirely at the ``hi`` face — the
    footprint is asymmetric, which displaces the object by ``dx/2``.
    """
    dx = A_WR90 / cells
    coords = np.arange(cells + 1) * dx
    occ = _occupied(8 * dx, coords)

    assert occ[0] == 0
    assert occ[-1] == 7, "hi face at node 8 must contribute no cell"
    assert len(occ) == 8
    realized = (occ[-1] - occ[0]) * dx
    assert realized == pytest.approx(8 * dx - dx, rel=1e-6), (
        "realized extent must be one cell short of the drawn extent")


@pytest.mark.parametrize("cells", [30, 60])
@pytest.mark.parametrize("d_phys", [7.620e-3, 12.192e-3, 18.288e-3])
def test_fins_drawn_to_nominal_aperture_open_one_cell_too_wide(cells, d_phys):
    """Drawing to the nominal opening yields an ELECTRICAL opening d + dx.

    This is the #493 mechanism. It also displaces the aperture by half a
    cell, because the shortfall sits only at the lo fin's ``hi`` face.
    """
    dx = A_WR90 / cells
    coords = np.arange(cells + 1) * dx
    d_c = int(round(d_phys / dx))
    fin_c = (cells - d_c) // 2

    aperture, free = _fin_pair_aperture(fin_c * dx, coords, dx)

    assert aperture == pytest.approx(d_phys + dx, rel=1e-6), (
        f"expected d + dx, got d + {(aperture - d_phys) / dx:.2f}*dx")
    # Asymmetry witness: the free span's centre is half a cell below the
    # guide centre node.
    assert 0.5 * (free[0] + free[-1]) == pytest.approx(cells / 2 - 0.5)


@pytest.mark.parametrize("cells", [30, 60])
@pytest.mark.parametrize("d_phys", [7.620e-3, 12.192e-3, 18.288e-3])
def test_half_cell_outward_offset_restores_nominal_electrical_aperture(
        cells, d_phys):
    """The documented recipe: interior faces on cell midpoints.

    Non-firing control for the test above — same geometry, corrected
    corners, aperture exactly nominal and centred.
    """
    dx = A_WR90 / cells
    coords = np.arange(cells + 1) * dx
    d_c = int(round(d_phys / dx))
    fin_c = (cells - d_c) // 2

    aperture, free = _fin_pair_aperture((fin_c + 0.5) * dx, coords, dx)

    assert aperture == pytest.approx(d_phys, rel=1e-6)
    assert 0.5 * (free[0] + free[-1]) == pytest.approx(cells / 2)


def test_half_cell_inward_offset_opens_two_cells_too_wide():
    """Offsetting interior faces the WRONG way gives d + 2*dx.

    This is the variant that produced #493's headline 4-6x |S11| envelope
    inflation: a half-cell offset applied to make footprints deterministic,
    but toward the metal instead of into the opening.
    """
    cells, d_phys = 30, 12.192e-3
    dx = A_WR90 / cells
    coords = np.arange(cells + 1) * dx
    fin_c = (cells - int(round(d_phys / dx))) // 2

    aperture, _ = _fin_pair_aperture((fin_c - 0.5) * dx, coords, dx)

    assert aperture == pytest.approx(d_phys + 2 * dx, rel=1e-6)


def test_node_plane_corner_is_a_single_float32_ulp_knife_edge():
    """Why the recipe says "cell midpoint" and not "on the node plane".

    Masks are evaluated at float32 precision (x64 is off by design), so one
    ULP of the corner value — ~5e-10 m at dx = 0.762 mm, 6e-7 of a cell —
    moves the footprint by a whole cell. A corner computed as ``a - n*dx``
    can therefore rasterize differently from an algebraically identical
    ``m*dx``. Midpoint corners sit half a cell from either edge.
    """
    dx = 0.762e-3
    coords = np.arange(31) * dx
    node8 = np.float32(8 * dx)
    ulp = float(np.nextafter(node8, np.float32(1)) - node8)
    assert ulp < 1e-9 * 1.0, ulp  # sanity: sub-nanometre

    below = _occupied(float(np.nextafter(node8, np.float32(0))), coords)
    above = _occupied(float(np.nextafter(node8, np.float32(1))), coords)

    assert below[-1] == 7
    assert above[-1] == 8, "one float32 ULP must flip the footprint by a cell"
    # And the midpoint recipe is insensitive to that perturbation.
    mid = 8 * dx + 0.5 * dx
    assert _occupied(mid, coords)[-1] == _occupied(mid + 8 * ulp, coords)[-1]


def test_drawn_vs_realized_gap_cannot_discriminate_a_correct_drawing():
    """Why #493 ships as documentation and not as a rasterized-vs-drawn check.

    Issue #493 floated an advisory that fires when a PEC volume's rasterized
    opening differs from its drawn opening by >= 1 cell. That predicate has
    no discriminating power: the half-open rule shifts the realized aperture
    up by exactly one cell relative to the gap between the drawn faces
    REGARDLESS of where those faces sit, so it reads +1 cell for the correct
    midpoint recipe and for both defective drawings alike. The defect is
    ``realized != INTENDED``, and the intended dimension is never
    communicated to the simulator, so it cannot be recovered from geometry.
    """
    cells, d_phys = 30, 12.192e-3
    dx = A_WR90 / cells
    coords = np.arange(cells + 1) * dx
    fin_c = (cells - int(round(d_phys / dx))) // 2

    diffs = {}
    for label, hi_face in (("nominal", fin_c * dx),
                           ("inward", (fin_c - 0.5) * dx),
                           ("outward", (fin_c + 0.5) * dx)):
        realized, _ = _fin_pair_aperture(hi_face, coords, dx)
        drawn = (A_WR90 - hi_face) - hi_face
        diffs[label] = (realized - drawn) / dx

    assert diffs["outward"] == pytest.approx(1.0, abs=1e-6)
    assert diffs["nominal"] == pytest.approx(1.0, abs=1e-6)
    assert diffs["inward"] == pytest.approx(1.0, abs=1e-6)


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
