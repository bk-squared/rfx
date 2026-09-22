"""The graded grid answers the per-cell metric interface (step 0a).

A graded mesh has a different cell size at almost every node, and the Yee
update divides by TWO different metrics: the H update by the local cell it
differences across, the E update by the dual spacing of the node it sits on.
Today a consumer gets those by reaching into ``dx_arr`` / ``inv_dx`` / ``dz``
and re-deriving the rule, or -- the defect step 0 exists to close -- by
reading the scalar ``grid.dx``, which is the BOUNDARY cell and not the cell
at the site.

This file is judge (ii) and judge (iv) of the design note
(``docs/design_notes/20260922_nu_grid_core_predeclaration.md``): on every
fixture shape the non-uniform test suite builds, the new accessors reproduce
the arrays the solver already carries -- ``inv_dx``/``inv_dx_h`` through
``_profile_to_inv_arrays`` and ``e_node_dual_spacing_at`` at every node --
bit for bit; a constructor handed cells that do not add up to the declared
profile refuses; and a traced axis asked for an index refuses instead of
answering from a nominal uniform mesh it is not running (gap G9).

No consumer moves here. ``position_to_index`` keeps its uniform fallback,
``cpml.py`` keeps its one scalar, and every number the solver produces is
unchanged; 0a adds surface only.

Fixture shapes
--------------
``FIXTURES`` reproduces every ``make_nonuniform_grid`` call shape in
``tests/unit/nonuniform`` (18 call sites in 7 modules, grepped), plus two the
suite does not have and the end rules need: a pad-free axis whose FIRST two
cells differ, and an axis whose lo and hi pads differ. Without the first, the
leading dual entry ``dual[0] = d[0]`` is invisible -- an absorber pad
replicates the boundary cell, so ``d[0] == d[1]`` and the end rule and the
mean agree by accident.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.geometry.rasterize_grid import (
    ExactNodeSpineMissingWarning,
    coords_from_nonuniform_grid,
)
from rfx.nonuniform import (
    NonUniformGrid,
    _append_bounding_node,
    _pad_profile,
    _profile_to_inv_arrays,
    assert_cells_span_declared_profile,
    e_node_dual_spacing_at,
    e_node_dual_spacings,
    interior_cells,
    make_band_profile,
    make_nonuniform_grid,
    position_to_index,
)

AXES = ("x", "y", "z")

# The 2:1 transition profile the source/port dual-spacing suite uses.
D = 0.5e-3
PROF_2TO1 = np.concatenate([np.full(5, D), np.full(6, 2.0 * D), np.full(5, D)])
# Coarse-fine-coarse in plane, the shape test_nonuniform_xy builds.
XY_STEP = np.concatenate([np.full(10, 1e-3), np.full(10, 0.5e-3),
                          np.full(10, 1e-3)])


def _band_xy():
    return make_band_profile([0.0, 10e-3, 12e-3, 27e-3],
                             [1e-3, 2e-4, 1e-3],
                             max_ratio=1.3, boundary_cell=1e-3)


FIXTURES = {
    # --- shapes grepped from tests/unit/nonuniform ---
    "api_basic": lambda: make_nonuniform_grid(
        (0.05, 0.05), np.array([0.4e-3] * 4 + [0.5e-3] * 10), 0.5e-3, 12),
    "api_cfl_10x": lambda: make_nonuniform_grid(
        (0.01, 0.01), np.array([0.1e-3] * 4 + [1e-3] * 5), 1e-3, 8),
    "api_source": lambda: make_nonuniform_grid(
        (0.02, 0.02), np.array([0.4e-3] * 4 + [0.5e-3] * 5), 0.5e-3, 8),
    "api_dv_mixed": lambda: make_nonuniform_grid(
        (0.02, 0.02), np.array([0.2e-3] * 4 + [0.5e-3] * 5), 0.5e-3, 8),
    "dual_2to1_padfree": lambda: make_nonuniform_grid(
        (8e-3, 8e-3), PROF_2TO1, D, cpml_layers=0, dx_profile=PROF_2TO1),
    "xy_step": lambda: make_nonuniform_grid(
        (0, 0), np.full(8, 0.5e-3), dx=1e-3, cpml_layers=8,
        dx_profile=XY_STEP, dy_profile=XY_STEP),
    "until_decay": lambda: make_nonuniform_grid(
        (0.008, 0.008), np.array([0.4e-3] * 12 + [0.6e-3] * 12), 0.5e-3,
        cpml_layers=4),
    "band_builder_xy": lambda: make_nonuniform_grid(
        (27e-3, 27e-3), np.full(10, 1e-3), 1e-3, cpml_layers=8,
        dx_profile=_band_xy(), dy_profile=_band_xy()),
    # --- two shapes the suite does not build, and the end rules need ---
    "padfree_graded_end": lambda: make_nonuniform_grid(
        (4e-3, 4e-3),
        np.concatenate([np.full(1, 1e-4), np.full(6, 2e-4), np.full(4, 1e-4)]),
        2e-4, cpml_layers=0),
    "asymmetric_pads": lambda: make_nonuniform_grid(
        (6e-3, 6e-3), np.array([1e-4] * 5 + [3e-4] * 8), 2e-4, cpml_layers=6,
        pec_faces={"z_lo", "x_lo"}),
}

GRADED_BAND = "band_builder_xy"


def _layout(grid, axis):
    ax = AXES.index(axis)
    n = (grid.nx, grid.ny, grid.nz)[ax]
    pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[ax]
    pad_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)[ax]
    return n, pad_lo, pad_hi


def _spine(grid, axis):
    """The grid's own float64 field, read WITHOUT going through ``cells``."""
    return (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)[AXES.index(axis)]


def _inv_pair(grid, axis):
    return ((grid.inv_dx, grid.inv_dx_h), (grid.inv_dy, grid.inv_dy_h),
            (grid.inv_dz, grid.inv_dz_h))[AXES.index(axis)]


ALL = [(name, axis) for name in sorted(FIXTURES) for axis in AXES]


# ---------------------------------------------------------------------------
# cells -- the float64 source of truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,axis", ALL)
def test_cells_is_the_float64_spine_over_the_padded_axis(name, axis):
    grid = FIXTURES[name]()
    n, _, _ = _layout(grid, axis)
    cells = grid.cells(axis)
    assert cells.dtype == np.float64
    assert cells.shape == (n,)
    assert np.array_equal(cells, np.asarray(_spine(grid, axis)))
    assert grid.is_traced(axis) is False


@pytest.mark.parametrize("name,axis", ALL)
def test_the_last_cell_is_the_bounding_node_duplicate(name, axis):
    """#562: N cells need N+1 nodes, so one boundary cell is duplicated to
    supply the last node. It carries no physical extent -- its H term is the
    one the stencil zeroes -- which is why ``sum(cells)`` overshoots."""
    grid = FIXTURES[name]()
    cells = grid.cells(axis)
    assert cells[-1] == cells[-2]
    _, pad_lo, pad_hi = _layout(grid, axis)
    realized = interior_cells(cells, pad_lo, pad_hi)
    assert float(cells.sum()) > float(realized.sum())


# ---------------------------------------------------------------------------
# Judge (ii) -- the two inverse-metric arrays derive from cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,axis", ALL)
def test_inverse_metric_arrays_derive_from_cells_bit_for_bit(name, axis):
    """``_profile_to_inv_arrays(cells(axis))`` reproduces what the grid holds.

    Not circular: the grid built ``inv_*`` at construction from its own
    padded profile, before any accessor existed. If ``cells`` returned the
    float32 store, a different slice, or dropped the bounding node, these
    arrays would not land on the same bits.
    """
    grid = FIXTURES[name]()
    inv_e, inv_h = _inv_pair(grid, axis)
    got_e, got_h = _profile_to_inv_arrays(grid.cells(axis))
    assert np.array_equal(np.asarray(got_e), np.asarray(inv_e))
    assert np.array_equal(np.asarray(got_h), np.asarray(inv_h))


@pytest.mark.parametrize("name,axis", ALL)
def test_the_two_end_entry_rules_hold(name, axis):
    """``inv_d_h[N-1] = 0`` and ``inv_d_e[0] = 1/d[0]``.

    The trailing zero is the forward difference having no ``d[N]`` to reach.
    The leading entry is the mirror rule -- treat ``d[-1]`` as ``d[0]`` -- and
    it is BIT-IDENTICAL to evaluating the general formula that way,
    ``2/(d[0]+d[0])``, because doubling and halving are exact in binary
    floating point. So the "override" is not a choice of a different number;
    it is the general rule evaluated at a mirrored boundary.
    """
    grid = FIXTURES[name]()
    inv_e, inv_h = _inv_pair(grid, axis)
    d0 = np.float32(grid.cells(axis)[0])
    assert float(np.asarray(inv_h)[-1]) == 0.0
    assert float(np.asarray(inv_e)[0]) == float(np.float32(1.0) / d0)
    assert float(np.asarray(inv_e)[0]) == float(np.float32(2.0) / (d0 + d0))


# ---------------------------------------------------------------------------
# Judge (ii) -- duals reproduce the scalar helper at every node
# ---------------------------------------------------------------------------

def _check_duals(grid, axis):
    """The duals oracle, factored out so a mutation run can re-enter it."""
    cells = grid.cells(axis)
    longhand = np.concatenate([cells[:1], 0.5 * (cells[:-1] + cells[1:])])
    duals = grid.duals(axis)
    assert duals.dtype == np.float64
    assert np.array_equal(duals, longhand)
    for k in range(cells.size):
        assert float(duals[k]) == float(e_node_dual_spacing_at(cells, k)), k


@pytest.mark.parametrize("name,axis", ALL)
def test_duals_reproduce_the_scalar_helper_at_every_node(name, axis):
    _check_duals(FIXTURES[name](), axis)


@pytest.mark.parametrize("name,axis", ALL)
def test_duals_match_the_vector_helper_to_float32_roundoff(name, axis):
    """``e_node_dual_spacings`` is the solver-facing spelling and goes through
    ``jnp``, which is float32 by default; ``duals`` stays in float64. The two
    agree to float32 roundoff and not to the bit -- the same gap
    ``test_nonuniform_source_port_dual_spacing`` already records between the
    vector and scalar helpers, inherited, not introduced here.
    """
    grid = FIXTURES[name]()
    vector = np.asarray(e_node_dual_spacings(grid.cells(axis)))
    np.testing.assert_allclose(vector, grid.duals(axis), rtol=1e-6, atol=0)


@pytest.mark.parametrize("name,axis", ALL)
def test_a_constant_axis_has_dual_equal_to_primal_bit_for_bit(name, axis):
    grid = FIXTURES[name]()
    if not grid.is_constant(axis):
        pytest.skip("axis is graded")
    assert np.array_equal(grid.duals(axis), grid.cells(axis))


# ---------------------------------------------------------------------------
# nodes and indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,axis", ALL)
def test_node_of_reproduces_the_coordinate_spine(name, axis):
    grid = FIXTURES[name]()
    n, pad_lo, _ = _layout(grid, axis)
    spine = np.asarray(getattr(coords_from_nonuniform_grid(grid), axis))
    for i in range(n):
        assert grid.node_of(axis, i) == float(spine[i]), i
    assert grid.node_of(axis, pad_lo) == 0.0


@pytest.mark.parametrize("name,axis", ALL)
def test_node_of_is_the_running_sum_of_the_cells(name, axis):
    """Written longhand in the test so the judge does not route through the
    same producer the accessor calls. A CONSTANT axis is exempt: it takes the
    closed form on purpose (#807), and the running sum differs from it in the
    last bits -- which is the point."""
    grid = FIXTURES[name]()
    if grid.is_constant(axis):
        pytest.skip("constant axis takes the closed form, see #807")
    n, pad_lo, _ = _layout(grid, axis)
    cells = grid.cells(axis)
    edges = np.insert(np.cumsum(cells), 0, 0.0)
    expected = edges[:-1] - edges[pad_lo]
    for i in range(n):
        assert grid.node_of(axis, i) == float(expected[i]), i


@pytest.mark.parametrize("name,axis", ALL)
def test_index_of_inverts_node_of_over_the_interior(name, axis):
    grid = FIXTURES[name]()
    n, pad_lo, pad_hi = _layout(grid, axis)
    for i in range(pad_lo, n - pad_hi):
        assert grid.index_of(axis, grid.node_of(axis, i)) == i, i


@pytest.mark.parametrize("name,axis", ALL)
def test_index_of_agrees_with_position_to_index(name, axis):
    """The accessor must land on the cell the existing lookup lands on, or a
    consumer moved onto it in 0b changes which cell it touches."""
    grid = FIXTURES[name]()
    ax = AXES.index(axis)
    n, pad_lo, pad_hi = _layout(grid, axis)
    for i in range(pad_lo, n - pad_hi):
        pos = [0.0, 0.0, 0.0]
        pos[ax] = grid.node_of(axis, i)
        assert grid.index_of(axis, pos[ax]) == position_to_index(
            grid, tuple(pos))[ax], i


@pytest.mark.parametrize("name,axis", ALL)
def test_boundary_cell_is_the_face_cell(name, axis):
    grid = FIXTURES[name]()
    cells = grid.cells(axis)
    assert grid.boundary_cell(axis, "lo") == float(cells[0])
    assert grid.boundary_cell(axis, "hi") == float(cells[-1])


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_scalar_fields_are_the_x_and_y_boundary_cells(name):
    """``dx``/``dy`` are documented as the BOUNDARY cell. Pinned so 0b's
    ``boundary_cell`` substitution is a rename and not a change of number."""
    grid = FIXTURES[name]()
    for axis, scalar in (("x", grid.dx), ("y", grid.dy)):
        assert grid.boundary_cell(axis, "lo") == scalar
        assert grid.boundary_cell(axis, "hi") == scalar


@pytest.mark.parametrize("name,axis", ALL)
def test_is_constant_says_what_the_cells_say(name, axis):
    grid = FIXTURES[name]()
    cells = grid.cells(axis)
    assert grid.is_constant(axis) is bool(np.all(cells == cells[0]))


def test_the_fixture_set_covers_both_answers():
    """A predicate that is never False on any fixture is not being tested."""
    seen = {grid.is_constant(axis)
            for name in FIXTURES for grid in [FIXTURES[name]()]
            for axis in AXES}
    assert seen == {True, False}


# ---------------------------------------------------------------------------
# Judge (iv) -- the constructor refuses cells that do not add up
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,axis", ALL)
def test_the_realized_interior_spans_the_declared_profile(name, axis):
    """What ``make_nonuniform_grid`` now asserts, restated on the built grid."""
    grid = FIXTURES[name]()
    _, pad_lo, pad_hi = _layout(grid, axis)
    realized = interior_cells(grid.cells(axis), pad_lo, pad_hi)
    edges = np.insert(np.cumsum(realized), 0, 0.0)
    assert grid.node_of(axis, pad_lo + realized.size) == pytest.approx(
        float(edges[-1]), abs=1e-12)


def test_mutation_a_a_bad_cell_array_is_refused():
    """(a) The check itself: hand it cells that do not add up.

    If the assertion were a no-op -- deleted, or weakened to a warning -- this
    goes red, because nothing else in the suite looks at the sum.
    """
    profile = np.full(10, 1e-3)
    padded = _append_bounding_node(_pad_profile(profile, 4, 4))
    tampered = padded.copy()
    tampered[6] *= 1.5        # one interior cell 50 % too wide
    with pytest.raises(ValueError, match="span"):
        assert_cells_span_declared_profile(tampered, 4, 4, profile, "z")
    # and the untampered array passes, so the check is not simply always red
    assert_cells_span_declared_profile(padded, 4, 4, profile, "z")


def test_mutation_b_the_pre_562_missing_bounding_node_is_refused():
    """(b) Helper calls kept, defect revived: drop ``_append_bounding_node``.

    Before #562 the padded array carried only N nodes for N cells, the
    stencil zeroed the last cell's H term, and the realized wall-to-wall
    extent came out one cell short -- +2.47 % of TM110 on a 45x39-cell PEC
    box, +37 MHz of WR-90's TE101 centre frequency from the guide-width axis
    alone. ``_pad_profile`` and ``interior_cells`` are both still called here;
    only the duplicate is gone.
    """
    profile = np.full(10, 1e-3)
    no_bounding_node = _pad_profile(profile, 4, 4)
    with pytest.raises(ValueError, match="interior cells"):
        assert_cells_span_declared_profile(no_bounding_node, 4, 4, profile, "z")


def test_mutation_b_an_extra_trailing_cell_is_refused():
    """(b) The same defect from the other side: the pre-#562 slice, which read
    ``d[pad_lo:len-pad_hi]`` and pulled the duplicate into the interior."""
    profile = np.full(10, 1e-3)
    doubled = np.concatenate(
        [_append_bounding_node(_pad_profile(profile, 4, 4)), profile[-1:]])
    with pytest.raises(ValueError, match="interior cells"):
        assert_cells_span_declared_profile(doubled, 4, 4, profile, "z")


# ---------------------------------------------------------------------------
# Judge (iv) -- a traced axis refuses to be indexed (G9)
# ---------------------------------------------------------------------------

def test_a_traced_axis_refuses_index_of_and_keeps_the_gradient():
    """Decision 6. ``position_to_index`` answers a traced axis by substituting
    a nominal uniform mesh of ``fallback_dx`` cells, so it returns an index
    for a mesh the solver is not running -- silently the wrong cell wherever
    the traced profile is graded (G9). The accessor refuses instead.

    Everything else on a traced axis keeps working, including the gradient:
    the mesh-as-design-variable path (#1190) differentiates through ``duals``,
    ``node_of`` and ``boundary_cell``.
    """
    seen = {}

    def f(profile):
        grid = make_nonuniform_grid((10e-3, 10e-3), profile, 5e-4,
                                    cpml_layers=6)
        seen["traced_z"] = grid.is_traced("z")
        seen["traced_x"] = grid.is_traced("x")
        seen["const_z"] = grid.is_constant("z")
        seen["const_x"] = grid.is_constant("x")
        with pytest.raises(ValueError, match="G9"):
            grid.index_of("z", 1e-3)
        # the concrete axes still answer
        assert grid.index_of("x", 0.0) == grid.pad_x_lo
        return (jnp.sum(grid.duals("z")) + grid.node_of("z", 10)
                + grid.boundary_cell("z", "lo"))

    profile = jnp.asarray(np.array([2e-4] * 6 + [5e-5] * 8 + [2e-4] * 6),
                          dtype=jnp.float32)
    value, grad = jax.value_and_grad(f)(profile)

    assert seen == {"traced_z": True, "traced_x": False,
                    "const_z": False, "const_x": True}
    assert np.isfinite(float(value))
    grad = np.asarray(grad)
    assert np.all(np.isfinite(grad))
    assert float(np.abs(grad).sum()) > 0.0, (
        "no gradient reached the traced profile, so the accessors broke the "
        "mesh-as-design-variable path they are supposed to leave alone"
    )


def test_a_traced_axis_is_not_reported_constant():
    """Constancy is undecidable on the host for a tracer, and ``False`` is the
    safe answer: decision 4 routes a non-constant axis to the non-uniform
    kernel, which is correct for a constant mesh too, while ``True`` would
    hand a possibly graded mesh to a kernel that folds one scalar."""
    def f(profile):
        grid = make_nonuniform_grid((10e-3, 10e-3), profile, 5e-4,
                                    cpml_layers=0)
        assert grid.is_constant("z") is False
        return jnp.sum(grid.cells("z"))

    jax.jit(f)(jnp.full(8, 5e-4, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# A grid built by hand has no float64 spine
# ---------------------------------------------------------------------------

def test_a_hand_built_grid_warns_instead_of_silently_degrading():
    """``NonUniformGrid(...)`` called directly leaves the spine fields None.
    ``cells`` then widens the float32 store and says so -- the same warning
    ``coords_from_nonuniform_grid`` emits, for the same reason: the pre-#802
    node line is ~1e-10 m off, which flips half-open inclusion at
    node-aligned faces."""
    built = FIXTURES["api_source"]()
    by_hand = built._replace(dx_arr_f64=None, dy_arr_f64=None, dz_f64=None)
    with pytest.warns(ExactNodeSpineMissingWarning):
        widened = by_hand.cells("z")
    assert widened.dtype == np.float64
    assert np.array_equal(widened, np.asarray(built.dz, dtype=np.float64))
    assert isinstance(by_hand, NonUniformGrid)


# ---------------------------------------------------------------------------
# Mutations on the metric rules -- (b), helper calls kept
# ---------------------------------------------------------------------------

def test_mutation_b_duals_revived_to_the_primal_goes_red(monkeypatch):
    """(b) The #669 defect: fold a sheet or a port by the PRIMAL cell instead
    of the dual spacing. Every helper call stays; only the rule inside
    ``dual_spacings_from_cells`` changes, which is what a refactor would
    plausibly get wrong.

    On a graded axis the two differ wherever the neighbouring cells do -- a
    Leontovich sheet stamped on a 0.25/0.50 mm transition node measured an
    attenuation ratio of 1.2021 against the matched-mesh case, where a
    mesh-independent sheet must give 1.000.

    The separation is asserted, not assumed, so a fixture change that made
    the two rules nearly agree could not leave this test passing on a
    coincidence. On the band-builder x axis, 14 of its 59 entries differ,
    the largest by 109.08 um and the smallest by 23.77 um, against cells
    that run 200 um to 1 mm.
    """
    grid = FIXTURES[GRADED_BAND]()
    cells = grid.cells("x")
    duals = grid.duals("x")
    differ = duals != cells
    gaps = np.abs(duals - cells)[differ]
    assert cells.size == 59
    assert int(differ.sum()) == 14
    assert gaps.max() == pytest.approx(1.09077e-4, rel=1e-4)
    assert gaps.min() == pytest.approx(2.37703e-5, rel=1e-4)

    _check_duals(grid, "x")          # green before the mutation

    monkeypatch.setattr("rfx.nonuniform._dual_spacings_from_cells",
                        lambda cells: np.asarray(cells, dtype=np.float64))
    with pytest.raises(AssertionError):
        _check_duals(grid, "x")


def test_mutation_b_the_leading_dual_end_rule_goes_red(monkeypatch):
    """(b) ``dual[0] = d[0]`` replaced by the general mean ``(d[0]+d[1])/2``.

    Only visible on an axis whose first two cells differ, which is why the
    fixture set carries a pad-free graded-end shape: an absorber pad
    replicates the boundary cell, so on every padded fixture ``d[0] == d[1]``
    and the mutation is silent. Recorded here because a judge that cannot see
    a defect on the fixtures it runs on is not a judge.
    """
    def defective(cells):
        d = np.asarray(cells, dtype=np.float64)
        out = np.empty_like(d)
        out[0] = 0.5 * (d[0] + d[1])
        out[1:] = 0.5 * (d[:-1] + d[1:])
        return out

    padded = FIXTURES["api_source"]()
    pad_free = FIXTURES["padfree_graded_end"]()
    monkeypatch.setattr("rfx.nonuniform._dual_spacings_from_cells", defective)

    _check_duals(padded, "z")        # silent: the pad makes d[0] == d[1]
    with pytest.raises(AssertionError):
        _check_duals(pad_free, "z")


def test_mutation_b_the_core_c2_metric_swap_goes_red():
    """(b) The pre-CORE-C2 defect: give the E update the local cell width and
    the H update the mean. Both arrays still come from
    ``_profile_to_inv_arrays``-shaped arithmetic on the same cells; only which
    metric goes where is swapped, which scaled the curl by
    ``2 d[k]/(d[k]+d[k+-1])`` on every graded cell.
    """
    grid = FIXTURES["padfree_graded_end"]()
    cells = grid.cells("z")
    arr = jnp.asarray(cells, dtype=jnp.float32)
    inv_local = 1.0 / arr
    inv_mean = 2.0 / (arr[:-1] + arr[1:])
    swapped_e = inv_local
    swapped_h = jnp.concatenate([inv_mean, jnp.zeros(1, dtype=jnp.float32)])

    good_e, good_h = _profile_to_inv_arrays(cells)
    assert np.array_equal(np.asarray(good_e), np.asarray(grid.inv_dz))
    assert not np.array_equal(np.asarray(swapped_e), np.asarray(grid.inv_dz))
    assert not np.array_equal(np.asarray(swapped_h), np.asarray(grid.inv_dz_h))


def test_mutation_b_the_trailing_h_end_rule_goes_red():
    """(b) ``inv_d_h[N-1] = 1/d[N-1]`` instead of 0. The forward difference has
    no ``d[N]``, so a nonzero entry there differences against a node that does
    not exist."""
    grid = FIXTURES["padfree_graded_end"]()
    cells = grid.cells("z")
    _, good_h = _profile_to_inv_arrays(cells)
    revived = np.asarray(good_h).copy()
    revived[-1] = np.float32(1.0) / np.float32(cells[-1])
    assert float(revived[-1]) != 0.0
    assert not np.array_equal(revived, np.asarray(grid.inv_dz_h))


# ---------------------------------------------------------------------------
# Recorded facts -- measured, not asserted as contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,axis", ALL)
def test_the_widened_float32_store_is_not_the_spine(name, axis):
    """Two consumers reach the dual rule through the float32 store widened
    back to float64 (``rfx/sources/sources.py:31-41``,
    ``rfx/sources/msl_port.py:446-453``) rather than through the float64
    spine ``cells`` returns. This records how far apart the two spellings
    are, so 0b's bit-identity judges are read against a known figure rather
    than against an assumption that they agree.

    The bound is generous on purpose: this is a measurement, not a contract.
    The size of the gap is pinned in
    ``test_the_spine_and_the_store_differ_by_the_recorded_amount``.
    """
    grid = FIXTURES[name]()
    store = np.asarray(
        (grid.dx_arr, grid.dy_arr, grid.dz)[AXES.index(axis)],
        dtype=np.float64)
    spine = grid.cells(axis)
    np.testing.assert_allclose(store, spine, rtol=1e-6, atol=0)

    from_store = np.array([float(e_node_dual_spacing_at(store, k))
                           for k in range(store.size)])
    np.testing.assert_allclose(from_store, grid.duals(axis), rtol=1e-6,
                               atol=0)


@pytest.mark.parametrize("name,axis", ALL)
def test_the_inverse_arrays_cannot_tell_the_spine_from_the_store(name, axis):
    """A limit of the judge above, measured rather than assumed.

    ``_profile_to_inv_arrays`` casts to float32 before it does anything else,
    and on all ten fixture shapes the float64 spine and the widened float32
    store round to the same float32 cells -- so the inverse-array half of
    judge (ii) is BLIND to which of the two ``cells`` returns. It pins that
    the derivation rule is right, not that the source array is.

    What pins the source is
    ``test_cells_is_the_float64_spine_over_the_padded_axis``, which compares
    against the grid's own float64 field directly, and the dual rule, which
    is computed in float64 and does see the difference. The size of that
    difference is measured in
    ``test_the_spine_and_the_store_differ_by_the_recorded_amount``.

    A red result here is not a regression. It means some fixture's two
    sources now round apart, the inverse judge has become discriminating, and
    this docstring should say so.
    """
    grid = FIXTURES[name]()
    store = np.asarray(
        (grid.dx_arr, grid.dy_arr, grid.dz)[AXES.index(axis)],
        dtype=np.float64)
    inv_e, inv_h = _inv_pair(grid, axis)
    from_store_e, from_store_h = _profile_to_inv_arrays(store)
    assert np.array_equal(np.asarray(from_store_e), np.asarray(inv_e))
    assert np.array_equal(np.asarray(from_store_h), np.asarray(inv_h))


def test_the_spine_and_the_store_differ_by_the_recorded_amount():
    """How far apart the float64 spine and the widened float32 store are.

    Two consumers reach the dual rule through the store rather than the
    spine, so this is the figure 0b's bit-identity judges are read against.
    It is measured over every entry of every fixture axis at once, because
    per-axis it is a handful of last bits and the population is what makes
    it meaningful.

    Every one of the 1201 entries differs in float64 arithmetic; only 10 are
    still apart once both spellings are rounded to the float32 the solver
    stores. The largest float64 gap among those ten is 2.09e-11 m; 4.75e-11 m
    is the largest over all 1201, on meshes whose cells run 50 um to 1 mm.
    Relative to the smallest cell that is about 4e-7 -- far below anything a
    user reads, and far above zero, which is why the dual judge sees the
    source array and the inverse-metric judge does not.
    """
    total = 0
    differ_in_f32 = 0
    equal_in_f64 = 0
    gaps_of_the_differing = []
    gaps_overall = []
    for name in sorted(FIXTURES):
        grid = FIXTURES[name]()
        for axis in AXES:
            store = np.asarray(
                (grid.dx_arr, grid.dy_arr, grid.dz)[AXES.index(axis)],
                dtype=np.float64)
            from_spine = grid.duals(axis)
            from_store = np.array(
                [float(e_node_dual_spacing_at(store, k))
                 for k in range(store.size)])
            gaps = np.abs(from_spine - from_store)
            apart = (from_spine.astype(np.float32)
                     != from_store.astype(np.float32))
            total += store.size
            differ_in_f32 += int(apart.sum())
            equal_in_f64 += int((from_spine == from_store).sum())
            gaps_overall.append(gaps.max())
            if apart.any():
                gaps_of_the_differing.append(gaps[apart].max())

    assert total == 1201, (
        f"the fixture population moved to {total} entries; the figures in "
        "this docstring and in the PR body were measured over 1201"
    )
    assert equal_in_f64 == 0
    assert differ_in_f32 == 10
    assert max(gaps_of_the_differing) == pytest.approx(2.092e-11, rel=1e-3)
    assert max(gaps_overall) == pytest.approx(4.750e-11, rel=1e-3)


@pytest.mark.parametrize("name,axis", ALL)
def test_index_of_refuses_a_coordinate_outside_the_domain(name, axis):
    """A position the grid has no node for is an error, not the nearest face.

    ``argmin`` over an edge list always returns something, so the legacy
    ``position_to_index`` answers a coordinate a metre outside a 6 mm domain
    with the last face index, and whatever the caller was placing lands
    there. The accessor refuses instead, which is what the uniform grid has
    always done.
    """
    grid = FIXTURES[name]()
    n, pad_lo, pad_hi = _layout(grid, axis)
    last = n - pad_hi - 1
    span = grid.node_of(axis, last)

    for outside in (-1.0, 1e9, span + span + 1e-3):
        with pytest.raises(ValueError, match="outside"):
            grid.index_of(axis, outside)

    # Both ends inclusive, and float dust at a face is not "outside": the
    # grid only claims to place a node to 1e-12 m, so it cannot tell a
    # coordinate 5e-13 m past the face from the face.
    assert grid.index_of(axis, 0.0) == pad_lo
    assert grid.index_of(axis, span) == last
    assert grid.index_of(axis, span + 5e-13) == last
    assert grid.index_of(axis, -5e-13) == pad_lo


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_legacy_lookup_still_clamps(name):
    """0a moves no consumer, so ``position_to_index`` keeps its clamp.

    Pinned as a fact, not endorsed: this is the behaviour the accessor
    replaces, and 0b retires it. If this test goes red because the clamp is
    gone, the accessor's refusal is the replacement and this test should go
    with it.
    """
    grid = FIXTURES[name]()
    far = position_to_index(grid, (1e9, 1e9, 1e9))
    assert all(isinstance(i, int) for i in far)
    with pytest.raises(ValueError, match="outside"):
        grid.index_of("z", 1e9)
