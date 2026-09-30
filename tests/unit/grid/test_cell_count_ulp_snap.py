"""``cells_spanning`` absorbs float dust and nothing else (issue #1070).

A declared domain length is an arithmetic expression. ``38*h + 2*(pad*h)``
can land one ULP above an exact multiple of ``dx``, and ``ceil`` then buys a
cell the declared domain does not reach. Nothing fills it -- it is outside
every declared Box -- and at a face abutting an absorber the CPML pad
extension replicates that vacuum outward, so the structure ends in a vacuum
facet inside its own absorber.

One test per row of the corner-case table in the PR, because the risk in a
tolerance is not the case it was written for but the case it swallows by
accident. The two sides that matter are named explicitly everywhere:
what must snap DOWN to the integer, and what must still take the cell.

Nothing here builds a grid; these are statements about one arithmetic rule.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx._grid_metric import NODE_TIE_REL
from rfx.grid import CELL_COUNT_ULP_BUDGET, cells_spanning

EPS = float(np.finfo(float).eps)

#: The #1070 rig: RO4003C at h = 0.787 mm, dx = h/4, domain 38h + 2*(pad*h).
H = 0.787e-3
DX = H / 4.0


def _ratio(n: int, ulps: int) -> float:
    """``n`` displaced upward by exactly *ulps* ULPs."""
    value = float(n)
    for _ in range(ulps):
        value = math.nextafter(value, math.inf)
    return value


# --- the band the rule absorbs, and where it stops -------------------------

@pytest.mark.parametrize("k", [1, 2, 4, 8, 14])
def test_a_few_ulps_above_an_integer_is_float_dust(k: int) -> None:
    """Up to 14 ULP at r = 232: the summed-expression band."""
    assert cells_spanning(_ratio(232, k) * DX, DX) == 232


@pytest.mark.parametrize("k", [15, 32, 1000])
def test_old_ulp_cutoffs_inside_the_node_band_do_not_add_a_cell(k: int) -> None:
    """#1138 gives sizing the Box's 1e-9-cell band: all three now need 232.

    These exceed the old 14.5-ULP limit but are still on the Box's face.
    """
    assert cells_spanning(_ratio(232, k) * DX, DX) == 232


def test_the_ulp_budget_remains_a_floor_for_large_cell_counts() -> None:
    """At a million cells 14 ULP exceeds 1e-9 cell but fits the old budget."""
    r = _ratio(1000000, 14)
    assert r - 1000000 > NODE_TIE_REL
    assert cells_spanning(r * DX, DX, ulp_budget=CELL_COUNT_ULP_BUDGET) == 1000000
    assert cells_spanning(r * DX, DX, ulp_budget=1) == 1000001


@pytest.mark.parametrize("outside,expected", [(False, 232), (True, 233)])
def test_the_node_band_boundary_keeps_a_real_overhang(outside, expected) -> None:
    """Representable lengths bracket the shared 1e-9-cell band."""
    length = np.nextafter(232 + NODE_TIE_REL, np.inf if outside else 232)
    assert (length - 232 > NODE_TIE_REL) == outside
    assert cells_spanning(length, 1.0) == expected


# --- the defect itself -----------------------------------------------------

def test_the_rig_that_found_this_sizes_to_the_declared_length() -> None:
    """``38*h + 2*(10*h)`` over ``h/4``: 232.00000000000003, one ULP up.

    ``ceil`` gives 233 and the extra node is the #1070 vacuum facet.
    """
    length = 38 * H + 2 * (10 * H)
    assert length / DX == 232.00000000000003, repr(length / DX)
    assert math.ceil(length / DX) == 233
    assert cells_spanning(length, DX) == 232


@pytest.mark.parametrize("pad,axis_terms,expected", [
    (6, 38, 200), (8, 38, 216), (10, 38, 232), (12, 38, 248),
    (6, 23, 140), (8, 23, 156), (10, 23, 172), (12, 23, 188),
])
def test_every_rig_in_the_issue_sizes_to_its_declared_length(
        pad: int, axis_terms: int, expected: int) -> None:
    """All eight axis/pad combinations of the issue's table, including the
    four whose ratio is already exact and must not move."""
    length = axis_terms * H + 2 * (pad * H)
    assert cells_spanning(length, DX) == expected


# --- what must NOT be absorbed --------------------------------------------

def test_a_genuine_half_cell_still_takes_the_cell() -> None:
    """``round`` alone would give 232 here, twice over: 0.5 is exactly
    representable and Python rounds halves to even. The tolerance test is
    what rejects it, which is why the rule is not ``round`` with a guard."""
    assert round(232.5) == 232
    assert cells_spanning(232.5 * DX, DX) == 233


@pytest.mark.parametrize("excess,label", [
    (0.01, "a hundredth of a cell"),
    (1e-6 * 232, "one part in a million of the length"),
    (1e-9 * 232, "one part in a billion of the length"),
])
def test_a_declared_sub_cell_overhang_still_takes_the_cell(
        excess: float, label: str) -> None:
    """The smallest overhang is 2.32e-7 cells, 232 times the #1138 band."""
    assert cells_spanning((232.0 + excess) * DX, DX) == 233, label


def test_a_large_domain_keeps_dust_but_not_a_real_overhang() -> None:
    """Three ULP of dust snaps; a declared 0.001-cell overhang still grows.

    #1138 uses the larger of the node band and the arithmetic ULP budget.
    """
    dust = _ratio(1000000, 3)
    assert cells_spanning(dust * DX, DX) == 1000000
    assert round(dust, 9) == 1000000.0        # an absolute round agrees here
    real = 1000000.0 * (1 + 1e-9)
    assert cells_spanning(real * DX, DX) == 1000001
    assert round(real, 9) == 1000000.001      # ... and disagrees here


# --- the small and degenerate end -----------------------------------------

def test_just_below_an_integer_is_unchanged() -> None:
    """The WR-90 taper: 22.86/1.27 comes out just UNDER 18, and ``ceil``
    already gives 18. The rule must not turn that into 17."""
    assert 22.86e-3 / 1.27e-3 == 17.999999999999996
    assert cells_spanning(22.86e-3, 1.27e-3) == 18


@pytest.mark.parametrize("ratio,expected", [
    (0.9999999999999999, 1),
    (0.5, 1),
    (0.01, 1),
])
def test_a_domain_smaller_than_one_cell_still_gets_a_cell(
        ratio: float, expected: int) -> None:
    """A positive fraction of a cell still needs at least one cell."""
    assert cells_spanning(ratio * DX, DX) == expected


@pytest.mark.parametrize("ratio", [0.0, -1.0, -232.00000000000003])
def test_zero_and_negative_lengths_behave_as_they_did(ratio: float) -> None:
    """Stated, not fixed. ``Grid`` does not validate a non-positive domain
    today and this change is not where that starts; the rule returns what
    ``ceil`` returned, so nothing downstream sees a new value."""
    assert cells_spanning(ratio * DX, DX) == int(math.ceil(ratio))


@pytest.mark.parametrize("ratio", [1e-17, 1e-16, 1.7e-15, 1.8e-15, 1e-9])
def test_a_positive_length_below_the_dust_band_still_gets_a_cell(
        ratio: float) -> None:
    """Zero is reachable from above (review of PR #1136, G).

    For ``0 < r <= 1e-9`` the nearest integer is 0 and the dust test passes,
    so an unguarded snap returned 0 cells for a real length where ``ceil``
    returned 1. Everywhere else the snap gives back a cell the ratio did not
    need; here it would have taken away one the length does need, which is a
    different thing and not what the rule is for.
    """
    assert cells_spanning(ratio * DX, DX) == 1 == int(math.ceil(ratio))


# --- cell sizes that are not exactly representable -------------------------

@pytest.mark.parametrize("dx,terms,expected", [
    (1.27e-3, 60, 60),
    (1.27e-3, 18, 18),
    (84.6667e-6, 40, 40),
    (84.6667e-6, 58, 58),
    (0.787e-3 / 4, 232, 232),
])
def test_an_exact_multiple_of_an_inexact_dx_sizes_exactly(
        dx: float, terms: int, expected: int) -> None:
    """Both sides of the integer, on cell sizes that are not representable in
    binary. ``terms * dx`` is the length a script actually writes."""
    assert cells_spanning(terms * dx, dx) == expected
    assert cells_spanning((terms + 0.5) * dx, dx) == expected + 1
