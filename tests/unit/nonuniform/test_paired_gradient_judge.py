"""Independent analytic controls for the separately declared paired judge."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from validation.research.multiband_nu import adq_designvar as legacy
from validation.research.multiband_nu.paired_gradient import (
    InvalidLadder,
    LossPair,
    judge_direction,
)


def _samples(function: Callable[[float], float]) -> list[LossPair]:
    return [
        {
            "h": h,
            "loss_plus": float(np.float32(function(h))),
            "loss_minus": float(np.float32(function(-h))),
        }
        for h in (2.0 ** -k for k in range(17, 2, -1))
    ]


@pytest.mark.parametrize(
    ("derivative", "expected"),
    [(1.0, "HELD"), (1.1, "FIRED"), (0.0, "FIRED")],
)
def test_affine_reference_resolves_correct_wrong_and_zero_gradients(
    derivative: float, expected: str,
) -> None:
    # Given: an independently specified affine function, exactly dyadic in f32.
    points = _samples(lambda h: 1.0 + h)
    assert legacy.fd_budget(points, derivative)["verdict"] == "INCONCLUSIVE"

    # When: the new protocol judges the unchanged samples and proposed gradient.
    result = judge_direction(points, 1.0, derivative)

    # Then: resolving power comes from the reference, not the proposed gradient.
    assert result.verdict == expected
    assert result.eligible_indices
    assert {row.model for row in result.windows if row.eligible} == {"plateau"}


@pytest.mark.parametrize(("factor", "expected"), [(1.0, "HELD"), (1.1, "FIRED")])
def test_smooth_polynomial_against_its_analytic_derivative(factor: float, expected: str) -> None:
    # Given: the derivative is the polynomial's linear coefficient, not an FD.
    points = _samples(lambda h: 0.1306 + 0.37*h + 2.9*h*h + 0.5*h**3)
    # When.
    result = judge_direction(points, float(np.float32(0.1306)), factor*0.37)
    # Then.
    assert result.verdict == expected


@pytest.mark.parametrize("derivative", [1.0, 1.1, 0.0])
def test_cubic_curvature_and_direction_reversal(derivative: float) -> None:
    # Given: curvature changes the one-sided residual, not derivative correctness.
    points = _samples(lambda h: 1.0 + h + 10*h*h + 1000*h**3)
    reverse: list[LossPair] = [
        {"h": p["h"], "loss_plus": p["loss_minus"], "loss_minus": p["loss_plus"]}
        for p in points
    ]
    # When: reverse the direction and candidate together, preserving all noise.
    positive = judge_direction(points, 1.0, derivative)
    negative = judge_direction(reverse, 1.0, -derivative)
    # Then: no choice of direction can rescue or condemn the same derivative.
    assert positive.verdict == ("HELD" if derivative == 1.0 else "FIRED")
    assert negative.verdict == positive.verdict
    assert negative.eligible_indices == positive.eligible_indices
    assert negative.narrowest_index == positive.narrowest_index
    assert [r.band for r in negative.windows] == [r.band for r in positive.windows]


def test_reference_qualification_does_not_consult_the_candidate() -> None:
    # Given: one fixed primal ladder and candidates with radically different sizes.
    points = _samples(lambda h: 1.0 + h + 10*h*h + 1000*h**3)
    # When.
    results = [judge_direction(points, 1.0, g) for g in (1.0, 0.0, 1e6)]
    # Then: reference formation is identical even when the verdict changes.
    assert results[0].eligible_indices
    assert all(r.eligible_indices == results[0].eligible_indices for r in results)
    assert all(r.reference_interval == results[0].reference_interval for r in results)
    assert all(r.narrowest_index == results[0].narrowest_index for r in results)
    assert [r.verdict for r in results] == ["HELD", "FIRED", "FIRED"]


@pytest.mark.parametrize("factor", [2.0**-10, 2.0**10])
def test_power_of_two_loss_units_preserve_the_judgment(factor: float) -> None:
    # Given: the same mathematical observable in different loss units.
    points = _samples(lambda h: 1.0 + h + 10*h*h + 1000*h**3)
    scaled: list[LossPair] = [
        {"h": p["h"], "loss_plus": factor*p["loss_plus"],
         "loss_minus": factor*p["loss_minus"]} for p in points
    ]
    # When: scale both the derivative and measured floor with the observable.
    original = judge_direction(points, 1.0, 1.0, sigma=1e-7)
    changed = judge_direction(scaled, factor, factor, sigma=factor*1e-7)
    # Then.
    assert changed.verdict == original.verdict == "HELD"
    assert changed.eligible_indices == original.eligible_indices
    assert changed.narrowest_index == original.narrowest_index
    assert [r.band for r in changed.windows] == [factor*r.band for r in original.windows]


def test_kink_is_inconclusive_despite_an_exact_central_difference() -> None:
    # Given: the central quotient is exactly 1, but no derivative exists at zero.
    points = _samples(lambda h: 1.0 + h + abs(h)/2)
    # When.
    result = judge_direction(points, 1.0, 1.0)
    # Then: the one-sided gap cannot be mistaken for a smooth affine response.
    assert result.verdict == "INCONCLUSIVE"
    assert not result.eligible_indices
    assert any(not row.smooth for row in result.windows)


@pytest.mark.parametrize(
    ("function", "derivative"),
    [(lambda h: 1.0 + 1e-9*h, 1e-9), (lambda h: 1.0 + h*h, 0.0)],
)
def test_unresolved_or_zero_reference_is_inconclusive(
    function: Callable[[float], float], derivative: float,
) -> None:
    # Given: either f32 cannot resolve the response, or the reference is zero.
    points = _samples(function)
    # When.
    result = judge_direction(points, 1.0, derivative)
    # Then: no unearned gradient verification.
    assert result.verdict == "INCONCLUSIVE"
    assert not result.eligible_indices


def test_unreliable_floor_cannot_produce_a_verdict() -> None:
    # Given: an otherwise resolving affine response with rejected floor evidence.
    points = _samples(lambda h: 1.0 + h)
    # When.
    result = judge_direction(points, 1.0, 1.0, sigma=1e-7, floor_reliable=False)
    # Then.
    assert result.verdict == "INCONCLUSIVE"
    assert result.reason == "unreliable_floor"
    assert not result.eligible_indices


def test_disagreeing_reference_windows_are_not_an_ad_failure() -> None:
    # Given: two disjoint, individually resolving FD plateaus.
    points = _samples(lambda h: 1.0 + h)
    for p in points[7:]:
        p["loss_plus"], p["loss_minus"] = 1.0 + 2*p["h"], 1.0 - 2*p["h"]
    # When: one candidate agrees with only one plateau.
    result = judge_direction(points, 1.0, 1.0)
    # Then: the contradictory reference is not allowed to judge AD.
    assert result.verdict == "INCONCLUSIVE"
    assert result.reason == "inconsistent_reference"


def test_result_owns_samples_and_keeps_every_reference_candidate() -> None:
    # Given: a caller-owned mutable ladder.
    points = _samples(lambda h: 1.0 + h)
    expected = (points[0]["h"], points[0]["loss_plus"], points[0]["loss_minus"])
    # When: change caller data after judging.
    result = judge_direction(points, 1.0, 1.0)
    points[0]["loss_plus"] = 123.0
    # Then: neither retained input nor rejected-window evidence disappears.
    assert result.samples[0] == expected
    assert len(result.windows) == len(points) - 2


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_step_is_an_error(step: float) -> None:
    # Given.
    points = _samples(lambda h: 1.0 + h)
    points[3]["h"] = step
    # When / Then.
    with pytest.raises(InvalidLadder) as error:
        judge_direction(points, 1.0, 1.0)
    assert error.value.field == "h"


def test_non_dyadic_ladder_is_an_error() -> None:
    # Given.
    points = _samples(lambda h: 1.0 + h)
    points[3]["h"] *= 1.25
    # When / Then.
    with pytest.raises(InvalidLadder):
        judge_direction(points, 1.0, 1.0)


@pytest.mark.parametrize("sigma", [-1.0, float("nan"), float("inf")])
def test_invalid_floor_is_an_error(sigma: float) -> None:
    # Given.
    points = _samples(lambda h: 1.0 + h)
    # When / Then.
    with pytest.raises(InvalidLadder) as error:
        judge_direction(points, 1.0, 1.0, sigma=sigma)
    assert error.value.field == "sigma"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2*float(np.finfo(np.float32).max)])
def test_nonfinite_loss_is_an_error(value: float) -> None:
    # Given.
    points = _samples(lambda h: 1.0 + h)
    points[3]["loss_plus"] = value
    # When / Then.
    with pytest.raises(InvalidLadder) as error:
        judge_direction(points, 1.0, 1.0)
    assert error.value.field == "loss"


@pytest.mark.parametrize("derivative", [float("nan"), float("inf")])
def test_nonfinite_candidate_is_an_error(derivative: float) -> None:
    # Given.
    points = _samples(lambda h: 1.0 + h)
    # When / Then.
    with pytest.raises(InvalidLadder) as error:
        judge_direction(points, 1.0, derivative)
    assert error.value.field == "derivative"
