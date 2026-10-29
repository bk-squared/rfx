"""Recorded boundary-value changes and removals must be rejected.
"""

from copy import deepcopy
import json

import pytest

from tests.contracts.boundary_baseline import ROOT
from tests.contracts.boundary_cases import build
from tests.contracts.boundary_compare import compare_values


BASELINE = json.loads((ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.json").read_text())


@pytest.mark.parametrize("quantity,before,after,direction", [
    ("h_plane_m", .0005, .0015, "away"),
    ("h_plane_m", .0005, 0, "toward"),
    ("h_zero_planes_m", [.0005], [.0005, .0015], "away"),
    ("h_zero_planes_m", [.0005], [0], "toward"),
    ("e_zero_planes_m", [0], [.001], "away"),
    ("e_plane_m", 0, .001, "away"),
    ("period_m", .025, .026, "away"),
    ("period_m", .025, .024, "toward"),
])
def test_recorded_value_change_fails_outside_departure_xfail(quantity, before, after, direction):
    case = "periodic-xy" if quantity == "period_m" else "pec" if quantity.startswith("e_") else "pmc-pec"
    baseline = deepcopy(next(c["faces"] for c in BASELINE["cells"] if c["case"] == case and c["entry"] == "run"))
    baseline["x_lo"][quantity] = before
    if quantity == "h_plane_m":
        baseline["x_lo"]["h_zero"] = True  # synthetic legacy wall for the comparison guard
    measured = deepcopy(baseline)
    measured["x_lo"][quantity] = after
    sim, _ = build(case, "run")
    with pytest.raises(AssertionError, match=direction) as exc:
        compare_values(sim.boundary_model(), sim._build_grid(), measured, baseline)
    assert type(exc.value) is AssertionError
    assert "the step" in str(exc.value).lower() and "--update" in str(exc.value)


@pytest.mark.parametrize("quantity,empty", [("e_zero_planes_m", []), ("h_zero_planes_m", []), ("period_m", None)])
def test_removed_recorded_value_fails(quantity, empty):
    case = "periodic-xy" if quantity == "period_m" else "pec" if quantity.startswith("e_") else "pmc-pec"
    baseline = deepcopy(next(c["faces"] for c in BASELINE["cells"] if c["case"] == case and c["entry"] == "run"))
    if quantity == "h_zero_planes_m":
        baseline["x_lo"][quantity] = [.0005]  # synthetic recorded legacy plane
    measured = deepcopy(baseline)
    measured["x_lo"][quantity] = empty
    sim, _ = build(case, "run")
    with pytest.raises(AssertionError, match="presence changed"):
        compare_values(sim.boundary_model(), sim._build_grid(), measured, baseline)
