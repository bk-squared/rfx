"""B1 refusal prefixes and reproducible record writes."""

from copy import deepcopy
import json

import pytest

from tests.contracts.boundary_baseline import ROOT, refusal_prefix, validate_class_changes, write_baseline
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
    baseline = next(c["faces"] for c in BASELINE["cells"] if c["case"] == case and c["entry"] == "run")
    measured = deepcopy(baseline)
    measured["x_lo"][quantity] = empty
    sim, _ = build(case, "run")
    with pytest.raises(AssertionError, match="presence changed"):
        compare_values(sim.boundary_model(), sim._build_grid(), measured, baseline)


def test_regeneration_accepts_only_the_two_backing_corrections():
    current = deepcopy(next(c for c in BASELINE["cells"] if (c["case"], c["entry"]) == ("tfsf", "distributed")))
    current["departures"] = [d for d in current["departures"] if (d["face"], d["code"]) not in
                              (("x_lo", "f"), ("x_hi", "f"))]
    old = deepcopy(current)
    for face in ("x_lo", "x_hi"):
        old["departures"].append(dict(face=face, code="f"))
        old["faces"][face]["e_zero"] = False
        current["faces"][face]["e_zero"] = True
    previous = dict(cells=[old])
    validate_class_changes(previous, [current])
    validate_class_changes(dict(cells=[current]), [current])
    current["departures"].append(dict(face="y_lo", code="a"))
    with pytest.raises(AssertionError, match="STOP.*classification changed"):
        validate_class_changes(previous, [current])


def test_regeneration_rejects_face_class_change_without_departure_change():
    old = next(c for c in BASELINE["cells"] if (c["case"], c["entry"]) == ("pec", "run"))
    current = deepcopy(old)
    current["faces"]["x_lo"]["h_zero"] = not old["faces"]["x_lo"]["h_zero"]
    with pytest.raises(AssertionError, match="STOP.*h_zero"):
        validate_class_changes(dict(cells=[old]), [current])


def test_baseline_rewrite_keeps_both_leader_conclusions(tmp_path):
    baseline = deepcopy(BASELINE)
    baseline["conclusion"] = "Leader's JSON text.\nKeep punctuation and spacing."
    baseline["kernel_base_commit"] = "new-base"
    markdown = (ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.md").read_text()
    markdown = markdown.rsplit("\n\n", 1)[0] + "\n\nLeader's Markdown text.  Keep two spaces.\n"
    write_baseline(baseline, markdown, tmp_path)
    assert json.loads((tmp_path / "MATRIX.json").read_text())["conclusion"] == baseline["conclusion"]
    updated = (tmp_path / "MATRIX.md").read_text()
    assert updated.split("## Stopped comparisons", 1)[1] == markdown.split("## Stopped comparisons", 1)[1]
    assert "Kernel base: `new-base`" in updated


def test_refusal_prefix_ignores_count_and_explanation():
    assert refusal_prefix("[run] preflight found 1 blocking error(s): old explanation") == "[run] preflight found"
    assert refusal_prefix("[run] preflight found 2 blocking error(s): new explanation") == "[run] preflight found"
    with pytest.raises(AssertionError, match="Unregistered refusal"):
        refusal_prefix("unrelated numerical error")
