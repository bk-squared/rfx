import json

import numpy as np
import pytest

import rfx
from rfx.pareto import (
    ParetoFront,
    ParetoPoint,
    epsilon_constraint_mask,
    pareto_front,
    pareto_mask,
    select_epsilon_constrained,
    weighted_scalarization,
)
from rfx.sweep import SweepResult


def test_pareto_mask_handles_mixed_minimize_maximize_objectives():
    # Columns: |S11| to minimize, bandwidth to maximize, footprint to minimize.
    objectives = np.array([
        [0.20, 1.0, 10.0],  # dominated by B
        [0.10, 1.5, 9.0],   # B: non-dominated
        [0.08, 1.0, 12.0],  # C: trades better S11 for worse area/bandwidth
        [0.12, 2.0, 11.0],  # D: trades bandwidth for worse S11/area
        [0.10, 1.5, 9.0],   # duplicate non-dominated point
    ])

    mask = pareto_mask(objectives, minimize=(True, False, True))

    assert mask.tolist() == [False, True, True, True, True]

def test_pareto_mask_handles_all_maximize_and_tolerance_boundaries():
    maximize_objectives = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    close_minimize = np.array([
        [1.00],
        [1.05],
    ])

    assert pareto_mask(maximize_objectives, minimize=False).tolist() == [
        False,
        True,
        True,
    ]
    assert pareto_mask(close_minimize, atol=0.0).tolist() == [True, False]
    assert pareto_mask(close_minimize, atol=0.10).tolist() == [True, True]



def test_pareto_front_serializes_ids_metadata_and_indices():
    objectives = np.array([
        [0.20, 1.0],
        [0.10, 1.5],
        [0.08, 1.0],
    ])

    front = pareto_front(
        objectives,
        minimize=(True, False),
        ids=["wide", "balanced", "narrow"],
        objective_names=["s11", "bandwidth"],
        metadata=[{"width_um": 80}, {"width_um": 120}, {"width_um": 60}],
    )
    artifact = front.to_dict()

    assert isinstance(front, ParetoFront)
    assert front.evidence_class == "pareto_front"
    assert front.front_indices == (1, 2)
    assert front.dominated_indices == (0,)
    assert [point.id for point in front.points] == ["balanced", "narrow"]
    assert front.points[0].source_index == 1
    assert front.points[0].metadata == {"width_um": 120}
    assert artifact["points"][0]["source_index"] == 1
    assert json.loads(front.to_json()) == artifact

def test_pareto_front_normalizes_numpy_metadata_for_json():
    front = pareto_front(
        [[1.0, 2.0]],
        metadata=[
            {
                "np_int": np.int64(7),
                "array": np.array([1.0, 2.0]),
                "nested": {"flag": np.bool_(True)},
            }
        ],
    )
    metadata = front.to_dict()["points"][0]["metadata"]

    assert metadata == {
        "np_int": 7,
        "array": [1.0, 2.0],
        "nested": {"flag": True},
    }
    assert json.loads(front.to_json())["points"][0]["metadata"] == metadata



def test_weighted_scalarization_is_lower_better_and_can_normalize():
    objectives = np.array([
        [0.20, 1.0, 10.0],
        [0.10, 1.5, 9.0],
        [0.12, 2.0, 11.0],
    ])

    raw_scores = weighted_scalarization(
        objectives,
        weights=[1.0, 1.0, 0.0],
        minimize=(True, False, True),
    )
    normalized_scores = weighted_scalarization(
        objectives,
        weights=[1.0, 1.0, 0.0],
        minimize=(True, False, True),
        normalize=True,
    )

    assert raw_scores.tolist() == pytest.approx([-0.8, -1.4, -1.88])
    assert int(np.argmin(raw_scores)) == 2
    assert int(np.argmin(normalized_scores)) == 2

def test_scalarization_and_epsilon_edge_cases():
    constant = np.array([
        [1.0, 3.0],
        [1.0, 2.0],
    ])

    scores = weighted_scalarization(
        constant,
        weights=[1.0, 1.0],
        normalize=True,
    )

    assert scores.tolist() == pytest.approx([1.0, 0.0])
    assert epsilon_constraint_mask(constant, {}).tolist() == [True, True]
    assert select_epsilon_constrained(
        constant,
        primary_index=1,
        epsilons={},
    ) == 1


def test_epsilon_constraint_selects_best_feasible_primary_objective():
    objectives = np.array([
        [0.20, 1.0, 10.0],
        [0.10, 1.5, 9.0],
        [0.08, 1.0, 12.0],
        [0.12, 2.0, 11.0],
    ])

    feasible = epsilon_constraint_mask(
        objectives,
        {1: 1.4, 2: 11.0},
        minimize=(True, False, True),
    )
    selected = select_epsilon_constrained(
        objectives,
        primary_index=0,
        epsilons={1: 1.4, 2: 11.0},
        minimize=(True, False, True),
    )

    assert feasible.tolist() == [False, True, False, True]
    assert selected == 1
    assert select_epsilon_constrained(
        objectives,
        primary_index=0,
        epsilons={1: 3.0},
        minimize=(True, False, True),
    ) is None

def test_sweep_result_builds_pareto_front_from_metric_mapping():
    sweep = SweepResult(
        results=[object(), object(), object()],
        param_name="width_um",
        param_values=np.array([80, 120, 160]),
    )

    front = sweep.pareto_front(
        {
            "s11": np.array([0.20, 0.10, 0.08]),
            "bandwidth": np.array([1.0, 1.5, 1.0]),
        },
        minimize=(True, False),
    )

    assert front.objective_names == ("s11", "bandwidth")
    assert front.front_indices == (1, 2)
    assert [point.id for point in front.points] == [
        "width_um=120",
        "width_um=160",
    ]
    assert front.points[0].metadata == {"width_um": 120}


def test_sweep_result_builds_pareto_front_from_builtin_metric_names():
    class Result:
        def __init__(self, s11_mag):
            self.s_params = np.array([[[s11_mag]]], dtype=complex)

    sweep = SweepResult(
        results=[Result(0.5), Result(0.1), Result(0.2)],
        param_name="gap_um",
        param_values=np.array([1, 2, 3]),
    )

    front = sweep.pareto_front(("s11_min_db",))

    assert front.front_indices == (1,)
    assert front.points[0].id == "gap_um=2"
    with pytest.raises(ValueError, match="unknown sweep metric"):
        sweep.pareto_front(("missing_metric",))
    with pytest.raises(ValueError, match="shape"):
        sweep.pareto_front({"bad": np.array([1.0, 2.0])})


def test_pareto_helpers_validate_shapes_and_finite_inputs():
    with pytest.raises(ValueError, match="2D"):
        pareto_mask([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        pareto_mask([[1.0, np.nan]])
    with pytest.raises(ValueError, match="minimize"):
        pareto_mask([[1.0, 2.0]], minimize=(True, False, True))
    with pytest.raises(ValueError, match="weight"):
        weighted_scalarization([[1.0, 2.0]], weights=[0.0, 0.0])
    with pytest.raises(IndexError, match="out of range"):
        epsilon_constraint_mask([[1.0, 2.0]], {2: 1.0})
    with pytest.raises(ValueError, match="ids"):
        pareto_front([[1.0, 2.0]], ids=[])


def test_pareto_public_export_identity_and_strict_json():
    assert rfx.ParetoFront is ParetoFront
    assert rfx.ParetoPoint is ParetoPoint
    assert rfx.pareto_front is pareto_front
    assert rfx.pareto_mask is pareto_mask
    assert rfx.weighted_scalarization is weighted_scalarization
    assert rfx.epsilon_constraint_mask is epsilon_constraint_mask
    assert rfx.select_epsilon_constrained is select_epsilon_constrained

    bad = ParetoFront(
        points=(ParetoPoint(source_index=0, id="bad", objectives=(float("nan"),)),),
        minimize=(True,),
        objective_names=("bad",),
        source_count=1,
        front_indices=(0,),
        dominated_indices=(),
    )
    with pytest.raises(ValueError, match="Out of range"):
        bad.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        bad.to_json(allow_nan=True)
