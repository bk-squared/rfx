"""Pareto-front utilities for multi-objective RF design sweeps.

The helpers in this module are intentionally optimizer-agnostic. They operate on
objective arrays produced by sweeps, scalarized inverse-design runs, or analytic
tests, then return JSON-serializable Pareto artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

def _json_safe(value: object) -> object:
    """Convert common NumPy metadata values into JSON-native objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value

class ParetoPoint(NamedTuple):
    """One candidate in a Pareto front."""

    source_index: int
    id: str
    objectives: tuple[float, ...]
    metadata: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable Pareto point."""
        return {
            "source_index": int(self.source_index),
            "id": self.id,
            "objectives": [float(value) for value in self.objectives],
            "metadata": None if self.metadata is None else _json_safe(self.metadata),
        }


class ParetoFront(NamedTuple):
    """Pareto frontier artifact for a finite set of objective vectors."""

    points: tuple[ParetoPoint, ...]
    minimize: tuple[bool, ...]
    objective_names: tuple[str, ...]
    source_count: int
    front_indices: tuple[int, ...]
    dominated_indices: tuple[int, ...]

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this design-sweep artifact."""
        return "pareto_front"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable Pareto-front artifact."""
        return {
            "evidence_class": self.evidence_class,
            "points": [point.to_dict() for point in self.points],
            "minimize": list(self.minimize),
            "objective_names": list(self.objective_names),
            "source_count": int(self.source_count),
            "front_indices": list(self.front_indices),
            "dominated_indices": list(self.dominated_indices),
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the Pareto-front artifact with non-finite floats rejected."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


def _as_2d_objectives(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"objectives must be a 2D array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("objectives must contain at least one point")
    if arr.shape[1] == 0:
        raise ValueError("objectives must contain at least one objective")
    if not np.all(np.isfinite(arr)):
        raise ValueError("objectives must be finite")
    return arr


def _orientation(minimize: bool | Sequence[bool], n_objectives: int) -> tuple[bool, ...]:
    if isinstance(minimize, (bool, np.bool_)):
        return (bool(minimize),) * n_objectives
    oriented = tuple(bool(item) for item in minimize)
    if len(oriented) != n_objectives:
        raise ValueError(
            f"minimize must have {n_objectives} entries, got {len(oriented)}"
        )
    return oriented


def _oriented_values(values: np.ndarray, minimize: tuple[bool, ...]) -> np.ndarray:
    signs = np.array([1.0 if flag else -1.0 for flag in minimize], dtype=float)
    return values * signs


def _require_nonnegative_finite(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def pareto_mask(
    objectives: Sequence[Sequence[float]] | np.ndarray,
    *,
    minimize: bool | Sequence[bool] = True,
    atol: float = 0.0,
) -> np.ndarray:
    """Return a boolean mask selecting non-dominated objective rows.

    ``minimize`` may be one bool for all objectives or one bool per objective.
    Objectives marked ``False`` are treated as maximization objectives. ``atol``
    is a dominance tolerance in the raw objective units after orientation.
    """
    values = _as_2d_objectives(objectives)
    minimize_tuple = _orientation(minimize, values.shape[1])
    oriented = _oriented_values(values, minimize_tuple)
    tolerance = _require_nonnegative_finite("atol", atol)

    n_points = oriented.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        candidate = oriented[i]
        no_worse = np.all(oriented <= candidate + tolerance, axis=1)
        strictly_better = np.any(oriented < candidate - tolerance, axis=1)
        dominated_by_other = no_worse & strictly_better
        dominated_by_other[i] = False
        if np.any(dominated_by_other):
            mask[i] = False
    return mask


def pareto_front(
    objectives: Sequence[Sequence[float]] | np.ndarray,
    *,
    minimize: bool | Sequence[bool] = True,
    ids: Sequence[str] | None = None,
    objective_names: Sequence[str] | None = None,
    metadata: Sequence[Mapping[str, object] | None] | None = None,
    atol: float = 0.0,
) -> ParetoFront:
    """Build a JSON-serializable Pareto-front artifact."""
    values = _as_2d_objectives(objectives)
    n_points, n_objectives = values.shape
    minimize_tuple = _orientation(minimize, n_objectives)
    mask = pareto_mask(values, minimize=minimize_tuple, atol=atol)

    if ids is None:
        ids_tuple = tuple(str(i) for i in range(n_points))
    else:
        ids_tuple = tuple(ids)
        if len(ids_tuple) != n_points:
            raise ValueError(f"ids must have {n_points} entries, got {len(ids_tuple)}")

    if objective_names is None:
        names_tuple = tuple(f"objective_{i}" for i in range(n_objectives))
    else:
        names_tuple = tuple(objective_names)
        if len(names_tuple) != n_objectives:
            raise ValueError(
                f"objective_names must have {n_objectives} entries, got {len(names_tuple)}"
            )

    if metadata is None:
        metadata_tuple: tuple[Mapping[str, object] | None, ...] = (None,) * n_points
    else:
        metadata_tuple = tuple(metadata)
        if len(metadata_tuple) != n_points:
            raise ValueError(
                f"metadata must have {n_points} entries, got {len(metadata_tuple)}"
            )

    front_indices = tuple(int(idx) for idx in np.flatnonzero(mask))
    dominated_indices = tuple(int(idx) for idx in np.flatnonzero(~mask))
    points = tuple(
        ParetoPoint(
            source_index=idx,
            id=ids_tuple[idx],
            objectives=tuple(float(value) for value in values[idx]),
            metadata=metadata_tuple[idx],
        )
        for idx in front_indices
    )
    return ParetoFront(
        points=points,
        minimize=minimize_tuple,
        objective_names=names_tuple,
        source_count=n_points,
        front_indices=front_indices,
        dominated_indices=dominated_indices,
    )


def weighted_scalarization(
    objectives: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float],
    *,
    minimize: bool | Sequence[bool] = True,
    normalize: bool = False,
) -> np.ndarray:
    """Return lower-is-better weighted scalarization scores.

    Maximization objectives are sign-flipped before scoring. When ``normalize``
    is true, each oriented objective is min-max normalized; constant objectives
    contribute zero after normalization.
    """
    values = _as_2d_objectives(objectives)
    minimize_tuple = _orientation(minimize, values.shape[1])
    oriented = _oriented_values(values, minimize_tuple)
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.shape != (values.shape[1],):
        raise ValueError(
            f"weights must have shape ({values.shape[1]},), got {weights_arr.shape}"
        )
    if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(weights_arr > 0.0):
        raise ValueError("at least one weight must be positive")

    if normalize:
        mins = np.min(oriented, axis=0)
        ranges = np.max(oriented, axis=0) - mins
        oriented = np.divide(
            oriented - mins,
            ranges,
            out=np.zeros_like(oriented),
            where=ranges > 0.0,
        )
    return oriented @ weights_arr


def epsilon_constraint_mask(
    objectives: Sequence[Sequence[float]] | np.ndarray,
    epsilons: Mapping[int, float],
    *,
    minimize: bool | Sequence[bool] = True,
    atol: float = 0.0,
) -> np.ndarray:
    """Return points satisfying raw epsilon constraints per objective.

    For minimization objectives the constraint is ``objective <= epsilon``; for
    maximization objectives it is ``objective >= epsilon``. Objective indices not
    present in ``epsilons`` are unconstrained.
    """
    values = _as_2d_objectives(objectives)
    minimize_tuple = _orientation(minimize, values.shape[1])
    tolerance = _require_nonnegative_finite("atol", atol)
    mask = np.ones(values.shape[0], dtype=bool)
    for idx, epsilon in epsilons.items():
        if idx < 0 or idx >= values.shape[1]:
            raise IndexError(f"epsilon objective index {idx} out of range")
        eps = float(epsilon)
        if not np.isfinite(eps):
            raise ValueError("epsilon values must be finite")
        if minimize_tuple[idx]:
            mask &= values[:, idx] <= eps + tolerance
        else:
            mask &= values[:, idx] >= eps - tolerance
    return mask


def select_epsilon_constrained(
    objectives: Sequence[Sequence[float]] | np.ndarray,
    *,
    primary_index: int,
    epsilons: Mapping[int, float],
    minimize: bool | Sequence[bool] = True,
    atol: float = 0.0,
) -> int | None:
    """Select the best feasible point by one primary objective.

    Returns ``None`` when no point satisfies the epsilon constraints.
    """
    values = _as_2d_objectives(objectives)
    if primary_index < 0 or primary_index >= values.shape[1]:
        raise IndexError(f"primary_index {primary_index} out of range")
    minimize_tuple = _orientation(minimize, values.shape[1])
    feasible = epsilon_constraint_mask(
        values,
        epsilons,
        minimize=minimize_tuple,
        atol=atol,
    )
    indices = np.flatnonzero(feasible)
    if len(indices) == 0:
        return None
    primary = values[indices, primary_index]
    local_idx = int(np.argmin(primary) if minimize_tuple[primary_index] else np.argmax(primary))
    return int(indices[local_idx])


__all__ = [
    "ParetoFront",
    "ParetoPoint",
    "epsilon_constraint_mask",
    "pareto_front",
    "pareto_mask",
    "select_epsilon_constrained",
    "weighted_scalarization",
]
