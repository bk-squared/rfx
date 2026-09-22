"""Record-length witness for a gradient taken through a finite time record.

Import contract: this module is a leaf of ``rfx.api``. It imports only
stdlib / jax / numpy, never ``rfx.api`` or ``. import`` the package, so
``rfx/api/__init__.py`` stays the sole composition point.

Why this exists
---------------
Every frequency-domain observable this simulator differentiates is a DFT of a
record of finite length ``T``. If a resonance is still ringing when the record
ends, each DFT bin keeps a leftover term whose phase is ``(w - w_r) * T``. A
parameter that moves the resonance spins that phase, and the spin appears in
the DERIVATIVE multiplied by ``T`` even while the term itself is small in the
VALUE. The value can therefore be converged to a fraction of a percent while
its gradient is wrong by tens of percent.

The ring-down witness (``settling_verdict``, the -40 dB end/peak probe-energy
bar) answers the question about the value. It does not answer this one, and
neither does an AD-versus-finite-difference check: both sides of that check
differentiate the SAME truncated record and agree to the precision floor at
every record length.

The check that does see it is here: take the same gradient from a record
``factor`` times longer and compare, per parameter leaf and per frequency bin.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "GradientRecordLengthWitness",
    "gradient_record_length_witness",
]


def _leaf_paths(tree) -> list[str]:
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [jax.tree_util.keystr(path) or "<root>" for path, _ in leaves_with_path]


def _settling_from_aux(aux) -> float | None:
    """The optional aux protocol, in one place.

    ``aux`` is whatever the objective returned as its second output. Only one
    shape of it is read: a mapping carrying a numeric ``"settling_db"``. Every
    other aux (including ``None``, a tuple, a dataclass) is accepted and simply
    contributes no settling level -- the witness never requires one.
    """
    if not isinstance(aux, Mapping):
        return None
    if "settling_db" not in aux:
        return None
    value = aux["settling_db"]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class GradientRecordLengthWitness:
    """What two record lengths said about one gradient.

    Every array carries a leading bin axis of length ``n_bins`` (``1`` when the
    objective returned a scalar, in which case ``observable_is_scalar`` is
    True), so a scalar and a per-frequency objective are read the same way.

    Attributes
    ----------
    n_steps, n_steps_long : int
        The two record lengths, in timesteps.
    factor : float
        ``n_steps_long = ceil(factor * n_steps)``.
    tol : float
        The caller's declared bar on ``worst``.
    passed : bool
        ``worst <= tol``. A gradient whose witness did not pass is not
        reportable, whatever the shorter record's settling level says.
    worst : float
        Largest relative gradient change over every bin, leaf and element.
    worst_leaf : str or None
        Parameter-tree path of the leaf holding ``worst``.
    worst_bin : int or None
        Bin index holding ``worst``.
    value, value_long : numpy.ndarray
        The observable at each length, shaped ``(n_bins,)``.
    value_rel_change : numpy.ndarray
        ``|value_long - value| / max(|value_long|, floor)``, shaped
        ``(n_bins,)``. Reported so the value's convergence and the gradient's
        can be read side by side; it is NOT part of the verdict.
    worst_value_rel_change : float
        Largest entry of ``value_rel_change``.
    grad, grad_long : dict
        Leaf path -> gradient array shaped ``(n_bins,) + leaf.shape``.
    grad_rel_change : dict
        Leaf path -> ``|g_long - g| / max(|g_long|, floor)``, same shape.
    settling_db, settling_db_long : float or None
        Ring-down levels of the two records, when the objective's aux supplied
        them. ``None`` means the objective did not report one, never that the
        record settled.
    floor_frac : float
        The divide-by-zero floor, as a fraction of the largest ``|gradient|``
        across leaves in that bin, at either record length.
    """

    n_steps: int
    n_steps_long: int
    factor: float
    tol: float
    passed: bool
    worst: float
    worst_leaf: str | None
    worst_bin: int | None
    value: np.ndarray
    value_long: np.ndarray
    value_rel_change: np.ndarray
    worst_value_rel_change: float
    grad: dict[str, np.ndarray]
    grad_long: dict[str, np.ndarray]
    grad_rel_change: dict[str, np.ndarray]
    settling_db: float | None
    settling_db_long: float | None
    floor_frac: float
    observable_is_scalar: bool

    def summary(self) -> str:
        """One line for a log or a PR body."""
        verdict = "PASS" if self.passed else "FAIL"
        where = ""
        if self.worst_leaf is not None:
            where = f" at {self.worst_leaf} bin {self.worst_bin}"
            if self.observable_is_scalar:
                where = f" at {self.worst_leaf}"
        settle = ""
        if self.settling_db is not None:
            settle = f", record settled to {self.settling_db:.1f} dB"
        return (
            f"gradient record-length witness {verdict}: "
            f"{self.n_steps} -> {self.n_steps_long} steps moved the gradient "
            f"by {self.worst * 100:.2f}%{where} (tol {self.tol * 100:.2f}%), "
            f"the value by {self.worst_value_rel_change * 100:.3f}%{settle}"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def gradient_record_length_witness(
    objective: Callable[[Any, int], Any],
    params: Any,
    n_steps: int,
    *,
    tol: float,
    factor: float = 2.0,
    has_aux: bool = False,
    floor_frac: float = 1e-3,
) -> GradientRecordLengthWitness:
    """Is this gradient converged in RECORD LENGTH, not just in value?

    Differentiates ``objective`` at ``n_steps`` and again at
    ``ceil(factor * n_steps)`` and reports how far the gradient moved, per
    parameter leaf and per frequency bin. Physically: a resonance that is still
    ringing when the record ends leaves a term in every DFT bin whose phase is
    ``(w - w_r) * T``; a parameter that moves ``w_r`` spins that phase, and the
    spin enters the derivative multiplied by ``T``. The value hides it, the
    gradient does not.

    Parameters
    ----------
    objective : callable
        ``objective(params, n_steps)`` returning the observable -- a scalar, or
        a 1-D array over frequency bins. With ``has_aux=True`` it returns
        ``(observable, aux)`` instead. It must be differentiable w.r.t. its
        first argument, and ``n_steps`` must be the number of timesteps of the
        record it takes the observable from.
    params : pytree
        The differentiation point. Any JAX pytree; the report is per leaf.
    n_steps : int
        The record length under test, in timesteps.
    tol : float
        Required. The largest relative gradient change that still counts as
        converged. There is deliberately NO default: the right bar depends on
        the structure's Q, on how far the parameter moves the resonance and on
        what the gradient is for, and a default here would be a number nobody
        measured. Measurements to calibrate against, all with ``factor=2``:

        * Grounded-slab patch, graded z, float32, 50 ohm wire port,
          ``d ln|S11|^2 / d ln eps_r``: a 2200-step record settling to
          -39.7 dB (so a PASS on the -40 dB bar) moved 9.7 % at 5.75 GHz and
          23 % at 6.5 GHz; a local-permittivity parameter on the same record
          moved 8.7 % and 38 %. At 4400 steps (-75 dB) the worst was 2.1 %.
        * The same board driven by a soft source instead of a port settles far
          more slowly (-40.9 dB in 6000 steps where the port reaches -101 dB):
          there ``d ln U(0) / d ln eps_r`` read 5.79 against a converged 6.80
          (15 % low) and a pattern-ratio gradient 0.331 against 0.070 (5x).
        * The committed cavity fixture
          (``tests/unit/autodiff/test_gradient_record_length_witness.py``):
          1600 steps, settling -46.7 dB, value converged to 0.013 %, gradient
          moved 16.8 %; 3200 steps, settling -89.1 dB, gradient moved 0.20 %.

        A bar of a few percent is what the -75 dB rows above support; tighten
        or loosen it against your own sweep, and record which.
    factor : float, default 2.0
        Record-length ratio of the long arm. Must be greater than 1. The
        witness only sees ringing that the LONGER record resolves: a record
        too short by more than ``factor`` can still pass, because both arms
        then carry a similar leftover. Pair this with the -40 dB settling rule
        (``settling_verdict``); it does not replace it.
    has_aux : bool, default False
        Whether ``objective`` returns ``(observable, aux)``. The witness reads
        exactly one thing out of ``aux``: if it is a mapping with a finite
        numeric ``"settling_db"``, that value is recorded as the record's
        ring-down level. Anything else is accepted and reported as ``None``.
    floor_frac : float, default 1e-3
        Divide-by-zero floor for the relative change, as a fraction of the
        largest ``|gradient|`` across ALL leaves in that bin, at either record
        length. A leaf whose gradient is negligible against the dominant one
        therefore cannot fail the witness on its own numerical noise.

    Returns
    -------
    GradientRecordLengthWitness

    Notes
    -----
    Cost: two differentiated runs, one of them ``factor`` times longer, and one
    compile each, because the record length is a traced-in constant. Roughly
    ``1 + factor`` times the steps of the run being checked plus a second
    compile. This is an opt-in diagnostic, not something ``forward()`` does on
    its own.

    Examples
    --------
    >>> def objective(p, n):  # doctest: +SKIP
    ...     result = sim.forward(eps_override=base * jnp.exp(p), n_steps=n)
    ...     return ln_power_at_bins(result.time_series)
    >>> w = gradient_record_length_witness(  # doctest: +SKIP
    ...     objective, 0.0, 1600, tol=0.05)
    >>> w.passed, w.worst  # doctest: +SKIP
    (False, 0.166)
    """
    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool):
        raise TypeError(f"n_steps must be an int, got {type(n_steps).__name__}")
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError(
            f"factor must be a finite number greater than 1, got {factor}: "
            "the second arm has to be a LONGER record than the first."
        )
    tol = float(tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a positive finite number, got {tol}")
    floor_frac = float(floor_frac)
    if not math.isfinite(floor_frac) or floor_frac < 0.0:
        raise ValueError(
            f"floor_frac must be a non-negative finite number, got {floor_frac}"
        )

    n_steps_long = int(math.ceil(factor * n_steps))
    if n_steps_long <= n_steps:
        raise ValueError(
            f"ceil(factor * n_steps) = {n_steps_long} is not longer than "
            f"n_steps = {n_steps}; raise factor or n_steps."
        )

    short = _differentiate(objective, params, n_steps, has_aux=has_aux)
    long = _differentiate(objective, params, n_steps_long, has_aux=has_aux)

    if short.n_bins != long.n_bins:
        raise ValueError(
            "the objective returned a different number of bins at the two "
            f"record lengths ({short.n_bins} at {n_steps}, {long.n_bins} at "
            f"{n_steps_long}); the observable must not depend on n_steps in "
            "shape, only in content."
        )
    if short.paths != long.paths:
        raise ValueError(
            "the objective returned a different parameter tree at the two "
            "record lengths; leaves at "
            f"{n_steps}: {short.paths}, at {n_steps_long}: {long.paths}"
        )

    n_bins = short.n_bins
    # Per-bin scale: the largest |gradient| anywhere in the tree, at either
    # record length. ``floor_frac`` of it is the floor in the denominator
    # below, so (a) a leaf whose gradient is negligible against the dominant
    # one cannot fail on its own noise, and (b) a gradient that is exactly zero
    # on the long record but not on the short one still reads as a change
    # instead of dividing 0 by 0.
    scale = np.zeros(n_bins, dtype=np.float64)
    for arrs in (long.grads, short.grads):
        for arr in arrs.values():
            mag = np.abs(np.asarray(arr, dtype=np.float64)).reshape(n_bins, -1)
            if mag.size:
                scale = np.maximum(scale, mag.max(axis=1))
    floor = floor_frac * scale

    grad_rel_change: dict[str, np.ndarray] = {}
    worst = 0.0
    worst_leaf: str | None = None
    worst_bin: int | None = None
    for path in short.paths:
        g_s = np.asarray(short.grads[path], dtype=np.float64)
        g_l = np.asarray(long.grads[path], dtype=np.float64)
        den = np.maximum(
            np.abs(g_l), floor.reshape((n_bins,) + (1,) * (g_l.ndim - 1))
        )
        rel = np.where(den > 0.0, np.abs(g_l - g_s) / np.where(den > 0.0, den, 1.0), 0.0)
        grad_rel_change[path] = rel
        if rel.size:
            flat = rel.reshape(n_bins, -1)
            leaf_worst = float(flat.max())
            if leaf_worst > worst or worst_leaf is None:
                worst = leaf_worst
                worst_leaf = path
                worst_bin = int(np.unravel_index(int(flat.argmax()), flat.shape)[0])

    v_s = np.asarray(short.value, dtype=np.float64)
    v_l = np.asarray(long.value, dtype=np.float64)
    v_scale = (
        float(max(np.abs(v_l).max(), np.abs(v_s).max())) if v_l.size else 0.0
    )
    v_den = np.maximum(np.abs(v_l), floor_frac * v_scale)
    value_rel_change = np.where(
        v_den > 0.0, np.abs(v_l - v_s) / np.where(v_den > 0.0, v_den, 1.0), 0.0
    )

    return GradientRecordLengthWitness(
        n_steps=n_steps,
        n_steps_long=n_steps_long,
        factor=factor,
        tol=tol,
        passed=bool(worst <= tol),
        worst=float(worst),
        worst_leaf=worst_leaf,
        worst_bin=worst_bin,
        value=v_s,
        value_long=v_l,
        value_rel_change=value_rel_change,
        worst_value_rel_change=(
            float(value_rel_change.max()) if value_rel_change.size else 0.0
        ),
        grad={k: np.asarray(v, dtype=np.float64) for k, v in short.grads.items()},
        grad_long={k: np.asarray(v, dtype=np.float64) for k, v in long.grads.items()},
        grad_rel_change=grad_rel_change,
        settling_db=short.settling_db,
        settling_db_long=long.settling_db,
        floor_frac=floor_frac,
        observable_is_scalar=short.is_scalar,
    )


@dataclass(frozen=True)
class _Arm:
    """One record length's value, per-bin gradients and settling level."""

    value: np.ndarray
    grads: dict[str, np.ndarray]
    paths: list[str]
    n_bins: int
    is_scalar: bool
    settling_db: float | None


def _differentiate(objective, params, n_steps: int, *, has_aux: bool) -> _Arm:
    """Reverse-mode gradients of one objective call, one row per bin.

    ``jax.vjp`` with a cotangent of 1 IS ``value_and_grad`` for a scalar
    observable; for a 1-D observable the same linearisation is pulled back once
    per bin, which is one forward record and ``n_bins`` backward passes rather
    than ``n_bins`` separate runs.
    """

    def wrapped(p):
        return objective(p, n_steps)

    if has_aux:
        out, pullback, aux = jax.vjp(wrapped, params, has_aux=True)
    else:
        out, pullback = jax.vjp(wrapped, params)
        aux = None

    value = np.asarray(out)
    if value.ndim == 0:
        is_scalar = True
        n_bins = 1
        cotangents = [np.ones((), dtype=value.dtype)]
    elif value.ndim == 1:
        is_scalar = False
        n_bins = int(value.shape[0])
        if n_bins == 0:
            raise ValueError(
                "the objective returned an empty observable; it must return a "
                "scalar or a 1-D array with at least one bin."
            )
        eye = np.eye(n_bins, dtype=value.dtype)
        cotangents = [eye[i] for i in range(n_bins)]
    else:
        raise ValueError(
            "the objective must return a scalar or a 1-D array over frequency "
            f"bins; got shape {value.shape}. Reduce or flatten it yourself, so "
            "the witness reports the bins you mean."
        )

    paths = _leaf_paths(params)
    rows: dict[str, list[np.ndarray]] = {p: [] for p in paths}
    for cotangent in cotangents:
        (grad_tree,) = pullback(jnp.asarray(cotangent))
        leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(grad_tree)
        grad_paths = [
            jax.tree_util.keystr(path) or "<root>" for path, _ in leaves_with_path
        ]
        if grad_paths != paths:
            raise ValueError(
                "the gradient tree does not match the parameter tree: leaves "
                f"{grad_paths} against {paths}."
            )
        for (_, leaf), name in zip(leaves_with_path, paths):
            rows[name].append(np.asarray(leaf, dtype=np.float64))

    grads = {name: np.stack(vals, axis=0) for name, vals in rows.items()}
    return _Arm(
        value=value.reshape(n_bins),
        grads=grads,
        paths=paths,
        n_bins=n_bins,
        is_scalar=is_scalar,
        settling_db=_settling_from_aux(aux),
    )
