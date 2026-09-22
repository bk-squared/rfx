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
``factor`` times longer and compare, per frequency bin.

How the two arms are compared
-----------------------------
The VERDICT is one number per bin and it is norm-level: the whole gradient
vector over every parameter element, ``||g_long - g_short|| / ||g_long||``, with
no floor in it. A per-element ratio cannot be the verdict, because a element
whose gradient is a millionth of the dominant one moves by 100 % on its own
rounding and says nothing about the descent direction; a per-cell permittivity
leaf produced a 366 % headline that way while the cells that had really moved
read 4 %. The direction is reported beside it as the cosine between the two
arms' gradient vectors.

The per-element table is still reported, floored against the dominant element
so rounding does not dominate it, but it is there to LOCATE where a change sits
once the verdict has already failed. It is not the verdict.

Two cautions on reading the verdict. Each bin is normalised by its own norm, so
a bin whose gradient is negligible against the other bins' -- a bin at a null,
or one sitting exactly on a stationary point of the observable -- reads large on
its own rounding; read the worst bin together with its absolute size. And the
number is taken in the metric of the parameters handed in: the same two records
read 11 % on (eps_global, eps_local) and 1.8 % on the 1936 per-cell leaf of the
same board, because the projection onto two scalars concentrates the change.
Witness the parameters you actually optimise; a finer parameterisation gives
the more permissive answer.
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


def _reject_unsupported_params(params) -> None:
    """Refuse parameter leaves this witness cannot differentiate meaningfully.

    Both cases used to die deep inside numpy with "setting an array element
    with a sequence", which says nothing about the parameter that caused it.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in leaves_with_path:
        name = jax.tree_util.keystr(path) or "<root>"
        try:
            dtype = jnp.asarray(leaf).dtype
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"parameter leaf {name} is not an array or a number "
                f"({type(leaf).__name__}): {error}"
            ) from error
        if np.issubdtype(dtype, np.complexfloating):
            raise ValueError(
                f"parameter leaf {name} is complex ({dtype}). This witness "
                "compares the sensitivity of an observable to REAL design "
                "parameters -- a permittivity, a dimension, a component value "
                "-- and a complex parameter has two independent real "
                "directions the report cannot name. Split it into its real and "
                "imaginary parts as separate real leaves."
            )
        if np.issubdtype(dtype, np.integer) or dtype == np.dtype(bool):
            raise ValueError(
                f"parameter leaf {name} has dtype {dtype}, which carries no "
                "gradient (JAX gives it a float0 tangent). Cast it to a float "
                "dtype before differentiating."
            )


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
        THE VERDICT. The largest ``rel_by_bin`` over the bins -- a norm-level
        relative change of the whole gradient vector, with no floor in it.
    worst_bin : int or None
        Bin index holding ``worst``.
    worst_leaf : str or None
        Within ``worst_bin``, the leaf contributing the largest
        ``||g_long - g_short||``. Where the change sits, not a second verdict.
    rel_by_bin : numpy.ndarray
        ``||g_long - g_short|| / ||g_long||`` per bin, over every parameter
        element of every leaf concatenated, shaped ``(n_bins,)``. ``inf`` where
        the long record's gradient is identically zero and the short one's is
        not; ``0.0`` where both are.
    cosine_by_bin : numpy.ndarray
        Per bin, the cosine between the two arms' gradient vectors (the real
        part of the Hermitian inner product, normalised). ``1.0`` means the
        descent DIRECTION is unchanged and only the step length moved; a value
        below 1 means the direction itself turned. ``nan`` where either arm's
        gradient is identically zero.
    value, value_long : numpy.ndarray
        The observable at each length, shaped ``(n_bins,)``. Complex when the
        objective returned a complex observable.
    value_rel_change : numpy.ndarray
        ``|value_long - value| / max(|value_long|, floor)``, shaped
        ``(n_bins,)``, with ``|.|`` the complex magnitude where the observable
        is complex. Reported so the value's convergence and the gradient's can
        be read side by side; it is NOT part of the verdict.
    worst_value_rel_change : float
        Largest entry of ``value_rel_change``.
    grad, grad_long : dict
        Leaf path -> gradient array shaped ``(n_bins,) + leaf.shape``. Complex
        (``dObservable/dp`` as ``dRe/dp + i dIm/dp``) for a complex observable.
    grad_rel_change : dict
        Leaf path -> per-ELEMENT ``|g_long - g| / max(|g_long|, floor)``, same
        shape. A REPORT, not the verdict: it locates where a change sits once
        ``worst`` has already failed. An element whose gradient is a tiny
        fraction of the dominant one reads a large ratio off its own rounding,
        which is what the floor and the norm-level verdict exist to keep out of
        the decision.
    worst_elementwise : float
        Largest entry of ``grad_rel_change``, with
        ``worst_elementwise_leaf`` / ``worst_elementwise_bin`` locating it.
        Reported for the same reason the table is.
    settling_db, settling_db_long : float or None
        Ring-down levels of the two records, when the objective's aux supplied
        them. ``None`` means the objective did not report one, never that the
        record settled.
    floor_frac : float
        The floor used in ``grad_rel_change`` only, as a fraction of the
        largest ``|gradient|`` across leaves in that bin, at either record
        length.
    observable_is_scalar, observable_is_complex : bool
        What the objective returned.
    """

    n_steps: int
    n_steps_long: int
    factor: float
    tol: float
    passed: bool
    worst: float
    worst_bin: int | None
    worst_leaf: str | None
    rel_by_bin: np.ndarray
    cosine_by_bin: np.ndarray
    value: np.ndarray
    value_long: np.ndarray
    value_rel_change: np.ndarray
    worst_value_rel_change: float
    grad: dict[str, np.ndarray]
    grad_long: dict[str, np.ndarray]
    grad_rel_change: dict[str, np.ndarray]
    worst_elementwise: float
    worst_elementwise_leaf: str | None
    worst_elementwise_bin: int | None
    settling_db: float | None
    settling_db_long: float | None
    floor_frac: float
    observable_is_scalar: bool
    observable_is_complex: bool

    def summary(self) -> str:
        """One line for a log or a PR body."""
        verdict = "PASS" if self.passed else "FAIL"
        where = ""
        if self.worst_bin is not None and not self.observable_is_scalar:
            where = f" in bin {self.worst_bin}"
        cosine = ""
        if self.worst_bin is not None:
            value = float(self.cosine_by_bin[self.worst_bin])
            if math.isfinite(value):
                cosine = f", direction cos {value:.4f}"
        settle = ""
        if self.settling_db is not None:
            settle = f", record settled to {self.settling_db:.1f} dB"
        return (
            f"gradient record-length witness {verdict}: "
            f"{self.n_steps} -> {self.n_steps_long} steps moved the gradient "
            f"by {self.worst * 100:.2f}%{where} (tol {self.tol * 100:.2f}%)"
            f"{cosine}, the value by "
            f"{self.worst_value_rel_change * 100:.3f}%{settle}"
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
    ``ceil(factor * n_steps)`` and asks whether the gradient VECTOR moved.
    Physically: a resonance that is still ringing when the record ends leaves a
    term in every DFT bin whose phase is ``(w - w_r) * T``; a parameter that
    moves ``w_r`` spins that phase, and the spin enters the derivative
    multiplied by ``T``. The value hides it, the gradient does not.

    The verdict is ``||g_long - g_short|| / ||g_long||`` per bin, over every
    parameter element of every leaf, with no floor in it, and the cosine
    between the two arms is reported beside it so a change of step LENGTH can
    be told from a change of DIRECTION. The per-element table
    (``grad_rel_change``) is a report for locating a failure, not the verdict --
    see the class docstring.

    Parameters
    ----------
    objective : callable
        ``objective(params, n_steps)`` returning the observable -- a scalar or
        a 1-D array over frequency bins, real or COMPLEX (an S-parameter, a DFT
        phasor). For a complex observable the full complex sensitivity
        ``dRe/dp + i dIm/dp`` is compared, not its real part: a parameter that
        rotates a phasor at constant magnitude moves only the imaginary part.
        With ``has_aux=True`` it returns ``(observable, aux)`` instead. It must
        be differentiable w.r.t. its first argument, and ``n_steps`` must be the
        number of timesteps of the record it takes the observable from.
    params : pytree
        The differentiation point. Any JAX pytree of REAL float leaves -- a
        permittivity, a dimension, a component value. Complex and integer leaves
        are refused with a message rather than silently halved or zeroed.
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
          more slowly (-40.9 dB in 6000 steps where the port reaches -101.5 dB):
          there ``d ln U(0) / d ln eps_r`` read 5.79 against a converged 6.80
          (15 % low) and a pattern-ratio gradient 0.331 against 0.070 (5x).
        * The committed cavity fixture
          (``tests/unit/autodiff/test_gradient_record_length_witness.py``):
          1600 steps, settling -46.7 dB, the POWER converged to 0.399 %,
          the gradient vector moved 11.0 %
          (its direction by cos 0.9989); 3200 steps, settling -89.1 dB, gradient
          moved 0.12 %.

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
        Applies to the per-element REPORT table only, never to the verdict. It
        is the floor in that table's denominator, as a fraction of the largest
        ``|gradient|`` across all leaves in that bin at either record length,
        so an element carrying no sensitivity does not fill the table with
        ratios off its own rounding. The verdict is a ratio of norms and needs
        no floor.

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
    (False, 0.110)
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
    is_complex = short.is_complex or long.is_complex
    dtype = np.complex128 if is_complex else np.float64
    g_short = {k: np.asarray(v, dtype=dtype) for k, v in short.grads.items()}
    g_long = {k: np.asarray(v, dtype=dtype) for k, v in long.grads.items()}

    # ---- THE VERDICT: norm-level, per bin, no floor -----------------------
    # One number per bin over the whole gradient vector. A per-element ratio
    # cannot decide this: an element carrying a millionth of the dominant
    # sensitivity moves by 100 % on its own rounding, and a per-cell
    # permittivity leaf then reports hundreds of percent while the cells that
    # actually moved report a few. The norm answers the question an optimizer
    # asks -- did the gradient VECTOR move -- and the cosine beside it says
    # whether the direction turned or only the length.
    rel_by_bin = np.zeros(n_bins, dtype=np.float64)
    cosine_by_bin = np.full(n_bins, np.nan, dtype=np.float64)
    leaf_delta_norm = np.zeros((n_bins, len(short.paths)), dtype=np.float64)
    for i in range(n_bins):
        vec_s = _bin_vector(g_short, short.paths, i, dtype)
        vec_l = _bin_vector(g_long, short.paths, i, dtype)
        delta = float(np.linalg.norm(vec_l - vec_s))
        norm_l = float(np.linalg.norm(vec_l))
        norm_s = float(np.linalg.norm(vec_s))
        if delta == 0.0:
            rel_by_bin[i] = 0.0
        elif norm_l == 0.0:
            # The long record says the observable does not depend on the
            # parameter here and the short record says it does. That is not a
            # small relative change, and dividing by the floor would dress it
            # up as one.
            rel_by_bin[i] = math.inf
        else:
            rel_by_bin[i] = delta / norm_l
        if norm_l > 0.0 and norm_s > 0.0:
            cosine_by_bin[i] = float(
                np.real(np.vdot(vec_s, vec_l)) / (norm_s * norm_l)
            )
        for j, path in enumerate(short.paths):
            leaf_delta_norm[i, j] = float(
                np.linalg.norm(
                    np.asarray(g_long[path][i], dtype=dtype).ravel()
                    - np.asarray(g_short[path][i], dtype=dtype).ravel()
                )
            )

    worst_bin: int | None = None
    worst = 0.0
    worst_leaf: str | None = None
    if n_bins:
        worst_bin = int(np.argmax(rel_by_bin))
        worst = float(rel_by_bin[worst_bin])
        if short.paths:
            worst_leaf = short.paths[int(np.argmax(leaf_delta_norm[worst_bin]))]

    # ---- THE REPORT: per-element table, floored ---------------------------
    # Floored per bin against the dominant element of that bin, at either
    # record length, so the table is readable rather than dominated by
    # rounding on elements that carry no sensitivity. Read it to find WHERE a
    # failed verdict sits; it decides nothing.
    scale = np.zeros(n_bins, dtype=np.float64)
    for arrs in (g_long, g_short):
        for arr in arrs.values():
            mag = np.abs(arr).reshape(n_bins, -1)
            if mag.size:
                scale = np.maximum(scale, mag.max(axis=1))
    floor = floor_frac * scale

    grad_rel_change: dict[str, np.ndarray] = {}
    worst_elementwise = 0.0
    worst_elementwise_leaf: str | None = None
    worst_elementwise_bin: int | None = None
    for path in short.paths:
        g_s = g_short[path]
        g_l = g_long[path]
        den = np.maximum(
            np.abs(g_l), floor.reshape((n_bins,) + (1,) * (g_l.ndim - 1))
        )
        rel = np.where(
            den > 0.0, np.abs(g_l - g_s) / np.where(den > 0.0, den, 1.0), 0.0
        )
        grad_rel_change[path] = rel
        if rel.size:
            flat = rel.reshape(n_bins, -1)
            leaf_worst = float(flat.max())
            if leaf_worst > worst_elementwise or worst_elementwise_leaf is None:
                worst_elementwise = leaf_worst
                worst_elementwise_leaf = path
                worst_elementwise_bin = int(
                    np.unravel_index(int(flat.argmax()), flat.shape)[0]
                )

    v_s = np.asarray(short.value, dtype=dtype)
    v_l = np.asarray(long.value, dtype=dtype)
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
        worst=worst,
        worst_bin=worst_bin,
        worst_leaf=worst_leaf,
        rel_by_bin=rel_by_bin,
        cosine_by_bin=cosine_by_bin,
        value=v_s,
        value_long=v_l,
        value_rel_change=value_rel_change,
        worst_value_rel_change=(
            float(value_rel_change.max()) if value_rel_change.size else 0.0
        ),
        grad=g_short,
        grad_long=g_long,
        grad_rel_change=grad_rel_change,
        worst_elementwise=worst_elementwise,
        worst_elementwise_leaf=worst_elementwise_leaf,
        worst_elementwise_bin=worst_elementwise_bin,
        settling_db=short.settling_db,
        settling_db_long=long.settling_db,
        floor_frac=floor_frac,
        observable_is_scalar=short.is_scalar,
        observable_is_complex=is_complex,
    )


def _bin_vector(grads, paths, bin_index: int, dtype) -> np.ndarray:
    """Every parameter element of every leaf for one bin, as one flat vector."""
    parts = [
        np.asarray(grads[path][bin_index], dtype=dtype).ravel() for path in paths
    ]
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts)


@dataclass(frozen=True)
class _Arm:
    """One record length's value, per-bin gradients and settling level."""

    value: np.ndarray
    grads: dict[str, np.ndarray]
    paths: list[str]
    n_bins: int
    is_scalar: bool
    is_complex: bool
    settling_db: float | None


def _pullback_leaves(pullback, cotangent, paths) -> list[np.ndarray]:
    """One pullback, checked against the parameter tree, as flat leaf arrays."""
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
    return [np.asarray(leaf, dtype=np.float64) for _, leaf in leaves_with_path]


def _differentiate(objective, params, n_steps: int, *, has_aux: bool) -> _Arm:
    """Reverse-mode sensitivities of one objective call, one row per bin.

    ``jax.vjp`` with a cotangent of 1 IS ``value_and_grad`` for a real scalar
    observable; for a 1-D observable the same linearisation is pulled back once
    per bin, which is one forward record and ``n_bins`` backward passes rather
    than ``n_bins`` separate runs.

    A COMPLEX observable needs two pullbacks per bin. rfx's S-parameters and
    every DFT phasor are complex, and half of a complex sensitivity is not a
    sensitivity: a parameter that rotates a phasor without changing its
    magnitude moves the imaginary part and nothing else. JAX's convention for
    a real input and a complex output is
    ``pullback(c) = Re(c) dRe(y)/dp - Im(c) dIm(y)/dp`` (measured, jax 0.10.2),
    so the complex sensitivity is ``pullback(1) - 1j * pullback(1j)``.
    """
    _reject_unsupported_params(params)

    def wrapped(p):
        return objective(p, n_steps)

    if has_aux:
        out, pullback, aux = jax.vjp(wrapped, params, has_aux=True)
    else:
        out, pullback = jax.vjp(wrapped, params)
        aux = None
        if not isinstance(out, (jax.Array, np.ndarray, int, float, complex)):
            raise ValueError(
                "the objective returned a "
                f"{type(out).__name__}, not a single array or scalar "
                "observable. If it returns (observable, aux), pass "
                "has_aux=True; the witness reads no aux unless you say so."
            )

    value = np.asarray(out)
    is_complex = bool(np.iscomplexobj(value))
    if value.ndim == 0:
        is_scalar = True
        n_bins = 1
        basis = [np.ones((), dtype=value.dtype)]
    elif value.ndim == 1:
        is_scalar = False
        n_bins = int(value.shape[0])
        if n_bins == 0:
            raise ValueError(
                "the objective returned an empty observable; it must return a "
                "scalar or a 1-D array with at least one bin."
            )
        eye = np.eye(n_bins, dtype=value.dtype)
        basis = [eye[i] for i in range(n_bins)]
    else:
        raise ValueError(
            "the objective must return a scalar or a 1-D array over frequency "
            f"bins; got shape {value.shape}. Reduce or flatten it yourself, so "
            "the witness reports the bins you mean."
        )

    paths = _leaf_paths(params)
    rows: dict[str, list[np.ndarray]] = {p: [] for p in paths}
    for cotangent in basis:
        real_part = _pullback_leaves(pullback, cotangent, paths)
        if is_complex:
            imag_part = _pullback_leaves(pullback, cotangent * 1j, paths)
            leaves = [a - 1j * b for a, b in zip(real_part, imag_part)]
        else:
            leaves = real_part
        for name, leaf in zip(paths, leaves):
            rows[name].append(leaf)

    grads = {name: np.stack(vals, axis=0) for name, vals in rows.items()}
    return _Arm(
        value=value.reshape(n_bins),
        grads=grads,
        paths=paths,
        n_bins=n_bins,
        is_scalar=is_scalar,
        is_complex=is_complex,
        settling_db=_settling_from_aux(aux),
    )
