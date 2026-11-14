"""Few-parameter minimisation through the shared optimiser loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx.errors import AccuracyWarning, ConfigurationWarning, RfxRefusal
from rfx.optimize import _run_optimizer
from rfx.optimizers import _Adam, _MMA, _ProjectedAdam, _box_in_dtype


@dataclass(frozen=True)
class MinimizeResult:
    """An evaluated solution and its finite objective and parameter histories.

    ``nfev`` includes a non-finite evaluation, but excludes ``setup_forwards``.
    ``solves`` counts forward, reverse and forward-mode passes in the loop.
    ``scale`` is the initial largest entry for worst-case problems, else None.
    ``subproblem_caps`` counts unconverged MMA subproblems' cap events.
    """

    x: jax.Array
    fun: float
    nfev: int
    history: list[float]
    x_history: jax.Array
    solves: dict[str, int]
    setup_forwards: int
    method: str
    jacobian: str
    subproblem_caps: int
    stopped_nonfinite: bool
    scale: float | None


@dataclass(frozen=True)
class _RecordingMMA(_MMA):
    scale: float | None = None
    history: list[float] = field(default_factory=list, compare=False)
    caps: list[int] = field(default_factory=list, compare=False)

    def tell(self, state, request, value, grad, cons_values, cons_grads):
        if self.scale is not None:
            largest = float(np.max(np.asarray(cons_values))) * self.scale
            self.history.append(largest)
            if largest <= 0:
                warnings.warn(
                    f"minimize: worst_case largest entry <= 0 at iteration {state.k + 1}",
                    AccuracyWarning, stacklevel=2,
                )
        updated = super().tell(state, request, value, grad, cons_values, cons_grads)
        self.caps.append(updated.caps_hit - state.caps_hit)
        return updated


def _bounds_arrays(bounds, n):
    try:
        if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
            lo, hi = (np.broadcast_to(np.asarray(b, dtype=np.float64), (n,))
                      for b in (bounds.lb, bounds.ub))
        else:
            pairs = np.asarray(bounds, dtype=np.float64)
            if pairs.shape != (n, 2):
                raise RfxRefusal("bounds must contain one (min, max) pair per x0 element")
            lo, hi = pairs[:, 0], pairs[:, 1]
    except (TypeError, ValueError) as exc:
        raise RfxRefusal("bounds must contain finite (min, max) pairs matching x0") from exc
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(lo >= hi):
        raise RfxRefusal("bounds must be finite with min < max")
    return lo, hi


def minimize(
    fun: Callable[[jax.Array], jax.Array],
    x0: object,
    *,
    bounds: object | None = None,
    method: str = "adam",
    worst_case: bool = False,
    n_iters: int = 50,
    lr: float | None = None,
    best_iterate: bool = False,
    jit: bool = True,
) -> MinimizeResult:
    """Minimise a JAX-traceable objective over a few scalar parameters.

    Parameters
    ----------
    fun : callable
        Scalar objective of a one-dimensional x. With worst_case=True, return
        a nonempty vector of nonnegative quantities, such as power, |S|^2 or
        reflectance; convert dB to linear power. Its initial maximum must be
        finite and positive and all initial entries finite and nonnegative.
    x0 : array-like
        Finite, nonempty one-dimensional start. Converted to JAX's default
        float (float32, or float64 within the caller's x64 scope). A float64
        NumPy or JAX array converted to float32 emits ConfigurationWarning.
    bounds : sequence of (min, max) pairs or object with lb and ub, optional
        Finite, strictly ordered inclusive bounds, one pair per parameter.
        Scalar lb or ub broadcasts. x0 must lie inside the original bounds.
        Values passed to fun are clipped to the inward-rounded box in x's
        dtype, preserving the gradient of the parameter map.
    method : {"adam", "projected_adam", "mma"}
        Case-insensitive. Adam uses x directly without bounds, or latent u
        with x = lo + (hi - lo) sigmoid(u). The initial logit is clipped to
        +/-13.8, placing a boundary start about 1e-6 of the box inside it.
        Projected Adam and MMA require bounds and use p in [0, 1]. MMA uses
        the standard moving-asymptote constants and no learning rate.
    worst_case : bool
        MMA only: minimise the largest entry using epigraph rows fun(x)/s,
        with a0=1, a=1, c=1000, d=1 and s the initial maximum. One setup
        forward computes both s and the row count.
        For n parameters and m entries, use forward mode if n < m + 2,
        otherwise reverse mode. Scalar objectives use value-and-gradient.
    n_iters : int
        Positive evaluation budget, default 50. There is no convergence test.
    lr : float, optional
        Defaults to 0.01 for either Adam method; refused for MMA. A step on u
        for bounded Adam, in x's units for unbounded Adam, and on p for
        projected Adam. At the midpoint, a quarter of a latent learning rate
        gives the same first design step on p.
    best_iterate : bool
        Return the finite evaluated point with the lowest objective instead
        of the last finite evaluated point.
    jit : bool
        Compile the differentiated evaluation once, with the shared loop's
        fallback for objectives that cannot be compiled.

    Returns
    -------
    MinimizeResult
        x is the evaluated iterate, one update before the point optimize
        returns. Under jit=True, x and x_history are the evaluated iterates
        mapped outside the compiled function, and can differ from the points
        fun received by one unit in the last place; fun is the value at the
        received point. history holds finite objectives (the unscaled
        maximum for worst_case); x_history has shape (k, n). nfev includes a
        non-finite evaluation and excludes setup_forwards, which is 1 for
        worst_case and 0 otherwise. solves counts only loop passes. jacobian
        is "value_and_grad", "forward" or "reverse". On a non-finite first
        evaluation, x is the mapped start, fun is nan and both histories are
        empty. stopped_nonfinite reports such a stop at any evaluation.

    Raises
    ------
    RfxRefusal
        Invalid method, shape, bounds, start, budget, or incompatible options;
        worst_case initial entries violate the finite nonnegative contract.

    See also: optimize, scipy.optimize.minimize; minimize_s11 and related
    objective factories construct objectives rather than run this loop.

    Notes
    -----
    Method guide: written with the example's reproduction run.
    """
    if not isinstance(method, str) or method.lower() not in ("adam", "projected_adam", "mma"):
        raise RfxRefusal("method must be adam, projected_adam or mma; for SciPy methods use scipy.optimize.minimize")
    method = method.lower()
    if worst_case and method != "mma":
        raise RfxRefusal('worst_case requires method="mma", or use a soft-max of the entries')
    if method != "adam" and bounds is None:
        raise RfxRefusal(f"bounds are required for method={method!r}")
    if method == "mma" and lr is not None:
        raise RfxRefusal("lr is not accepted for method='mma'")
    if isinstance(n_iters, (bool, np.bool_)) or not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
        raise RfxRefusal("n_iters must be a positive integer")
    try:
        original = np.asarray(x0, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RfxRefusal("x0 must be a finite one-dimensional real array") from exc
    if original.ndim != 1 or original.size == 0 or not np.all(np.isfinite(original)):
        raise RfxRefusal("x0 must be finite, nonempty and one-dimensional")
    box = None if bounds is None else _bounds_arrays(bounds, original.size)
    if box is not None and (np.any(original < box[0]) or np.any(original > box[1])):
        raise RfxRefusal("x0 must lie inside bounds")
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    if isinstance(x0, (np.ndarray, jax.Array)) and x0.dtype == np.dtype("float64") and dtype == jnp.float32:
        warnings.warn("minimize: x0 converted from float64 to float32", ConfigurationWarning, stacklevel=2)
    start = jnp.asarray(x0, dtype=dtype)
    if not bool(jnp.all(jnp.isfinite(start))):
        raise RfxRefusal("x0 must remain finite in the default float dtype")

    if box is not None:
        lo64, hi64 = box
        try:
            lower, upper = _box_in_dtype(lo64, hi64, np.dtype(dtype))
        except ValueError as exc:
            raise RfxRefusal(f"bounds: {exc}") from exc
        lo, hi = jnp.asarray(lo64, dtype=dtype), jnp.asarray(hi64, dtype=dtype)
        if not bool(jnp.all(jnp.isfinite(hi - lo))):
            raise RfxRefusal("bounds must have finite width in the default float dtype")
        lower, upper = jnp.asarray(lower), jnp.asarray(upper)

        p0 = (original - lo64) / (hi64 - lo64)
        if method == "adam":
            with np.errstate(divide="ignore"):
                p0 = np.clip(np.log(p0 / (1 - p0)), -13.8, 13.8)
        start = jnp.asarray(p0, dtype=dtype)

    def physical(point):
        if box is None:
            return point
        p = jax.nn.sigmoid(point) if method == "adam" else point
        raw = lo + (hi - lo) * p
        return raw + jax.lax.stop_gradient(jnp.clip(raw, lower, upper) - raw)

    m = 0
    scale = None
    if worst_case:
        values = np.asarray(fun(physical(start)))
        if values.ndim != 1 or values.size == 0:
            raise RfxRefusal("fun must return a nonempty vector with worst_case=True, or a scalar otherwise")
        m = values.size
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.max(values) <= 0:
            raise RfxRefusal("worst_case fun entries at x0 must be finite and nonnegative with a positive maximum")
        scale = float(np.max(values))

    def objective(point):
        values = fun(physical(point))
        if worst_case:
            return jnp.concatenate((jnp.zeros(1, dtype=values.dtype), values / scale))
        if jnp.shape(values) != ():
            raise RfxRefusal("fun must return a nonempty vector with worst_case=True, or a scalar otherwise")
        return values

    rate = 0.01 if lr is None else lr
    opt = (_Adam(rate) if method == "adam" else _ProjectedAdam(rate)
           if method == "projected_adam" else _RecordingMMA(m=m, a=1.0, scale=scale))
    mode = "fwd" if m and start.size < m + 2 else "rev"
    _, history, visited, solves = _run_optimizer(
        opt, objective, start, (0.0, 1.0) if opt.native_box else None,
        n_iters=n_iters, m=m, jacobian=mode, jit=jit,
        keep_iterates=True, name="minimize",
    )
    if worst_case:
        history = opt.history
    nfev = solves["forward_mode"] // start.size if mode == "fwd" else solves["forward"]
    x_history = jnp.stack([physical(point) for point in visited]) if visited else jnp.empty((0, start.size), dtype=dtype)
    index = int(np.argmin(history)) if best_iterate and history else -1
    return MinimizeResult(
        x=x_history[index] if history else physical(start),
        fun=history[index] if history else float("nan"), nfev=nfev,
        history=history, x_history=x_history, solves=solves,
        setup_forwards=int(worst_case), method=method,
        jacobian="value_and_grad" if not m else "forward" if mode == "fwd" else "reverse",
        subproblem_caps=sum(opt.caps) if isinstance(opt, _RecordingMMA) else 0,
        stopped_nonfinite=nfev > len(history), scale=scale,
    )
