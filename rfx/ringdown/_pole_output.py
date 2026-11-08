"""Traced ringdown pole outputs and target-based selection."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfx.ringdown import RingdownReport


def select_pole(poles, mask, f_target, window_s):
    """Return traced (frequency Hz, 1/Q) of exactly one pole near f_target.

    Candidates have mask 1, positive imaginary part and negative real part.
    The radius is 1/window_s; use the MAIN window length for both windows.
    Selection is stop-gradient; the gathered pole retains the completion's
    derivative. No unique candidate gives NaN values and gradients. Mask 2
    is unresolved and never selected. See :func:`pole_selection_reason` for
    a concrete reason. Matching across windows assumes isolated neighbours;
    a wrong match in general must be checked with gradient_witness.
    """
    import jax
    import jax.numpy as jnp
    poles = jnp.asarray(poles)
    if poles.size == 0:
        nan = jnp.real(jnp.sum(poles)) * jnp.nan
        return nan, nan
    fixed = jax.lax.stop_gradient(poles)
    eligible = (jnp.asarray(mask) == 1) & (fixed.imag > 0) & (fixed.real < 0)
    distance = jnp.abs(fixed.imag / (2 * jnp.pi) - f_target)
    ok = jnp.sum(eligible & (distance <= 1 / window_s)) == 1
    index = jnp.argmin(jnp.where(eligible, distance, jnp.inf))
    pole = poles[index]
    factor = jnp.where(ok, 1., jnp.nan)
    return pole.imag / (2 * jnp.pi) * factor, -2 * pole.real / pole.imag * factor


def pole_selection_reason(poles, mask, f_target, window_s, status=None):
    """Host-only reason for non-selection, or None; status names failed windows."""
    if status is not None and int(status) != 0:
        from rfx.ringdown import _status_reason
        return "failed window: " + _status_reason(status, kept_count=-1, pole_budget=-1)
    poles, mask = np.asarray(poles), np.asarray(mask)
    near = np.abs(poles.imag / (2 * np.pi) - f_target) <= 1 / window_s
    count = np.count_nonzero(near & (mask == 1) & (poles.imag > 0) & (poles.real < 0))
    if count == 1:
        return None
    if count:
        return f"{count} candidates within delta"
    unresolved = np.any(near & (mask == 2) & (poles.imag > 0))
    return "none within delta" + ("; unresolved mask-2 pole within delta" if unresolved else "")


class RingdownForwardResult:
    """``ForwardResult.ringdown``: the completed S-parameters, traced and differentiable.

    ``s_params`` is the completion from the main window
    ``[window_start T, T]`` and ``s_params_long`` the one from the window that
    starts twice as early, ``[window_start/2 T, T]`` (the two ``WE``
    compares); both have the shape and dtype of ``ForwardResult.s_params`` and
    its bins ``freqs``. Their VALUE is ``run(ringdown=...)``'s completion of
    the same record, computed in the working precision; their DERIVATIVE is
    that of the infinite-record spectrum (poles by the frozen Gauss-Newton
    implicit derivative, residues re-fitted). When the host identification
    fails on a window, that array is NaN and the report names the failure.

    ``report`` is a :class:`RingdownReport` computed on the host the first time
    it is read, from a CONCRETE result (the float64 completion, W0, W1, WE and
    the rest, as ``run()``'s, plus ``traced``: the traced value against the
    float64 one). It is ``None`` while the arrays are traced, like
    ``ForwardResult.settling_db``, and it never warns.

    ``poles`` and ``poles_long`` are read-only working-complex arrays, with
    int32 ``pole_mask`` and ``pole_mask_long`` (0 padding, 1 completed,
    2 unresolved). They share the completion derivative and failure factor.
    Under vmap they acquire a leading batch axis.

    A JAX pytree: the arrays are its leaves.
    """

    __slots__ = ("s_params", "s_params_long", "freqs", "_channels", "_accs",
                 "_s_plain", "_status", "_consistency", "_pole_counts", "_pole_budgets",
                 "_ctx", "_report", "_poles", "_poles_long", "_pole_mask", "_pole_mask_long")

    def __init__(self, s_params, s_params_long, freqs, channels, accs, s_plain,
                 status, consistency, pole_counts, pole_budgets, ctx, *, poles=None, poles_long=None,
                 pole_mask=None, pole_mask_long=None):
        self.s_params = s_params
        self.s_params_long = s_params_long
        self.freqs = freqs
        self._channels = channels
        self._accs = accs
        self._s_plain = s_plain
        self._status = status
        self._consistency = consistency
        self._pole_counts = pole_counts
        self._pole_budgets = pole_budgets
        self._ctx = ctx
        self._report = None
        from rfx.ringdown._jax import working_dtypes
        import jax.numpy as jnp
        _, cdt = working_dtypes()
        self._poles = jnp.empty(0, cdt) if poles is None else poles
        self._poles_long = jnp.empty(0, cdt) if poles_long is None else poles_long
        self._pole_mask = jnp.empty(0, jnp.int32) if pole_mask is None else pole_mask
        self._pole_mask_long = jnp.empty(0, jnp.int32) if pole_mask_long is None else pole_mask_long

    def tree_flatten(self):
        return ((self.s_params, self.s_params_long, self.freqs, self._channels,
                 self._accs, self._s_plain, self._status, self._consistency,
                 self._pole_counts, self._pole_budgets, self.poles, self.poles_long,
                 self.pole_mask, self.pole_mask_long), self._ctx)

    @classmethod
    def tree_unflatten(cls, ctx, children):
        return cls(*children[:10], ctx, poles=children[10], poles_long=children[11],
                   pole_mask=children[12], pole_mask_long=children[13])

    @property
    def poles(self):
        """Main-window traced poles in exp(s t), in 1/s; padding follows host order."""
        return self._poles

    @property
    def poles_long(self):
        """Earlier-window poles with the completion's own implicit derivative."""
        return self._poles_long

    @property
    def pole_mask(self):
        """Main-window int32 mask: 0 padding, 1 completed, 2 unresolved."""
        return self._pole_mask

    @property
    def pole_mask_long(self):
        """Earlier-window int32 mask with the same three states."""
        return self._pole_mask_long

    def _pole_window_s(self):
        plan = self._ctx.plan
        return (plan.n_steps - plan.n_start) * plan.grid.dt

    def pole(self, f_target, window="main"):
        """Return (f, 1/Q) using the main window's resolution for either window.

        Both values and their derivatives are NaN without a unique completed
        positive-frequency decaying pole. See :func:`pole_selection_reason`.
        """
        if window not in ("main", "long"):
            raise ValueError("window must be 'main' or 'long'")
        if np.ndim(self._channels) != 2:
            raise ValueError(
                "RingdownForwardResult.pole reads one result; this one is batched "
                f"(its port channels have shape {tuple(np.shape(self._channels))}, "
                "one record is (n_steps, channels)), as jax.vmap returns it. Take "
                "one element first: jax.tree_util.tree_map(lambda a: a[i], result).")
        poles, mask = ((self.poles, self.pole_mask) if window == "main" else
                       (self.poles_long, self.pole_mask_long))
        return select_pole(poles, mask, f_target, self._pole_window_s())

    @property
    def report(self) -> RingdownReport | None:
        if self._report is not None:
            return self._report
        import jax

        from rfx.core.jax_utils import is_tracer
        leaves = jax.tree_util.tree_leaves(
            (self.s_params, self.s_params_long, self._channels, self._accs,
             self._s_plain))
        if any(is_tracer(x) for x in leaves):
            return None
        if np.ndim(self._channels) != 2:
            raise ValueError(
                "RingdownForwardResult.report reads one result; this one is batched "
                f"(its port channels have shape {tuple(np.shape(self._channels))}, "
                "one record is (n_steps, channels)), as jax.vmap returns it. Take "
                "one element first: jax.tree_util.tree_map(lambda a: a[i], result).")
        self._report = self._ctx.report(self)
        return self._report

    def __repr__(self):
        return (f"RingdownForwardResult(s_params={self.s_params!r}, "
                f"s_params_long={self.s_params_long!r}, freqs={self.freqs!r}, "
                f"poles={self.poles!r}, poles_long={self.poles_long!r}, "
                f"pole_mask={self.pole_mask!r}, pole_mask_long={self.pole_mask_long!r})")


def _register_pytree():
    import jax
    jax.tree_util.register_pytree_node_class(RingdownForwardResult)


_register_pytree()

