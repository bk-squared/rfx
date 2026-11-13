"""Private ask/tell optimisers for the inverse-design loops."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings
from typing import Protocol, TypeVar

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class _Request:
    x: jnp.ndarray
    need_grad: bool
    iterate: bool


_State = TypeVar("_State")


class _Optimizer(Protocol[_State]):
    native_box: bool

    def init(self, x0: jnp.ndarray, bounds) -> _State: ...
    def ask(self, state: _State) -> _Request: ...
    def tell(self, state: _State, request: _Request, value, grad,
             cons_values, cons_grads) -> _State: ...
    def restart(self, state: _State) -> _State: ...


@dataclass(frozen=True)
class _AdamState:
    x: jnp.ndarray
    m: jnp.ndarray
    v: jnp.ndarray
    # Keep bias correction in Python double precision, outside grad_fn's jit.
    it: int


@dataclass(frozen=True)
class _Adam(_Optimizer[_AdamState]):
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    step_clamp: float | None = None
    native_box = False

    def init(self, x0: jnp.ndarray, bounds) -> _AdamState:
        return _AdamState(x0, jnp.zeros_like(x0), jnp.zeros_like(x0), 0)

    def ask(self, state: _AdamState) -> _Request:
        return _Request(state.x, need_grad=True, iterate=True)

    def tell(self, state: _AdamState, request: _Request, value, grad,
             cons_values, cons_grads) -> _AdamState:
        # Local import keeps the shared arithmetic in optimize without a cycle.
        from rfx.optimize import _adam_update

        step, m, v = _adam_update(
            state.m, state.v, grad, state.it, self.lr,
            self.beta1, self.beta2, self.eps, self.step_clamp,
        )
        return _AdamState(request.x - step, m, v, state.it + 1)

    def restart(self, state: _AdamState) -> _AdamState:
        return state


# Implemented from the equations in Svanberg, MMA and GCMMA (2007),
# sections 3 and 5. No MATLAB implementation or GPL port is used here.
@dataclass(frozen=True)
class _MMASubproblemResult:
    x: np.ndarray
    y: np.ndarray
    z: float
    lam: np.ndarray
    xi: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    zeta: float
    s: np.ndarray
    epsilon: float
    newton_steps: int
    caps_hit: int


def _mma_subproblem(low, upp, alpha, beta, p, q, r, a0, a, c, d,
                    *, max_halvings=50, max_newton=500):
    """Solve (5.1) in host float64; p/q/r include the objective row.

    Return primal variables and the dual/slack witness for (5.9). The constant
    r[0] does not affect the solution; b in section 5 is -r[1:].
    """
    low, upp, alpha, beta, p, q, r, a, c, d = (
        np.asarray(v, dtype=np.float64)
        for v in (low, upp, alpha, beta, p, q, r, a, c, d)
    )
    n, m = low.size, a.size
    x = (alpha + beta) / 2
    # The order of w follows (5.9), with scalar z and zeta in length-one arrays.
    w = (x, np.ones(m), np.ones(1), np.ones(m),
         np.maximum(1, 1 / (x - alpha)), np.maximum(1, 1 / (beta - x)),
         np.maximum(1, c / 2), np.ones(1), np.ones(m))

    def quantities(w):
        x, y, z, lam, xi, eta, mu, zeta, s = w
        invu, invl = 1 / (upp - x), 1 / (x - low)
        pl = p[0] + np.einsum('i,ij->j', lam, p[1:])
        ql = q[0] + np.einsum('i,ij->j', lam, q[1:])
        psi = pl * invu**2 - ql * invl**2
        g = np.einsum('ij,j->i', p[1:], invu) + np.einsum('ij,j->i', q[1:], invl)
        return invu, invl, pl, ql, psi, g

    def residual(w, eps, data=None):
        x, y, z, lam, xi, eta, mu, zeta, s = w
        if data is None:
            data = quantities(w)
        *_, psi, g = data
        res = np.concatenate((psi - xi + eta, c + d * y - lam - mu,
                               a0 - zeta - a @ lam,
                               g - a * z - y + s + r[1:],
                               xi * (x - alpha) - eps,
                               eta * (beta - x) - eps, mu * y - eps,
                               zeta * z - eps, lam * s - eps))
        return res, data

    steps, caps = 0, 0
    eps = 1.0
    res, data = residual(w, eps)
    while True:
        if np.max(np.abs(res)) < 0.9 * eps:
            if eps <= 1e-7:
                break
            # Pin the last level to avoid an extra decade from binary rounding.
            eps = max(1e-7, 0.1 * eps)
            res, data = residual(w, eps, data)
            continue
        if steps >= max_newton:
            caps += 1
            warnings.warn('MMA subproblem reached max_newton', RuntimeWarning, stacklevel=2)
            break
        x, y, z, lam, xi, eta, mu, zeta, s = w
        # Reuse the quantities evaluated by the accepted line-search trial.
        invu, invl, pl, ql, psi, g = data
        G = p[1:] * invu**2 - q[1:] * invl**2
        # Diagonal blocks and reduced residuals (5.15)-(5.18).
        Dx = 2 * pl * (invu * invu * invu) + 2 * ql * (invl * invl * invl) + xi / (x - alpha) + eta / (beta - x)
        Dy = d + mu / y
        Dly = s / lam + 1 / Dy
        tx = psi - eps / (x - alpha) + eps / (beta - x)
        ty = c + d * y - lam - eps / y
        tz = a0 - a @ lam - eps / z
        tly = g - a * z - y + r[1:] + eps / lam + ty / Dy
        if n > m:
            # (5.20): retain z in the (m+1)-square system.
            matrix = np.empty((m + 1, m + 1))
            matrix[:m, :m] = np.diag(Dly) + np.einsum('ij,kj,j->ik', G, G, 1 / Dx)
            matrix[:m, m] = a
            matrix[m, :m] = a
            matrix[m, m] = -zeta[0] / z[0]
            rhs = np.concatenate((tly - np.einsum('ij,j->i', G, tx / Dx), tz))
            sol = np.linalg.solve(matrix, rhs)
            dl, dz = sol[:m], sol[m:]
            dx = -(tx + np.einsum('i,ij->j', dl, G)) / Dx
        else:
            # (5.22): retain z in the (n+1)-square system.
            matrix = np.empty((n + 1, n + 1))
            matrix[:n, :n] = np.diag(Dx) + np.einsum('ij,ik,i->jk', G, G, 1 / Dly)
            cross = -np.einsum('ij,i->j', G, a / Dly)
            matrix[:n, n] = cross
            matrix[n, :n] = cross
            matrix[n, n] = zeta[0] / z[0] + np.sum(a * a / Dly)
            rhs = np.concatenate((-tx - np.einsum('ij,i->j', G, tly / Dly),
                                  -tz + np.sum(a * tly / Dly)))
            sol = np.linalg.solve(matrix, rhs)
            dx, dz = sol[:n], sol[n:]
            dl = (np.einsum('ij,j->i', G, dx) - a * dz + tly) / Dly
        dy = (dl - ty) / Dy
        # Recover the other five directions via (5.13).
        dw = (dx, dy, dz, dl,
              -xi * dx / (x - alpha) - xi + eps / (x - alpha),
              eta * dx / (beta - x) - eta + eps / (beta - x),
              -mu * dy / y - mu + eps / y,
              -zeta * dz / z - zeta + eps / z,
              -s * dl / lam - s + eps / lam)
        # Largest t <= 1 retaining at least 1% of each positive distance (§5.4).
        rate = max(0.0, float(np.max(-dx / (x - alpha))),
                   float(np.max(dx / (beta - x))))
        for v, dv in zip(w[1:], dw[1:]):
            rate = max(rate, float(np.max(-dv / v, initial=0)))
        t = min(1.0, 0.99 / rate) if rate > 0 else 1.0
        norm2 = np.einsum('i,i->', res, res)
        accepted = False
        for halvings in range(max_halvings + 1):
            trial = tuple(v + t * dv for v, dv in zip(w, dw))
            trial_res, trial_data = residual(trial, eps)
            if np.einsum('i,i->', trial_res, trial_res) < norm2:
                w, res, data = trial, trial_res, trial_data
                accepted = True
                break
            if halvings < max_halvings:
                t *= 0.5
        steps += 1
        if not accepted:
            caps += 1
            warnings.warn('MMA subproblem reached max_halvings', RuntimeWarning, stacklevel=2)
            break
    x, y, z, lam, xi, eta, mu, zeta, s = w
    return _MMASubproblemResult(x, y, float(z[0]), lam, xi, eta, mu,
                               float(zeta[0]), s, eps, steps, caps)


@dataclass(frozen=True)
class _MMAState:
    x: jnp.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    xold1: np.ndarray | None
    xold2: np.ndarray | None
    low: np.ndarray
    upp: np.ndarray
    k: int
    y: np.ndarray
    z: float
    newton_steps_last: int
    caps_hit: int


@dataclass(frozen=True)
class _MMA(_Optimizer[_MMAState]):
    m: int
    a0: float = 1.0
    a: object = None
    c: object = 1000.0
    d: object = 1.0
    move: float = 0.5
    albefa: float = 0.1
    asyinit: float = 0.5
    asydecr: float = 0.7
    asyincr: float = 1.2
    raa0: float = 1e-5
    max_halvings: int = 50
    max_newton: int = 500
    native_box = True

    def __post_init__(self):
        if self.m < 0 or int(self.m) != self.m:
            raise ValueError('MMA m must be a nonnegative integer')
        for name in ('a', 'c', 'd'):
            value = getattr(self, name)
            arr = np.broadcast_to(np.asarray(0 if value is None else value, dtype=np.float64), (self.m,)).copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        a, c, d = (np.asarray(v) for v in (self.a, self.c, self.d))
        if (not np.isfinite(self.a0) or self.a0 <= 0
                or any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in (a, c, d))
                or np.any(c + d <= 0) or np.any((a > 0) & (a * c <= self.a0))):
            raise ValueError('MMA coefficients must satisfy Svanberg (1.1)')
        if (not 0 < self.albefa < 1 or self.move <= 0 or self.asyinit <= 0
                or self.asydecr <= 0 or self.asyincr <= 0 or self.raa0 <= 0
                or self.max_halvings < 0 or self.max_newton < 1):
            raise ValueError('Invalid MMA move, asymptote, regularization or cap parameter')

    @staticmethod
    def _design_point(x, dtype, shape, xmin, xmax):
        # Round the solution before storing history or asking for an evaluation.
        rounded = np.asarray(x, dtype=dtype)
        lower, upper = xmin.astype(dtype), xmax.astype(dtype)
        # A nearest-rounded bound can lie outside the original float64 box.
        lower = np.where(lower < xmin, np.nextafter(lower, np.asarray(np.inf, dtype=dtype)), lower)
        upper = np.where(upper > xmax, np.nextafter(upper, np.asarray(-np.inf, dtype=dtype)), upper)
        if np.any(lower > upper):
            raise ValueError('MMA rounded box is empty in the design dtype')
        rounded = np.clip(rounded, lower, upper)
        return jnp.asarray(rounded.reshape(shape), dtype=dtype)

    def init(self, x0: jnp.ndarray, bounds) -> _MMAState:
        if bounds is None:
            raise ValueError('MMA is a box method: bounds are required')
        x = np.asarray(x0, dtype=np.float64).ravel()
        xmin, xmax = (np.broadcast_to(np.asarray(b, dtype=np.float64), x0.shape).ravel().copy() for b in bounds)
        if (x.size == 0 or not np.all(np.isfinite(x))
                or not np.all(np.isfinite(xmin)) or not np.all(np.isfinite(xmax))
                or np.any(xmin >= xmax)):
            raise ValueError('MMA requires finite points and finite, strictly ordered bounds')
        point = self._design_point(np.clip(x, xmin, xmax), x0.dtype, x0.shape, xmin, xmax)
        x = np.asarray(point, dtype=np.float64).ravel()
        return _MMAState(point, xmin, xmax, None, None,
                         x - self.asyinit * (xmax - xmin), x + self.asyinit * (xmax - xmin),
                         0, np.zeros(self.m), 0.0, 0, 0)

    def ask(self, state: _MMAState) -> _Request:
        return _Request(state.x, need_grad=True, iterate=True)

    def tell(self, state: _MMAState, request: _Request, value, grad,
             cons_values, cons_grads) -> _MMAState:
        x = np.asarray(request.x, dtype=np.float64).ravel()
        gradient = np.asarray(grad, dtype=np.float64)
        f0 = np.asarray(value, dtype=np.float64)
        if gradient.shape != request.x.shape or f0.shape != ():
            raise ValueError('MMA needs a scalar objective and a gradient shaped like x')
        if self.m == 0 and cons_values is None and cons_grads is None:
            cons_values, cons_grads = np.empty(0), np.empty((0, *request.x.shape))
        f = np.asarray(cons_values, dtype=np.float64)
        df = np.asarray(cons_grads, dtype=np.float64)
        if f.shape != (self.m,) or df.shape != (self.m, *request.x.shape):
            raise ValueError('MMA constraint values/gradients must have shapes (m,) and (m, *x.shape)')
        width = state.xmax - state.xmin
        k = state.k + 1
        if k <= 2:
            low, upp = x - self.asyinit * width, x + self.asyinit * width
        else:
            assert state.xold1 is not None and state.xold2 is not None
            trend = (x - state.xold1) * (state.xold1 - state.xold2)
            factor = np.where(trend < 0, self.asydecr, np.where(trend > 0, self.asyincr, 1))
            low = np.clip(x - factor * (state.xold1 - state.low), x - 10 * width, x - 0.01 * width)
            upp = np.clip(x + factor * (state.upp - state.xold1), x + 0.01 * width, x + 10 * width)
        alpha = np.maximum.reduce((state.xmin, low + self.albefa * (x - low), x - self.move * width))
        beta = np.minimum.reduce((state.xmax, upp - self.albefa * (upp - x), x + self.move * width))
        derivatives = np.concatenate((gradient.reshape(1, -1), df.reshape(self.m, x.size)))
        positive, negative = np.maximum(derivatives, 0), np.maximum(-derivatives, 0)
        # (3.3)-(3.5), including the constraint rows without rescaling their data.
        p = (upp - x)**2 * (1.001 * positive + 0.001 * negative + self.raa0 / width)
        q = (x - low)**2 * (0.001 * positive + 1.001 * negative + self.raa0 / width)
        r = np.concatenate((f0.reshape(1), f)) - np.sum(p / (upp - x) + q / (x - low), axis=1)
        result = _mma_subproblem(low, upp, alpha, beta, p, q, r,
                                 self.a0, self.a, self.c, self.d,
                                 max_halvings=self.max_halvings, max_newton=self.max_newton)
        point = self._design_point(result.x, request.x.dtype, request.x.shape, state.xmin, state.xmax)
        return _MMAState(point, state.xmin, state.xmax, x.copy(), state.xold1,
                         low, upp, k, result.y, result.z, result.newton_steps,
                         state.caps_hit + result.caps_hit)

    def restart(self, state: _MMAState) -> _MMAState:
        return replace(state, k=0, xold1=None, xold2=None)
