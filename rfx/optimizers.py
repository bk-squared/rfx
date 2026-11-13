"""Private ask/tell optimisers for the inverse-design loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import jax.numpy as jnp


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
