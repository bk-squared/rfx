"""Compile the requested scan's single-step pullback on abstract inputs."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def compile_pullback(program):
    """Return one step's compiler memory analysis, without executing it.

    The scan body is already flattened by JAX. Its constants, coefficients,
    carry and step operands are all dynamic abstract inputs here; no values
    from the traced forward are allocated or run. Integer outputs receive
    float0 cotangents, exactly as in JAX's transpose of the original step.
    """
    body = program.jaxpr
    count = len(body.constvars)

    def step(*values):
        return jax.core.eval_jaxpr(body, values[:count], *values[count:])

    def pullback(primals, cotangents):
        _, backward = jax.vjp(step, *primals)
        return backward(cotangents)

    primals = tuple(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
                    for v in (*body.constvars, *body.invars))
    cotangents = [jax.ShapeDtypeStruct(v.aval.shape,
        v.aval.dtype if jnp.issubdtype(v.aval.dtype, jnp.inexact) else jax.dtypes.float0)
        for v in body.outvars]
    return jax.jit(pullback).lower(primals, cotangents).compile().memory_analysis()


# Compiled (argument, temp) bytes by step-program text. Equal text is the
# same program on the same shapes, so its memory analysis is the same; the
# closed-box planner asks once per candidate length for one or two programs.
_COMPILED_STEP_BYTES = {}
_COMPILED_STEP_LIMIT = 32


def _step_bytes(program):
    key = (jax.default_backend(), str(program))
    if key not in _COMPILED_STEP_BYTES:
        memory = compile_pullback(program)
        if len(_COMPILED_STEP_BYTES) >= _COMPILED_STEP_LIMIT:
            _COMPILED_STEP_BYTES.pop(next(iter(_COMPILED_STEP_BYTES)))
        _COMPILED_STEP_BYTES[key] = (memory.argument_size_in_bytes, memory.temp_size_in_bytes)
    return _COMPILED_STEP_BYTES[key]


def pullback_terms(program, fallback_bytes):
    """Read c and w; a failed compile keeps the labelled per-cell fallback."""
    try:
        argument, working = _step_bytes(program)
        fixed = argument + working
    except Exception as error:
        warning = ("One-step pullback compile unavailable "
                   f"({type(error).__name__}: {error}); AD_MEMORY_C0_PER_CELL fallback "
                   "used for c; w=0, so each extra cotangent adds carry_bytes only.")
        return fallback_bytes, 0, "per-cell fallback (GPU, jax 0.10.2)", [warning]
    return fixed, working, "compiled one-step pullback (estimate)", []
