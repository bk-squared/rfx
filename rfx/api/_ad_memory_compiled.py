"""Compile-only device peak for the measured graded AD forward paths."""
from __future__ import annotations

from numbers import Integral
import os

import jax
import jax.numpy as jnp

from rfx.api._ad_memory_types import ADMemoryUnavailable
from rfx.api._compiled_memory import _normalize_compiled_memory_analysis
from rfx.api._validators import _require_integral_param
from rfx.backend import backend_class
from rfx.stepping.remat import active_levels, validate_cotangents


def _allocator_rounding(temp_bytes):
    """BFC growth-region excess and declared maximum, in bytes."""
    t = ((temp_bytes + 255) // 256) * 256
    region = max(2 * 1024**2, 1 << max(0, t - 1).bit_length())
    excess = region - t
    limit = 128 * 1024**2
    return (excess if excess < limit else 0), min(t, limit)


def _default_device():
    device = jax.config.jax_default_device
    if isinstance(device, str):
        return jax.local_devices(backend=device)[0]
    return device or jax.local_devices()[0]


def _resident_bytes(device):
    try:
        stats = device.memory_stats()
        value = None if stats is None else stats.get("bytes_in_use")
    except Exception as error:
        raise ADMemoryUnavailable("device memory_stats bytes_in_use unavailable") from error
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ADMemoryUnavailable("device memory_stats bytes_in_use unavailable")
    return int(value)


def _inexact_leaves(value):
    # Program outputs only: a NumPy leaf (N3's frequency array in
    # waveguide_sparams) is a constant with no pullback memory.
    return tuple(leaf for leaf in jax.tree_util.tree_leaves(value)
                 if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact))


def _measured_outputs(result):
    measured = ("time_series", "waveguide_ports", "waveguide_sparams")
    for name, value in result._asdict().items():
        if name not in (*measured, "grid") and _inexact_leaves(value):
            raise ADMemoryUnavailable(f"compiled AD memory: output {name} not measured")
    return tuple(leaf for name in measured
                 for leaf in _inexact_leaves(getattr(result, name)))


def compiled_estimate(sim, n_steps, *, available_memory_gb, checkpoint_every,
                      checkpoint_segments, n_warmup, design_box, design,
                      checkpoint, distributed, cotangents):
    """Compile, never run, the forward's pullback and estimate its device peak.

    The inputs are abstract float32 permittivity and conductivity grids plus
    one cotangent per inexact time_series/waveguide port record leaf. Multiple
    cotangents vmap the pullback. Other inexact output arrays except grid
    coordinates are unmeasured and refused, as are warmup steps and a plan
    that does not segment the run.

    R_res is read after compilation while traced, lowered and compiled objects
    are alive. It includes values evaluated during tracing (device arrays held
    by the program) and arrays the process already holds. Existing buffers
    contribute to the run's peak only if they remain live when it runs.
    No executable is returned or stored. The analytic fields are unchanged.
    """
    if not sim._uses_nonuniform_mesh:
        raise ADMemoryUnavailable("compiled AD memory: uniform path not measured")
    if distributed:
        raise ADMemoryUnavailable("compiled AD memory: distributed path not measured")
    if getattr(sim, "_refinement", None) is not None:
        raise ADMemoryUnavailable("compiled AD memory: subgridded path not measured")
    device = _default_device()
    if backend_class(device.platform).name != "gpu":
        raise ADMemoryUnavailable(f"compiled AD memory: backend {device.platform} not measured; requires GPU")
    _resident_bytes(device)  # Availability refusal precedes allocator admission.
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise ADMemoryUnavailable("compiled AD memory: preallocation not measured; XLA_PYTHON_CLIENT_PREALLOCATE=false required")
    if design != "permittivity" or design_box is not None:
        raise ADMemoryUnavailable("compiled AD memory: design other than whole-grid permittivity not measured")
    if sim._precision != "float32":
        raise ADMemoryUnavailable("compiled AD memory: float64 not measured; requires float32")
    lengths = checkpoint_every if isinstance(checkpoint_every, tuple) else (checkpoint_every,)
    if (checkpoint_segments is not None or len(lengths) not in (1, 2)
            or (isinstance(checkpoint_every, tuple) and len(lengths) != 2)
            or any(not isinstance(v, Integral) or isinstance(v, bool) or v <= 0 for v in lengths)):
        raise ADMemoryUnavailable("compiled AD memory: checkpoint plan not measured; requires checkpoint_every int or two-level tuple")
    if n_warmup != 0:
        raise ADMemoryUnavailable("compiled AD memory: n_warmup not measured; requires 0")
    n_steps_i = _require_integral_param("n_steps", n_steps)
    # The graded loop's own test (stepping/graded_loop.py): an unsegmented scan
    # runs instead, with or without the per-step checkpoint; not measured.
    if not (active_levels(n_steps_i, checkpoint_every) if isinstance(checkpoint_every, tuple)
            else 0 < checkpoint_every < n_steps_i):
        raise ADMemoryUnavailable("compiled AD memory: checkpoint plan does not segment the run; unsegmented scan not measured")
    cotangents = validate_cotangents(cotangents)
    grid = sim._build_realized_grid()
    inputs = tuple(jax.ShapeDtypeStruct(grid.shape, jnp.float32) for _ in range(2))

    def forward(eps, sigma):
        return _measured_outputs(sim.forward(
            n_steps=n_steps, checkpoint=checkpoint, checkpoint_every=checkpoint_every,
            checkpoint_segments=checkpoint_segments, n_warmup=n_warmup,
            eps_override=eps, sigma_override=sigma, skip_preflight=True))

    outputs = jax.eval_shape(forward, *inputs)
    cotangent_inputs = tuple(jax.ShapeDtypeStruct(
        ((cotangents,) + leaf.shape) if cotangents > 1 else leaf.shape, leaf.dtype)
        for leaf in outputs)
    analytic = sim.estimate_ad_memory(
        n_steps, available_memory_gb=available_memory_gb, checkpoint_every=checkpoint_every,
        checkpoint_segments=checkpoint_segments, n_warmup=n_warmup, design_box=design_box,
        design=design, checkpoint=checkpoint, distributed=distributed, cotangents=cotangents)

    def pullback(eps, sigma, cotangent):
        _, pull = jax.vjp(forward, eps, sigma)
        return jax.vmap(pull)(cotangent) if cotangents > 1 else pull(cotangent)

    traced = jax.jit(pullback).trace(*inputs, cotangent_inputs)
    lowered = traced.lower()
    compiled = lowered.compile()
    resident = _resident_bytes(device)
    analysis = _normalize_compiled_memory_analysis(compiled)
    if analysis.status != "complete":
        raise ADMemoryUnavailable(analysis.reason)
    fields = analysis.fields
    if fields["alias_size_in_bytes"] != 0:
        raise ADMemoryUnavailable("compiled AD memory: nonzero alias_size_in_bytes not measured")
    compiled_bytes = sum(fields[name] for name in (
        "temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes"))
    rounding, rounding_max = _allocator_rounding(fields["temp_size_in_bytes"])
    return analytic._replace(
        compiled_bytes=compiled_bytes, resident_bytes=resident,
        allocator_rounding_bytes=rounding, allocator_rounding_max_bytes=rounding_max,
        compiled_total_bytes=compiled_bytes + resident + rounding)
