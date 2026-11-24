"""Host checks for concrete operands and backend admission for traced ones."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from rfx.backend import backend_class, host_cpu_device

_STAGED = object()


def _host_callbacks_refusal(feature, check_name):
    """Return the refusal for a backend that cannot run a host callback.

    A GPU process runs callbacks on its CPU device; with JAX_PLATFORMS
    lacking ``cpu`` it has none, and the callback fails only after the solve.
    """
    backend = backend_class()
    if not backend.host_callbacks:
        return NotImplementedError(
            f"{feature}: {check_name}: this backend ({backend.name}) cannot run "
            "a host check inside compiled code")
    if backend.name == "cpu":
        return None
    try:
        host_cpu_device()
    except RuntimeError as exc:
        refusal = RuntimeError(
            f"{feature}: {check_name}: a host check inside compiled code runs on "
            f"this process's CPU device, and JAX has none (JAX_PLATFORMS="
            f"{os.environ.get('JAX_PLATFORMS', '')!r}); add cpu, e.g. "
            "JAX_PLATFORMS=cuda,cpu")
        refusal.__cause__ = exc
        return refusal
    return None


def host_callbacks_admitted():
    """Whether this process can run a host callback inside compiled code."""
    return _host_callbacks_refusal("", "") is None


def require_host_callbacks(feature, check_name):
    """Refuse at trace time a backend that cannot run a host callback."""
    refusal = _host_callbacks_refusal(feature, check_name)
    if refusal is not None:
        raise refusal



def _callback_transport(check, operands):
    """Preserve traced bits and x64 scope on host callback worker threads.

    Checks observe primals, so their integer transport has no tangent.
    """
    x64_enabled = jax.config.jax_enable_x64
    cpu_device = host_cpu_device()
    leaves, structure = jax.tree.flatten(operands)
    descriptions = []
    transported = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        dtype = np.dtype(array.dtype)
        wide = dtype in (np.dtype('float64'), np.dtype('int64'),
                         np.dtype('uint64'), np.dtype('complex128'))
        descriptions.append((dtype, array.shape) if wide else None)
        if not wide:
            transported.append(leaf)
            continue
        array = jax.lax.stop_gradient(array)
        if dtype == np.dtype('complex128'):
            transported.append(tuple(jax.lax.bitcast_convert_type(part, jnp.uint32)
                                     for part in (array.real, array.imag)))
        else:
            transported.append(jax.lax.bitcast_convert_type(array, jnp.uint32))

    def receive(*values):
        rebuilt = []
        with jax.enable_x64(x64_enabled):
            for value, description in zip(values, descriptions):
                if description is not None:
                    dtype, shape = description
                    if dtype == np.dtype('complex128'):
                        parts = [np.ascontiguousarray(np.asarray(part)).view(np.float64)
                                 .reshape(shape) for part in value]
                        value = np.ascontiguousarray(np.stack(parts, axis=-1)).view(dtype)
                    else:
                        value = np.ascontiguousarray(np.asarray(value)).view(dtype)
                    value = value.reshape(shape)
                    value = jax.device_put(value, cpu_device)
                rebuilt.append(value)
            return check(*jax.tree.unflatten(structure, rebuilt))

    return receive, tuple(transported)


def _host_with_concrete_leaves(check, leaves, tree):
    """Split leaves: JAX arrays go through the callback, the rest stay as given.

    Returns the host function and the staged leaves. Tracers and concrete
    ``jax.Array`` leaves are staged: a device array keeps its dtype and values
    through the callback, and a host function that read device memory from
    inside a running program could deadlock on GPU. Every other leaf (Python
    scalars, NumPy arrays, None, other objects) reaches ``check`` as the same
    object the caller passed, so a traced call sees the concrete operands an
    eager call sees. The closure holds no tracer and no device array.
    """
    positions = tuple(i for i, value in enumerate(leaves)
                      if isinstance(value, (jax.core.Tracer, jax.Array)))
    held = [_STAGED if i in positions else value for i, value in enumerate(leaves)]
    staged = [leaves[i] for i in positions]

    def host(*values):
        filled = list(held)
        for i, value in zip(positions, values):
            filled[i] = value
        return check(*jax.tree.unflatten(tree, filled))

    return host, staged


def traced_check(feature, check_name, check, *operands, ordered=False,
                 result_shape=None):
    """Run concrete checks now; retain callbacks (and optional tokens) in traces.

    Inspect pytree leaves: records can contain arrays nested in dictionaries
    or material tuples. ``result_shape`` retains an IO token's data dependency.
    Only tracer and ``jax.Array`` leaves pass through the callback; Python
    scalars and NumPy arrays reach the host function unchanged, with their
    own types and dtypes.
    """
    leaves, tree = jax.tree.flatten(operands)
    if not any(isinstance(value, jax.core.Tracer) for value in leaves):
        return check(*operands)
    require_host_callbacks(feature, check_name)
    host, staged = _host_with_concrete_leaves(check, leaves, tree)
    host, staged = _callback_transport(host, tuple(staged))
    if result_shape is not None:
        from jax.experimental import io_callback
        return io_callback(host, result_shape, *staged, ordered=ordered)
    return jax.debug.callback(host, *staged, ordered=ordered)
