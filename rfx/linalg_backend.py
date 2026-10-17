"""Complex linear solves using real block systems on the mps backend."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def _is_complex(*arrays):
    return any(jnp.iscomplexobj(a) for a in arrays)


def _block_solve(a, b):
    """Solve complex matrix RHSs (or an unbatched vector) in the real dtype."""
    if not _is_complex(a, b):
        return jnp.linalg.solve(a, b)
    dtype = jnp.result_type(a, b)
    a, b = jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    n = a.shape[-1]
    ar, ai = jnp.real(a), jnp.imag(a)
    matrix = jnp.concatenate([
        jnp.concatenate([ar, -ai], axis=-1),
        jnp.concatenate([ai, ar], axis=-1),
    ], axis=-2)
    rhs = jnp.concatenate([jnp.real(b), jnp.imag(b)], axis=-2)
    y = jnp.linalg.solve(matrix, rhs)
    x = (y[..., :n, :] + 1j * y[..., n:, :]).astype(dtype)
    return x[:, 0] if vector else x


def complex_solve(a, b):
    """Use the native solve except for complex operands on mps at trace time."""
    if jax.default_backend() == "mps" and _is_complex(a, b):
        return _block_solve(a, b)
    return jnp.linalg.solve(a, b)


def complex_inv(a):
    """Use a real block solve for a complex inverse on mps."""
    if jax.default_backend() == "mps" and _is_complex(a):
        return _block_solve(a, jnp.eye(a.shape[-1], dtype=a.dtype))
    return jnp.linalg.inv(a)


def complex_solve_triangular(r, b, *, lower):
    """The real block form is general, even when the complex input is triangular."""
    if jax.default_backend() == "mps" and _is_complex(r, b):
        return _block_solve(r, b)
    return jsp_linalg.solve_triangular(r, b, lower=lower)
