"""Precision admission and matrix-product precision for rfx.

On Ampere-class and newer NVIDIA GPUs, JAX's default precision for a float32
(and complex64) matrix product is TF32: the operands are rounded to a 10-bit
mantissa before multiplying, about 1e-3 relative per product. On an RTX 3090
that moved a completed S-parameter by up to 4e-4, twenty times the float32
bar of the ring-down tests; with ``HIGHEST`` the same tests agree to 1e-6
(rfx #1364; records ``rfx-archive`` ``rfx/records/20260929-tf32-ab/``).
CPUs and older GPUs compute the same thing either way.

Every ``jnp.einsum`` / ``jnp.matmul`` / ``jnp.dot`` / ``lax.conv*`` in rfx passes
``precision=HIGHEST``. A library routine whose derivative issues its own
products (``jnp.linalg.qr``) is called under
``jax.default_matmul_precision("highest")``. ``tests/contracts/test_matmul_precision.py``
lowers the entry points that reach them, including a gradient program, and
requires every ``stablehlo.dot_general`` and ``stablehlo.convolution`` to carry
``HIGHEST``, so a new contraction without it turns that test red on CPU.
"""

from __future__ import annotations

import jax

HIGHEST = jax.lax.Precision.HIGHEST


FLOAT64_MODE_ERROR = (
    "precision='float64' needs JAX 64-bit mode, which is off in this process: "
    "rfx would compute in float32. Build and run the Simulation inside a scoped "
    "`with jax.enable_x64():` block (do not switch it on at module level)."
)


# The jax-mps plugin (Apple-silicon GPUs) has no float64: a float64 transfer
# fails, a float64 value inside jit is cast to float32, and a complex128
# transfer returns wrong values without an error (#1824). Under JAX 64-bit
# mode a float32 run also forms complex128 spectra (DFT accumulators, S), so
# the mode itself is refused there, whatever the precision.
FLOAT64_MPS_ERROR = (
    "precision='float64' and JAX 64-bit mode are not available on the "
    "Apple-silicon 'mps' JAX backend: the device has no float64, so rfx would "
    "compute in float32 or return wrong complex128 values. Use "
    "precision='float32' with 64-bit mode off on this backend, or run float64 "
    "on the CPU with JAX_PLATFORMS=cpu."
)


def resolve_field_dtype(precision):
    """Resolve storage without allocating or changing the caller's JAX mode."""
    import jax.numpy as jnp

    if (precision == "float64" or jax.config.x64_enabled) and jax.default_backend() == "mps":
        raise ValueError(FLOAT64_MPS_ERROR)
    if precision == "float64":
        if not jax.config.x64_enabled:
            raise ValueError(FLOAT64_MODE_ERROR)
        return jnp.float64
    return jnp.float16 if precision == "mixed" else None
