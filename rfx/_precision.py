"""Matrix-product precision for every JAX contraction rfx issues.

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
