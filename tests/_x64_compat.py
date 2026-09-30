"""Version-robust scoped-x64 context (local jax-drift class, see ledger).

``jax.experimental.enable_x64`` (the scoped context manager this repo's
AD/referee tests use per the "never flip x64 at module level" rule) was
removed in newer JAX releases. The push-only Python 3.10 compatibility lane
resolves JAX 0.6.2, which exports it; required CI uses Python 3.11 and JAX
0.10.2, where the bare import fails at COLLECTION, aborting whole-tree
``-k`` runs before a single test executes. This shim keeps the upstream
context manager when it exists and otherwise provides the same semantics the sanctioned way:
a per-scope flip of ``jax_enable_x64`` with guaranteed restore — exactly
the "scope x64 per-test (fixture/context)" pattern the repo rule
prescribes, never a module-level flip.
"""
from __future__ import annotations

try:  # JAX <= 0.7 ships the scoped context manager here
    from jax.experimental import enable_x64  # noqa: F401
except ImportError:
    try:  # JAX >= 0.8 moved it to the top level, same signature: enable_x64(new_val=True)
        from jax import enable_x64  # noqa: F401
    except ImportError:  # neither: same semantics, scoped + restored
        import contextlib

        import jax

        @contextlib.contextmanager
        def enable_x64(new_val=True):
            prev = bool(jax.config.read("jax_enable_x64"))
            jax.config.update("jax_enable_x64", bool(new_val))
            try:
                yield
            finally:
                jax.config.update("jax_enable_x64", prev)
