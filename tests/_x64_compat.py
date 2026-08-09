"""Version-robust scoped-x64 context (local jax-drift class, see ledger).

``jax.experimental.enable_x64`` (the scoped context manager this repo's
AD/referee tests use per the "never flip x64 at module level" rule) was
removed in newer JAX releases. CI (python 3.10) still resolves a JAX that
exports it; a drifted local environment (python 3.11, newer JAX) fails at
COLLECTION on the bare import, aborting whole-tree ``-k`` runs before a
single test executes. This shim keeps the upstream context manager when
it exists and otherwise provides the same semantics the sanctioned way:
a per-scope flip of ``jax_enable_x64`` with guaranteed restore — exactly
the "scope x64 per-test (fixture/context)" pattern the repo rule
prescribes, never a module-level flip.
"""
from __future__ import annotations

try:  # JAX still ships the scoped context manager
    from jax.experimental import enable_x64  # noqa: F401
except ImportError:  # newer JAX removed it — same semantics, scoped + restored
    import contextlib

    import jax

    @contextlib.contextmanager
    def enable_x64():
        prev = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", prev)
