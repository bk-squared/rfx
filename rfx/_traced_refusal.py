"""One exception type for a traced structure input a host decision cannot read.

A declared structure input (a shape dimension, mesh cell widths, a monitor's
cells) traced under ``jax.jit``/``jax.vmap`` reaches a host-side decision that
needs a concrete value. Every such refusal raises a subclass of
``TracedStructureRefusal``, so a caller that can retry eagerly
(``optimize(jit=True)``) keys on this type and not on whichever JAX error the
refusal happens to share.
"""

from __future__ import annotations

from rfx.errors import RfxUnsupported


class TracedStructureRefusal(RfxUnsupported):
    """A traced structure input refused by name.

    The fallback retries once eagerly; a refusal that also fires on the eager
    gradient trace then reaches the caller unchanged.
    """
