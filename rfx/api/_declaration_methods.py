"""Keep concrete declaration calculations concrete under a caller's JAX trace."""

from functools import wraps

import jax


def wrap_declaration(method):
    """Evaluate concrete inputs now while preserving operations on traced inputs."""
    @wraps(method)
    def declaration(self, *args, **kwargs):
        with jax.ensure_compile_time_eval():
            return method(self, *args, **kwargs)

    setattr(declaration, "_rfx_declaration_setup", True)
    return declaration
