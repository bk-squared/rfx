"""Rejection of removed ``Simulation.forward()`` keyword arguments.

Moved out of ``rfx/api/_execute.py`` unchanged (file-size ratchet, #1512 PR).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Arc-audit follow-up item 5(b): a removed public kwarg (currently just
# `design_mask`, issue #625) used to surface as the bare Python default --
# `TypeError: _ExecuteMixin.forward() got an unexpected keyword argument
# 'design_mask'` -- which leaks this mixin's internal class name (the
# public surface is `Simulation.forward`, never `_ExecuteMixin.forward`)
# and gives no reason or replacement. `forward()` now accepts
# `**_removed_kwargs` and calls this helper so a removed kwarg gets a
# `TypeError` naming the actual reason and the one-line migration path
# instead. Deliberately still a `TypeError` (not some other exception
# type) -- `tests/unit/autodiff/test_design_mask_removed.py` pins "the kwarg is
# rejected" via `pytest.raises(TypeError, match="design_mask")`, and this
# only changes the MESSAGE, not the exception class or that contract.
# ---------------------------------------------------------------------------
_REMOVED_FORWARD_KWARGS: dict = {
    "design_mask": (
        "removed entirely in issue #625, not deprecated: it was measured "
        "to save ZERO reverse-mode AD memory (partial-eval residuals have "
        "whole-array granularity) while corrupting the gradient in every "
        "configuration tested. For memory relief use checkpoint_every / "
        "checkpoint_segments; to restrict which cells carry a derivative, "
        "wrap eps_override yourself: eps = jnp.where(region, eps, "
        "jax.lax.stop_gradient(eps)) -- see CHANGELOG.md 'Removed — "
        "design_mask' and docs/public/guide/memory-reduction.mdx."
    ),
}


def _reject_removed_forward_kwargs(removed_kwargs: dict) -> None:
    """Raise a TypeError for an unrecognised forward() kwarg, naming the
    reason and replacement for a KNOWN removed one (see
    ``_REMOVED_FORWARD_KWARGS``) instead of leaking ``_ExecuteMixin`` (the
    internal mixin, not the public ``Simulation.forward`` surface)."""
    parts = []
    for name in removed_kwargs:
        reason = _REMOVED_FORWARD_KWARGS.get(name)
        if reason is not None:
            parts.append(f"'{name}' was {reason}")
        else:
            parts.append(f"'{name}' is not a recognised forward() keyword argument")
    raise TypeError(
        "Simulation.forward() got unexpected keyword argument(s) "
        f"{sorted(removed_kwargs)}: " + " | ".join(parts)
    )
