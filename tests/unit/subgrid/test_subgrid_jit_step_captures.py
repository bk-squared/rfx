"""Guards for the run_subgridded_jit step_fn capture contract.

``run_subgridded_jit`` hoists its ~477-line scan body into ``_make_step_fn``,
passing the captured locals through a ``ctx`` dict.  ``ctx.get`` keeps closure
semantics for conditionally-bound captures, but it would also silently mask a
*required* capture (renamed or accidentally conditional) as ``None``.
``_STEP_FN_REQUIRED_CAPTURES`` lists captures verified to be unconditionally
bound; the runner fails loudly at ctx-build time if one is missing.
"""

from __future__ import annotations

from rfx.subgridding.jit_runner import (
    _STEP_FN_CAPTURES,
    _STEP_FN_REQUIRED_CAPTURES,
)


def test_required_captures_are_declared_step_fn_captures():
    """Every required capture must also be a declared step_fn capture."""
    extra = _STEP_FN_REQUIRED_CAPTURES - set(_STEP_FN_CAPTURES)
    assert extra == frozenset(), f"required captures not in _STEP_FN_CAPTURES: {extra}"


def test_required_capture_set_is_non_empty():
    """The required-capture guard must actually guard something."""
    assert len(_STEP_FN_REQUIRED_CAPTURES) > 0
