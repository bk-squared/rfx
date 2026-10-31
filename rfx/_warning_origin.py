"""Attribute solve warnings to the user across internal and JAX frames."""

import sys


_SKIPPED = ("rfx", "jax", "jaxlib")


def user_stacklevel():
    """Return the level for ``warnings.warn(..., stacklevel=user_stacklevel())``.

    The function calling warn is level 1, not this helper's frame. Skip rfx
    frames and JAX frames, by module name (not path: a user folder ending in
    ``jax`` or an IPython cell is the user's), so directly jitting a bound rfx
    method still names its user's call site.
    """
    frame, level = sys._getframe(1), 1
    try:
        while frame.f_back is not None and frame.f_globals.get(
                "__name__", "").partition(".")[0] in _SKIPPED:
            frame, level = frame.f_back, level + 1
        return level
    finally:
        del frame
