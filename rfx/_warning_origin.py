"""Attribute solve warnings to the user across internal and JAX frames."""

import os
import sys


_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


def user_stacklevel():
    """Return the level for ``warnings.warn(..., stacklevel=user_stacklevel())``.

    The function calling warn is level 1, not this helper's frame. Skip rfx
    package files and, like preflight.line_stub, JAX and synthetic frames so
    directly jitting a bound rfx method still names its user's call site.
    """
    frame, level = sys._getframe(1), 1
    try:
        while frame.f_back is not None and (
                os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE)
                or "jax" + os.sep in frame.f_code.co_filename
                or frame.f_code.co_filename.startswith("<")):
            frame, level = frame.f_back, level + 1
        return level
    finally:
        del frame
