"""Errors raised by the design-interop layer."""

from __future__ import annotations


class UnsupportedDesignFeature(ValueError):
    """Raised when a design cannot be represented without silent loss.

    The interop layer refuses rather than approximating: a design description
    that quietly drops a shape parameter, a dispersion pole, or a mesh profile
    is worse than no description at all, because the resulting comparison looks
    valid while comparing a different structure.
    """


__all__ = ["UnsupportedDesignFeature"]
