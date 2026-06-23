"""Geometry shape builder for the config-driven CLI.

v1 supports the axis-aligned ``box`` only. Every other CSG primitive
(cylinder, sphere, polyline wire, ...) raises a clear
:class:`NotImplementedError` naming the unsupported shape so the failure
is loud and obviously a scope boundary, not a silent drop.
"""

from __future__ import annotations

from rfx.geometry.csg import Box

# Shapes the YAML schema knows how to build. Kept as a name->builder map so
# the unsupported-shape error can enumerate what *is* available.
_SUPPORTED_SHAPES = ("box",)


def shape_from_config(cfg: dict):
    """Build a CSG shape from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``shape`` (currently only ``"box"``) and the shape's
        geometry keys. For a box: ``bounds: [[x0,y0,z0], [x1,y1,z1]]``.

    Returns
    -------
    rfx.geometry.csg.Box
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"geometry entry must be a mapping, got {type(cfg).__name__}"
        )
    if "shape" not in cfg:
        raise KeyError(
            "geometry entry is missing required key 'shape' "
            f"(supported: {list(_SUPPORTED_SHAPES)})"
        )
    shape = cfg["shape"]
    if shape != "box":
        raise NotImplementedError(
            f"Unsupported geometry shape {shape!r}. The config CLI v1 "
            f"supports only {list(_SUPPORTED_SHAPES)}. Use the Python API "
            f"for {shape!r}."
        )
    if "bounds" not in cfg:
        raise KeyError(
            "box geometry is missing required key 'bounds' "
            "(expected [[x0,y0,z0], [x1,y1,z1]])"
        )
    bounds = cfg["bounds"]
    try:
        lo, hi = bounds
        corner_lo = tuple(float(v) for v in lo)
        corner_hi = tuple(float(v) for v in hi)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"box 'bounds' must be [[x0,y0,z0], [x1,y1,z1]], got {bounds!r}"
        ) from exc
    if len(corner_lo) != 3 or len(corner_hi) != 3:
        raise ValueError(
            f"box corners must each have 3 coordinates, got "
            f"lo={corner_lo}, hi={corner_hi}"
        )
    return Box(corner_lo, corner_hi)
