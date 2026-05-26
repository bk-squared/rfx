"""CSG geometry primitives for defining simulation structures."""

from rfx.geometry.csg import Box, Cylinder, Sphere, union, difference, intersection, rasterize  # noqa: F401
from rfx.geometry.curved import CurvedPatch  # noqa: F401
from rfx.geometry.via import Via  # noqa: F401
from rfx.geometry.conformal import (
    compute_conformal_weights,  # noqa: F401
    compute_conformal_weights_sdf,  # noqa: F401
    clamp_conformal_weights,  # noqa: F401
    apply_conformal_pec,  # noqa: F401
    conformal_eps_correction,  # noqa: F401
)
