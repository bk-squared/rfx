"""CSG geometry primitives for defining simulation structures."""

from rfx.geometry.csg import Box, Cylinder, Sphere, union, difference, intersection, rasterize
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.geometry.conformal import (
    compute_conformal_weights,
    compute_conformal_weights_sdf,
    clamp_conformal_weights,
    apply_conformal_pec,
    conformal_eps_correction,
)
