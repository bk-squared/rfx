"""Design-description interop for rfx.

This package carries the round-trip-complete description of a simulation
*setup* (geometry parameters, materials, excitations, observables, mesh
directives) as distinct from:

- ``rfx.artifacts.build_scene_artifact`` — a review/provenance **summary** of
  runtime state, explicitly "not a CAD export";
- ``rfx.io.export_geometry_json`` — the older lightweight geometry dump;
- ``rfx.surrogate.export_geometry_sdf`` — a signed-distance encoding for
  machine-learning consumers.

Status: **provisional**. The shape codec is round-trip tested; the wider design
description and external-solver emitters are under construction.
"""

from __future__ import annotations

from rfx.interop._design import (
    DESIGN_SCHEMA_VERSION,
    SUPPORTED_WAVEFORM_KINDS,
    design_to_dict,
    design_to_json,
    simulation_from_design,
)
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.interop._materials import (
    material_from_dict,
    material_to_dict,
    materials_from_dict,
    materials_to_dict,
)
from rfx.interop._shapes import (
    SUPPORTED_SHAPE_KINDS,
    shape_field_names,
    shape_from_dict,
    shape_kind_of,
    shape_to_dict,
)

__all__ = [
    "DESIGN_SCHEMA_VERSION",
    "SUPPORTED_SHAPE_KINDS",
    "SUPPORTED_WAVEFORM_KINDS",
    "UnsupportedDesignFeature",
    "design_to_dict",
    "design_to_json",
    "material_from_dict",
    "material_to_dict",
    "materials_from_dict",
    "materials_to_dict",
    "shape_field_names",
    "shape_from_dict",
    "shape_kind_of",
    "shape_to_dict",
    "simulation_from_design",
]
