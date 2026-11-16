from rfx.measurement.farfield import ETA_0 as ETA_0
from rfx.measurement.farfield import NTFFBox as NTFFBox
from rfx.measurement.farfield import NTFFData as NTFFData
from rfx.measurement.farfield import FarFieldResult as FarFieldResult
from rfx.measurement.farfield import _cell_widths as _cell_widths
from rfx.measurement.farfield import _normal_weight as _normal_weight
from rfx.measurement.farfield import _AXIS_NAMES as _AXIS_NAMES
from rfx.measurement.farfield import _ntff_face_pads as _ntff_face_pads
from rfx.measurement.farfield import _grid_axis_counts as _grid_axis_counts
from rfx.measurement.farfield import _axis_node_positions as _axis_node_positions
from rfx.measurement.farfield import _face_centre_margin_failures as _face_centre_margin_failures
from rfx.measurement.farfield import _raise_face_centre_margin as _raise_face_centre_margin
from rfx.measurement.farfield import with_face_centre_collocation as with_face_centre_collocation
from rfx.measurement.farfield import make_ntff_box as make_ntff_box
from rfx.measurement.farfield import ntff_accum_dtype as ntff_accum_dtype
from rfx.measurement.farfield import init_ntff_data as init_ntff_data
from rfx.measurement.farfield import _require_face_centre_margin as _require_face_centre_margin
from rfx.measurement.farfield import _AXIS_FACE_ATTRS as _AXIS_FACE_ATTRS
from rfx.measurement.farfield import _AXIS_INDEX as _AXIS_INDEX
from rfx.measurement.farfield import NTFFBoxPlacementError as NTFFBoxPlacementError
from rfx.measurement.farfield import _face_sample_offsets as _face_sample_offsets
from rfx.measurement.farfield import _sample_names as _sample_names
from rfx.measurement.farfield import _listed as _listed
from rfx.measurement.farfield import require_box_encloses_injected_region as require_box_encloses_injected_region
from rfx.measurement.farfield import _raise_face_inside_injected_region as _raise_face_inside_injected_region
from rfx.measurement.farfield import accumulate_ntff as accumulate_ntff
from rfx.measurement.farfield import _surface_currents as _surface_currents
from rfx.measurement.farfield import _face_positions as _face_positions
from rfx.measurement.farfield import radiation_pattern as radiation_pattern
from rfx.measurement.farfield import directivity as directivity
from rfx.measurement.farfield import axial_ratio as axial_ratio
from rfx.measurement.farfield import axial_ratio_dB as axial_ratio_dB
from rfx.measurement.farfield import polarization_tilt as polarization_tilt
from rfx.measurement.farfield import polarization_sense as polarization_sense
from rfx.measurement.farfield import C0 as C0
from rfx.measurement.farfield import EPS_0 as EPS_0
from rfx.measurement.farfield import MU_0 as MU_0
from rfx.measurement.farfield import Grid as Grid
from rfx.measurement.farfield import integrate_radiated_power as integrate_radiated_power


# Lazy compatibility exports keep record definitions free of import cycles.
_LAZY_EXPORTS = {
    "compute_far_field": "rfx.measurement.farfield",
    "compute_far_field_jax": "rfx.measurement.farfield",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)
