"""S-parameter helpers live in rfx.measurement.sparam_assembly and rfx.measurement.sparam_advisories; this module re-exports them."""

from __future__ import annotations
from rfx._warning_origin import user_stacklevel

from rfx.errors import AccuracyWarning, ConfigurationWarning
from rfx.model.graded_admission import host_cells

import jax
import jax.numpy as jnp
import numpy as np

from rfx.sparams._settling_warning import _warn_if_ringdown_truncated

try:  # Public API on current JAX; the 0.4.x GPU image uses the old location.
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64

from rfx._internal._grid_metric import cells_crossed, distinct_cell_sizes
from rfx.core.jax_utils import complex_solve, is_tracer
from rfx.measurement.phasors import exp_j
from rfx.sparams._mixed_checks import _mixed_reciprocity_deviation, _warn_partial_nonpassive
from rfx.sparams._grid_metrics import absorber_depth, junction_absorber_depth, node_distance

from rfx.model.graded_grid import NonUniformGrid, interior_cells

from rfx.sparams._grid_metrics import _msl_axis_spacing, _msl_axis_cell_f64

from rfx.measurement.sparam_assembly import (
    _msl_cell_profile,
    msl_modal_voltage,
    msl_solve_s_from_waves,
    _msl_wave_split_reliability,
    _project_passive,
    _assemble_mixed_power_wave_s,
    _mixed_flux_magnitude_override,
    _assemble_coaxial_two_port_from_voltages,
    _ladder_split_witness,
    _assemble_coax_msl_transition_from_voltages,
    _register_msl_h_planes,
    _collocated_msl_h,
    _msl_power_wave_scales,
)

from rfx.measurement.sparam_advisories import (
    _warn_msl_wave_split_unreliable,
    _warn_msl_beta_scan_railed,
    _SETTLING_WITNESS_DB,
    settling_verdict,
    _validate_extra_flux_monitor_entries,
    _nu_shift_span_cells,
    _assert_nu_shift_span_in_one_grading_zone,
    _resolve_msl_auto_offsets,
    _sigma_max_excess,
    _warn_if_sigma_max_excess,
    _warn_if_passivity_projected,
    WAVEGUIDE_RECIPROCITY_ADVISORY_TOL,
    _reciprocity_advisory_message,
    _warn_if_nonpassive_smatrix,
    _finalize_sparam_result,
    _C0_SPARAMS,
    _warn_junction_probe_clearance,
    _warn_junction_cpml_thickness,
    _warn_ntff_box_dropped,
    _FAR_PORT_LAMBDA_G_FRACTION,
    _warn_thin_absorber_vs_guide_wavelength,
)
