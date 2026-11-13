"""Field probe records live in rfx.measurement.probes; this module re-exports them."""

from __future__ import annotations
from rfx.errors import AccuracyWarning; from rfx._warning_origin import user_stacklevel  # noqa: E702
from rfx._internal._grid_metric import field_index
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from rfx.core.dft_utils import dft_window_weight as _dft_window_weight
from rfx.measurement.dft import phase as _dft_phase, accumulate as _accumulate
from rfx.measurement.plan import field_channel
from rfx.grid import Grid
from rfx.stepping.probe_loop import resolve_extractor_axes
from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, init_cpml
from rfx.sources.sources import LumpedPort, _wire_port_live_cells, _wire_port_cells, WirePort
from rfx.sparams.passivity_warning import warn_if_nonpassive_lumped_s11

from rfx.measurement.probes import (
    FieldMonitor,
    sample_field,
    DFTProbe,
    init_dft_probe,
    update_dft_probe,
    _project_port,
    port_voltage,
    _port_curl_boundary,
    _bwd_h,
    port_current,
    _metric_ampere_loop,
    _ampere_loop,
    _ampere_loop_components,
    _ampere_loop_values,
    _port_voltage_value,
    SParamProbe,
    PortVIReplayBundle,
    WirePortVIReplayBundle,
    init_sparam_probe,
    update_sparam_probe,
    update_lumped_drive_ref_probe,
    PreDecisionLumpedDiagonalWarning,
    driven_port_reflection,
    extract_s11,
    DFTPlaneProbe,
    init_dft_plane_probe,
    update_dft_plane_probe,
    FluxMonitor,
    _FLUX_COMPONENTS,
    init_flux_monitor,
    update_flux_monitor,
    flux_spectrum,
    _warn_if_flux_subnormal_flush,
    _FLUX_ACCUMULATOR_FIELDS,
    _FLUX_METADATA_FIELDS,
    _flux_meta_equal,
    subtract_flux_monitors,
    _wire_port_live_mid,
    wire_port_voltage,
    wire_port_gap_voltage,
    wire_port_current,
    init_wire_sparam_probe,
    update_wire_sparam_probe,
    update_wire_drive_ref_probe,
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
    extract_s_matrix,
    extract_lumped_s11,
    extract_s11_normalised,
    extract_s_matrix_wire,
)
