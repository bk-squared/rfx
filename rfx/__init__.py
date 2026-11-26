"""rfx — JAX-based RF FDTD electromagnetic simulator.

The runtime guard must execute before the public JAX-dependent imports.
"""
# ruff: noqa: F401, E402

__version__ = "2.3.0.dev0"

import sys as _sys

import jax as _jax
import jaxlib as _jaxlib

from rfx._runtime_floor import check as _check_runtime_floor

_check_runtime_floor(_sys.version_info, _jax.__version__, _jaxlib.__version__)

from rfx.errors import AccuracyWarning, ConfigurationWarning, EdgeAwareProfileRefusal, RfxError, RfxRefusal, RfxUnsupported, RfxWarning
from rfx.diagnostic_records import Diagnostic
from rfx.grid import Grid
from rfx.simulation import run, run_until_decay, make_source, make_probe, make_port_source, SimResult
from rfx.api import (
    Simulation, Result, ForwardResult, WaveguideSParamResult, WaveguideSMatrixResult,
    MSLProbeClearance, MSLSMatrixResult, MixedSMatrixResult, MATERIAL_LIBRARY,
    AD_MemoryEstimate, ADMemoryPlan, ADMemoryComponent,
    ADMemoryActionHint, ADMemoryExplainabilityReport,
    ADMemoryPreflightReport, ADCompiledMemoryCertificate,
    MeshIntelligenceReport,
    GradientRecordLengthWitness, gradient_record_length_witness,
)
from rfx.ad_diagnostics import (
    ADParserHealth,
    ADResidualGroup,
    ADResidualInspection,
    ADResidualRecord,
    ADSavedResidualDiagnosticReport,
    diagnose_ad_saved_residuals,
    inspect_ad_saved_residuals,
    parse_saved_residual_line,
)
from rfx.pareto import (
    ParetoFront,
    ParetoPoint,
    epsilon_constraint_mask,
    pareto_front,
    pareto_mask,
    select_epsilon_constrained,
    weighted_scalarization,
)
from rfx.jax_checks import (
    check_bounds,
    check_courant_number,
    check_finite,
    check_positive,
    checkify_invariants,
)
from rfx.geometry.csg import Box, OrientedBox, Sphere, Cylinder, PolylineWire
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.mesh_import import MeshShape
from rfx.subgridding.validation import SubgridValidationIssue, SubgridValidationReport
from rfx.geometry.via import Via
from rfx.sources.sources import GaussianPulse, ModulatedGaussian, CWSource, CustomWaveform
from rfx.sources.coaxial_port import (
    CoaxialPort, CoaxialPlaneSourceSpec,
    CoaxialTEMCartesianPlaneVI, CoaxialTEMReferencePlaneVI,
    build_coaxial_tem_plane_source_specs,
    extract_coaxial_plane_vi_from_dft,
    coaxial_load_reflection,
    coaxial_tem_capacitance_per_m,
    coaxial_tem_characteristic_impedance, coaxial_tem_inductance_per_m,
    coaxial_tem_phase_constant, coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
)
from rfx.sources.waveguide_port import (
    WaveguidePort, WaveguidePortConfig,
    init_waveguide_port, inject_waveguide_port, update_waveguide_port_probe,
    apply_waveguide_port_h, apply_waveguide_port_e,
    extract_waveguide_port_waves, extract_waveguide_s_matrix,
    extract_waveguide_sparams, extract_waveguide_s11, extract_waveguide_s21,
    waveguide_plane_positions,
    solve_rectangular_modes, init_multimode_waveguide_port,
    extract_multimode_s_matrix,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole, lorentz_pole, LorentzPoleMap, drude_pole_map, lorentz_pole_map
from rfx.materials.relative_strength import RelativeStrength
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.boundaries.pec import (
    SheetSpec, WireSpec, edge_is_pec, realized_pec_edge_masks,
    realized_wall_planes,
)
from rfx.farfield import (
    NTFFBox,
    NTFFData,
    FarFieldResult,
    make_ntff_box,
    radiation_pattern,
    directivity,
    axial_ratio,
    axial_ratio_dB,
    polarization_tilt,
    polarization_sense,
)
from rfx.measurement.farfield import (
    compute_far_field,
    compute_far_field_jax,
)
from rfx.current_moments import current_moment_far_field
from rfx.measurement.rcs import compute_rcs
from rfx.rcs import compute_rcs_jax, RCSResult, ScatteringResponse
from rfx.antenna import (
    antenna_gain, antenna_gain_dB, antenna_efficiency,
    half_power_beamwidth, front_to_back_ratio,
    antenna_bandwidth, BandwidthResult,
    plot_antenna_summary,
)
from rfx.ntff_surface import NTFFSurface, export_ntff_surface
from rfx.optimize import (
    DesignRegion,
    OptimizeResult,
    optimize,
    GradientCheckResult,
    gradient_check,
    ProgressiveStage,
    ProgressiveOptimizeResult,
    progressive_optimize,
)
from rfx._minimize import minimize, MinimizeResult
from rfx.optimizers import MMASubproblemCapWarning
from rfx.topology import (
    TopologyDesignRegion, TopologyResult,
    topology_optimize, apply_density_filter, apply_projection, density_to_eps,
)
from rfx.io import (
    TouchstoneData, read_touchstone, read_touchstone_full, write_touchstone,
    network_quality_metrics,
    save_optimization_result, load_optimization_result,
    save_far_field, export_radiation_pattern,
    export_geometry_json, save_experiment_report,
    save_simulation_dataset, save_optimization_trajectory,
)
from rfx.batch import (
    BATCH_MANIFEST_SCHEMA, BatchCaseResult, ParameterSweep,
    case_id_from_params, run_batch_with_manifest,
    summarize_batch_manifest,
)
from rfx.artifacts import (
    ArtifactBundle,
    build_scene_artifact,
    build_runtime_report,
    render_artifact_markdown,
    validate_artifact_report,
    export_artifact_bundle,
)
from rfx.deembed import deembed_port_extension, deembed_thru
from rfx.validation import (
    PortDumpMetadata, PortReplayComparison, PortSMatrixObservable,
    PortValidationIssue, PortValidationReport, PortVIDump,
    compare_replayed_smatrix, load_port_vi_dump_npz, normalize_port_smatrix,
    replay_smatrix_from_port_vi_dump, replay_smatrix_from_vi_dump,
    save_port_vi_dump_npz, validate_port_smatrix, assert_port_smatrix_valid,
)
from rfx.visualize import (
    plot_field_slice, plot_s_params, plot_radiation_pattern, plot_time_series,
    plot_rcs,
)
from rfx.smith import plot_smith
from rfx.visualize3d import (
    plot_geometry_3d, plot_field_3d, save_field_vtk, save_screenshot,
)
from rfx.animation import save_field_animation
from rfx.simulation import SnapshotSpec
from rfx.snapshots import SnapshotAxes, snapshot_axes, field_sample_coords
from rfx.checkpoint import (
    save_state, load_state, save_snapshots, load_snapshots,
    save_materials, load_materials,
)
from rfx.optimize_objectives import (
    minimize_s11,
    maximize_s21,
    target_impedance,
    maximize_bandwidth,
    maximize_directivity,
    minimize_reflected_energy,
    maximize_transmitted_energy,
    steer_probe_array,
)
from rfx.observables import dft_field, field_energy, field_softmax, jacobian_fwd
from rfx.eigenmode import WaveguideMode, solve_waveguide_modes
from rfx.lumped import LumpedRLCSpec, RLCState, RLCCellMeta
from rfx.nonuniform import NonUniformGrid, make_nonuniform_grid, run_nonuniform, make_current_source, make_band_profile
from rfx.auto_config import auto_configure, SimConfig, analyze_features, smooth_grading, apply_thirds_rule
from rfx.mesh_planner import MeshPlan, plan_mesh, plan_simulation_mesh
from rfx.harminv import (
    matrix_pencil, matrix_pencil_from_probe, MatrixPencilMode,
)
from rfx.probes.probes import (
    wire_port_voltage, wire_port_current,
    init_wire_sparam_probe, update_wire_sparam_probe,
    extract_s_matrix_wire,
    FluxMonitor, init_flux_monitor, update_flux_monitor, flux_spectrum,
    subtract_flux_monitors,
)
from rfx.sparams._port_reference import renormalize_power_waves
from rfx.sweep import parametric_sweep, SweepResult, plot_sweep
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult
from rfx.pcb import PCBLayer, Stackup
from rfx.microstrip import (
    microstrip_impedance, microstrip_eps_eff, microstrip_width,
)
from rfx.floquet import (
    FloquetPort,
    floquet_wave_vector,
    FloquetDFTAccumulator,
    init_floquet_dft,
    extract_floquet_modes,
    compute_floquet_s_params,
)
from rfx.measurement.floquet import (
    floquet_phase_shift,
    update_floquet_dft,
)
from rfx.stepping.floquet import (
    inject_floquet_source,
)
from rfx.material_fit import (
    load_material_csv, fit_debye, fit_lorentz,
    eval_debye, eval_lorentz, plot_material_fit,
    DebyeFitResult, LorentzFitResult,
)
from rfx.differentiable_material_fit import (
    differentiable_material_fit, MaterialFitResult, sparam_loss,
)
from rfx.surrogate import export_training_data, export_geometry_sdf
from rfx.convergence import (
    convergence_study, ConvergenceResult, richardson_extrapolation,
    quick_convergence,
)

__all__ = [
    "RfxError", "RfxRefusal", "RfxUnsupported", "EdgeAwareProfileRefusal",
    "RfxWarning", "AccuracyWarning", "ConfigurationWarning",
    "Diagnostic",
    "Grid", "NonUniformGrid", "make_nonuniform_grid",
    "make_band_profile",
    "Simulation", "run", "run_until_decay", "run_nonuniform",
    "make_source", "make_probe", "make_port_source", "make_current_source",
    "SimResult", "SnapshotSpec",
    "Result", "ForwardResult", "WaveguideSParamResult", "WaveguideSMatrixResult",
    "MSLProbeClearance", "MSLSMatrixResult", "MixedSMatrixResult",
    "AD_MemoryEstimate", "ADMemoryPlan", "ADMemoryComponent",
    "ADMemoryActionHint",
    "ADMemoryExplainabilityReport", "MeshIntelligenceReport",
    "ADMemoryPreflightReport",
    "ADCompiledMemoryCertificate",
    "ADParserHealth", "ADResidualGroup",
    "ADResidualInspection", "ADResidualRecord",
    "ADSavedResidualDiagnosticReport", "diagnose_ad_saved_residuals",
    "inspect_ad_saved_residuals", "parse_saved_residual_line",
    "ParetoFront", "ParetoPoint", "pareto_front", "pareto_mask",
    "weighted_scalarization", "epsilon_constraint_mask",
    "select_epsilon_constrained",
    "checkify_invariants", "check_finite", "check_positive",
    "check_bounds", "check_courant_number",
    "Box", "Sphere", "Cylinder", "PolylineWire", "CurvedPatch", "Via",
    "MeshShape",
    "PCBLayer", "Stackup",
    "microstrip_impedance", "microstrip_eps_eff", "microstrip_width",
    "GaussianPulse", "ModulatedGaussian", "CWSource", "CustomWaveform",
    "WaveguidePort", "WaveguidePortConfig", "CoaxialPort",
    "FloquetPort",
    "extract_waveguide_s_matrix", "extract_waveguide_sparams",
    "extract_waveguide_s11", "extract_waveguide_s21",
    "extract_s_matrix_wire", "compute_floquet_s_params",
    "compute_rcs", "compute_rcs_jax", "RCSResult", "ScatteringResponse",
    "DebyePole", "LorentzPole", "RelativeStrength", "drude_pole", "lorentz_pole",
    "LorentzPoleMap", "drude_pole_map", "lorentz_pole_map",
    "ThinConductor", "MATERIAL_LIBRARY",
    "realized_pec_edge_masks", "realized_wall_planes", "edge_is_pec",
    "load_material_csv", "fit_debye", "fit_lorentz", "eval_debye", "eval_lorentz",
    "plot_material_fit", "DebyeFitResult", "LorentzFitResult",
    "differentiable_material_fit", "MaterialFitResult", "sparam_loss",
    "NTFFBox", "NTFFData", "FarFieldResult", "make_ntff_box",
    "compute_far_field", "compute_far_field_jax", "radiation_pattern", "directivity",
    "axial_ratio", "axial_ratio_dB", "polarization_tilt", "polarization_sense",
    "antenna_gain", "antenna_gain_dB", "antenna_efficiency",
    "half_power_beamwidth", "front_to_back_ratio",
    "antenna_bandwidth", "BandwidthResult", "plot_antenna_summary",
    "DesignRegion", "OptimizeResult", "optimize",
    "minimize", "MinimizeResult", "MMASubproblemCapWarning",
    "GradientCheckResult", "gradient_check",
    "ProgressiveStage", "ProgressiveOptimizeResult", "progressive_optimize",
    "TopologyDesignRegion", "TopologyResult", "topology_optimize", "density_to_eps",
    "minimize_s11", "maximize_s21", "target_impedance", "maximize_bandwidth",
    "maximize_directivity", "minimize_reflected_energy",
    "maximize_transmitted_energy", "steer_probe_array",
    "dft_field", "field_energy", "field_softmax", "jacobian_fwd",
    "parametric_sweep", "SweepResult", "plot_sweep",
    "vmap_material_sweep", "VmapSweepResult",
    "ParameterSweep", "BatchCaseResult", "run_batch_with_manifest",
    "summarize_batch_manifest",
    "wire_port_voltage", "wire_port_current",
    "FluxMonitor", "flux_spectrum", "subtract_flux_monitors",
    "matrix_pencil", "matrix_pencil_from_probe", "MatrixPencilMode",
    "WaveguideMode", "solve_waveguide_modes",
    "LumpedRLCSpec",
    "auto_configure", "SimConfig", "analyze_features",
    "MeshPlan", "plan_mesh", "plan_simulation_mesh",
    "convergence_study", "ConvergenceResult", "richardson_extrapolation",
    "quick_convergence",
    "deembed_port_extension", "deembed_thru",
    "PortValidationIssue", "PortValidationReport",
    "validate_port_smatrix", "assert_port_smatrix_valid",
    "normalize_port_smatrix", "compare_replayed_smatrix",
    "replay_smatrix_from_vi_dump",
    "TouchstoneData", "read_touchstone", "read_touchstone_full", "write_touchstone",
    "network_quality_metrics",
    "save_optimization_result", "load_optimization_result",
    "save_far_field", "export_radiation_pattern", "export_geometry_json",
    "save_experiment_report", "save_simulation_dataset",
    "ArtifactBundle", "export_artifact_bundle",
    "save_state", "load_state",
    "plot_field_slice",
    "plot_s_params", "plot_radiation_pattern",
    "plot_time_series", "plot_rcs", "plot_smith",
    "plot_geometry_3d", "plot_field_3d", "save_field_vtk", "save_field_animation",
    "export_training_data", "export_geometry_sdf",
]

from rfx.fidelity import fidelity_report  # noqa: E402,F401
