"""rfx — JAX-based RF FDTD electromagnetic simulator."""
# ruff: noqa: F401

__version__ = "1.6.5"

from rfx.grid import Grid
from rfx.simulation import run, run_until_decay, make_source, make_probe, make_port_source, SimResult
from rfx.adi import ADIState2D, ADIState3D, thomas_solve, adi_step_2d, run_adi_2d, adi_step_3d, run_adi_3d
from rfx.api import (
    Simulation, Result, WaveguideSParamResult, WaveguideSMatrixResult,
    MSLSMatrixResult, CoaxialSMatrixResult, MATERIAL_LIBRARY,
    AD_MemoryEstimate, ADMemoryPlan, MeshIntelligenceReport,
)
from rfx.geometry.csg import Box, Sphere, Cylinder, PolylineWire
from rfx.geometry.curved import CurvedPatch
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
from rfx.materials.lorentz import LorentzPole, drude_pole, lorentz_pole
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.farfield import (
    NTFFBox, NTFFData, FarFieldResult,
    make_ntff_box, compute_far_field, compute_far_field_jax,
    radiation_pattern, directivity,
    axial_ratio, axial_ratio_dB, polarization_tilt, polarization_sense,
)
from rfx.rcs import compute_rcs, RCSResult
from rfx.antenna import (
    antenna_gain, antenna_gain_dB, antenna_efficiency,
    half_power_beamwidth, front_to_back_ratio,
    antenna_bandwidth, BandwidthResult,
    plot_antenna_summary,
)
from rfx.gpu import device_info, benchmark
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
    case_id_from_params, run_batch, run_batch_with_manifest,
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
    save_field_animation as _save_field_animation_legacy,
)
from rfx.animation import save_field_animation
from rfx.simulation import SnapshotSpec
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
try:
    from rfx.eigenmode import WaveguideMode, solve_waveguide_modes
except ImportError:
    # scipy is optional; eigenmode solver is unavailable without it.
    # Accessing WaveguideMode or solve_waveguide_modes will raise ImportError.
    def _eigenmode_unavailable(*args, **kwargs):
        raise ImportError(
            "rfx.eigenmode requires scipy. Install it with: pip install scipy"
        )
    WaveguideMode = _eigenmode_unavailable  # type: ignore[assignment]
    solve_waveguide_modes = _eigenmode_unavailable  # type: ignore[assignment]
from rfx.lumped import LumpedRLCSpec, RLCState, RLCCellMeta
from rfx.nonuniform import NonUniformGrid, make_nonuniform_grid, run_nonuniform, make_current_source
from rfx.auto_config import auto_configure, SimConfig, analyze_features, smooth_grading, apply_thirds_rule
from rfx.mesh_planner import MeshPlan, plan_mesh, plan_simulation_mesh
from rfx.harminv import harminv, harminv_from_probe, HarminvMode
from rfx.probes.probes import (
    wire_port_voltage, wire_port_current,
    init_wire_sparam_probe, update_wire_sparam_probe,
    extract_s_matrix_wire,
    FluxMonitor, init_flux_monitor, update_flux_monitor, flux_spectrum,
)
from rfx.sweep import parametric_sweep, SweepResult, plot_sweep
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult
from rfx.pcb import PCBLayer, Stackup
from rfx.floquet import (
    FloquetPort,
    floquet_phase_shift,
    floquet_wave_vector,
    FloquetDFTAccumulator,
    init_floquet_dft,
    update_floquet_dft,
    inject_floquet_source,
    extract_floquet_modes,
    compute_floquet_s_params,
)
from rfx.material_fit import (
    load_material_csv, fit_debye, fit_lorentz,
    eval_debye, eval_lorentz, plot_material_fit,
    DebyeFitResult, LorentzFitResult,
)
from rfx.differentiable_material_fit import (
    differentiable_material_fit, MaterialFitResult, sparam_loss,
)
from rfx.ris import RISUnitCell, RISSweepResult
from rfx.amr import compute_error_indicator, suggest_refinement_regions, auto_refine
from rfx.surrogate import export_training_data, export_geometry_sdf
from rfx.convergence import (
    convergence_study, ConvergenceResult, richardson_extrapolation,
    quick_convergence,
)

# ---------------------------------------------------------------------------
# Curated public surface (W5.5).
#
# ``__all__`` is the *front door*: ``from rfx import *`` and rendered API
# references use it. It is intentionally a strict subset of the ~245 flat
# symbols imported above — every back-compat import is KEPT (so existing
# ``import rfx; rfx.<name>`` keeps working), but per-step kernel internals,
# low-level bookkeeping/state classes, and dump/replay round-trip helpers are
# omitted from the star-import surface.
#
# Excluded categories (still importable by explicit attribute access):
#   - per-step Yee/port kernels: init_*/update_*/inject_*/apply_waveguide_port_*,
#     adi_step_2d/3d, thomas_solve
#   - low-level state/bookkeeping classes: ADIState2D/3D, RLCState, RLCCellMeta,
#     Coaxial*PlaneVI, CoaxialPlaneSourceSpec, FloquetDFTAccumulator,
#     PortVIDump/PortDumpMetadata/PortSMatrixObservable/PortReplayComparison,
#     SubgridValidationIssue/Report, BATCH_MANIFEST_SCHEMA
#   - low-level coaxial/floquet/waveguide plane helpers and modal builders
#     (coaxial_tem_*, *_plane_vi*, extract_floquet_modes, extract_multimode_*,
#     init_multimode_waveguide_port, solve_rectangular_modes,
#     waveguide_plane_positions, floquet_phase_shift/wave_vector)
#   - dump/replay round-trip + secondary IO: save_*/load_*_npz,
#     save_snapshots/load_snapshots, save_materials/load_materials,
#     replay_smatrix_from_port_vi_dump, save_optimization_trajectory,
#     render_artifact_markdown, validate_artifact_report, build_*_report/artifact
__all__ = [
    # grid / core simulation entry points
    "Grid", "NonUniformGrid", "make_nonuniform_grid",
    "Simulation", "run", "run_until_decay", "run_nonuniform",
    "make_source", "make_probe", "make_port_source", "make_current_source",
    "SimResult", "SnapshotSpec",
    # result + S-matrix types
    "Result", "WaveguideSParamResult", "WaveguideSMatrixResult",
    "MSLSMatrixResult", "CoaxialSMatrixResult",
    "AD_MemoryEstimate", "ADMemoryPlan", "MeshIntelligenceReport",
    # geometry
    "Box", "Sphere", "Cylinder", "PolylineWire", "CurvedPatch", "Via",
    "PCBLayer", "Stackup",
    # sources / waveforms / ports (builder classes; not per-step kernels)
    "GaussianPulse", "ModulatedGaussian", "CWSource", "CustomWaveform",
    "WaveguidePort", "WaveguidePortConfig", "CoaxialPort",
    "FloquetPort", "RISUnitCell", "RISSweepResult",
    # high-level S-parameter / port extractors (AD-contract-classified surface)
    "extract_waveguide_s_matrix", "extract_waveguide_sparams",
    "extract_waveguide_s11", "extract_waveguide_s21",
    "extract_s_matrix_wire", "compute_floquet_s_params",
    "compute_rcs", "RCSResult",
    # materials / dispersion / fitting
    "DebyePole", "LorentzPole", "drude_pole", "lorentz_pole",
    "ThinConductor", "MATERIAL_LIBRARY",
    "load_material_csv", "fit_debye", "fit_lorentz", "eval_debye", "eval_lorentz",
    "plot_material_fit", "DebyeFitResult", "LorentzFitResult",
    "differentiable_material_fit", "MaterialFitResult", "sparam_loss",
    # far-field / antenna
    "NTFFBox", "NTFFData", "FarFieldResult", "make_ntff_box",
    "compute_far_field", "compute_far_field_jax", "radiation_pattern", "directivity",
    "axial_ratio", "axial_ratio_dB", "polarization_tilt", "polarization_sense",
    "antenna_gain", "antenna_gain_dB", "antenna_efficiency",
    "half_power_beamwidth", "front_to_back_ratio",
    "antenna_bandwidth", "BandwidthResult", "plot_antenna_summary",
    # optimization / inverse design + objectives
    "DesignRegion", "OptimizeResult", "optimize",
    "GradientCheckResult", "gradient_check",
    "ProgressiveStage", "ProgressiveOptimizeResult", "progressive_optimize",
    "TopologyDesignRegion", "TopologyResult", "topology_optimize", "density_to_eps",
    "minimize_s11", "maximize_s21", "target_impedance", "maximize_bandwidth",
    "maximize_directivity", "minimize_reflected_energy",
    "maximize_transmitted_energy", "steer_probe_array",
    # sweeps / batch
    "parametric_sweep", "SweepResult", "plot_sweep",
    "vmap_material_sweep", "VmapSweepResult",
    "ParameterSweep", "BatchCaseResult", "run_batch", "run_batch_with_manifest",
    "summarize_batch_manifest",
    # probes / measurements / spectral
    "wire_port_voltage", "wire_port_current",
    "FluxMonitor", "flux_spectrum",
    "harminv", "harminv_from_probe", "HarminvMode",
    # eigenmode (optional scipy)
    "WaveguideMode", "solve_waveguide_modes",
    # lumped
    "LumpedRLCSpec",
    # auto config / mesh planning / convergence / amr
    "auto_configure", "SimConfig", "analyze_features",
    "MeshPlan", "plan_mesh", "plan_simulation_mesh",
    "convergence_study", "ConvergenceResult", "richardson_extrapolation",
    "quick_convergence",
    "compute_error_indicator", "suggest_refinement_regions", "auto_refine",
    # de-embedding
    "deembed_port_extension", "deembed_thru",
    # validation entry points
    "PortValidationIssue", "PortValidationReport",
    "validate_port_smatrix", "assert_port_smatrix_valid",
    "normalize_port_smatrix", "compare_replayed_smatrix",
    "replay_smatrix_from_vi_dump",
    # io / touchstone / artifacts / checkpoint
    "TouchstoneData", "read_touchstone", "read_touchstone_full", "write_touchstone",
    "network_quality_metrics",
    "save_optimization_result", "load_optimization_result",
    "save_far_field", "export_radiation_pattern", "export_geometry_json",
    "save_experiment_report", "save_simulation_dataset",
    "ArtifactBundle", "export_artifact_bundle",
    "save_state", "load_state",
    # visualization
    "plot_field_slice", "plot_s_params", "plot_radiation_pattern",
    "plot_time_series", "plot_rcs", "plot_smith",
    "plot_geometry_3d", "plot_field_3d", "save_field_vtk", "save_field_animation",
    # gpu / surrogate
    "device_info", "benchmark",
    "export_training_data", "export_geometry_sdf",
]
