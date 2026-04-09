"""rfx — JAX-based RF FDTD electromagnetic simulator."""

__version__ = "1.4.0"

from rfx.grid import Grid
from rfx.simulation import run, run_until_decay, make_source, make_probe, make_port_source, SimResult
from rfx.adi import ADIState2D, ADIState3D, thomas_solve, adi_step_2d, run_adi_2d, adi_step_3d, run_adi_3d
from rfx.api import Simulation, Result, WaveguideSParamResult, WaveguideSMatrixResult, MATERIAL_LIBRARY
from rfx.geometry.csg import Box, Sphere, Cylinder, PolylineWire
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.sources.sources import GaussianPulse, ModulatedGaussian, CWSource, CustomWaveform
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.waveguide_port import (
    WaveguidePort, WaveguidePortConfig,
    init_waveguide_port, inject_waveguide_port, update_waveguide_port_probe,
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
from rfx.optimize import DesignRegion, OptimizeResult, optimize, GradientCheckResult, gradient_check
from rfx.topology import (
    TopologyDesignRegion, TopologyResult,
    topology_optimize, apply_density_filter, apply_projection, density_to_eps,
)
from rfx.io import (
    read_touchstone, write_touchstone,
    save_optimization_result, load_optimization_result,
    save_far_field, export_radiation_pattern,
    export_geometry_json, save_experiment_report,
    save_simulation_dataset, save_optimization_trajectory,
)
from rfx.deembed import deembed_port_extension, deembed_thru
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
