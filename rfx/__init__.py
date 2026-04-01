"""rfx — JAX-based RF FDTD electromagnetic simulator."""

__version__ = "0.1.0"

from rfx.grid import Grid
from rfx.simulation import run, run_until_decay, make_source, make_probe, make_port_source, SimResult
from rfx.api import Simulation, Result, WaveguideSParamResult, WaveguideSMatrixResult, MATERIAL_LIBRARY
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.sources.sources import GaussianPulse, CWSource, CustomWaveform
from rfx.sources.waveguide_port import (
    WaveguidePort, WaveguidePortConfig,
    init_waveguide_port, inject_waveguide_port, update_waveguide_port_probe,
    extract_waveguide_port_waves, extract_waveguide_s_matrix,
    extract_waveguide_sparams, extract_waveguide_s11, extract_waveguide_s21,
    waveguide_plane_positions,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole, lorentz_pole
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.farfield import (
    NTFFBox, NTFFData, FarFieldResult,
    make_ntff_box, compute_far_field, radiation_pattern, directivity,
)
from rfx.rcs import compute_rcs, RCSResult
from rfx.gpu import device_info, benchmark
from rfx.optimize import DesignRegion, OptimizeResult, optimize
from rfx.io import read_touchstone, write_touchstone
from rfx.visualize import (
    plot_field_slice, plot_s_params, plot_radiation_pattern, plot_time_series,
    plot_rcs,
)
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
)
from rfx.eigenmode import WaveguideMode, solve_waveguide_modes
