"""rfx — JAX-based RF FDTD electromagnetic simulator."""

__version__ = "0.1.0"

from rfx.grid import Grid
from rfx.simulation import run, make_source, make_probe, make_port_source, SimResult
from rfx.api import Simulation, Result, MATERIAL_LIBRARY
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole, lorentz_pole
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
from rfx.farfield import (
    NTFFBox, NTFFData, FarFieldResult,
    make_ntff_box, compute_far_field, radiation_pattern, directivity,
)
from rfx.gpu import device_info, benchmark
from rfx.optimize import DesignRegion, OptimizeResult, optimize
from rfx.io import read_touchstone, write_touchstone
from rfx.visualize import (
    plot_field_slice, plot_s_params, plot_radiation_pattern, plot_time_series,
)
