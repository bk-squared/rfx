"""Excitation sources for FDTD simulation."""
# ruff: noqa: F401

from rfx.sources.sources import GaussianPulse, add_point_source, add_lumped_port
from rfx.sources.coaxial_port import (
    CoaxialPort,
    CoaxialPlaneSourceSpec,
    CoaxialTEMCartesianPlaneVI,
    build_coaxial_tem_plane_source_specs,
    extract_coaxial_plane_vi_from_dft,
    CoaxialTEMReferencePlaneVI,
    coaxial_load_reflection,
    coaxial_tem_capacitance_per_m,
    coaxial_tem_characteristic_impedance,
    coaxial_tem_inductance_per_m,
    coaxial_tem_phase_constant,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi,
    CoaxialLineReflection,
    coaxial_line_plane_voltage,
    coaxial_line_reflection_from_plane_voltages,
    stamp_coaxial_line,
    stamp_coaxial_short_plane,
    stamp_coaxial_annular_resistor,
    SMA_PIN_RADIUS,
    SMA_OUTER_RADIUS,
    PTFE_EPS_R,
)

# rfx.sources.waveguide_port imports the model, stepping and measurement
# layers, and those layers import submodules of this package; an eager import
# here would load that file whenever a layer loads (rfx#1818).
_LAZY_EXPORTS = {
    "WaveguidePort": "rfx.sources.waveguide_port",
    "WaveguidePortConfig": "rfx.sources.waveguide_port",
    "init_waveguide_port": "rfx.sources.waveguide_port",
    "inject_waveguide_port": "rfx.sources.waveguide_port",
    "apply_waveguide_port_h": "rfx.sources.waveguide_port",
    "apply_waveguide_port_e": "rfx.sources.waveguide_port",
    "update_waveguide_port_probe": "rfx.sources.waveguide_port",
    "extract_waveguide_port_waves": "rfx.sources.waveguide_port",
    "extract_waveguide_s_matrix": "rfx.sources.waveguide_port",
    "extract_waveguide_sparams": "rfx.sources.waveguide_port",
    "extract_waveguide_s11": "rfx.sources.waveguide_port",
    "extract_waveguide_s21": "rfx.sources.waveguide_port",
    "waveguide_plane_positions": "rfx.sources.waveguide_port",
    "solve_rectangular_modes": "rfx.sources.waveguide_port",
    "init_multimode_waveguide_port": "rfx.sources.waveguide_port",
    "extract_multimode_s_matrix": "rfx.sources.waveguide_port",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
