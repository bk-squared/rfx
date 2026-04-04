"""Excitation sources for FDTD simulation."""

from rfx.sources.sources import GaussianPulse, add_point_source, add_lumped_port
from rfx.sources.waveguide_port import (
    WaveguidePort,
    WaveguidePortConfig,
    init_waveguide_port,
    inject_waveguide_port,
    update_waveguide_port_probe,
    extract_waveguide_port_waves,
    extract_waveguide_s_matrix,
    extract_waveguide_sparams,
    extract_waveguide_s11,
    extract_waveguide_s21,
    waveguide_plane_positions,
    solve_rectangular_modes,
    init_multimode_waveguide_port,
    extract_multimode_s_matrix,
)
