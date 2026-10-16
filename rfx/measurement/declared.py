"""Package the measurements a user declared for either execution entry."""

import numpy as np

from rfx.core.jax_utils import is_tracer


def package_declared_measurements(sim, grid, *, flux_monitors, waveguide_cfgs):
    """Return named declared monitors, port records and reported-plane S."""
    from rfx.api import WaveguideSParamResult
    from rfx.sources.waveguide_port import (
        extract_waveguide_sparams, waveguide_plane_positions,
    )
    from rfx.sparams._waveguide_planes import resolve_reported_planes

    def concrete_array(value):
        return value if is_tracer(value) else np.asarray(value)

    flux_result = (
        {entry.name: mon for entry, mon in zip(sim._flux_monitors, flux_monitors)}
        if sim._flux_monitors else None
    )
    ports_result = (
        {entry.name: cfg for entry, cfg in zip(sim._waveguide_ports, waveguide_cfgs)}
        if sim._waveguide_ports else None
    )
    sparams_result = None
    if sim._waveguide_ports:
        sparams_result = {}
        for entry, cfg in zip(sim._waveguide_ports, waveguide_cfgs):
            plane_positions = waveguide_plane_positions(cfg)
            source_plane = plane_positions["source"]
            measured_reference_plane = plane_positions["reference"]
            measured_probe_plane = plane_positions["probe"]
            reference_plane, probe_plane, calibration_preset = resolve_reported_planes(
                entry, cfg, grid=grid,
            )
            s11, s21 = extract_waveguide_sparams(
                cfg,
                ref_shift=reference_plane - measured_reference_plane,
                probe_shift=probe_plane - measured_probe_plane,
            )
            sparams_result[entry.name] = WaveguideSParamResult(
                freqs=concrete_array(cfg.freqs),
                s11=concrete_array(s11),
                s21=concrete_array(s21),
                calibration_preset=calibration_preset,
                source_plane=float(source_plane),
                measured_reference_plane=measured_reference_plane,
                measured_probe_plane=measured_probe_plane,
                reference_plane=reference_plane,
                probe_plane=probe_plane,
            )
    return {
        "flux_monitors": flux_result,
        "waveguide_ports": ports_result,
        "waveguide_sparams": sparams_result,
    }
