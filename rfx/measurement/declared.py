"""Package the measurements a user declared for either execution entry."""

import warnings

import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.errors import ConfigurationWarning
from rfx._warning_origin import user_stacklevel
from rfx.measurement.waveguide_power import driven_waveguide_indices
from rfx.sparams.passivity_warning import warn_if_singular_value_exceeds_one


def package_declared_measurements(sim, grid, *, entry, flux_monitors, waveguide_cfgs, waveguide_reference_only=False):
    """Return named declared monitors, port records and reported-plane S."""
    from rfx.api._spec import WaveguideSParamResult
    from rfx.measurement.waveguide_s import (
        extract_waveguide_sparams, extract_waveguide_s_column, waveguide_plane_positions,
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
    if sim._waveguide_ports and not waveguide_reference_only:
        sparams_result = {}
        driven = driven_waveguide_indices(sim._waveguide_ports)
        drive_idx = driven[0] if len(driven) == 1 else None
        driven_port = sim._waveguide_ports[drive_idx].name if drive_idx is not None else None
        reported = [resolve_reported_planes(port, cfg, grid=grid)
                    for port, cfg in zip(sim._waveguide_ports, waveguide_cfgs)]
        column = None
        if drive_idx is not None and len(waveguide_cfgs) > 1:
            ref_shifts = [planes[0] - waveguide_plane_positions(cfg)["reference"]
                          for planes, cfg in zip(reported, waveguide_cfgs)]
            column = extract_waveguide_s_column(waveguide_cfgs, drive_idx, ref_shifts)
        undriven = [port.name for i, port in enumerate(sim._waveguide_ports) if i not in driven]
        if undriven:
            warnings.warn(
                "Waveguide ports with no incident wave: " + ", ".join(undriven)
                + "; their own s11 and s21 entries are NaN.",
                ConfigurationWarning, stacklevel=user_stacklevel(),
            )
        for index, (port, cfg) in enumerate(zip(sim._waveguide_ports, waveguide_cfgs)):
            plane_positions = waveguide_plane_positions(cfg)
            source_plane = plane_positions["source"]
            measured_reference_plane = plane_positions["reference"]
            measured_probe_plane = plane_positions["probe"]
            reference_plane, probe_plane, calibration_preset = reported[index]
            s11, s21 = extract_waveguide_sparams(
                cfg,
                ref_shift=reference_plane - measured_reference_plane,
                probe_shift=probe_plane - measured_probe_plane,
            )
            if index not in driven:
                s11, s21 = jnp.full_like(s11, jnp.nan), jnp.full_like(s21, jnp.nan)
            s_driven = (jnp.full_like(s11, jnp.nan) if drive_idx is None else
                        s11 if column is None else column[index])
            sparams_result[port.name] = WaveguideSParamResult(
                freqs=concrete_array(cfg.freqs),
                s11=concrete_array(s11),
                s21=concrete_array(s21),
                calibration_preset=calibration_preset,
                source_plane=float(source_plane),
                measured_reference_plane=measured_reference_plane,
                measured_probe_plane=measured_probe_plane,
                reference_plane=reference_plane,
                probe_plane=probe_plane,
                driven_port=driven_port,
                s_driven=concrete_array(s_driven),
            )
        measurements = list(sparams_result.values())
        if measurements and not any(
            is_tracer(value.s11) or is_tracer(value.freqs) for value in measurements
        ):
            reflections = [value.s11 for value in measurements]
            frequencies = [value.freqs for value in measurements]
            if len(measurements) == 1:
                s11, freqs = reflections[0], frequencies[0]
            else:
                s11, freqs = np.stack(reflections), frequencies[0]
            warn_if_singular_value_exceeds_one(
                s11, freqs, extractor=f"{entry} waveguide_sparams",
            )
    return {
        "flux_monitors": flux_result,
        "waveguide_ports": ports_result,
        "waveguide_sparams": sparams_result,
    }
