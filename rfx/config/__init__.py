"""Config-driven front-end for rfx.

Build and run an rfx :class:`~rfx.api.Simulation` from a YAML file or a plain
``dict`` without writing Python:

>>> from rfx.config import simulation_from_yaml, run_and_save
>>> sim = simulation_from_yaml("sim.yaml")
>>> result = run_and_save("sim.yaml", "result.h5")

MVP scope is uniform-grid 3D microstrip / patch (lumped ports + point
sources, box geometry). Unsupported features raise a clear
``NotImplementedError`` naming the offending key.
"""

from __future__ import annotations

from ._shapes import shape_from_config
from ._waveforms import WAVEFORM_REGISTRY, waveform_from_config
from .loader import (
    execution_to_run_kwargs,
    simulation_from_dict,
    simulation_from_yaml,
)
from .runner import run_and_save

__all__ = [
    "simulation_from_yaml",
    "simulation_from_dict",
    "run_and_save",
    "execution_to_run_kwargs",
    "waveform_from_config",
    "WAVEFORM_REGISTRY",
    "shape_from_config",
]
