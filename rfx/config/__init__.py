"""Legacy YAML compatibility adapter for rfx.

Build and run an rfx :class:`~rfx.api.Simulation` from a YAML file or a plain
``dict`` without writing Python:

>>> from rfx.config import simulation_from_yaml, run_and_save
>>> sim = simulation_from_yaml("sim.yaml")
>>> result = run_and_save("sim.yaml", "result.h5")

The canonical Studio/agent contract is now ``rfx-experiment/v2`` in
``rfx.experiments``. This package remains stable for existing YAML and CLI
users, translating the legacy uniform-grid microstrip/patch subset directly to
the public builder API. It is deliberately not extended with new Studio
variants; unsupported features raise a clear ``NotImplementedError``.
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
