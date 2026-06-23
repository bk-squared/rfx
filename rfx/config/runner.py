"""Run a config-defined simulation and persist the result to HDF5."""

from __future__ import annotations

from pathlib import Path

import yaml

from rfx.io import save_simulation_dataset

from .loader import execution_to_run_kwargs, simulation_from_dict


def run_and_save(yaml_path, output, **run_kwargs):
    """Build a simulation from YAML, run it, and save the dataset.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML config.
    output : str or Path
        Destination HDF5 file (written via
        :func:`rfx.io.save_simulation_dataset`).
    **run_kwargs
        Override / supplement the ``execution`` block in the YAML. Anything
        passed here wins over the file's ``execution`` values.

    Returns
    -------
    rfx.api.Result
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"config file not found: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config file {yaml_path} must contain a top-level mapping"
        )

    sim = simulation_from_dict(cfg)

    merged_kwargs = execution_to_run_kwargs(cfg.get("execution"))
    merged_kwargs.update(run_kwargs)

    result = sim.run(**merged_kwargs)

    save_simulation_dataset(output, sim, result)
    return result
