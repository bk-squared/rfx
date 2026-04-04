"""Batch simulation and parameter sweep utilities for ML data generation."""

from __future__ import annotations

import itertools
from typing import Callable, Any

import numpy as np


class ParameterSweep:
    """Define a multi-dimensional parameter sweep.

    Parameters are given as keyword arguments, each mapping to an array
    of values.  The sweep iterates over the full Cartesian product.

    Example
    -------
    >>> sweep = ParameterSweep(width=[0.01, 0.02], eps_r=[2.0, 4.0])
    >>> sweep.total
    4
    >>> list(sweep.combinations())
    [{'width': 0.01, 'eps_r': 2.0}, {'width': 0.01, 'eps_r': 4.0}, ...]
    """

    def __init__(self, **params):
        self._keys = list(params.keys())
        self._values = [np.asarray(v).ravel() for v in params.values()]

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def total(self) -> int:
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def combinations(self):
        """Iterate over all parameter combinations as dicts."""
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, [float(c) for c in combo]))


def run_batch(
    sim_factory: Callable[..., Any],
    sweep: ParameterSweep,
    *,
    run_kwargs: dict | None = None,
) -> list[tuple[dict, Any]]:
    """Run simulations for all parameter combinations.

    Parameters
    ----------
    sim_factory : callable
        Function that takes keyword args from the sweep and returns
        a configured Simulation object (or any object with a .run() method).
    sweep : ParameterSweep
        Parameter sweep definition.
    run_kwargs : dict or None
        Extra keyword arguments passed to sim.run().

    Returns
    -------
    list of (params_dict, result) tuples
    """
    if run_kwargs is None:
        run_kwargs = {}
    results = []
    for i, params in enumerate(sweep.combinations()):
        sim = sim_factory(**params)
        result = sim.run(**run_kwargs)
        results.append((params, result))
    return results


class SimulationDataset:
    """Structured dataset from batch simulation results.

    Parameters
    ----------
    inputs : ndarray, shape (n_samples, n_params)
    outputs : ndarray, shape (n_samples, n_outputs)
    input_keys : list of str
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 input_keys: list[str]):
        self.inputs = inputs
        self.outputs = outputs
        self.input_keys = input_keys

    @classmethod
    def from_results(
        cls,
        results: list[tuple[dict, Any]],
        input_keys: list[str],
        output_fn: Callable,
    ) -> "SimulationDataset":
        """Build dataset from run_batch() results.

        Parameters
        ----------
        results : list of (params_dict, Result)
        input_keys : which param keys to use as inputs
        output_fn : callable(Result) -> 1D array of output values
        """
        X_rows, Y_rows = [], []
        for params, result in results:
            X_rows.append([params[k] for k in input_keys])
            Y_rows.append(np.asarray(output_fn(result)).ravel())
        return cls(
            inputs=np.array(X_rows),
            outputs=np.array(Y_rows),
            input_keys=input_keys,
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) arrays."""
        return self.inputs, self.outputs

    def to_hdf5(self, path: str):
        """Save to HDF5 file."""
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("inputs", data=self.inputs)
            f.create_dataset("outputs", data=self.outputs)
            f.attrs["input_keys"] = self.input_keys

    def to_csv(self, path: str):
        """Save to CSV file."""
        header = ",".join(self.input_keys +
                          [f"y{i}" for i in range(self.outputs.shape[1])])
        data = np.hstack([self.inputs, self.outputs])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
