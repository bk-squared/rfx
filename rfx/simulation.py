"""Top-level simulation API: build, run, and extract results."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import FDTDState, MaterialArrays, init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec


class Simulation:
    """FDTD simulation runner.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays | None
        Material property arrays. Defaults to free space.
    boundary : str
        Boundary type: "pec" or "cpml". Default "pec".
    """

    def __init__(
        self,
        grid: Grid,
        materials: MaterialArrays | None = None,
        boundary: str = "pec",
    ):
        self.grid = grid
        self.materials = materials if materials is not None else init_materials(grid.shape)
        self.boundary = boundary
        self._source_fns: list[Callable] = []
        self._probe_fns: list[Callable] = []

    def add_source(self, fn: Callable) -> None:
        """Register a source function: fn(state, grid, step, dt) -> state."""
        self._source_fns.append(fn)

    def add_probe(self, fn: Callable) -> None:
        """Register a probe function: fn(state, grid, step, dt) -> scalar."""
        self._probe_fns.append(fn)

    def _step_fn(self, dt: float, dx: float):
        """Build a single-timestep function for jax.lax.scan."""
        materials = self.materials
        boundary = self.boundary
        source_fns = self._source_fns

        def step(state: FDTDState, _) -> tuple[FDTDState, jnp.ndarray]:
            # H update
            state = update_h(state, materials, dt, dx)

            # E update
            state = update_e(state, materials, dt, dx)

            # Boundary conditions
            if boundary == "pec":
                state = apply_pec(state)
            elif boundary == "cpml":
                raise NotImplementedError(
                    "CPML boundary not yet integrated into Simulation.run(). "
                    "Use boundary='pec' for Stage 1, or apply CPML manually."
                )

            # Sources
            for src_fn in source_fns:
                state = src_fn(state, dt)

            # Probe sample (Ez at center for default)
            return state, jnp.array(0.0)

        return step

    def run(self, num_steps: int | None = None) -> tuple[FDTDState, jnp.ndarray]:
        """Run the simulation.

        Parameters
        ----------
        num_steps : int | None
            Number of timesteps. Defaults to grid.num_timesteps().

        Returns
        -------
        final_state : FDTDState
        probe_data : jnp.ndarray
        """
        if num_steps is None:
            num_steps = self.grid.num_timesteps()

        state = init_state(self.grid.shape)
        step_fn = self._step_fn(self.grid.dt, self.grid.dx)

        final_state, probe_data = jax.lax.scan(
            step_fn, state, jnp.arange(num_steps)
        )

        return final_state, probe_data

    def run_with_probes(
        self,
        probe_component: str,
        probe_index: tuple[int, int, int],
        num_steps: int | None = None,
    ) -> tuple[FDTDState, jnp.ndarray]:
        """Run simulation and record a field component at a point each step.

        Returns
        -------
        final_state : FDTDState
        time_series : jnp.ndarray of shape (num_steps,)
        """
        if num_steps is None:
            num_steps = self.grid.num_timesteps()

        state = init_state(self.grid.shape)
        materials = self.materials
        boundary = self.boundary
        source_fns = self._source_fns
        dt = self.grid.dt
        dx = self.grid.dx
        i, j, k = probe_index

        def step(state: FDTDState, _) -> tuple[FDTDState, jnp.ndarray]:
            state = update_h(state, materials, dt, dx)
            state = update_e(state, materials, dt, dx)

            if boundary == "pec":
                state = apply_pec(state)
            elif boundary == "cpml":
                raise NotImplementedError(
                    "CPML boundary not yet integrated. Use boundary='pec'."
                )

            for src_fn in source_fns:
                state = src_fn(state, dt)

            sample = getattr(state, probe_component)[i, j, k]
            return state, sample

        final_state, time_series = jax.lax.scan(
            step, state, jnp.arange(num_steps)
        )

        return final_state, time_series
