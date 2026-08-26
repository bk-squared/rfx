"""Sparse DFT-region contracts for MSL port extraction."""

from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.probes import init_dft_plane_probe, update_dft_plane_probe


def _state(ez: jnp.ndarray, step: int) -> SimpleNamespace:
    return SimpleNamespace(ez=ez, step=jnp.asarray(step, dtype=jnp.int32))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sparse_monitor_matches_full_plane(axis):
    """A cropped plane must retain exactly the requested DFT samples."""
    shape = (5, 7, 8)
    region = (2, 5, 1, 7)
    freqs = jnp.asarray([0.7e9, 1.1e9, 1.9e9], dtype=jnp.float32)
    full = init_dft_plane_probe(
        axis=axis,
        index=2,
        component="ez",
        freqs=freqs,
        grid_shape=shape,
    )
    sparse = init_dft_plane_probe(
        axis=axis,
        index=2,
        component="ez",
        freqs=freqs,
        grid_shape=shape,
        region=region,
    )

    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    for step in range(4):
        state = _state(base * (step + 1), step)
        full = update_dft_plane_probe(full, state, dt=2.5e-12)
        sparse = update_dft_plane_probe(sparse, state, dt=2.5e-12)

    lo1, hi1, lo2, hi2 = region
    expected = full.accumulator[:, lo1:hi1, lo2:hi2]
    np.testing.assert_array_equal(np.asarray(sparse.accumulator), np.asarray(expected))
    assert sparse.accumulator.size < full.accumulator.size


def test_sparse_monitor_gradient_matches_full_plane():
    """Cropping must preserve gradients of objectives over retained samples."""
    shape = (4, 6, 7)
    region = (1, 5, 2, 6)
    freqs = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 10.0

    def objective(scale: jnp.ndarray, *, sparse: bool) -> jnp.ndarray:
        if sparse:
            probe = init_dft_plane_probe(
                axis=0,
                index=1,
                component="ez",
                freqs=freqs,
                grid_shape=shape,
                region=region,
            )
        else:
            probe = init_dft_plane_probe(
                axis=0,
                index=1,
                component="ez",
                freqs=freqs,
                grid_shape=shape,
            )
        for step in range(3):
            probe = update_dft_plane_probe(
                probe,
                _state(base * scale * (step + 1), step),
                dt=0.1,
            )
        acc = probe.accumulator
        if not sparse:
            lo1, hi1, lo2, hi2 = region
            acc = acc[:, lo1:hi1, lo2:hi2]
        return jnp.sum(jnp.abs(acc) ** 2)

    scale = jnp.asarray(0.8, dtype=jnp.float32)
    grad_full = jax.grad(lambda value: objective(value, sparse=False))(scale)
    grad_sparse = jax.grad(lambda value: objective(value, sparse=True))(scale)

    assert jnp.isfinite(grad_sparse)
    assert abs(float(grad_sparse)) > 1e-3
    np.testing.assert_allclose(
        np.asarray(grad_sparse),
        np.asarray(grad_full),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_msl_registers_sparse_regions():
    """The production MSL path must opt every internal plane into cropping."""
    from tests.test_msl_sparam_ad import (
        _build_thru_line_sim,
        _fake_run_for_theta,
    )

    sim = _build_thru_line_sim()
    fake_run = _fake_run_for_theta(jnp.asarray(0.7, dtype=jnp.float32))
    observed: list[tuple[int, int, object]] = []

    def audited_run(self, *, n_steps=None, num_periods=1.0,
                    compute_s_params=False):
        grid = self._build_grid()
        for entry in self._dft_planes:
            if entry.axis == "x":
                full_shape = (grid.ny, grid.nz)
            elif entry.axis == "y":
                full_shape = (grid.nx, grid.nz)
            else:
                full_shape = (grid.nx, grid.ny)
            region = self._dft_plane_regions.get(entry.name)
            sparse_cells = (
                int(np.prod(full_shape))
                if region is None
                else (region[1] - region[0]) * (region[3] - region[2])
            )
            observed.append((int(np.prod(full_shape)), sparse_cells, region))
        return fake_run(
            self,
            n_steps=n_steps,
            num_periods=num_periods,
            compute_s_params=compute_s_params,
        )

    sim.run = MethodType(audited_run, sim)
    sim.compute_msl_s_matrix(
        n_steps=1,
        freqs=jnp.asarray([1.0e9], dtype=jnp.float32),
        num_periods=1.0,
        enforce_passivity=False,
    )

    assert observed
    assert all(region is not None for _, _, region in observed)
    assert sum(sparse for _, sparse, _ in observed) < sum(
        full for full, _, _ in observed
    )
