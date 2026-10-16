"""One realized model owns at most one tensor operand set (M4b P2)."""
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from rfx import Simulation, simulation
from rfx.core.yee import MaterialArrays
from rfx.model.materials import validate_components, with_components
from rfx.model.tensor_operands import TensorOperands, attach_tensor


def test_one_tensor_set_per_model_and_raw_run_without_tensor():
    grid = Simulation(freq_max=15e9, domain=(.012, .012, .012), dx=.002,
                      boundary="pec")._build_grid()
    cells = MaterialArrays(jnp.ones(grid.shape), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    materials = with_components(cells, grid, periodic=(False,) * 3)
    first = TensorOperands.of(eps=(cells.eps_r,) * 3)
    second = TensorOperands.of(eps=(cells.eps_r,) * 3)
    attached = attach_tensor(materials, first)
    assert attached.components.tensor is first
    assert attach_tensor(attached, first).components.tensor is first
    assert attached.components.cells_view is materials.components.cells_view
    assert attached.components.periodic is materials.components.periodic
    assert attached.components.grid_key is materials.components.grid_key
    validate_components(attached, periodic=(False,) * 3, grid=grid)
    with pytest.raises(ValueError, match="two tensor operand sets for one model"):
        attach_tensor(attached, second)
    with pytest.raises(ValueError, match="require realized material components"):
        attach_tensor(cells, first)
    assert TensorOperands.of(inv_eps_smooth=True) is None

    captured = []
    setup = simulation._build_step_setup

    def record(*args, **kwargs):
        result = setup(*args, **kwargs)
        captured.append(result.ctx_kwargs["materials"].components.tensor)
        return result

    with patch.object(simulation, "_build_step_setup", record):
        simulation.run(grid, cells, 2, boundary="pec")
    assert captured == [None]
