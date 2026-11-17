"""Node-index refusals (P1).

CPU budget: <60 s per file; declarations only, no built mesh or field steps.
"""
from dataclasses import asdict

import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx._traced_refusal import TracedStructureRefusal


def _declare(**overrides):
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001,
                     boundary="pec")
    indices = dict(normal_axis=2, plane_index=1, i_range=(2, 5), j_range=(3, 6))
    indices.update(overrides)
    sim.add_pinned_sheet(**indices)
    return sim._pinned_sheets[0]


@pytest.mark.parametrize("field", ["normal_axis", "plane_index", "i_range", "j_range"])
@pytest.mark.parametrize("transform", ["grad", "jvp", "jit", "vmap"])
def test_traced_node_indices_refuse(field, transform):
    """Physical anchor: a sheet moves by whole nodes, so its index has no derivative."""
    def objective(index):
        value = (index, 6) if field.endswith("range") else index
        _declare(**{field: value})
        return index * 0.

    with pytest.raises(TracedStructureRefusal) as caught:
        if transform == "jvp":
            jax.jvp(objective, (1.,), (1.,))
        elif transform == "vmap":
            jax.vmap(objective)(jnp.array([1., 2.]))
        else:
            getattr(jax, transform)(objective)(1.)
    assert "add_pinned_sheet" in str(caught.value)
    assert field in str(caught.value)
    assert "whole nodes" in str(caught.value)


def test_python_indices_keep_the_concrete_sheet():
    """Physical anchor: fixed integer endpoints preserve the same node footprint."""
    sheet = _declare(i_range=(5, 2), j_range=(6, 3))
    assert asdict(sheet) == dict(normal_axis=2, plane_index=1,
                                 i_range=(2, 5), j_range=(3, 6),
                                 sigma_bulk=5.8e7, thickness=35e-6, eps_r=1.,
                                 surface_impedance_f0=None, name=None)
    assert all(type(v) is int for v in (sheet.normal_axis, sheet.plane_index,
                                       *sheet.i_range, *sheet.j_range))
