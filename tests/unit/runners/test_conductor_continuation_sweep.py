"""The material sweep applies a face-reaching sheet after each pad fill."""
import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation
from rfx.vmap_sweep import _build_batched_materials


def test_face_reaching_thin_conductor_matches_individual_assemblies():
    def build(eps):
        sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                         boundary="cpml", cpml_layers=2)
        sim.add_material("substrate", eps_r=eps)
        sim.add(Box((0., 2., 2.), (8., 6., 4.)), material="substrate")
        sim.add_thin_conductor(Box((0., 2., 4.), (8., 6., 4.)),
                               sigma_bulk=100., thickness=.01)
        grid = sim._build_grid()
        mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
        return sim, grid, mats
    sim, grid, base = build(2.)
    values = [2., 3., 5.]
    batch = _build_batched_materials(sim, grid, base, "substrate.eps_r",
                                     jnp.asarray(values))
    for i, value in enumerate(values):
        _, _, expected = build(value)
        for name in ("eps_r", "sigma", "mu_r"):
            np.testing.assert_array_equal(np.asarray(getattr(batch, name)[i]),
                                          np.asarray(getattr(expected, name)))
        # Both paths being empty in the pad must fail too.
        assert np.any(np.asarray(expected.sigma)[0] > 0)
        assert np.any(np.asarray(batch.sigma)[i, -2] > 0)
