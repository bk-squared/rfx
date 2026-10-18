"""A graded current drive in a pole medium rounds the same eager and under a user jit (#1856).

With a traced permittivity the drive coefficient Cb is formed inside the user's program. XLA compiles
``Cb / dV`` there as a multiply by the float32 reciprocal of dV; an eager call divided, so the source edge
differed by 1-2 float32 ULP from the first driven step (graded CPML box, Debye block, source inside it).
"""
import jax
import jax.numpy as jnp
import numpy as np

from tests.unit.runners.test_distributed_edge_averaged_materials import _debye_box, _mm


def test_traced_override_drive_is_trace_invariant_in_a_debye_medium():
    sim = _debye_box('cpml', 'fwd', 'debye')
    sim.add_probe(_mm(4, 6, 6), 'ez')  # the source edge
    eps = jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)

    def series(e):
        return sim.forward(eps_override=e, n_steps=20, skip_preflight=True, checkpoint=False).time_series

    eager = np.asarray(series(eps))
    compiled = np.asarray(jax.jit(series)(eps))
    assert np.max(np.abs(eager)) > 0
    np.testing.assert_array_equal(eager.view(np.uint32), compiled.view(np.uint32))
