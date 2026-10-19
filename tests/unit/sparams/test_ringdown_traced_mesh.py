"""forward(ringdown=) through a traced graded cell profile.

The patch board of ``test_wire_port_on_traced_mesh``: a 50 ohm wire port feeds
a node-pinned patch, and ``delta`` stretches the cells inside its +x edge. The
ring-down probes resolve on the nominal mesh positions use on a traced axis,
and the host identification sizes the port V/I on it, a different combination
of the same recorded samples and so of the same poles; residues and S stay traced. So the traced call equals
the concrete one at a deformed profile, and its gradient passes the
early-start witness.
"""
import jax
import jax.numpy as jnp
import numpy as np

from rfx.ringdown import RingdownSpec, gradient_witness
from tests.unit.autodiff.test_wire_port_on_traced_mesh import BIN, DX, FREQS, _board

N_STEPS = 1500     # the source is off over [T/8, T]: the early-start witness can be formed


def _forward(delta):
    return _board(delta).forward(n_steps=N_STEPS, skip_preflight=True,
                                 port_s11_freqs=FREQS, ringdown=RingdownSpec())


def _objective(s):
    return jnp.log(jnp.abs(jnp.asarray(s).reshape(-1)[BIN]) ** 2 + 1e-30)


def test_traced_profile_equals_concrete_at_a_deformation():
    delta = 0.5 * DX
    concrete = np.asarray(_forward(delta).ringdown.s_params, np.complex128)
    traced = np.asarray(jax.jit(lambda d: _forward(d).ringdown.s_params)(jnp.float32(delta)),
                        np.complex128)
    assert np.isfinite(traced).all()
    assert np.abs(traced - concrete).max() <= 1e-4 * np.abs(concrete).max()


def test_traced_profile_gradient_passes_the_early_start_witness():
    def both(d):
        r = _forward(d)
        return (_objective(r.ringdown.s_params), _objective(r.ringdown.s_params_long)), r.ringdown

    _, pullback, ringdown = jax.vjp(both, jnp.float32(0.25 * DX), has_aux=True)
    (g,) = pullback((jnp.float32(1.0), jnp.float32(0.0)))
    (g_long,) = pullback((jnp.float32(0.0), jnp.float32(1.0)))
    assert np.isfinite(float(g)) and float(g) != 0.0
    witness = gradient_witness(g, g_long, against='early_start', ringdown=ringdown)
    assert witness.judged and witness.ok, witness
