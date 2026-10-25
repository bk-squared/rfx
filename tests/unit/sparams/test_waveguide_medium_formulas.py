"""Independent SI and Yee formulas, including magnetic and TM terms."""
import numpy as np
import pytest
import jax.numpy as jnp
from rfx.model import PortMedium
from rfx.model.waveguide_modes import _compute_beta, _compute_mode_impedance


@pytest.mark.parametrize('medium', [PortMedium(), PortMedium(2.2,1.), PortMedium(2.2,2.)])
@pytest.mark.parametrize('discrete', [False,True])
@pytest.mark.parametrize('mode', ['TE','TM'])
def test_hand_written_modal_formulas(medium, discrete, mode):
    # Hand-written expected expressions. No implementation helper supplies kc,
    # c, beta, or Z. Cutoff is specified in the filled medium.
    eps0, mu0 = 8.8541878128e-12, 1.25663706212e-6
    speed = 1 / np.sqrt(eps0 * mu0 * medium.eps_r * medium.mu_r)
    f, fc = np.array([5e9,7e9,9e9]), 3e9
    omega, kc = 2*np.pi*f, 2*np.pi*fc/speed
    dt, dx = (0.5e-12, 0.5e-3) if discrete else (0.,0.)
    if discrete:
        beta = 2/dx*np.arcsin(dx/2*np.sqrt((np.sin(omega*dt/2)/(speed*dt/2))**2-kc**2))
        expected = (mu0*medium.mu_r*dx*np.sin(omega*dt/2)/(dt*np.sin(beta*dx/2)) if mode=='TE'
                    else dt*np.sin(beta*dx/2)/(eps0*medium.eps_r*dx*np.sin(omega*dt/2)))
    else:
        beta = np.sqrt((omega/speed)**2-kc**2)
        expected = omega*mu0*medium.mu_r/beta if mode=='TE' else beta/(omega*eps0*medium.eps_r)
    np.testing.assert_allclose(_compute_beta(jnp.asarray(f),fc,medium=medium,dt=dt,dx=dx),beta,rtol=2e-6)
    np.testing.assert_allclose(_compute_mode_impedance(jnp.asarray(f),fc,mode,medium=medium,dt=dt,dx=dx),expected,rtol=2e-6)


def test_medium_keyword_required():
    with pytest.raises(TypeError,match='medium'):
        _compute_beta(jnp.array([8e9]),3e9)
    with pytest.raises(TypeError,match='medium'):
        _compute_mode_impedance(jnp.array([8e9]),3e9,'TE')
