"""Hand-written copies of main 2d11d09f9's arithmetic, independent of rfx helpers.

Keep operation order: this reference pins bit identity, including evanescent
sentinels and the separate NumPy source-table / JAX measurement arithmetic.
"""
import jax.numpy as jnp
import numpy as np

EPS0, MU0 = 8.8541878128e-12, 1.25663706212e-6
C = 1.0 / np.sqrt(EPS0 * MU0)


def main_beta(f, fc, dt, dx):
    omega = 2 * jnp.pi * f
    kc = 2 * jnp.pi * fc / C
    if dt > 0 and dx > 0:
        temporal = jnp.sin(omega * .5 * dt) / (C * .5 * dt)
        squared = temporal ** 2 - kc ** 2
        arg = jnp.clip(.5 * dx * jnp.sqrt(jnp.maximum(squared, 0.0)),
                       -1.0 + 1e-12, 1.0 - 1e-12)
        return jnp.where(squared >= 0, (2.0 / dx) * jnp.arcsin(arg),
                         1j * jnp.sqrt(jnp.maximum(-squared, 0.0)))
    squared = (omega / C) ** 2 - kc ** 2
    return jnp.where(squared >= 0, jnp.sqrt(jnp.maximum(squared, 0.0)),
                     1j * jnp.sqrt(jnp.maximum(-squared, 0.0)))


def main_impedance(f, fc, mode, dt, dx):
    omega = 2 * jnp.pi * f
    beta = main_beta(f, fc, dt, dx)
    safe_beta = jnp.where(jnp.abs(beta) > 1e-30, beta, 1e-30 * jnp.ones_like(beta))
    safe_omega = jnp.where(jnp.abs(omega) > 1e-30, omega, 1e-30 * jnp.ones_like(omega))
    if dt > 0 and dx > 0:
        sw, sb = jnp.sin(omega * .5 * dt), jnp.sin(beta * .5 * dx)
        if mode == 'TE':
            z = MU0 * dx * sw / jnp.maximum(dt * sb, 1e-30)
            return jnp.where(f > fc, z, jnp.asarray(1e30, dtype=jnp.float32))
        z = dt * sb / jnp.maximum(EPS0 * dx * sw, 1e-30)
        return jnp.where(f > fc, z, jnp.asarray(0., dtype=jnp.float32))
    return omega * MU0 / safe_beta if mode == 'TE' else beta / (safe_omega * EPS0)


def main_tables(fc, dt, dx, n, t0, tau, f0, profile, mode, dtype):
    t = np.arange(n, dtype=np.float64) * float(dt)
    tt = t - t0
    raw = np.cos(2.0 * np.pi * f0 * tt) * np.exp(-(((t-t0)/tau) ** 2))
    raw = np.where(np.abs(tt) > t0, 0.0, raw)
    spectrum = np.fft.rfft(raw)
    f = np.fft.rfftfreq(n, d=float(dt))
    omega = 2.0 * np.pi * f
    kc = 2.0 * np.pi * fc / C
    propagating = omega > (2.0 * np.pi * fc)
    if profile == 'discrete':
        temporal = np.sin(omega * .5 * float(dt)) / (.5 * float(dt))
        squared = np.maximum((temporal/C) ** 2 - kc ** 2, 0.0) * (.5*float(dx)) ** 2
        beta = np.where(propagating, (2.0/float(dx))*np.arcsin(np.sqrt(np.minimum(squared, 1.0))), 0.0)
        sw, sb = np.sin(omega*.5*float(dt)), np.sin(beta*.5*float(dx))
        z = (MU0*float(dx)*sw / np.maximum(float(dt)*sb, 1e-30) if mode=='TE'
             else float(dt)*sb / np.maximum(EPS0*float(dx)*sw, 1e-30))
    else:
        beta = np.where(propagating, np.sqrt(np.maximum((omega/C)**2-kc**2, 0.0)), 0.0)
        z = (omega*MU0/np.maximum(beta, 1e-30) if mode=='TE'
             else beta/np.maximum(omega*EPS0, 1e-30))
    z = np.where(propagating, z, np.inf if mode=='TE' else 0.0)
    phase = np.exp(+1j*beta*.5*float(dx) + 1j*omega*.5*float(dt))
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(propagating, spectrum/z*phase, 0.0)
    return (jnp.asarray(np.fft.irfft(spectrum,n),dtype=dtype),
            jnp.asarray(np.fft.irfft(h,n),dtype=dtype))
