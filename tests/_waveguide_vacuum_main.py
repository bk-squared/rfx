"""Hand-written copies of main 2d11d09f9's arithmetic, independent of rfx helpers.

Keep operation order: this reference pins bit identity of beta and Z, including evanescent
sentinels (the source tables changed on purpose in #2013).
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
