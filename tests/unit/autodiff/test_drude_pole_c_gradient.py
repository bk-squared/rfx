"""A Drude pole's omega_p reaches the time step only through c.

drude_pole gives omega_0 = 0 and kappa = omega_p**2. A dropped derivative
of c therefore makes reverse-mode AD differ from a central difference of
the finite-record J; cutting c entirely gives AD = 0. J is a fixed finite
sum, so no record-length witness is needed (design A12/A12c, C1).

At N1 = 20,980, the measured new-code AD and FD were both 7.163866e17,
with |AD/FD - 1| = 4.7e-8. The case took 10.8 s and 1.74 GiB on cpu 4.
Cutting the y/z c derivatives gave AD = -0.362 FD; cutting core c gave
AD = 0 exactly. These observations motivate the bar, not numeric pins.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation, drude_pole
from tests._x64_compat import enable_x64
from tests.oracle._drude_plasma_scene import DOMAIN, DX, PROBES, SOURCE, FREQ_MAX, FP, waveform


def test_drude_pole_c_gradient_matches_finite_difference():
    with enable_x64():
        def objective(omega_p):
            sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
                             boundary="pec", precision="float64")
            sim.add_material("plasma", eps_r=1.0, sigma=0.0,
                             lorentz_poles=[drude_pole(omega_p, gamma=1.063e7)])
            sim.add(Box((0.0, 0.0, 0.0), DOMAIN), material="plasma")
            sim.add_source(SOURCE, component="ez", waveform=waveform(),
                           amplitude_kind="current")
            for position in PROBES:
                sim.add_probe(position, component="ez")
            # 20,980 steps is one period of f_cf at scene P's dt (design A12).
            result = sim.forward(n_steps=20980, checkpoint=False)
            return jnp.sum(result.time_series**2)

        omega = jnp.asarray(2 * np.pi * FP, dtype=jnp.float64)
        j0 = float(objective(omega))
        ad = float(jax.grad(objective)(omega))
        w_plus, w_minus = omega * (1 + 1e-4), omega * (1 - 1e-4)
        fd = float((objective(w_plus) - objective(w_minus)) / (w_plus - w_minus))
        assert np.isfinite(j0), f"C1 bar: J is non-finite: {j0}"
        assert np.isfinite(ad), f"C1 bar: AD is non-finite: {ad}"
        assert np.isfinite(fd) and fd != 0, f"C1 bar: FD must be finite and nonzero: {fd}"
        ratio = ad / fd
        print(f"C1: J={j0:.17g} AD={ad:.17g} FD={fd:.17g} AD/FD={ratio:.17g}",
              file=sys.stderr, flush=True)
        assert abs(ratio - 1) <= 1e-2, f"C1 bar: |AD/FD - 1|={abs(ratio - 1):.17g} > 1e-2 (AD/FD={ratio:.17g})"
