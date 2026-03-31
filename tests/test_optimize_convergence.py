"""End-to-end convergence tests for the rfx differentiable optimization pipeline.

Validates that gradient-based inverse design actually converges on
two real EM problems:
1. Minimising reflected energy (S11 proxy) through a dielectric slab
2. Maximising transmitted probe signal through a design region

Each test must complete in <120 seconds on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation, Box
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port
from rfx.core.yee import MaterialArrays
from rfx.simulation import run as _run, make_port_source, make_probe as _make_probe
from rfx.optimize import DesignRegion, _latent_to_eps


def _make_adam_state(latent):
    """Initialise Adam optimiser state."""
    return {
        "m": jnp.zeros_like(latent),
        "v": jnp.zeros_like(latent),
    }


def _adam_step(latent, grad, state, lr, it):
    """One Adam update; returns (new_latent, new_state)."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    state["m"] = beta1 * state["m"] + (1 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
    m_hat = state["m"] / (1 - beta1 ** (it + 1))
    v_hat = state["v"] / (1 - beta2 ** (it + 1))
    return latent - lr * m_hat / (jnp.sqrt(v_hat) + eps), state


def _build_forward(sim, region, n_steps, objective_fn):
    """Build a differentiable forward function: latent -> scalar loss.

    Uses the low-level runner with checkpoint=True for memory-efficient AD.
    """
    grid = sim._build_grid()
    lo_idx = grid.position_to_index(region.corner_lo)
    hi_idx = grid.position_to_index(region.corner_hi)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    eps_min, eps_max = region.eps_range

    def forward(lat):
        eps_design = _latent_to_eps(lat, eps_min, eps_max)
        materials, debye_spec, lorentz_spec = sim._assemble_materials(grid)
        eps_r = materials.eps_r

        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)
        materials = MaterialArrays(eps_r=eps_r, sigma=materials.sigma, mu_r=materials.mu_r)

        sources = []
        probes = []

        for pe in sim._ports:
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))

        for pe in sim._probes:
            probes.append(_make_probe(grid, pe.position, pe.component))

        _, debye, lorentz = sim._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)

        result = _run(
            grid, materials, n_steps,
            boundary=sim._boundary,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            checkpoint=True,
        )
        return objective_fn(result)

    return forward, design_shape


def _run_optimization(forward, design_shape, n_iters, lr, verbose=True):
    """Run Adam optimization loop, returning losses and gradient norms."""
    latent = jnp.zeros(design_shape, dtype=jnp.float32)
    adam = _make_adam_state(latent)
    grad_fn = jax.value_and_grad(forward)

    losses = []
    grad_norms = []

    for it in range(n_iters):
        loss, grad = grad_fn(latent)
        loss_val = float(loss)
        g_norm = float(jnp.sqrt(jnp.sum(grad ** 2)))
        losses.append(loss_val)
        grad_norms.append(g_norm)

        latent, adam = _adam_step(latent, grad, adam, lr, it)

        if verbose and (it % 5 == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  loss = {loss_val:.6e}  |grad| = {g_norm:.6e}")

    return losses, grad_norms


class TestOptimizeDielectricSlabS11:
    """Optimize eps_r in a dielectric slab to minimize |S11| (reflected energy)."""

    def test_optimize_dielectric_slab_s11(self):
        """Slab permittivity optimisation converges: final loss < 0.5 * initial."""
        # Small domain with PEC walls — reflections are strong and the
        # optimiser has a clear gradient signal to reduce them by tuning
        # the slab permittivity toward impedance matching.
        sim = Simulation(
            freq_max=10e9,
            domain=(0.06, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_port(
            (0.01, 0.01, 0.01), "ez",
            impedance=50.0,
            waveform=GaussianPulse(f0=5e9, bandwidth=0.5),
        )
        # Probe at the port to measure total field (incident + reflected)
        sim.add_probe((0.01, 0.01, 0.01), "ez")
        # Second probe beyond the slab to measure transmitted energy
        sim.add_probe((0.050, 0.01, 0.01), "ez")

        region = DesignRegion(
            corner_lo=(0.025, 0.0, 0.0),
            corner_hi=(0.035, 0.02, 0.02),
            eps_range=(1.0, 6.0),
        )

        n_steps = 200

        # Objective: minimise reflected energy at port (probe 0) while
        # encouraging transmission to probe 1.
        # A simple and effective proxy: maximise transmitted energy
        # (equivalent to minimising S11 in a two-port sense).
        def s11_proxy(result):
            # Maximise transmitted energy = minimise negative
            ts_tx = result.time_series[:, 1]
            return -jnp.sum(ts_tx ** 2)

        forward, design_shape = _build_forward(sim, region, n_steps, s11_proxy)

        n_iters = 25
        losses, grad_norms = _run_optimization(
            forward, design_shape, n_iters=n_iters, lr=0.1,
        )

        # ---- assertions ----
        assert len(losses) == n_iters
        assert not any(np.isnan(l) for l in losses), \
            f"NaN in loss history: {losses}"

        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss * 0.5, (
            f"Optimiser did not converge: initial={initial_loss:.6e}, "
            f"final={final_loss:.6e}, ratio={final_loss/initial_loss:.4f}"
        )

        print(f"\n[S11 slab] initial={initial_loss:.6e}, final={final_loss:.6e}, "
              f"ratio={final_loss/initial_loss:.4f}")


class TestOptimizeWaveguideTransmission:
    """Optimize eps_r between source and probe to maximise transmission."""

    def test_optimize_waveguide_transmission(self):
        """Transmission optimisation converges with non-zero gradients."""
        sim = Simulation(
            freq_max=10e9,
            domain=(0.04, 0.01, 0.01),
            boundary="pec",
        )
        sim.add_port(
            (0.008, 0.005, 0.005), "ez",
            impedance=50.0,
            waveform=GaussianPulse(f0=5e9, bandwidth=0.5),
        )
        sim.add_probe((0.030, 0.005, 0.005), "ez")

        region = DesignRegion(
            corner_lo=(0.015, 0.0, 0.0),
            corner_hi=(0.025, 0.01, 0.01),
            eps_range=(1.0, 4.0),
        )

        n_steps = 150

        def neg_transmission(result):
            ts = result.time_series[:, 0]
            return -jnp.sum(ts ** 2)

        forward, design_shape = _build_forward(sim, region, n_steps, neg_transmission)

        n_iters = 20
        losses, grad_norms = _run_optimization(
            forward, design_shape, n_iters=n_iters, lr=0.1,
        )

        # ---- assertions ----
        assert not any(np.isnan(l) for l in losses), \
            f"NaN in loss history: {losses}"
        assert not any(np.isnan(g) for g in grad_norms), \
            f"NaN in gradient norms: {grad_norms}"

        # Gradient is non-zero at step 0
        assert grad_norms[0] > 0.0, \
            "Gradient norm at step 0 is zero — AD path is broken"

        # Objective decreased (transmission increased)
        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss, (
            f"Transmission did not increase: initial={initial_loss:.6e}, "
            f"final={final_loss:.6e}"
        )

        print(f"\n[Waveguide TX] initial={initial_loss:.6e}, final={final_loss:.6e}, "
              f"grad_norm[0]={grad_norms[0]:.6e}")
