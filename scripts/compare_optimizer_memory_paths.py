#!/usr/bin/env python3
"""Compare compiled memory stats for full-trace vs proxy-objective paths.

This script is intended for issue #13 / #18 validation work. It compares a
custom objective that consumes ``result.time_series`` directly against a built-in
proxy objective that reduces the differentiable contract to a scalar loss.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.optimize import DesignRegion, _latent_to_eps
from rfx.optimize_objectives import maximize_transmitted_energy
from rfx.sources.sources import GaussianPulse


def build_problem(domain: float, dx: float, n_steps: int):
    sim = Simulation(
        freq_max=10e9,
        domain=(domain, 0.02, 0.02),
        boundary="pec",
        dx=dx,
    )
    sim.add_port(
        (0.008, 0.01, 0.01),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=5e9, bandwidth=0.5),
    )
    sim.add_probe((domain - 0.01, 0.01, 0.01), "ez")

    region = DesignRegion(
        corner_lo=(0.015, 0.0, 0.0),
        corner_hi=(0.025, 0.02, 0.02),
        eps_range=(1.0, 4.0),
    )

    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    base_sigma = base_materials.sigma
    base_mu_r = base_materials.mu_r

    lo_idx = list(grid.position_to_index(region.corner_lo))
    hi_idx = list(grid.position_to_index(region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads[d])
    lo_idx = tuple(lo_idx)
    hi_idx = tuple(hi_idx)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    latent0 = jnp.zeros(design_shape, dtype=jnp.float32)

    def raw_materialized_forward(latent):
        eps_design = _latent_to_eps(latent, *region.eps_range)
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)
        from rfx.core.yee import MaterialArrays
        materials = MaterialArrays(eps_r=eps_r, sigma=base_sigma, mu_r=base_mu_r)
        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=True,
            pec_mask=base_pec_mask,
        )
        return result.time_series

    builtin_obj = maximize_transmitted_energy(output_probe_idx=0)

    def raw_proxy_forward(latent):
        eps_design = _latent_to_eps(latent, *region.eps_range)
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)
        from rfx.core.yee import MaterialArrays
        materials = MaterialArrays(eps_r=eps_r, sigma=base_sigma, mu_r=base_mu_r)
        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=True,
            pec_mask=base_pec_mask,
        )
        return builtin_obj(result)

    def materialized_forward(latent):
        return -jnp.sum(raw_materialized_forward(latent)[:, 0] ** 2)

    def proxy_forward(latent):
        return raw_proxy_forward(latent)

    return (
        latent0,
        raw_materialized_forward,
        raw_proxy_forward,
        materialized_forward,
        proxy_forward,
    )


def memory_stats(fn, latent0):
    compiled = jax.jit(jax.value_and_grad(fn)).lower(latent0).compile()
    stats = compiled.memory_analysis()
    return {
        "argument_bytes": stats.argument_size_in_bytes,
        "output_bytes": stats.output_size_in_bytes,
        "temp_bytes": stats.temp_size_in_bytes,
        "generated_code_bytes": stats.generated_code_size_in_bytes,
    }


def forward_memory_stats(fn, latent0):
    compiled = jax.jit(fn).lower(latent0).compile()
    stats = compiled.memory_analysis()
    return {
        "argument_bytes": stats.argument_size_in_bytes,
        "output_bytes": stats.output_size_in_bytes,
        "temp_bytes": stats.temp_size_in_bytes,
        "generated_code_bytes": stats.generated_code_size_in_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=float, default=0.06)
    parser.add_argument("--dx", type=float, default=0.001)
    parser.add_argument("--n-steps", type=int, default=400)
    args = parser.parse_args()

    (
        latent0,
        raw_materialized_forward,
        raw_proxy_forward,
        materialized_forward,
        proxy_forward,
    ) = build_problem(
        args.domain,
        args.dx,
        args.n_steps,
    )
    full_forward_stats = forward_memory_stats(raw_materialized_forward, latent0)
    proxy_forward_stats = forward_memory_stats(raw_proxy_forward, latent0)
    full_grad_stats = memory_stats(materialized_forward, latent0)
    proxy_grad_stats = memory_stats(proxy_forward, latent0)

    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / max(denominator, 1)

    payload = {
        "config": {
            "domain": args.domain,
            "dx": args.dx,
            "n_steps": args.n_steps,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
        },
        "forward_materialized": full_forward_stats,
        "forward_proxy_objective": proxy_forward_stats,
        "forward_output_ratio_full_over_proxy": _ratio(
            full_forward_stats["output_bytes"],
            proxy_forward_stats["output_bytes"],
        ),
        "grad_materialized": full_grad_stats,
        "grad_proxy_objective": proxy_grad_stats,
        "grad_temp_ratio_full_over_proxy": _ratio(
            full_grad_stats["temp_bytes"],
            proxy_grad_stats["temp_bytes"],
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
