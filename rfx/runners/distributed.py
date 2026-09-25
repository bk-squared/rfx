"""Retired: the ``jax.pmap`` multi-device runner (#1296).

``run_distributed`` here was the first distributed FDTD runner. It was not the
production lane: ``Simulation.run(devices=[...])`` dispatches to
``rfx.runners.distributed_v2.run_distributed`` (``shard_map``) for two or more
devices, on uniform and non-uniform grids, and a one-device run takes the
ordinary single-device lane. The pmap runner survived only because v2 imported
helpers from this module and handed a direct one-device call to it. It dropped
the declared-PEC cell mask and so refused any PEC volume (#1055), and it
refused an ``nx`` not divisible by the device count, which v2 pads.

Its helpers now live in ``rfx/runners/_distributed_common.py``, where v2 and
``distributed_nu`` import them. This module stays for one release so that an
old import path still resolves: it re-exports every helper name it used to
define or re-export, and its ``run_distributed`` raises with the replacement
named. The pmap runner itself is at commit ``f7b3270d`` for anyone who needs to
reproduce a record that names it.
"""

from __future__ import annotations

from rfx.runners._distributed_common import (  # noqa: F401 -- re-exported
    cpml_coeff_e_vacuum,
    cpml_coeff_h_vacuum,
    gather_array_x,
    split_array_x,
    split_poles_x,
    zeros_psi_stacked,
    _split_state,
    _split_materials,
    _split_debye_coeffs,
    _split_debye_state,
    _split_lorentz_coeffs,
    _split_lorentz_state,
    _update_h_local,
    _update_e_local,
    _update_e_debye_local,
    _update_e_lorentz_local,
    _update_e_local_with_dispersion,
    _init_cpml_distributed,
    _apply_cpml_e_distributed,
    _apply_cpml_h_distributed,
)


def run_distributed(*args, **kwargs):
    """Removed (#1296). Raises ``RuntimeError`` naming the replacement."""
    raise RuntimeError(
        "rfx.runners.distributed.run_distributed, the jax.pmap multi-device "
        "runner, was removed in #1296. Use Simulation.run(n_steps=..., "
        "devices=[...]), which runs the shard_map runner for two or more "
        "devices and the single-device lane for one; or, to call a runner "
        "directly, rfx.runners.distributed_v2.run_distributed(sim, n_steps=..., "
        "devices=[...]), which takes the same arguments. The shard_map runner "
        "pads an nx that the device count does not divide and realizes a "
        "declared PEC volume, both of which this runner refused.")
