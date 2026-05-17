"""Shared scaffolding for the distributed FDTD runners.

This module holds the genuinely-common, bit-identical helpers used by
``distributed.py``, ``distributed_v2.py`` and ``distributed_nu.py``.
It is a *mechanical de-duplication* surface only — no algebra change.

Scope rationale (Stage 1.5a):

* ``distributed_v2.py`` already imports the CPML loop body from
  ``distributed.py`` (``_apply_cpml_{e,h}_distributed``), so the
  uniform-CPML scaffolding is NOT triplicated and is not re-extracted
  here.
* ``distributed_nu.py`` carries the *non-uniform* CPML variant
  (axis-aware per-face spacing). That is a genuinely different kernel
  and is intentionally NOT merged.
* What IS byte-for-byte duplicated, and is extracted here:
  - the vacuum CPML field-update coefficients, and
  - the ``shard_map`` ghost-exchange inner body (duplicated verbatim
    between ``distributed_v2.py`` and ``distributed_nu.py``).
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import EPS_0, MU_0

__all__ = [
    "cpml_coeff_e_vacuum",
    "cpml_coeff_h_vacuum",
    "exchange_component_shmap",
]


# ---------------------------------------------------------------------------
# Vacuum CPML field-update coefficients
# ---------------------------------------------------------------------------
#
# Architect residual (a): the vacuum assumption is encoded in the NAME.
# The distributed CPML correction is applied where ``eps_r``/``mu_r`` are
# NOT in scope (``_apply_cpml_*_distributed`` and ``_apply_cpml_*_local_nu``
# receive only field arrays + profile coefficients). The vacuum form is
# therefore structural; these helpers + the guard test in
# ``tests/test_cpml_axis_params_refactor_bit_identical.py`` make that
# explicit and prevent a CORE-C2-class hidden-assumption helper. A
# per-cell-``eps_r`` CPML coefficient is a separate future effort and is
# deliberately NOT supported by this signature.


def cpml_coeff_e_vacuum(dt: float) -> float:
    """Vacuum CPML E-field update coefficient ``dt / eps_0``.

    Bit-identical to the pre-refactor inline literal
    ``dt / 8.854187817e-12`` (``EPS_0`` is exactly that value).

    Takes ONLY ``dt`` — no ``eps_r``/``materials`` argument. The vacuum
    assumption is intentional and load-bearing: see the module docstring.
    """
    return dt / EPS_0


def cpml_coeff_h_vacuum(dt: float) -> float:
    """Vacuum CPML H-field update coefficient ``dt / mu_0``.

    Returns ``dt / MU_0`` using the canonical ``rfx.core.yee.MU_0``
    (post-2019 SI value 1.25663706212e-6). The pre-refactor distributed
    runners held the literal ``1.2566370614e-6`` inline; Stage 3.5b
    consolidated every MU_0 site onto the canonical post-2019 constant.

    Takes ONLY ``dt`` — no ``mu_r``/``materials`` argument. The vacuum
    assumption is intentional and load-bearing: see the module docstring.
    """
    return dt / MU_0


# ---------------------------------------------------------------------------
# shard_map ghost-cell exchange (single field component)
# ---------------------------------------------------------------------------


def exchange_component_shmap(field, mesh, n_devices):
    """Exchange ghost cells for one field component using ``shard_map``.

    Extracted verbatim from ``distributed_v2.py::_exchange_component_shmap``
    and ``distributed_nu.py::_exchange_component_nu_shmap`` — those two
    bodies were byte-for-byte identical.

    ``field`` has global shape ``(nx_with_ghost*n_devices, ny, nz)`` when
    viewed from outside ``shard_map``; inside each shard sees
    ``(nx_local_with_ghost, ny, nz)``.

    Convention (matches the pmap version in ``distributed.py``):
    - ``field[0]``  = left ghost  <- left neighbour's rightmost real cell
    - ``field[-1]`` = right ghost <- right neighbour's leftmost real cell
    - real cells: ``field[1:-1]``
    """

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
        check_rep=False,
    )
    def _exchange(f):
        right_boundary = f[-2:-1, :, :]   # last real cell -> right neighbour's left ghost
        left_boundary = f[1:2, :, :]      # first real cell -> left neighbour's right ghost

        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_ghost_recv = lax.ppermute(right_boundary, "x", perm=perm_right)

        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_ghost_recv = lax.ppermute(left_boundary, "x", perm=perm_left)

        device_idx = lax.axis_index("x")

        left_ghost_val = jnp.where(device_idx > 0,
                                   left_ghost_recv,
                                   f[0:1, :, :])
        right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                    right_ghost_recv,
                                    f[-1:, :, :])

        f = f.at[0:1, :, :].set(left_ghost_val)
        f = f.at[-1:, :, :].set(right_ghost_val)
        return f

    return _exchange(field)
