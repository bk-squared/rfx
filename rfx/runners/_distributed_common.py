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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import EPS_0, MU_0

__all__ = [
    "cpml_coeff_e_vacuum",
    "cpml_coeff_h_vacuum",
    "exchange_component_shmap",
    "shard_stacked",
    "shard_stacked_poles",
    "shard_stacked_psi",
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
# ``tests/locks/test_cpml_axis_params_refactor_bit_identical.py`` make that
# explicit and prevent a CORE-C2-class hidden-assumption helper. A
# per-cell-``eps_r`` CPML coefficient is a separate future effort and is
# deliberately NOT supported by this signature.


def cpml_coeff_e_vacuum(dt: float) -> float:
    """Vacuum CPML E-field update coefficient ``dt / eps_0``.

    Returns ``dt / EPS_0`` using the canonical ``rfx.core.yee.EPS_0``
    (post-2019 SI value 8.8541878128e-12). The pre-refactor distributed
    runners held the literal ``8.854187817e-12`` inline; Stage 3.5b
    consolidated every EPS_0 site onto the canonical constant.

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


# ---------------------------------------------------------------------------
# Device-axis merge + x-slab sharding (host side, eager)
# ---------------------------------------------------------------------------
#
# These three take an array whose leading axis is the DEVICE axis, fold that
# axis into the next one, and ``jax.device_put`` the result onto an
# x-sharding.  They run at setup time, outside any ``jit``/``shard_map``
# trace, so lifting them out of their enclosing closures cannot change a
# jaxpr -- the only thing that moved into the signature is ``shd``, a
# ``NamedSharding`` the caller already had in hand (#1038 leg 1).
#
# The bodies are the pre-move bodies verbatim.  They are three FUNCTIONS and
# not one, on purpose: ``shard_stacked``'s generic ``*rest`` form would
# numerically subsume the other two, but that would be an algebra change on a
# leg whose whole contract is byte-identical motion.  Merging them is a later
# decision with its own evidence, not a side effect of de-duplication.


def shard_stacked(arr, shd):
    """Merge the device axis into x, then shard.

    ``(n_devices, nx_local, ny, nz) -> (n_devices*nx_local, ny, nz)``.

    Extracted verbatim from four byte-identical copies that carried two
    different names: ``distributed_nu.py::shard_debye_coeffs_x_slab._shard_3d``,
    ``distributed_nu.py::shard_lorentz_coeffs_x_slab._shard_3d``,
    ``distributed_nu.py::run_nonuniform_distributed_pec._shard_stacked`` and
    ``distributed_v2.py::run_distributed._shard_stacked``.  All four bodies
    hashed ``ab71430ea8cb``.
    """
    n_dev = arr.shape[0]
    rest = arr.shape[1:]
    return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)


def shard_stacked_poles(arr, shd):
    """Merge the device axis into the pole axis, then shard.

    ``(n_devices, n_poles, nx_local, ny, nz) ->``
    ``(n_devices*n_poles, nx_local, ny, nz)``, so ``P("x")`` hands each
    device ``(n_poles, nx_local, ny, nz)``.

    Extracted verbatim from the four ``_shard_4d`` copies in
    ``distributed_nu.py`` (``shard_debye_coeffs_x_slab``,
    ``shard_debye_state_x_slab``, ``shard_lorentz_coeffs_x_slab``,
    ``shard_lorentz_state_x_slab``), body ``2883a7c93bd2`` on all four.

    NOT the same text as ``distributed_v2.py::_shard_stacked_5d``, which
    computes the same thing through a named intermediate; that copy is
    outside this leg's byte-identical set and is deliberately left alone.
    """
    n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
    return jax.device_put(
        arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
    )


def shard_stacked_psi(arr, shd):
    """Merge the device axis into the CPML-depth axis, then shard.

    ``arr``: ``(n_devices, n_cpml, d1, d2)`` -- re-interpreted as
    ``(n_devices * n_cpml, d1, d2)`` so x-sharding distributes the first
    axis across devices correctly.  Each device owns ``n_cpml`` rows.

    Extracted verbatim from ``distributed_nu.py::shard_cpml_state_x_slab.
    _shard_psi`` and ``distributed_v2.py::_init_cpml_sharded._shard_psi``,
    body ``6ae90bb58fb5`` on both (the copies differed only in comments).
    """
    n_dev, n_c, d1, d2 = arr.shape
    merged = arr.reshape(n_dev * n_c, d1, d2)
    return jax.device_put(merged, shd)
