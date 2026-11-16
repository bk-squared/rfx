"""The AD-memory estimate reads a scan's carry from either form of scan's parameters.

jax 0.10 numbers a scan's operands (``num_consts``, ``num_carry``); jax 0.11
replaces both with the operand tree ``ft_in`` = (consts, carry, xs). Before
this helper, a Python 3.12 user on jax 0.11 got ``KeyError: 'num_consts'``
from ``estimate_ad_memory``. CI runs jax 0.10, so the jax 0.11 form is built
here from a real scan equation with the same operand counts.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from rfx.api._ad_carry import scan_carry_invars

# One closed-over constant, a carry of two field-like arrays of distinct
# shapes, and one scanned input: consts, carry and xs are all non-empty.
CONST = jnp.arange(5.0)
CARRY_SHAPES = [(4, 3, 2), (3, 2, 4)]


def _scan_eqn():
    def loop(ex, hy, drive):
        def body(carry, x):
            e, h = carry
            return (e + x * CONST.sum(), h - x), x

        return jax.lax.scan(body, (ex, hy), drive)

    traced = jax.make_jaxpr(loop)(jnp.zeros(CARRY_SHAPES[0]), jnp.zeros(CARRY_SHAPES[1]), jnp.ones(7))
    (eqn,) = [e for e in traced.jaxpr.eqns if e.primitive.name == "scan"]
    return eqn


class _OperandTree:
    """jax 0.11 ``ft_in``: ``unpack()`` gives the (consts, carry, xs) groups."""

    def __init__(self, *counts):
        self.counts = counts

    def unpack(self):
        return tuple([None] * n for n in self.counts)


def _shapes(invars):
    return [tuple(v.aval.shape) for v in invars]


def test_counted_form_reads_the_carry():
    eqn = _scan_eqn()
    if "num_consts" not in eqn.params:
        pytest.skip("this jax gives scan operands as a tree; the tree form is tested below")
    assert _shapes(scan_carry_invars(eqn)) == CARRY_SHAPES


def test_tree_form_reads_the_carry():
    eqn = _scan_eqn()
    n_xs = 1
    n_consts = len(eqn.invars) - len(CARRY_SHAPES) - n_xs
    assert n_consts >= 1
    params = {k: v for k, v in eqn.params.items() if k not in ("num_consts", "num_carry", "ft_in", "ft_out")}
    params["ft_in"] = _OperandTree(n_consts, len(CARRY_SHAPES), n_xs)
    tree_eqn = SimpleNamespace(params=params, invars=eqn.invars)
    assert _shapes(scan_carry_invars(tree_eqn)) == CARRY_SHAPES
