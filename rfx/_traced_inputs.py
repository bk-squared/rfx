"""Refuse traced static declarations before host-side decisions consume them."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from jax import tree_util

from rfx._traced_input_table import ROWS, VALUE_TYPES
from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer


class TracedStaticInput(TracedStructureRefusal):
    """A traced input selects static structure or a host run control."""


_REASONS = {
    "S1": "controls a count, choice or host threshold",
    "S2": "selects cells without a nominal twin",
    "S3": "selects a node index",
}
_REMEDIES = {
    "S1": "to vary it, call once per value",
    "S2": "a position selects cells and has no derivative here; differentiate geometry through a design region or eps_override",
    "S3": "the index selects a node; its gradient is zero by construction",
}


def _children(value, value_rows=None):
    if isinstance(value, dict):
        return tuple(value.keys()) + tuple(value.values())
    if isinstance(value, (tuple, list)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(getattr(value, field.name) for field in fields(value))
    if value_rows and type(value) in value_rows:
        return tuple(getattr(value, name) for name in value_rows[type(value)]
                     if hasattr(value, name))
    # Flatten only one level: a registered pytree cannot bypass the depth cap.
    root = True

    def only_root(item):
        nonlocal root
        if root:
            root = False
            return False
        return True

    leaves, _ = tree_util.tree_flatten(value, is_leaf=only_root)
    return () if len(leaves) == 1 and leaves[0] is value else leaves


def walk(value, depth=4, value_rows=None):
    """Visit the root and up to four container edges, without coercing leaves."""
    yield value
    if depth > 0 and not is_tracer(value):
        for child in _children(value, value_rows):
            yield from walk(child, depth - 1, value_rows)


def _has_tracer(value, depth=4, value_rows=None):
    return any(is_tracer(item) for item in walk(value, depth, value_rows))


def _refuse(owner, parameter, evidence):
    rule = evidence.split(":", 1)[0]
    raise TracedStaticInput(
        f"{owner}({parameter}=...) received a traced value "
        "(jax.jit, jax.vmap, jax.grad or jax.jvp). "
        f"{parameter} {_REASONS[rule]}; pass a Python number. {_REMEDIES[rule]}"
    )


# Data is indexed once; no callable is replaced and no input is retained.
_BY_OWNER = {}
for _owner, _parameter, _state, _evidence in ROWS:
    _BY_OWNER.setdefault(_owner, {})[_parameter] = (_state, _evidence)


def _value_rows(value):
    cls = type(value)
    if VALUE_TYPES.get(cls.__name__) == cls.__module__:
        return {cls: _BY_OWNER.get(cls.__name__, {})}
    return {}


def _check_value_fields(value, owner, parameter, depth=4):
    if is_tracer(value):
        return
    rows = _value_rows(value)
    for field, (state, evidence) in rows.get(type(value), {}).items():
        if state == "static" and depth > 0 and _has_tracer(getattr(value, field, None), depth - 1):
            _refuse(owner, f"{parameter}.{field}", evidence)
    if depth > 0:
        for child in _children(value, rows):
            _check_value_fields(child, owner, parameter, depth - 1)


def admit_traced_inputs(owner, arguments):
    """Refuse static arguments before host decisions, then return to the caller.

    The private materials entry inherits only S1 run-control rows from run and
    forward, since other names can hold resolved specs.
    Ringdown post-init supplies its fields through the instance in locals().
    """
    if owner == "Simulation._forward_from_materials":
        rows = {**_BY_OWNER.get("Simulation.run", {}), **_BY_OWNER.get("Simulation.forward", {})}
        rows = {parameter: row for parameter, row in rows.items()
                if row[0] == "static" and row[1].startswith("S1:")}
    else:
        rows = _BY_OWNER.get(owner, {})
    if owner == "RingdownSpec" and "self" in arguments:
        arguments = {field.name: getattr(arguments["self"], field.name) for field in fields(arguments["self"])}
    for parameter, value in arguments.items():
        if parameter in {"self", "cls"}:
            continue
        state, evidence = rows.get(parameter, rows.get("*", ("undecided", "")))
        if state == "static" and _has_tracer(value):
            _refuse(owner, parameter, evidence)
        if rows:
            _check_value_fields(value, owner, parameter)
    if owner == "Simulation.run":
        from rfx.geometry.conformal import validate_conformal_min_weight
        validate_conformal_min_weight(arguments["conformal_min_weight"])


def state_counts():
    """Return expanded parameter counts, including the undecided progress count."""
    from rfx._traced_input_table import state_counts as counts

    return counts()
