"""Refuse traced static declarations before host-side decisions consume them."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from jax import tree_util

from rfx._traced_input_table import ROWS, VALUE_TYPES
from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer


class TracedStaticInput(TracedStructureRefusal):
    """A traced input selects static structure or a host run control."""


class NonDifferentiableEntry(TracedStructureRefusal):
    """A traced input cannot enter a calculator without a differentiable route."""


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


def refuse_mixed_traced_entry(sim, arguments):
    def check(name, value):
        if _has_tracer(value):
            raise NonDifferentiableEntry(
                f"compute_mixed_s_matrix is not differentiable; {name} is traced. "
                "Use compute_msl_s_matrix(eps_override=...) for a microstrip-only model, "
                "or forward() for lumped ports."
            )

    for name, value in arguments.items():
        if name != "self":
            check(name, value)
    for name, material in sim._materials.items():
        for field in fields(material):
            check(f"material {name!r} {field.name}", getattr(material, field.name))
    for index, entry in enumerate(sim._geometry, 1):
        shape = entry.shape
        names = (field.name for field in fields(shape)) if is_dataclass(shape) else (
            name for name in _value_rows(shape).get(type(shape), {}) if hasattr(shape, name)
        )
        for name in names:
            check(f"shape {index} ({type(shape).__name__}) {name}", getattr(shape, name))
    for label, declarations in (("thin conductor", sim._thin_conductors), ("pinned sheet", sim._pinned_sheets)):
        for index, declaration in enumerate(declarations, 1):
            for field in fields(declaration):
                check(f"{label} {index} {field.name}", getattr(declaration, field.name))
    for index, port in enumerate((*sim._ports, *sim._msl_ports), 1):
        label = f"port {getattr(port, 'name', str(index))!r}"
        for field in fields(port):
            value = getattr(port, field.name)
            if field.name == "waveform":
                for item in walk(value):
                    names = (part.name for part in fields(item)) if is_dataclass(item) else (
                        name for name in _value_rows(item).get(type(item), {}) if hasattr(item, name)
                    )
                    for name in names:
                        check(f"{label} waveform {name}", getattr(item, name))
            check(f"{label} {field.name}", value)
    for name in ("dx_profile", "dy_profile", "dz_profile"):
        check(name, getattr(sim, f"_{name}"))
    check("dt", sim._dt_pin)


def _static_parts(owner, parameter, value):
    if (owner, parameter) == ("Simulation.forward", "design_box"):
        corner_lo = getattr(value, "corner_lo", None)
        corner_hi = getattr(value, "corner_hi", None)
        if corner_lo is not None and corner_hi is not None:
            return corner_lo, corner_hi
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return value
    return (value,)


def _refuse(owner, parameter, evidence):
    rule = evidence.split(":", 1)[0]
    raise TracedStaticInput(
        f"{owner}({parameter}=...) received a traced value "
        "(jax.jit, jax.vmap, jax.grad or jax.jvp). "
        f"{parameter} {_REASONS[rule]}; pass a Python number. {_REMEDIES[rule]}"
    )


_BY_OWNER: dict[str, dict[str, tuple[str, str]]] = {}
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
    Static S2 rows check only cell-selecting parts when resolved, otherwise the whole value.
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
        if state == "static" and any(_has_tracer(part) for part in _static_parts(owner, parameter, value)):
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
