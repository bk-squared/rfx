"""Shared resolution for deprecated module attributes that moved elsewhere."""

from importlib import import_module
import warnings


def resolve_moved(module_name, table, name):
    """Resolve a moved name, warning at the caller of the module's __getattr__."""
    if name not in table:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    target = table[name]
    warnings.warn(
        f"{module_name}.{name} moved to {target}.{name}; import it from there. "
        "The old path will be removed in rfx 3.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(import_module(target), name)
