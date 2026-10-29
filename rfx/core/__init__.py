"""FDTD core: Yee cell state and coefficient builders."""

from rfx.core.yee import (
    FDTDState, init_state, UpdateCoeffs, precompute_coeffs, update_he_fast,  # noqa: F401
)


# Moved to the stepping layer; the old import path resolves lazily (a module-level import would be circular).
_LAZY_EXPORTS = {
    "update_e": "rfx.stepping.yee",
    "update_h": "rfx.stepping.yee",
    "update_h_fast": "rfx.stepping.yee",
    "update_e_fast": "rfx.stepping.yee",
}


def __getattr__(name):
    from importlib import import_module
    import warnings

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    target = _LAZY_EXPORTS[name]
    warnings.warn(
        f"{__name__}.{name} moved to {target}.{name}; import it from there. "
        "The old path will be removed in rfx 3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(import_module(target), name)
