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
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)
