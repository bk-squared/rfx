"""Single home for soft-source amplitude semantics (issue #571, option 4).

Two named, boundary- and mesh-INDEPENDENT amplitude kinds:

``'current'``
    The waveform is a current I(t) in amperes; realized as
    ``E += Cb * I / dV`` on every path and boundary
    (``Cb = (dt/eps) / (1 + sigma*dt/(2*eps))``; ``dV`` = local cell
    volume). This is the non-uniform path's native convention
    (Meep-style, resolution-independent injected power) and the future
    default.

``'field'``
    The waveform is a raw E-field increment per step; realized as
    ``E += w(t)`` on every path and boundary. This is the uniform-PEC
    path's native convention.

The three LEGACY (``amplitude_kind=None``) contracts map onto these exactly:

====================  ====================  =====================================
path                  legacy native coeff   exact explicit spelling
====================  ====================  =====================================
NU (any boundary)     ``Cb/dV`` (current)   ``amplitude_kind='current'``, waveform
                                            unchanged
uniform + pec         ``1`` (field)         ``amplitude_kind='field'``, waveform
                                            unchanged
uniform + cpml/upml   ``Cb`` (NEITHER —     ``amplitude_kind='current'`` with the
                      third contract)       waveform amplitude multiplied by dV
                                            (``Cb*(w*dV)/dV == Cb*w``; exact up to
                                            one float multiply/divide pair)
====================  ====================  =====================================

The conversion FORMULA lives here and nowhere else. The source-building
helpers (``rfx.simulation.make_source`` — native ``'raw'``,
``rfx.simulation.make_j_source`` — native ``'cb'``,
``rfx.nonuniform.make_current_source`` — native ``'cb_over_dv'``) each
declare their native coefficient and, when :func:`needs_scale` says a
conversion applies, multiply the waveform by
:func:`source_amplitude_scale`. ``kind=None`` or a kind matching the
native convention skips the multiply entirely, keeping legacy output
bit-identical during the deprecation window.

Tracer safety (issue #571 dossier): whether to apply the multiply is
decided by :func:`needs_scale` on the PYTHON-LEVEL ``(kind, native)``
string pair — never by comparing the scale VALUE, which may be a JAX
tracer (``cb``/``dV`` are traced on the differentiable-materials /
mesh-as-design-variable paths and ``scale != 1.0`` would raise
``ConcretizationTypeError`` there).

2D-grid convention: a 2D grid is treated as ONE CELL DEEP, so
``dV = dx * dy * dz_one_cell`` with the missing axes duck-typed to
``dx`` (``getattr(grid, 'dy', dx)`` pattern, repo engineering
principle 3) — i.e. ``dV = dx**3`` on a cubic 2D grid, not the
per-unit-length ``dx**2``.
"""

from __future__ import annotations

SOURCE_AMPLITUDE_KINDS = ("field", "current")

# Native waveform coefficient of each helper, i.e. what multiplies w(t) in
# the E update when the helper is fed the waveform unscaled:
#   'raw'        E += w              (rfx.simulation.make_source)
#   'cb'         E += Cb * w         (rfx.simulation.make_j_source)
#   'cb_over_dv' E += Cb * w / dV    (rfx.nonuniform.make_current_source)
_NATIVES = ("raw", "cb", "cb_over_dv")

# Which named kind each native coefficient already realizes. 'cb' realizes
# NEITHER kind — it is the legacy open-uniform third contract, reachable
# only through amplitude_kind=None during the deprecation window.
_NATIVE_REALIZES = {"raw": "field", "cb": None, "cb_over_dv": "current"}


def validate_amplitude_kind(kind) -> None:
    """Raise ValueError unless ``kind`` is 'field', 'current' or None."""
    if kind is not None and kind not in SOURCE_AMPLITUDE_KINDS:
        raise ValueError(
            f"amplitude_kind must be 'field', 'current' or None, got {kind!r}")


def needs_scale(kind, native) -> bool:
    """Python-level dispatch: does ``kind`` require rescaling ``native``?

    Decides on the ``(kind, native)`` STRING pair only. Callers must gate
    the waveform multiply on this predicate rather than on the scale value,
    which may be a JAX tracer (see module docstring, tracer safety).
    Returns False for ``kind=None`` (legacy: bit-identical no-op) and for a
    kind the native coefficient already realizes.
    """
    if native not in _NATIVES:
        raise ValueError(f"unknown native coefficient {native!r}")
    if kind is None:
        return False
    validate_amplitude_kind(kind)
    return _NATIVE_REALIZES[native] != kind


def source_amplitude_scale(kind, native, *, cb, dV):
    """Scalar ``s`` such that <native helper>(s * w) realizes kind ``kind``.

    Parameters
    ----------
    kind : 'field' | 'current' | None
        Requested amplitude semantics. None = legacy = the helper's native
        convention; returns exactly 1.0.
    native : 'raw' | 'cb' | 'cb_over_dv'
        The calling helper's native waveform coefficient (module table).
    cb : float or jnp scalar
        ``(dt/eps)/(1 + sigma*dt/(2*eps))`` at the source cell — computed
        by the CALLER with its existing (tracer-safe) eps/sigma handling,
        so this module never touches materials and stays jax-agnostic.
    dV : float or jnp scalar
        Local cell volume at the source cell (``dx*dy*dz``; one cell deep
        on 2D grids — module docstring).

    Returns
    -------
    Exactly ``1.0`` when :func:`needs_scale` is False (bit-identity
    guarantee for the deprecation window); otherwise the target/native
    coefficient ratio: ``field<-cb: 1/Cb``, ``field<-cb_over_dv: dV/Cb``,
    ``current<-raw: Cb/dV``, ``current<-cb: 1/dV``.
    """
    if not needs_scale(kind, native):
        return 1.0
    if kind == "field":
        # native 'cb': 1/Cb;  native 'cb_over_dv': dV/Cb
        return (1.0 / cb) if native == "cb" else (dV / cb)
    # kind == 'current': native 'raw': Cb/dV;  native 'cb': 1/dV
    return (cb / dV) if native == "raw" else (1.0 / dV)


def legacy_kind_description(is_nonuniform: bool, boundary: str) -> str:
    """One-line concrete statement of what ``amplitude_kind=None`` means for
    THIS simulation's ``run()`` path — used verbatim in the ``add_source``
    DeprecationWarning. (The ``forward()`` uniform route always uses the
    Cb-normalized helper regardless of boundary — pre-existing route
    difference, documented at ``add_source``.)"""
    if is_nonuniform:
        return ("a CURRENT in amperes (E += Cb*I/dV; identical to "
                "amplitude_kind='current')")
    if boundary in ("cpml", "upml"):
        return ("a Cb-normalized field add (E += Cb*w; NOT one of the two "
                "named kinds — to keep today's numbers, multiply your "
                "waveform amplitude by the cell volume dV and pass "
                "amplitude_kind='current')")
    return ("a raw field increment (E += w; identical to "
            "amplitude_kind='field')")
