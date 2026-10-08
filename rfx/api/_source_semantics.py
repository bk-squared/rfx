"""Single home for soft-source amplitude semantics (issue #571, option 4).

Two named, boundary- and mesh-INDEPENDENT amplitude kinds:

``'current'``
    The waveform is a current moment I(t) in A·m; realized as
    ``E += Cb * I / dV`` on every path and boundary
    (Yee: ``Cb = (dt/eps) / (1 + sigma*dt/(2*eps))``; ADI refuses
    ``'current'`` and requires explicit ``'field'``;
    ``dV`` = local cell volume). This is the non-uniform path's native convention
    (Meep-style, resolution-independent injected power) and the declaration default.

``'field'``
    The waveform is a raw E-field increment per step; realized as
    ``E += w(t)`` on every path and boundary. This is the uniform-PEC
    path's native convention.

Declarations resolve ``None`` to ``current``. Low-level helpers retain their
native ``None`` contracts for internal callers:

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

Product tables are built by ``rfx.model.source_coefficients.source_table``.
It returns field samples directly, without coefficient cancellation, and
preserves the historical current product order for each low-level native.
``source_amplitude_scale`` remains the scalar conversion API for external
callers; it is not used to construct product field tables.

2D-grid convention: a 2D grid is treated as ONE CELL DEEP, so
``dV = dx * dy * dz_one_cell`` with the missing axes duck-typed to
``dx`` (``getattr(grid, 'dy', dx)`` pattern, repo engineering
principle 3) — i.e. ``dV = dx**3`` on a cubic 2D grid, not the
per-unit-length ``dx**2``.

Non-uniform meshes (issue #672): ``dV`` is the E node's CONTROL VOLUME,
not the product of three per-cell widths. An ``E_a`` component is an edge
along its own axis and sits on a node on the other two, so

    dV = d_a[idx_a] * dual_b[idx_b] * dual_c[idx_c],
    dual[k] = (d[k-1] + d[k]) / 2   (``rfx.nonuniform.e_node_dual_spacings``)

with ``(a, b, c)`` the component's own axis and the two transverse ones.
On a uniform profile ``dual == d`` bit-exactly, so the sentence above is
unchanged there.
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
# through low-level helper calls with amplitude_kind=None.
_NATIVE_REALIZES = {"raw": "field", "cb": None, "cb_over_dv": "current"}


def validate_amplitude_kind(kind) -> None:
    """Raise ValueError unless ``kind`` is 'field', 'current' or None."""
    if kind is not None and kind not in SOURCE_AMPLITUDE_KINDS:
        raise ValueError(
            f"amplitude_kind must be 'field', 'current' or None, got {kind!r}")


def resolve_amplitude_kind(kind) -> str:
    """Validate a soft-source declaration and resolve its default to current moments."""
    validate_amplitude_kind(kind)
    return "current" if kind is None else kind


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
    """Description of the declaration default, independent of mesh/boundary."""
    return "'current' (E += Cb*I/dV, I is a current moment in A·m) on every path"


def guard_float16_source_increment(waveform, field_dtype, amplitude_kind=None):
    """Refuse an oversized, fully scaled soft-source increment before stepping.

    ``waveform`` already includes the native drive coefficient and kind scaling.
    A traced build uses a host check whose returned token is a data
    dependency of the source samples, so stepping cannot precede the check.
    """
    import numpy as np

    if field_dtype is None or np.dtype(field_dtype) != np.dtype(np.float16):
        return waveform

    import jax
    import jax.numpy as jnp

    # Refuse only an increment that overflows float16 on its own, in one
    # step: an explicit 'current' source with a one-step increment of ~4e4
    # ran finite before this guard existed and must still run (#1442).
    limit = float(np.finfo(np.float16).max)

    def check(samples):
        peak = float(np.max(np.abs(np.asarray(samples)), initial=0))
        if not np.isfinite(peak) or peak > limit:
            raise ValueError(
                f"float16 soft-source one-step increment {peak:g} exceeds "
                f"the float16 maximum ({limit:g}); scale the waveform"
                + ("." if amplitude_kind == "field" else
                   " or declare amplitude_kind='field'."))
        return np.int32(0)

    if isinstance(waveform, jax.core.Tracer):
        from jax.experimental import io_callback
        token = io_callback(check, jax.ShapeDtypeStruct((), jnp.int32),
                            jax.lax.stop_gradient(waveform), ordered=False)
        return waveform + token.astype(waveform.dtype)
    check(waveform)
    return waveform
