"""Lumped RLC elements for FDTD via material modification + ADE.

Parallel topology
-----------------
R and C are folded into the cell's sigma and eps_r (unconditionally
stable, same approach as the existing LumpedPort for R).  L requires
an Auxiliary Differential Equation (ADE) updated each timestep.

Series topology  (true series RLC current tracking)
---------------------------------------------------
In series RLC, R, L, and C share a single series current I.
Instead of folding R and C into independent material properties,
we track the series current and capacitor charge via ADE.

For the series path, `setup_rlc_materials` does NOT fold R or C
into material arrays (only parallel topology does that).  The
series ADE handles R, L, and C together, and it is solved TOGETHER
with the field on the element's edge (issue #1163):

    Ampere at the edge (d = edge length, A = dual face, D0 = 1/Cb the
    edge's own E-update denominator, e_std = Ca*E^n + Cb*curl H):
        E^{n+1} = e_std - I_avg / (D0*A)
    element, trapezoidal in time (I_avg = (I^{n+1} + I^n)/2):
        d*(E^{n+1} + E^n)/2 = R*I_avg + L*(I^{n+1} - I^n)/dt
                              + (Q^n + Q^{n+1})/(2C)
        Q^{n+1} = Q^n + dt*I_avg

Eliminating E^{n+1} leaves one linear equation for I_avg:

    I_avg * (R + 2L/dt + dt/(2C) + kappa/2)
        = d*(e_std + E^n)/2 + (2L/dt)*I^n - Q^n/C,   kappa = d/(D0*A)

With L = 0 and C -> inf this is exactly the folded resistor's
semi-implicit sigma update.  The element's power
d*(E^{n+1}+E^n)/2 * I_avg = R*I_avg^2 + [change of L*I^2/2 + Q^2/(2C)]/dt
is passive for R >= 0, so the coupled update is stable for any R.

The update it replaces took the element current from e_std (the field
BEFORE the element's own current acted on it) and subtracted it in full
afterwards.  Measured on a line whose load reflection is a closed form,
that subtracts the edge's own impedance kappa (215 ohm in a vacuum
cubic cell) from R, so below ~215 ohm the element was a negative
resistance and the run diverged.

Parallel topology (inductor ADE only)
--------------------------------------
At the inductor cell, Ampere's law with current density J_L = I_L/dx^2:

    eps*(E^{n+1}-E^n)/dt + sigma*(E^{n+1}+E^n)/2
        = curl_H/dx - I_L^{n+1}/dx^2

Inductor relation (leapfrog):

    I_L^{n+1} = I_L^n + (dt*dx/L) * E^{n+1}

Substituting and solving for E^{n+1}:

    E^{n+1} = [D0*E_std - I_L^n/dx^2] / (D0 + gamma)

where:
    D0    = eps/dt + sigma/2         (standard Yee denominator)
    gamma = dt/(L*dx)                (inductor contribution)
    E_std = Ca*E^n + Cb*curl_H       (standard Yee update result)

The key insight: E_std already incorporates D0 in its coefficients,
so we can write the correction as a post-update rescaling:

    E^{n+1} = (D0 * E_std - I_L^n/dx^2) / (D0 + gamma)

Then update I_L:

    I_L^{n+1} = I_L^n + (dt*dx/L) * E^{n+1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import EPS_0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LumpedRLCSpec:
    """Lumped RLC element specification.

    Parameters
    ----------
    R : float
        Resistance in ohms. 0 = no resistive component.
    L : float
        Inductance in henries. 0 = no inductive component.
    C : float
        Capacitance in farads. 0 = no capacitive component.
    topology : str
        "series" or "parallel".  Series topology tracks a shared
        current through R, L, C via ADE (true series RLC).  Parallel
        topology folds R/C into material properties independently
        and uses an ADE only for L.
    position : (x, y, z) in metres
    component : str
        E-field component ("ex", "ey", or "ez").
    """
    R: float = 0.0
    L: float = 0.0
    C: float = 0.0
    topology: str = "series"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    component: str = "ez"


# ---------------------------------------------------------------------------
# ADE state (used for inductor current and/or capacitor charge tracking)
# ---------------------------------------------------------------------------

class RLCState(NamedTuple):
    """ADE auxiliary state for one lumped RLC element."""
    inductor_current: jnp.ndarray  # amperes: I_L (parallel); the series
                                   # current I^{n+1} (series with L) or
                                   # I^{n+1/2} (series without L)
    capacitor_charge: jnp.ndarray  # Q_C in coulombs


def init_rlc_state(dtype=None) -> RLCState:
    """Create zero-initialised RLC ADE state.

    Parameters
    ----------
    dtype : jnp dtype, optional
        Real dtype of the ADE carry (``inductor_current``,
        ``capacitor_charge``).  ``None`` (the default) follows the ambient
        JAX default float dtype with a ``float32`` floor, which is what the
        ADE update body actually produces — see the dtype note below.  The
        concrete ``run()`` path and the differentiable ``forward()`` lane both
        pass an explicit dtype from ``rlc_carry_dtype`` instead.

    Dtype note (issue #646)
    -----------------------
    This default used to be a hard ``jnp.float32`` pin, which reds every
    caller under ``jax_enable_x64``.  The promoting quantity is NOT the field
    dtype: ``build_rlc_meta`` derives ``D0``/``gamma``/``dt_dx_over_L``/
    ``dt_over_C_dx``/``dt`` from ``grid.dt``, which is a **numpy float64
    scalar**.  numpy scalars are strongly typed in JAX, so under x64 they
    promote the ADE update to float64 even when the fields are float32; with
    x64 off JAX clamps them to float32 and the pin happened to agree.  So the
    body returns the ambient default float dtype, and the carry must be
    allocated at that same dtype for ``lax.scan`` to close.

    ``promote_types(..., float32)`` keeps the float32 floor: this is a
    recursive accumulator, so it must never land in float16 under
    ``precision="mixed"``.
    """
    if dtype is None:
        dtype = jnp.promote_types(jnp.result_type(float), jnp.float32)
    return RLCState(
        inductor_current=jnp.array(0.0, dtype=dtype),
        capacitor_charge=jnp.array(0.0, dtype=dtype),
    )


def rlc_carry_dtype(metas, field_dtype=jnp.float32):
    """Real dtype for the RLC ADE scan carry given the per-element metas.

    The concrete ``run()`` path builds ``RLCCellMeta`` from ``grid.dt``, which
    is a **numpy float64 scalar**, so ``D0``/``gamma``/``dt_dx_over_L``/
    ``dt_over_C_dx`` are ``np.float64`` (they DO carry a ``.dtype``) and this
    returns ``float64``, not ``float32``.  With x64 off JAX clamps that back to
    float32 at allocation, which is why the result is byte-identical to the
    historical ``init_rlc_state()`` float32 pin; with x64 on it correctly
    returns float64, matching what the ADE body produces.  (An earlier version
    of this docstring claimed the metas were plain Python floats and that this
    returned float32 — measured false, issue #646.)

    The differentiable ``forward()`` lane builds metas with
    ``build_rlc_meta_traced``; when a component value is supplied as a tracer
    (via ``forward(rlc_values_override=...)``) the numeric coefficients carry a
    JAX dtype.  Under a scoped ``jax_enable_x64`` that dtype is ``float64``, so
    the ADE state update promotes to ``float64`` — this returns ``float64`` and
    keeps the scan carry consistent.
    """
    dtypes = [jnp.dtype(field_dtype)]
    for m in metas:
        for v in (m.D0, m.gamma, m.dt_dx_over_L, m.dt_over_C_dx, m.R):
            dt = getattr(v, "dtype", None)
            if dt is not None:
                dtypes.append(dt)
    return jnp.result_type(*dtypes)


# ---------------------------------------------------------------------------
# Precomputed per-element metadata
# ---------------------------------------------------------------------------

class RLCCellMeta(NamedTuple):
    """Grid-resolved metadata for one RLC element.

    Precomputed once and captured by the scan closure.
    """
    i: int
    j: int
    k: int
    component: str
    has_inductor: bool
    has_capacitor: bool     # True when C > 0
    gamma: float   # dt * d_par / (L * dual_area) — inductor ADE term (0 if L == 0)
    D0: float      # eps/dt + sigma/2 of the element EDGE's own E update
                   # (four-cell edge average plus lumped stamps, #1210;
                   # #1163) -- 1/Cb there.
    dx: float      # PRIMAL cell size along the component axis (the E edge
                   # the element voltage V = E*d_par is taken over)
    dt_dx_over_L: float  # dt * dx / L — for I_L update (0 if L == 0)
    dt_over_C_dx: float  # dt / (C * dx) — for capacitor ADE (0 if C == 0)
    dual_area: float     # dual_b * dual_c — the area of the dual face the
                         # element current pierces, i.e. the E node's discrete
                         # Ampere control-volume cross-section.  The element
                         # current becomes a current DENSITY through this area,
                         # never through dx**2 (which assumes a cubic cell).
    R: float               # resistance in ohms
    dt: float              # timestep in seconds
    is_series: bool        # True for series topology


def _series_needs_ade(spec: LumpedRLCSpec) -> bool:
    """Return True if this series element requires the ADE path.

    The series ADE is needed when multiple components share a current
    (e.g., R+C, R+L, L+C, R+L+C).  A standalone component (pure R,
    pure C, or pure L) can be handled by material folding / the
    standard inductor ADE without the full series current tracker.
    """
    # Count with explicit int() casts, NOT bool ``+``.  For plain-float specs
    # ``(spec.R > 0)`` is a Python bool and ``bool + bool`` already sums as an
    # integer, so this is byte-identical on the concrete path.  The int() cast
    # also makes the intent unambiguous for the traced lane, which always
    # consults the STATIC (plain-float) spec for topology so ``_series_needs_ade``
    # never sees a JAX bool array (whose ``+`` has OR semantics and miscounts).
    n_components = int(spec.R > 0) + int(spec.L > 0) + int(spec.C > 0)
    return n_components >= 2


def edge_update_denominator(materials, cell, component, dt,
                            periodic=(False, False, False), *,
                            as_float=False):
    """``D0 = eps/dt + sigma/2`` of the E update at one element's edge.

    A lumped element's current enters Ampere's law at its edge through the
    SAME coefficient the Yee update multiplies the curl by, ``Cb = 1/D0``, so
    the element's ``D0`` has to be built from the edge's own ``eps`` and
    ``sigma``: :func:`rfx.core.yee.cell_component_e_materials` at the
    element's cell and component -- the mean of the four incident cells'
    volume material plus the lumped stamps on this edge (this cell, this
    component: #1210, #1236), the values
    ``update_e`` turns into ``Cb``. Reading ``materials.eps_r[i, j, k]`` /
    ``sigma[i, j, k]`` instead is the single-cell value, which differs from
    what the update uses wherever the four cells around the edge are not one
    material: on an eps 1|4 surface the edge is 2.5 and the cell 4, and a
    series 100 ohm element then realized 160 ohm, a parallel 2 nH 3.2 nH
    (#1163 review). Where the four cells are one material the two reads are
    the same floats, so vacuum and uniform-dielectric fixtures keep their
    bytes.

    Used for series AND parallel elements. ``periodic`` must be the run's own
    flags (they decide the neighbour of a cell at index 0). ``as_float``
    reproduces the concrete builder's Python-float arithmetic; without it the
    arithmetic stays in the arrays' dtype, so a traced material keeps its
    gradient.
    """
    from rfx.core.yee import cell_component_e_materials
    eps_r, sigma = cell_component_e_materials(materials, cell, component,
                                              periodic)
    if as_float:
        eps_r, sigma = float(eps_r), float(sigma)
    return eps_r * EPS_0 / dt + sigma / 2.0


def _resolve_position_to_index(grid, position):
    """Resolve a physical (x, y, z) to grid indices.

    Duck-types over uniform ``Grid`` (method) and ``NonUniformGrid``
    (module-level helper in ``rfx.nonuniform``).
    """
    if hasattr(grid, "position_to_index"):
        return grid.position_to_index(position)
    from rfx.nonuniform import position_to_index as _nu_p2i
    return _nu_p2i(grid, position)


def setup_rlc_materials(grid, spec: LumpedRLCSpec, materials):
    """Fold R and C into material arrays at the element cell.

    For **parallel** topology:
    - R: adds ``sigma_R = d_par / (R * dual_b * dual_c)`` to the cell
      conductivity (:func:`rfx.sources.sources.port_sigma`).
    - C: adds ``eps_r_extra = C * d_par / (EPS_0 * dual_b * dual_c)`` to the
      cell permittivity.

    Both realize their lumped value through the E node's dual face, whose
    sides are the DUAL spacings on the two axes transverse to the element
    component — not the primal cell widths, and not one scalar cell size.
    The C fold used to read ``C / (d_par * EPS_0)``, which is
    ``C * d_par / (EPS_0 * d_par**2)``: right only on a CUBIC cell, so it is
    wrong on any anisotropic UNIFORM grid too, not only a graded one
    (issue #691; measured on a uniform 1.0/0.5/0.25 mm cell, a nominal 1 pF
    realized 0.125 pF as ``ex`` and 8.0 pF as ``ez``).

    For **series** topology with multiple components (R+C, R+L+C, etc.):
    - R and C are handled by the series ADE (shared current), so they
      are NOT folded into material arrays here.

    For **series** topology with a single component (pure R or pure C):
    - The component is folded into material arrays as in the parallel
      case, since a single component does not need current sharing.

    Returns updated MaterialArrays.
    """
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx

    # Series topology with multiple components: ADE handles R and C
    if spec.topology == "series" and _series_needs_ade(spec):
        return materials

    # Parallel topology, or series with a single component: fold into material
    # Use axis-aware formula: σ_R = d_parallel / (R · d_perp1 · d_perp2)
    from rfx.sources.sources import (
        port_sigma as _port_sigma,
        port_d_parallel as _d_par,
        port_dual_transverse as _dual_perp,
    )
    # #1210: an R or a C across one edge is a LUMPED stamp, not a cell
    # volume property -- it is excluded from the edge average and added back
    # at its own cell (see rfx.core.yee.MaterialArrays).
    from rfx.sources.sources import (
        stamp_lumped_sigma as _stamp_sigma, stamp_lumped_eps as _stamp_eps)
    if spec.R > 0:
        materials = _stamp_sigma(
            materials, (i, j, k),
            _port_sigma(grid, (i, j, k), spec.component, spec.R),
            spec.component)

    if spec.C > 0:
        d_par = _d_par(grid, (i, j, k), spec.component)
        dual_b, dual_c = _dual_perp(grid, (i, j, k), spec.component)
        materials = _stamp_eps(
            materials, (i, j, k),
            spec.C * d_par / (EPS_0 * dual_b * dual_c), spec.component)

    return materials


def build_rlc_meta(grid, spec: LumpedRLCSpec, materials, *,
                   periodic=(False, False, False)) -> RLCCellMeta:
    """Build per-element metadata from (modified) materials.

    Must be called AFTER ``setup_rlc_materials()`` and after every other
    fold into the material arrays (ports), with the materials the run's E
    update uses: an element takes ``D0`` from the edge's own update
    coefficient (:func:`edge_update_denominator`), and ``periodic`` is the
    run's periodic flags for that lookup.
    """
    from rfx.sources.sources import port_d_parallel as _d_par
    from rfx.sources.sources import port_dual_transverse as _dual_t
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx
    d_par = _d_par(grid, idx, spec.component)
    _b, _c = _dual_t(grid, idx, spec.component)
    dual_area = _b * _c
    dt = grid.dt

    has_inductor = spec.L > 0
    has_capacitor = spec.C > 0
    is_series = spec.topology == "series" and _series_needs_ade(spec)

    # #1163: the edge's own E-update denominator, not the cell's (series
    # and parallel alike).
    D0 = edge_update_denominator(
        materials, (i, j, k), spec.component, dt, periodic, as_float=True)

    if has_inductor:
        # gamma is the implicit self-coupling of the inductor into the E
        # update.  Eliminating I^{n+1} = I^n + (dt*d_par/L)*E^{n+1} from the
        # Ampere balance  D0*(E^{n+1} - E_std) = -I^{n+1}/dual_area  gives
        #     E^{n+1} = (D0*E_std - I^n/dual_area) / (D0 + gamma),
        #     gamma   = dt * d_par / (L * dual_area).
        # The old spelling dt/(L*d_par) is that with dual_area = d_par**2,
        # i.e. a CUBIC cell.  It was self-consistent with the equally cubic
        # I/(dx*dx) below, which is why the pair realized an inductance
        # L * d_par**2 / dual_area instead of L (issue #691 follow-up).
        gamma = dt * d_par / (spec.L * dual_area)
        dt_dx_over_L = dt * d_par / spec.L
    else:
        gamma = 0.0
        dt_dx_over_L = 0.0

    if has_capacitor:
        dt_over_C_dx = dt / (spec.C * d_par)
    else:
        dt_over_C_dx = 0.0

    return RLCCellMeta(
        i=i, j=j, k=k,
        component=spec.component,
        has_inductor=has_inductor,
        has_capacitor=has_capacitor,
        gamma=gamma,
        D0=D0,
        dx=d_par,
        dt_dx_over_L=dt_dx_over_L,
        dt_over_C_dx=dt_over_C_dx,
        dual_area=dual_area,
        R=spec.R,
        dt=dt,
        is_series=is_series,
    )


# ---------------------------------------------------------------------------
# Differentiable (traced) setup — WP 4-E
#
# The concrete ``setup_rlc_materials`` / ``build_rlc_meta`` above are the
# byte-identical ``run()`` path (Python ``float()`` coercions, float64 scalar
# arithmetic).  The two functions below are the parallel TRACED lane used only
# by the differentiable ``Simulation.forward()``.  They:
#   * decide the element TOPOLOGY (which components are present, series-vs-fold)
#     from the STATIC plain-float ``spec`` — so the Python ``if`` gates and
#     ``_series_needs_ade`` never see a JAX bool tracer, and
#   * feed the NUMERIC component VALUES (which may be tracers supplied via
#     ``forward(rlc_values_override=...)``) into the jnp-native coefficients
#     without any ``float()`` coercion, so ``jax.grad`` flows w.r.t. R/L/C.
# This mirrors the coax ``eps_scale`` (PR #261) and waveguide-flux AD (#148/#172)
# dual-path idiom: concrete path frozen, a separate traced path added.
# ---------------------------------------------------------------------------

def _resolve_value(static_val, override_val):
    """Pick the traced override when supplied, else the static spec float."""
    return static_val if override_val is None else override_val


def setup_rlc_materials_traced(grid, spec: LumpedRLCSpec, materials, *,
                               r_val=None, c_val=None):
    """jnp-native counterpart of ``setup_rlc_materials`` for ``forward()``.

    Topology decisions (series-needs-ADE, component presence) use the STATIC
    plain-float ``spec``.  Only the NUMERIC R/C values folded into the material
    arrays may be tracers (``r_val`` / ``c_val``); when ``None`` the static
    spec float is used, so a plain ``forward()`` on a sim with a registered RLC
    element correctly reflects the element (no more silent no-op).
    """
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx

    # Series topology with multiple components: ADE handles R and C (static).
    if spec.topology == "series" and _series_needs_ade(spec):
        return materials

    from rfx.sources.sources import (
        port_sigma as _port_sigma,
        port_d_parallel as _d_par,
        port_dual_transverse as _dual_perp,
    )
    # #1210: same lumped-stamp treatment as the concrete path above.
    from rfx.sources.sources import (
        stamp_lumped_sigma as _stamp_sigma, stamp_lumped_eps as _stamp_eps)
    if spec.R > 0:  # static presence check on the plain-float spec
        R = _resolve_value(spec.R, r_val)
        materials = _stamp_sigma(
            materials, (i, j, k),
            _port_sigma(grid, (i, j, k), spec.component, R), spec.component)

    if spec.C > 0:
        C = _resolve_value(spec.C, c_val)
        d_par = _d_par(grid, (i, j, k), spec.component)
        dual_b, dual_c = _dual_perp(grid, (i, j, k), spec.component)
        # Same dual-face fold as the concrete path (#691). d_par / dual_b /
        # dual_c are plain floats from the grid, so a traced C stays traced.
        materials = _stamp_eps(
            materials, (i, j, k), C * d_par / (EPS_0 * dual_b * dual_c),
            spec.component)

    return materials


def build_rlc_meta_traced(grid, spec: LumpedRLCSpec, materials, *,
                          r_val=None, l_val=None, c_val=None,
                          periodic=(False, False, False)) -> RLCCellMeta:
    """jnp-native counterpart of ``build_rlc_meta`` for ``forward()``.

    Structural fields (``i/j/k``, ``component``, ``has_inductor``,
    ``has_capacitor``, ``is_series``) are STATIC, resolved from the plain-float
    ``spec`` exactly as ``build_rlc_meta`` does — so the Python ``if`` gates
    below dispatch at trace time (no ``TracerBoolConversionError``).  The
    NUMERIC coefficients (``D0``, ``gamma``, ``dt_dx_over_L``,
    ``dt_over_C_dx``, ``R``) are computed jnp-natively with NO ``float()``
    coercion, so a traced ``r_val`` / ``l_val`` / ``c_val`` flows through
    ``jax.grad``.  Must be called AFTER ``setup_rlc_materials_traced``.
    """
    from rfx.sources.sources import port_d_parallel as _d_par
    from rfx.sources.sources import port_dual_transverse as _dual_t
    idx = _resolve_position_to_index(grid, spec.position)
    i, j, k = idx
    d_par = _d_par(grid, idx, spec.component)
    _b, _c = _dual_t(grid, idx, spec.component)
    dual_area = _b * _c
    dt = grid.dt

    # STATIC topology from the plain-float spec.
    has_inductor = spec.L > 0
    has_capacitor = spec.C > 0
    is_series = spec.topology == "series" and _series_needs_ade(spec)

    # No float() coercion: eps/sigma may carry a folded R/C tracer. #1163:
    # the edge's own E-update denominator (same as the concrete twin), for
    # series and parallel elements.
    D0 = edge_update_denominator(
        materials, (i, j, k), spec.component, dt, periodic)

    R = _resolve_value(spec.R, r_val)
    L = _resolve_value(spec.L, l_val)
    C = _resolve_value(spec.C, c_val)

    if has_inductor:
        # Same fold as the concrete twin above; see its comment.
        gamma = dt * d_par / (L * dual_area)
        dt_dx_over_L = dt * d_par / L
    else:
        gamma = 0.0
        dt_dx_over_L = 0.0

    if has_capacitor:
        dt_over_C_dx = dt / (C * d_par)
    else:
        dt_over_C_dx = 0.0

    return RLCCellMeta(
        i=i, j=j, k=k,
        component=spec.component,
        has_inductor=has_inductor,
        has_capacitor=has_capacitor,
        gamma=gamma,
        D0=D0,
        dx=d_par,
        dt_dx_over_L=dt_dx_over_L,
        dt_over_C_dx=dt_over_C_dx,
        dual_area=dual_area,
        R=R,
        dt=dt,
        is_series=is_series,
    )


# ---------------------------------------------------------------------------
# Per-timestep ADE update
# ---------------------------------------------------------------------------

def _update_parallel(state, rlc_state: RLCState, meta: RLCCellMeta):
    """Parallel topology: inductor ADE only; R/C are in material arrays.

    If ``meta.has_inductor`` is False, this is a no-op.
    """
    i, j, k = meta.i, meta.j, meta.k

    e_field = getattr(state, meta.component)
    e_std = e_field[i, j, k]

    i_L = rlc_state.inductor_current
    Q = rlc_state.capacitor_charge

    D0 = meta.D0
    gamma = meta.gamma

    # E^{n+1} = (D0 * E_std - I_L^n / dual_area) / (D0 + gamma)
    #
    # The element current becomes a current DENSITY through the dual face it
    # pierces -- ``dual_b * dual_c``, the E node's Ampere control-volume
    # cross-section -- NOT through ``d_par**2``, which is that area only on a
    # cubic cell.
    A = D0 + gamma
    e_new = jnp.where(
        meta.has_inductor,
        (D0 * e_std - i_L / meta.dual_area) / A,
        e_std,
    )

    # I_L^{n+1} = I_L^n + (dt * dx / L) * E^{n+1}
    i_L_new = jnp.where(
        meta.has_inductor,
        i_L + meta.dt_dx_over_L * e_new,
        i_L,
    )

    field_new = e_field.at[i, j, k].set(e_new)
    state_new = state._replace(**{meta.component: field_new})

    return state_new, RLCState(inductor_current=i_L_new, capacitor_charge=Q)


def _update_series(state, rlc_state: RLCState, meta: RLCCellMeta, e_prev):
    """Series topology: R, L, C share one current, solved WITH the edge field.

    ``state`` holds ``e_std`` at the element edge (the standard Yee update,
    ``Ca*E^n + Cb*curl H``, before the element's current has acted);
    ``e_prev`` is ``E^n``, the edge field at the START of this step.

    One implicit, trapezoidal step (issue #1163; derivation in the module
    docstring).  With ``I_avg = (I^{n+1} + I^n)/2``, ``kappa = d/(D0*A)``::

        I_avg * (R + 2L/dt + dt/(2C) + kappa/2)
            = d*(e_std + E^n)/2 + (2L/dt)*I^n - Q^n/C
        E^{n+1} = e_std - I_avg/(D0*A)
        Q^{n+1} = Q^n + dt*I_avg
        I^{n+1} = 2*I_avg - I^n        (with L; without L the element has
                                         no state current and I_avg is kept)

    ``D0`` is the edge's own ``1/Cb`` (:func:`edge_update_denominator`), so
    the current enters the field through exactly the coefficient the Yee
    update used.  With L = 0 and C -> inf the step is the folded resistor's
    semi-implicit update.

    Carries: ``inductor_current`` holds I^{n+1} with an inductor and I_avg
    (the current at n+1/2) without one; ``capacitor_charge`` holds Q^{n+1}.
    """
    i, j, k = meta.i, meta.j, meta.k

    e_field = getattr(state, meta.component)
    e_std = e_field[i, j, k]

    i_old = rlc_state.inductor_current
    q_old = rlc_state.capacitor_charge

    d = meta.dx
    dt = meta.dt
    # Field change per ampere of element current: Cb / A.
    per_amp = 1.0 / (meta.D0 * meta.dual_area)
    kappa = d * per_amp            # the edge's own impedance d/(D0*A)

    # has_inductor / has_capacitor are static Python bools (the traced lane
    # takes topology from the plain-float spec), so these are trace-time
    # branches and never divide by a zero coefficient.
    if meta.has_inductor:
        two_l_over_dt = 2.0 * d / meta.dt_dx_over_L      # dt_dx_over_L = dt*d/L
    else:
        two_l_over_dt = 0.0
    if meta.has_capacitor:
        half_dt_over_c = 0.5 * d * meta.dt_over_C_dx      # dt_over_C_dx = dt/(C*d)
        v_cap = q_old * (meta.dt_over_C_dx * d / dt)      # Q^n / C
    else:
        half_dt_over_c = 0.0
        v_cap = 0.0

    z_step = meta.R + two_l_over_dt + half_dt_over_c + 0.5 * kappa
    drive = 0.5 * d * (e_std + e_prev) + two_l_over_dt * i_old - v_cap
    i_avg = drive / z_step

    e_new = e_std - per_amp * i_avg

    if meta.has_capacitor:
        q_new = q_old + dt * i_avg
    else:
        q_new = q_old
    if meta.has_inductor:
        i_new = 2.0 * i_avg - i_old
    else:
        i_new = i_avg

    field_new = e_field.at[i, j, k].set(e_new.astype(e_field.dtype))
    state_new = state._replace(**{meta.component: field_new})

    return state_new, RLCState(
        inductor_current=jnp.asarray(i_new).astype(i_old.dtype),
        capacitor_charge=jnp.asarray(q_new).astype(q_old.dtype))


def update_rlc_element(state, rlc_state: RLCState, meta: RLCCellMeta,
                       e_prev=None):
    """Update ADE and correct the E-field at the element cell.

    Dispatches between series and parallel topology at Python trace
    time (``meta.is_series`` is a static bool), so only the needed
    code path is compiled into the XLA graph.

    Called AFTER the standard ``update_e()`` in the scan body.  ``e_prev`` is
    the element edge's field at the START of the step (``E^n``, read before
    the E update); the series update needs it, the parallel one does not.

    Returns (new_fdtd_state, new_rlc_state).
    """
    if meta.is_series:
        if e_prev is None:
            raise ValueError(
                "a series RLC element is solved together with its edge field "
                "(issue #1163) and needs E^n, the edge field at the start of "
                "the step: pass e_prev=state.<component>[i, j, k] read BEFORE "
                "the E update")
        return _update_series(state, rlc_state, meta, e_prev)
    return _update_parallel(state, rlc_state, meta)
