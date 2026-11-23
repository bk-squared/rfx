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
with the field on the element's edge:

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
R and C are folded into the edge's sigma and eps, which the Yee E update
already treats time-centred (sigma*(E^{n+1}+E^n)/2).  The inductor is
solved together with the edge field, time-centred the same way.
At the edge (d = edge length, A = dual face, D0 = eps/dt + sigma/2
of the edge = 1/Cb, e_std = Ca*E^n + Cb*curl H):

    Ampere:    D0*(E^{n+1} - e_std) = -I_avg/A,  I_avg = (I^{n+1} + I^n)/2
    inductor:  L*(I^{n+1} - I^n)/dt = d*(E^{n+1} + E^n)/2

Eliminating I^{n+1}, with gamma = dt*d/(L*A):

    E^{n+1} = e_std - (I^n/A + (gamma/4)*(e_std + E^n)) / (D0 + gamma/4)
    I^{n+1} = I^n + (dt*d/(2L)) * (E^{n+1} + E^n)

The element's power d*(E^{n+1}+E^n)/2 * I_avg is exactly the change of
L*I^2/2 per step, so the inductor stores and returns energy without loss;
its discrete impedance is j*w*L*tan(w*dt/2)/(w*dt/2), right to second order.

The update it replaces applied I^{n+1} over the step n -> n+1 with
I^{n+1} = I^n + (dt*d/L)*E^{n+1} (backward Euler, half a step late against
the centred field update).  That inductor has the impedance
j*w*L*sinc(w*dt/2)*exp(-j*w*dt): a series resistance ~ w^2*L*dt.  Measured
with a plane wave on a sheet of pure 2 nH elements on 1 mm cells it read
0.61 ohm at 2 GHz and 9.5 ohm at 8 GHz (Q ~ 10), halving with the cell.
"""

from __future__ import annotations

from rfx.errors import RfxRefusal, RfxUnsupported

from rfx._internal._grid_metric import field_index
from dataclasses import dataclass
from typing import NamedTuple
import math

import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import EPS_0, si_value_eps_r_grad


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


class RLCState(NamedTuple):
    """ADE auxiliary state for one lumped RLC element. ``inductor_current`` is in amperes and stores the parallel inductor current, the series current at n+1 with an inductor, or the series current at n+1/2 without an inductor."""
    inductor_current: jnp.ndarray
    capacitor_charge: jnp.ndarray


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

    Dtype note
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
    returns float64, matching what the ADE body produces.

    The differentiable ``forward()`` lane builds metas with
    ``build_rlc_meta``; when a component value is supplied as a tracer
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


class RLCCellMeta(NamedTuple):
    """Grid-resolved metadata for one RLC element.

    Precomputed once and captured by the scan closure.
    ``dual_area`` is the transverse dual-face area through which the element current enters Ampere's law."""
    i: int
    j: int
    k: int
    component: str
    has_inductor: bool
    has_capacitor: bool
    gamma: float
    D0: float
    dx: float
    dt_dx_over_L: float
    dt_over_C_dx: float
    dual_area: float
    R: float
    dt: float
    is_series: bool


def _series_needs_ade(spec: LumpedRLCSpec) -> bool:
    """Return True if this series element requires the ADE path.

    The series ADE is needed when multiple components share a current
    (e.g., R+C, R+L, L+C, R+L+C).  A standalone component (pure R,
    pure C, or pure L) can be handled by material folding / the
    standard inductor ADE without the full series current tracker.
    """
    n_components = int(spec.R > 0) + int(spec.L > 0) + int(spec.C > 0)
    return n_components >= 2


def edge_update_denominator(materials, cell, component, dt,
                            periodic=(False, False, False), *,
                            as_float=False, grid=None):
    """``D0 = eps/dt + sigma/2`` of the E update at one element's edge.

    A lumped element's current enters Ampere's law at its edge through the
    SAME coefficient the Yee update multiplies the curl by, ``Cb = 1/D0``, so
    the element's ``D0`` has to be built from the edge's own ``eps`` and
    ``sigma``: :func:`rfx.model.materials.e_update_material_at` at the
    element's cell and component -- the stored update operands including
    the lumped stamps on this edge (this cell, this component),
    the values
    ``update_e`` turns into ``Cb``. Reading ``materials.eps_r[i, j, k]`` /
    ``sigma[i, j, k]`` instead is the single-cell value, which differs from
    what the update uses wherever the four cells around the edge are not one
    material: on an eps 1|4 surface the edge is 2.5 and the cell 4, and a
    series 100 ohm element then realized 160 ohm, a parallel 2 nH 3.2 nH. Where the four cells are one material the two
    reads are
    the same floats, so vacuum and uniform-dielectric fixtures keep their
    bytes.

    Used for series AND parallel elements. ``periodic`` must be the run's own
    flags (they decide the neighbour of a cell at index 0). ``as_float``
    reproduces the concrete builder's Python-float arithmetic; without it the
    arithmetic stays in the arrays' dtype, so a traced material keeps its
    gradient.

    With an attached tensor the edge's operand is the one the step reads
    (:func:`rfx.model.materials.tensor_operand_at`): the diagonal, the
    UPML form (float32 diagonal, cell-owned conductivity), or the Stage-2
    inverse ``inv``, for which ``D0 = EPS_0/(dt*inv) + sigma/2`` is formed
    in the inverse form, never from a re-inverted permittivity. A frozen
    edge (``inv = 0``) has no finite ``D0``; an element there is refused
    before the first step, and its host value is ``inf`` until then.

    Under UPML (``components.source_upml``) the step's E coefficient is
    formed from cell-owned operands, never the edge mean, and ``D0`` reads
    them through :func:`rfx.model.materials.upml_operands_at`, the reader
    the drive coefficient uses. With no tensor ``D0 = 1/Cb``, ``Cb`` formed
    in UPML's float32 arithmetic (:func:`rfx.model.materials.upml_update_cb`),
    so the element and the step agree to the rounding of one division; on an
    eps 1|4 corner edge the edge mean is 1.75 and the owner cell 4.0, so
    reading the mean scaled the element's D0 by 1.75/4.0. With a tensor the
    float32 diagonal and the cell-owned conductivity enter
    ``eps/dt + sigma/2`` as before. ``D0 = 1/Cb`` holds on interior edges;
    an element on an absorber pad edge is refused before it is built
    (:func:`refuse_rlc_in_absorber_pad`).
    """
    from rfx.model.materials import (
        e_update_material_at, tensor_operand_at, upml_operands_at, upml_update_cb)
    operand = tensor_operand_at(materials, cell, component)
    if getattr(getattr(materials, "components", None), "source_upml", False):
        eps_r, sigma = upml_operands_at(materials, cell, component)
        if operand is None:
            cb = upml_update_cb(eps_r, sigma, dt)
            if as_float and not is_tracer(cb):
                return 1.0 / float(cb)
            return 1.0 / cb
        if as_float and not (is_tracer(eps_r) or is_tracer(sigma)):
            return _denominator_si(float(eps_r), float(sigma), dt)
        return si_value_eps_r_grad(_denominator_si, _denominator_eps_r,
                                   eps_r, sigma, dt)
    if operand is not None and operand[0] == "inv":
        axis = {"ex": 0, "ey": 1, "ez": 2}[str(component).lower()]
        inv = operand[1]
        sigma = materials.components.sigma_update[axis][tuple(cell)]
        if as_float and not (is_tracer(inv) or is_tracer(sigma)):
            inv = float(inv)
            return math.inf if inv == 0.0 else _denominator_inv_si(inv, float(sigma), dt)
        return si_value_eps_r_grad(_denominator_inv_si, _denominator_inv_eps_r,
                                   inv, sigma, dt)
    eps_r, sigma = e_update_material_at(materials, cell, component, periodic, grid=grid)
    if as_float and not (is_tracer(eps_r) or is_tracer(sigma)):
        return _denominator_si(float(eps_r), float(sigma), dt)
    return si_value_eps_r_grad(_denominator_si, _denominator_eps_r,
                               eps_r, sigma, dt)


def _denominator_si(eps_r, sigma, dt):
    return eps_r * EPS_0 / dt + sigma / 2.0


def _denominator_eps_r(eps_r, sigma, dt):
    return eps_r * (EPS_0 / dt) + sigma / 2.0


def _denominator_inv_si(inv, sigma, dt):
    return EPS_0 / (dt * inv) + sigma / 2.0


def _denominator_inv_eps_r(inv, sigma, dt):
    return (EPS_0 / dt) / inv + sigma / 2.0


def _resolve_position_to_index(grid, position):
    """Resolve a physical (x, y, z) to grid indices.

    Duck-types over uniform ``Grid`` (method) and ``NonUniformGrid``
    (module-level helper in ``rfx.nonuniform``).
    """
    if hasattr(grid, "position_to_index"):
        return grid.position_to_index(position)
    from rfx.nonuniform import position_to_index as _nu_p2i
    return _nu_p2i(grid, position)


def _rlc_inputs_traced(materials, *overrides):
    import jax
    return any(is_tracer(value) for value in
               (*overrides, *jax.tree_util.tree_leaves(materials))
               if value is not None)


def _resolve_value(static_val, override_val, *, traced):
    """Keep static values; coerce concrete overrides to host floats."""
    if override_val is None:
        return static_val
    return override_val if traced else float(override_val)


def setup_rlc_materials(grid, spec: LumpedRLCSpec, materials, *,
                        r_val=None, c_val=None):
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
    (measured on a uniform 1.0/0.5/0.25 mm cell, a nominal 1 pF
    realized 0.125 pF as ``ex`` and 8.0 pF as ``ez``).

    For **series** topology with multiple components (R+C, R+L+C, etc.):
    - R and C are handled by the series ADE (shared current), so they
      are NOT folded into material arrays here.

    For **series** topology with a single component (pure R or pure C):
    - The component is folded into material arrays as in the parallel
      case, since a single component does not need current sharing.

    Topology uses the static spec; optional R/C overrides supply numeric
    values. Concrete inputs use host arithmetic and traced inputs retain
    array arithmetic. Returns updated MaterialArrays.
    """
    idx = field_index(grid, spec.position, spec.component)
    i, j, k = idx

    if spec.topology == "series" and _series_needs_ade(spec):
        return materials

    from rfx.sources.sources import (
        port_sigma as _port_sigma,
        port_d_parallel as _d_par,
        port_dual_transverse as _dual_perp,
    )
    from rfx.sources.sources import (
        stamp_lumped_sigma as _stamp_sigma, stamp_lumped_eps as _stamp_eps)
    traced = _rlc_inputs_traced(materials, r_val, c_val)
    R = _resolve_value(spec.R, r_val, traced=traced)
    C = _resolve_value(spec.C, c_val, traced=traced)
    if spec.R > 0:
        materials = _stamp_sigma(
            materials, (i, j, k),
            _port_sigma(grid, (i, j, k), spec.component, R),
            spec.component)

    if spec.C > 0:
        d_par = _d_par(grid, (i, j, k), spec.component)
        dual_b, dual_c = _dual_perp(grid, (i, j, k), spec.component)
        materials = _stamp_eps(
            materials, (i, j, k),
            C * d_par / (EPS_0 * dual_b * dual_c), spec.component)

    return materials


def refuse_rlc_in_absorber_pad(grid, spec, cell, materials):
    """Refuse an element whose realized edge lies in a UPML absorber pad.

    Under UPML the step's ``Cb`` on such an edge includes the absorber's
    perpendicular conductivity; the element's ``D0`` is the interior
    ``1/Cb`` (:func:`edge_update_denominator`), so the element would solve
    its current against a different update than the one that steps the
    edge. Same predicate as the drives
    (:func:`rfx.model.source_coefficients.edge_in_absorber_pad`).
    """
    if not getattr(getattr(materials, "components", None), "source_upml", False):
        return
    from rfx.model.source_coefficients import edge_in_absorber_pad
    if edge_in_absorber_pad(grid, cell, spec.component):
        raise RfxRefusal(
            f"lumped RLC element at {tuple(spec.position)} ({spec.component}, "
            f"R={spec.R}, L={spec.L}, C={spec.C}, {spec.topology}) realizes edge "
            f"{tuple(int(v) for v in cell)}, which lies inside a UPML absorber pad; "
            "its update denominator does not include the absorber operator. "
            "Place the element inside the physical domain, or enlarge the domain.")


def build_rlc_meta(grid, spec: LumpedRLCSpec, materials, *,
                   r_val=None, l_val=None, c_val=None,
                   periodic=(False, False, False)) -> RLCCellMeta:
    """Build per-element metadata from (modified) materials.

    Must be called AFTER ``setup_rlc_materials()`` and after every other
    fold into the material arrays (ports), with the materials the run's E
    update uses: an element takes ``D0`` from the edge's own update
    coefficient (:func:`edge_update_denominator`), and ``periodic`` is the
    run's periodic flags for that lookup. Topology uses the static spec;
    optional R/L/C overrides supply numeric values. Concrete inputs keep
    host arithmetic and traced inputs retain array arithmetic.

    Under UPML (``materials.components.source_upml``) an element whose
    realized edge lies in an absorber pad is refused
    (:func:`refuse_rlc_in_absorber_pad`).
    """
    from rfx.sources.sources import port_d_parallel as _d_par
    from rfx.sources.sources import port_dual_transverse as _dual_t
    idx = field_index(grid, spec.position, spec.component)
    refuse_rlc_in_absorber_pad(grid, spec, idx, materials)
    i, j, k = idx
    d_par = _d_par(grid, idx, spec.component)
    _b, _c = _dual_t(grid, idx, spec.component)
    dual_area = _b * _c
    dt = grid.dt

    has_inductor = spec.L > 0
    has_capacitor = spec.C > 0
    is_series = spec.topology == "series" and _series_needs_ade(spec)

    traced = _rlc_inputs_traced(materials, r_val, l_val, c_val)
    R = _resolve_value(spec.R, r_val, traced=traced)
    L = _resolve_value(spec.L, l_val, traced=traced)
    C = _resolve_value(spec.C, c_val, traced=traced)

    D0 = edge_update_denominator(
        materials, (i, j, k), spec.component, dt, periodic, as_float=not traced, grid=grid)

    if has_inductor:
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


def _describe_spec(n, spec: LumpedRLCSpec) -> str:
    vals = ", ".join(f"{name}={float(v):g}" for name, v in
                     (("R", spec.R), ("L", spec.L), ("C", spec.C)) if float(v) > 0)
    pos = ", ".join(f"{float(p):g}" for p in spec.position)
    return (f"add_lumped_rlc #{n} ({vals}, topology={spec.topology!r}, "
            f"component={spec.component!r}, position=({pos}))")


def refuse_stacked_solved_elements(specs, metas) -> None:
    """Refuse two lumped elements with their OWN solve on one edge.

    An element with an inductor (parallel topology) or a series element with
    two or more components is solved together with its edge field in one
    implicit step, against ``e_std`` and ``E^n``. Two such elements on the
    same edge and component each solve against a field the other one changes
    in the same step: the pair is not the parallel circuit it declares, and
    the run gains energy (measured in a closed PEC box, float64: a parallel
    4 nH + 1 pF with another 4 nH on its edge x4.1 in 6400 steps; two series
    4 nH + 10 pF x4.0). Folded elements (parallel R and C, and a pure R
    or pure C declared "series") have no solve of their own -- they add
    into the edge's material, which the one solved element reads through
    ``D0`` -- so any number of them may share the edge with one solved
    element.

    ``specs`` are the declarations and ``metas`` the elements built from them
    (same order), i.e. AFTER each position has snapped to its realized edge.
    """
    first = {}
    for n, (spec, meta) in enumerate(zip(specs, metas)):
        if not (meta.is_series or meta.has_inductor):
            continue
        edge = (int(meta.i), int(meta.j), int(meta.k), meta.component)
        if edge in first:
            m = first[edge]
            raise RfxUnsupported(
                f"{_describe_spec(m, specs[m])} and {_describe_spec(n, spec)} "
                f"both land on the {edge[3]} edge of cell {edge[:3]}. Each is "
                "solved together with that edge's field in its own implicit "
                "step, and two such solves on one edge act on a field the other "
                "has already changed: the pair gains energy instead of being the "
                "parallel circuit it declares. Model them as ONE "
                "add_lumped_rlc with the combined value (two parallel inductors "
                "L1*L2/(L1+L2)), or place them on different edges. Folded "
                "elements -- a parallel R or C, or a pure R or pure C declared "
                "series -- may share the edge with one of them; a pure L is "
                "solved, whichever topology it is declared with.")
        first[edge] = n


def _parallel_values(e_std, rlc_state: RLCState, meta: RLCCellMeta,
                     e_prev=None):
    """Parallel topology: the inductor, solved WITH the edge field; R and C
    are folded into the material arrays.

    ``e_std`` is the field at the element edge (the standard Yee update,
    folded R and C included, before the inductor's current has acted);
    ``e_prev`` is ``E^n``, the edge field at the START of this step.

    One implicit, trapezoidal step (derivation in the module
    docstring), with ``gamma = dt*d/(L*A)``::

        E^{n+1} = e_std - (I^n/A + (gamma/4)*(e_std + E^n)) / (D0 + gamma/4)
        I^{n+1} = I^n + (dt*d/(2L)) * (E^{n+1} + E^n)

    so the field is loaded over the step by ``I_avg = (I^{n+1} + I^n)/2``,
    time-centred like the Yee update, and the inductor is lossless.  The
    current becomes a current DENSITY through the dual face it pierces,
    ``A = dual_b * dual_c`` (the E node's Ampere control-volume
    cross-section), never ``d_par**2``, which is that area only on a cubic
    cell.  ``D0`` is the edge's own ``1/Cb`` (:func:`edge_update_denominator`).

    Carries: ``inductor_current`` holds I^{n+1}.  Without an inductor this
    is a no-op (``has_inductor`` is a static Python bool).
    """
    if not meta.has_inductor:
        return e_std, rlc_state
    if e_prev is None:
        raise ValueError(
            "a parallel lumped inductor is solved together with its edge "
            "field and needs E^n, the edge field at the start "
            "of the step: pass e_prev=state.<component>[i, j, k] read BEFORE "
            "the E update")

    i_L = rlc_state.inductor_current

    D0 = meta.D0
    quarter_gamma = 0.25 * meta.gamma

    correction = ((i_L / meta.dual_area + quarter_gamma * (e_std + e_prev))
                  / (D0 + quarter_gamma))
    e_new = e_std - correction
    i_L_new = i_L + 0.5 * meta.dt_dx_over_L * (e_new + e_prev)

    return e_new, RLCState(
        inductor_current=jnp.asarray(i_L_new).astype(i_L.dtype),
        capacitor_charge=rlc_state.capacitor_charge)


def _series_values(e_std, rlc_state: RLCState, meta: RLCCellMeta, e_prev):
    """Series topology: R, L, C share one current, solved WITH the edge field.

    ``e_std`` is the field at the element edge (the standard Yee update,
    ``Ca*E^n + Cb*curl H``, before the element's current has acted);
    ``e_prev`` is ``E^n``, the edge field at the START of this step.

    One implicit, trapezoidal step (derivation in the module
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
    i_old = rlc_state.inductor_current
    q_old = rlc_state.capacitor_charge

    d = meta.dx
    dt = meta.dt
    per_amp = 1.0 / (meta.D0 * meta.dual_area)
    kappa = d * per_amp

    if meta.has_inductor:
        two_l_over_dt = 2.0 * d / meta.dt_dx_over_L
    else:
        two_l_over_dt = 0.0
    if meta.has_capacitor:
        half_dt_over_c = 0.5 * d * meta.dt_over_C_dx
        v_cap = q_old * (meta.dt_over_C_dx * d / dt)
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

    return e_new, RLCState(
        inductor_current=jnp.asarray(i_new).astype(i_old.dtype),
        capacitor_charge=jnp.asarray(q_new).astype(q_old.dtype))


def rlc_element_values(e_std, rlc_state: RLCState, meta: RLCCellMeta,
                       e_prev=None):
    """Compute the corrected edge scalar and the element ADE state.

    Dispatches between series and parallel topology at Python trace
    time (``meta.is_series`` is a static bool), so only the needed
    code path is compiled into the XLA graph.

    Called AFTER the standard ``update_e()`` in the scan body.  ``e_prev`` is
    the element edge's field at the START of the step (``E^n``, read before
    the E update); the series update and the parallel inductor need it, a folded-only parallel element does not.

    Returns (new_edge_value, new_rlc_state); does not write a field.
    """
    if meta.is_series:
        if e_prev is None:
            raise ValueError(
                "a series RLC element is solved together with its edge field"
                " and needs E^n, the edge field at the start of "
                "the step: pass e_prev=state.<component>[i, j, k] read BEFORE "
                "the E update")
        return _series_values(e_std, rlc_state, meta, e_prev)
    return _parallel_values(e_std, rlc_state, meta, e_prev)


def _update_element(state, rlc_state: RLCState, meta: RLCCellMeta,
                       e_prev=None):
    """Update one element; stepping batches these scalar solves by component."""
    field = getattr(state, meta.component)
    cell = (meta.i, meta.j, meta.k)
    value, updated = rlc_element_values(field[cell], rlc_state, meta, e_prev)
    if not (meta.is_series or meta.has_inductor):
        return state, updated
    return state._replace(**{meta.component: field.at[cell].set(
        value.astype(field.dtype))}), updated


def _update_parallel(state, rlc_state: RLCState, meta: RLCCellMeta, e_prev=None):
    """Single parallel-element field update (compatibility entry)."""
    return _update_element(state, rlc_state, meta, e_prev)


def _update_series(state, rlc_state: RLCState, meta: RLCCellMeta, e_prev):
    """Single series-element field update (compatibility entry)."""
    return _update_element(state, rlc_state, meta, e_prev)


def update_rlc_element(state, rlc_state: RLCState, meta: RLCCellMeta,
                       e_prev=None):
    """Update one element's field and ADE state using its declared topology."""
    if meta.is_series:
        return _update_series(state, rlc_state, meta, e_prev)
    return _update_parallel(state, rlc_state, meta, e_prev)
