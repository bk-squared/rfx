"""Differentiable observables built on registered DFT-plane probes,
plus a batched forward-mode Jacobian wrapper.

``Simulation.add_dft_plane_probe(...)`` registers a frequency-domain plane
accumulator that ``run()`` and ``forward()`` both populate into a name-keyed
dict on the result (``Result.dft_planes`` / ``ForwardResult.dft_planes``).
That accumulator mechanism already existed and is already JAX-traceable
end-to-end (``rfx/api/_execute.py`` packs it for ``forward()``, and
``rfx/runners/uniform.py`` / ``rfx/runners/nonuniform.py`` pack it for
``run()``); this module is the missing public **accessor + objective**
layer on top of it, in the same "factory returns ``callable(result) ->
...``" style as :mod:`rfx.optimize_objectives`.

- :func:`dft_field` -- pull one or more named DFT planes off a result as
  plain ``jax.numpy`` arrays (a result accessor, not an objective).
- :func:`field_energy` -- ``sum(|field|**2)`` over one or more planes; the
  simplest possible smooth objective.
- :func:`field_softmax` -- a temperature-controlled soft-max over
  space + frequency, for objectives that care about the WORST bin (e.g.
  "minimize peak leaked field anywhere behind a shield") rather than the
  average.
- :func:`jacobian_fwd` -- batch ``jax.jvp`` over a tangent basis to get a
  many-output Jacobian in one pass, for any ``sim_fn(params) -> observable``
  built on top of the three accessors above (or anything else that is
  ``jax``-differentiable). See its own docstring for what it computes and
  the paths it supports.

Both lanes, both entry points: ``result.dft_planes`` is a name-keyed dict on
the uniform AND non-uniform meshes, for both ``run()`` and ``forward()``
(``rfx/runners/uniform.py:859-866``, ``rfx/runners/nonuniform.py:1064-1068``,
``rfx/api/_execute.py:1579-1596``). Everything here is lane-agnostic: it
only reads ``result.dft_planes``, so it works unmodified on any of the four
combinations. The exceptions are the two DISTRIBUTED lanes, which are
fenced fail-loud (see "Fail-loud fences" below) because neither sharded
runner accumulates DFT-plane fields. The per-name value is duck-typed
(``.accumulator`` if present, else the value itself), so a name-keyed dict
of bare arrays (e.g. a vmap-batched sweep result) works too, not just
``DFTPlaneProbe`` objects.

What stays on the AD tape
--------------------------
Each registered plane's accumulator is a single ``(n_freqs, n1, n2)``
complex array (complex64, or complex128 under ``jax_enable_x64``) living in
the ``jax.lax.scan`` carry for the whole run (``rfx/simulation.py:857-858,
1512-1526, 1604-1605``) -- there is no intermediate per-step buffer to
discard. Reverse-mode AD through :func:`dft_field` / :func:`field_energy` /
:func:`field_softmax` therefore differentiates through that one carried
array, not through a stored time series. The window is always the
rectangular (unweighted) DFT window: ``add_dft_plane_probe`` never exposes
``dft_window`` / ``dft_window_alpha`` (unlike ``add_flux_monitor``), so
``DFTPlaneProbe.window`` stays at its ``"rect"`` default on every plane this
module can see. On the uniform lane, ``forward()`` also always carries the
full probe time-series tape alongside the DFT accumulators --
``emit_time_series=False`` is a non-uniform-lane-only option
(``rfx/api/_execute.py`` forward() docstring); a DFT-only uniform-lane
objective still pays for the time-series tape today.

Physical E/H stamps
-------------------
DFT planes use the shared measurement kernel: slot n has E at (n+1)*dt
and H at (n+1/2)*dt on both runners. H spectra already include the
leapfrog half-step. Combine them directly in E x H* or V/I expressions;
applying another half-step correction would rotate the current twice.
Magnitude-only objectives retain their existing convention.

Fail-loud fences
-----------------
Registering any DFT plane probe and then requesting a distributed run
raises ``NotImplementedError`` -- on both ``forward(distributed=True)`` and
``run(devices=[...])`` with ``len(devices) > 1`` -- because neither the
sharded non-uniform forward runner (``rfx.runners.distributed_nu``) nor the
sharded ``run()`` runner (``rfx.runners.distributed_v2``) accumulates DFT
planes at all; before this fence they were silently dropped (CHANGELOG,
back-compat break). This mirrors the pre-existing
``add_flux_monitor()``-on-distributed-NU-forward fence
(``rfx/api/_execute.py``) in both mechanism and wording.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.nn import logsumexp

from rfx.core.jax_utils import is_tracer

__all__ = ["dft_field", "field_energy", "field_softmax", "jacobian_fwd"]


def _select_planes(result, names, caller: str) -> dict:
    """Return ``{name: jnp array}`` for the requested plane name(s).

    Shared by every public function in this module. Raises ``ValueError``
    naming the producing call (``add_dft_plane_probe`` + ``run()``/
    ``forward()``) when the result carries no planes at all, and naming the
    specific missing name(s) when the result has planes but not all of the
    requested ones -- both in the ``optimize_objectives`` error style (name
    the field that's missing and the call that would populate it).

    Duck-types the per-name value: ``DFTPlaneProbe``-like objects (an
    ``.accumulator`` attribute, from ``run()``/``forward()``) and bare
    arrays (e.g. a vmap-batched sweep result whose ``dft_planes`` dict
    carries raw stacked arrays rather than probe objects) are both
    accepted -- ``getattr(value, "accumulator", value)``.
    """
    planes = getattr(result, "dft_planes", None)
    if not planes:
        raise ValueError(
            f"{caller} requires result.dft_planes, but got {planes!r}. "
            "DFT-plane observables need Simulation.add_dft_plane_probe(...) "
            "registered BEFORE run()/forward() -- neither call populates "
            "dft_planes on its own, and the distributed forward/run lanes "
            "reject registered planes outright (see rfx.observables module "
            "docstring, 'Fail-loud fences')."
        )
    name_list = [names] if isinstance(names, str) else list(names)
    if not name_list:
        raise ValueError(f"{caller}: names must be a non-empty name or list of names.")
    missing = [n for n in name_list if n not in planes]
    if missing:
        raise ValueError(
            f"{caller}: plane name(s) {missing} not found in result.dft_planes "
            f"(available: {sorted(planes)}). Check the `name=` argument you "
            "passed (or the auto-generated 'component_axis_index' default) to "
            "the matching add_dft_plane_probe(...) call."
        )
    return {n: jnp.asarray(getattr(planes[n], "accumulator", planes[n])) for n in name_list}


def dft_field(
    names: str | Sequence[str], *, stack: bool = True,
) -> Callable:
    """Result accessor for one or more registered DFT-plane probes.

    Parameters
    ----------
    names : str or sequence of str
        The ``name=`` (or auto-generated default) given to
        ``add_dft_plane_probe(...)``. A single name returns the raw
        ``(n_freqs, n1, n2)`` complex accumulator directly (no dict
        wrapping -- there is nothing to disambiguate). A sequence of names
        returns either a stacked array or a dict, controlled by *stack*.
    stack : bool
        When *names* is a sequence and ``stack=True`` (default), the
        accessor stacks all requested planes into one
        ``(n_names, n_freqs, n1, n2)`` array ALONG A NEW LEADING AXIS --
        but only when every requested plane has the same
        ``(n_freqs, n1, n2)`` shape (planes at different axes/coordinates
        commonly do not, e.g. a yz-plane vs an xz-plane monitor). When the
        shapes differ, stacking is impossible and the accessor RAISES
        ``ValueError`` naming the mismatched shapes and pointing at
        ``stack=False``; it never silently truncates or broadcasts. Pass
        ``stack=False`` explicitly to opt into the per-name
        ``dict[name] -> array`` form instead (also the unconditional return
        form when *names* is a sequence and ``stack=False``, even if the
        shapes happen to match). Mixed-dtype stacking (e.g. a complex64
        plane alongside a complex128 one, possible if planes were
        registered under different ``jax_enable_x64`` states) follows
        ordinary ``jnp.stack`` dtype promotion -- the result takes the
        wider dtype.

    Returns
    -------
    callable(result) -> jnp.ndarray | dict[str, jnp.ndarray]
        JAX-differentiable: the returned array(s) are views/stacks of the
        same accumulator that was carried through the FDTD scan, so
        ``jax.grad`` flows through them unchanged.

    Raises
    ------
    ValueError
        If *result* carries no ``dft_planes`` at all, is missing one of the
        requested *names*, or (list form, ``stack=True``) the requested
        planes' shapes do not match -- naming the fix in each case.
    """
    single = isinstance(names, str)
    name_list = [names] if single else list(names)

    def accessor(result):
        arrays = _select_planes(result, name_list, "dft_field")
        if single:
            return arrays[names]
        if stack:
            shapes = {a.shape for a in arrays.values()}
            if len(shapes) == 1:
                return jnp.stack([arrays[n] for n in name_list], axis=0)
            raise ValueError(
                f"dft_field({name_list!r}): accumulator shapes differ "
                f"({ {n: tuple(a.shape) for n, a in arrays.items()} }), so "
                "they cannot be stacked into one array. Pass stack=False to "
                "get a dict[name] -> array instead."
            )
        return arrays

    return accessor


def field_energy(
    names: str | Sequence[str], *, weights: dict[str, float] | None = None,
) -> Callable:
    """Objective factory: ``sum(|field|**2)`` over one or more DFT planes.

    Sums squared magnitude over every element (space AND frequency) of
    every requested plane. Trivially smooth (a sum of squares is C-infinity
    in the accumulator), so this is a good first objective to sanity-check
    a new DFT-plane setup with before reaching for :func:`field_softmax`.

    Parameters
    ----------
    names : str or sequence of str
        Plane name(s) to sum over -- see :func:`dft_field`. Shapes need not
        match between planes (unlike ``dft_field(..., stack=True)``): each
        plane's energy is summed independently, so mixing e.g. a yz-plane
        and an xz-plane monitor is fine.
    weights : dict[str, float] or None
        Optional per-plane scalar weight (default 1.0 for every plane not
        listed), applied before summing -- e.g. to down-weight a monitor
        plane closer to the source.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    name_list = [names] if isinstance(names, str) else list(names)
    _weights = weights or {}

    def objective(result) -> jnp.ndarray:
        arrays = _select_planes(result, name_list, "field_energy")
        total = jnp.asarray(0.0)
        for n in name_list:
            w = _weights.get(n, 1.0)
            total = total + w * jnp.sum(jnp.abs(arrays[n]) ** 2)
        return total

    return objective


def field_softmax(
    names: str | Sequence[str], *, beta: float = 1.0,
) -> Callable:
    """Objective factory: smooth (soft) max of ``|field|**2`` over space+frequency.

    Uses the standard log-sum-exp soft-max, AUTO-SCALED to the field's own
    magnitude each call: ``beta_eff = beta / stop_gradient(max(vals))``,
    ``softmax = logsumexp(beta_eff * vals) / beta_eff``, computed with
    ``jax.nn.logsumexp`` for numerical stability. As *beta* (equivalently
    ``beta_eff * max(vals)``, which the normalization pins to exactly
    *beta* by construction -- see "Auto-scaling", below) grows this
    converges to the true ``max(|field|**2)``; a smaller *beta* trades a
    looser max approximation for a smoother, better-conditioned gradient
    (useful early in an optimization run, e.g. the shielding case: minimize
    the worst-case leaked field anywhere behind a barrier, over all
    monitored frequencies, rather than the mean leaked field).

    Parameters
    ----------
    names : str or sequence of str
        Plane name(s) to pool over -- see :func:`dft_field`. All requested
        planes' elements (space AND frequency, across every plane) are
        pooled into ONE soft-max, so this composes the register-N-planes
        pattern: register one plane per physical monitor location, then
        soft-max over all of them at once for a single worst-case scalar.
    beta : float
        Soft-max sharpness (default 1.0), now DIMENSIONLESS -- see
        "Auto-scaling" below. Must be positive.

    Auto-scaling
    ------------
    Read this before raising *beta* to make the soft-max "safer".
    Earlier versions of this function computed ``logsumexp(beta*vals)/beta``
    directly against the RAW ``|field|**2`` values, so *beta* had to be
    hand-matched to whatever physical units/magnitude the accumulator
    happened to carry (rfx's own DFT-plane accumulators commonly sit
    around ``|field|**2`` ~1e-10 to 1e-22 depending on
    geometry/normalization). At the then-default ``beta=1.0`` the
    objective sat at ~100.000002% of the design-independent constant
    ``log(count)/beta`` (measured): both the value and its
    gradient were differentiating ROUNDING NOISE in that constant, not
    physics, and it could coincidentally still pass a single
    finite-difference check while failing at a different step size.

    This function now computes ``beta_eff = beta / stop_gradient(max(vals))``
    and uses ``beta_eff`` in place of a raw *beta* everywhere above.
    Because ``vals``' own current max scales out of the product exactly,
    ``beta_eff * max(vals) == beta`` ALWAYS (BEFORE any overflow -- see
    the next paragraph), regardless of the field's physical units -- the
    ``log(count)/beta`` swallow above cannot recur at any REPRESENTABLE
    (nonzero) field magnitude, at any *beta*, including the default. This
    is scoped deliberately: below the working dtype's smallest
    representable magnitude (~1.4e-45 for float32 subnormals) ``vals``
    itself underflows to a literal ``0`` array, the degenerate all-zero
    fallback below kicks in (``beta_eff = beta``, i.e. UNSCALED), and the
    original swallow pattern can in principle recur there -- rfx's own
    realistic DFT-plane field magnitudes (``|field|**2`` ~1e-10 to
    1e-22) sit ~23-35 orders of magnitude above that floor, so this is a
    dtype-representability edge, not a practical concern at any field
    magnitude this simulator actually produces. ``stop_gradient`` on
    the normalizer is deliberate: without it, the returned scalar would
    differentiate the (physically meaningless) rescaling itself rather
    than approximating ``max``; with it, the gradient is the standard
    softmax-weighted average at the CURRENT call's ``beta_eff`` (a real,
    directionally-correct signal, recomputed fresh -- and re-normalized --
    every call).

    What *beta* now controls (and does not control): a larger *beta*
    tightens the approximation -- the softmax value can exceed the true
    ``max(vals)`` by a factor of at most ``1 + log(count)/beta`` (default
    ``beta=1.0`` is therefore LOOSE for a field with many samples: e.g.
    ~14.8x at ``count=1e6``), so raise *beta* (5-50 is a reasonable range
    for most plane sizes) for a tighter approximation once you know you
    need one, exactly as before -- and the DEFAULT is now safe at any
    field magnitude, merely loose, never rounding-noise.

    .. warning::
        **The top-end overflow footgun MOVED, it did not disappear --
        and it now has TWO thresholds, not one.** Auto-scaling divides by
        ``max(vals)`` instead of multiplying by the raw field magnitude,
        so a large enough *beta* against a physically tiny field can
        still overflow -- now in computing ``beta_eff = beta /
        max(vals)`` itself, before ``vals`` is even touched. Measured at
        ``max(vals)`` ~2.24e-22 (this repo's realistic DFT-plane range):

        1. **Value overflow (``beta_eff`` itself overflows).** ``beta``
           at or above ~1e17 makes ``beta_eff`` exceed float32's range,
           previously returning ``value=nan, grad=nan`` silently, no
           exception. This function guards it: when ``beta_eff`` would
           be non-finite, it is clipped to the largest finite value the
           working dtype represents (``jnp.finfo(vals.dtype).max``)
           rather than propagating ``nan``. The clipped VALUE is
           accurate (still converges to the true ``max(vals)``, verified
           at ``beta`` up to 1e38).
        2. **Gradient collapse, measured to start EARLIER (~1e15-1e17)
           and NOT covered by the value clip above.** Even where the
           value stays finite and correct, the returned gradient
           silently drops to EXACTLY ``0.0`` somewhere in this range --
           the precise threshold is FIXTURE-DEPENDENT (measured to shift
           by up to 10x between two fixtures differing only in element
           values, so no single number is trustworthy across geometries)
           -- a sharp cliff, not a gradual falloff, correct to machine
           precision on one side and exactly ``0.0`` on the other. A
           reciprocal-of-a-huge-number precision limit in the VJP's
           ``1/beta_eff`` term, distinct from (and tighter than) the
           value-overflow threshold above. This is NOT fixed by the
           value clip (it happens at a lower *beta* than clipping even
           engages) and is not attempted here: it would need a
           different numerical formulation (e.g. normalizing ``vals`` by
           ``scale`` before scaling by *beta*, which trades this
           gradient cliff for a symmetric one on the VALUE side at
           extreme *beta* / tiny-field combinations -- no formulation
           tried avoids both).

        **Practical guidance**: both thresholds sit many orders of
        magnitude above the 5-50 dimensionless range recommended above,
        and above the 100-1e6 range this repo's own re-calibrated tests
        and examples actually use -- treat *beta* past ~1e6 as
        unverified, not merely "less sharp than useful."
    """
    if beta <= 0.0:
        raise ValueError(f"field_softmax: beta must be positive, got {beta}")
    name_list = [names] if isinstance(names, str) else list(names)

    def objective(result) -> jnp.ndarray:
        arrays = _select_planes(result, name_list, "field_softmax")
        vals = jnp.concatenate(
            [jnp.abs(arrays[n]).reshape(-1) ** 2 for n in name_list]
        )
        scale = jax.lax.stop_gradient(jnp.max(vals))
        scale_safe = jnp.where(scale > 0, scale, jnp.asarray(1.0, dtype=vals.dtype))
        beta_eff = beta / scale_safe
        beta_eff_finite = jnp.isfinite(beta_eff)
        beta_eff = jnp.where(
            beta_eff_finite, beta_eff, jnp.asarray(jnp.finfo(vals.dtype).max, dtype=vals.dtype)
        )
        return logsumexp(beta_eff * vals) / beta_eff

    return objective


def _identity_tangent_batch(params):
    """Build the ``tangents="identity"`` basis for :func:`jacobian_fwd`.

    Defined as per-ELEMENT identity over SCALAR leaves only -- deliberately
    NOT a per-leaf identity, and it fails loud rather than silently picking
    one of the two readings. Measured trap: on a
    ``{'scalar': (), 'field': (4, 4)}`` pytree, a per-leaf one-hot tangent
    (ones on the whole ``field`` leaf) returns 64.0 -- the SUM of the 16
    Jacobian entries for that leaf, not any single column of them. Because
    every leaf here is required to be scalar, "one basis vector per leaf"
    and "one basis vector per scalar element" are the same construction, so
    that trap cannot occur once the non-scalar check below passes.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    if not leaves:
        raise ValueError(
            "jacobian_fwd(tangents='identity'): params pytree has no "
            "leaves -- nothing to differentiate."
        )
    arrays = [jnp.asarray(leaf) for leaf in leaves]
    non_scalar = [
        (i, tuple(a.shape)) for i, a in enumerate(arrays) if a.shape != ()
    ]
    if non_scalar:
        raise ValueError(
            "jacobian_fwd(tangents='identity') requires every params leaf "
            f"to be a SCALAR (shape ()); found non-scalar leaf shape(s) "
            f"{non_scalar}. A per-leaf one-hot tangent on a multi-element "
            "leaf sums that leaf's Jacobian entries instead of isolating "
            "one column of them (measured: a {'scalar': (), 'field': (4, "
            "4)} pytree returned the SUM of 16 Jacobian entries, not one "
            "of them). Pass an explicit tangent pytree/matrix instead: "
            "same structure as params, each leaf carrying an extra "
            "leading n_t axis."
        )
    non_float = [
        (i, str(a.dtype)) for i, a in enumerate(arrays)
        if not jnp.issubdtype(a.dtype, jnp.floating)
    ]
    if non_float:
        raise ValueError(
            "jacobian_fwd(tangents='identity') requires every params leaf "
            f"to have a floating dtype; found non-floating leaf dtype(s) "
            f"{non_float}. An int/bool leaf has no continuous derivative "
            "(jax.jvp requires a float0 tangent for it, which "
            "tangents='identity' does not auto-build); pass an explicit "
            "tangent pytree instead, or drop the non-float leaf(s) from "
            "params."
        )
    n_t = len(leaves)
    tangent_leaves = [
        jnp.asarray(1.0, dtype=a.dtype) * (jnp.arange(n_t) == i).astype(a.dtype)
        for i, a in enumerate(arrays)
    ]
    return jax.tree_util.tree_unflatten(treedef, tangent_leaves), n_t


def _explicit_tangent_batch(params, tangents):
    """Validate and pass through an explicit tangent batch (the escape
    hatch for a non-all-scalar ``params`` pytree, e.g. a flat ``(n_p,)``
    design vector with an explicit ``(n_t, n_p)`` tangent matrix).

    ``tangents`` must share ``params``' pytree structure, with every leaf
    carrying one extra LEADING axis of a common size ``n_t`` (so
    ``tangents`` is exactly what you get by stacking ``n_t`` tangent
    pytrees, each shaped like ``params``, along a new axis 0).
    """
    param_leaves, param_treedef = jax.tree_util.tree_flatten(params)
    tangent_leaves, tangent_treedef = jax.tree_util.tree_flatten(tangents)
    if tangent_treedef != param_treedef:
        raise ValueError(
            "jacobian_fwd: explicit tangents must share params' pytree "
            f"structure (with a leading n_t axis on every leaf); got "
            f"tangents structure {tangent_treedef!r} vs params structure "
            f"{param_treedef!r}."
        )
    bad = []
    n_t_seen = set()
    for i, (p, t) in enumerate(zip(param_leaves, tangent_leaves)):
        p_shape = jnp.shape(p)
        t_shape = jnp.shape(t)
        if len(t_shape) != len(p_shape) + 1 or t_shape[1:] != p_shape:
            bad.append((i, tuple(p_shape), tuple(t_shape)))
        else:
            n_t_seen.add(t_shape[0])
    if bad:
        raise ValueError(
            "jacobian_fwd: explicit tangents leaf shapes must equal the "
            "matching params leaf shape with one leading n_t axis "
            "prepended; mismatched leaf(s) (index, params_shape, "
            f"tangents_shape) = {bad}."
        )
    if len(n_t_seen) != 1:
        raise ValueError(
            "jacobian_fwd: explicit tangents leaves disagree on n_t (the "
            f"leading axis size): {sorted(n_t_seen)}. Every leaf must "
            "share one n_t."
        )
    n_t = n_t_seen.pop()
    if n_t < 1:
        raise ValueError(f"jacobian_fwd: n_t must be >= 1, got {n_t}.")
    return tangents, n_t


def jacobian_fwd(
    sim_fn: Callable,
    params: Any,
    *,
    tangents: Any = "identity",
    batch_tangents: bool = True,
) -> tuple[Any, Any]:
    """Compute a value and its forward-mode Jacobian along tangent directions.

    The value is ``sim_fn(params)``. The Jacobian mirrors the value's pytree,
    with each leaf gaining a leading axis whose rows are tangent directions.
    The default identity basis differentiates with respect to each scalar
    parameter leaf in ``jax.tree_util.tree_flatten`` order.
    Use this for many outputs and few parameters: forward mode stores field
    and tangent states without retaining a reverse-mode time-step tape.

    Parameters
    ----------
    sim_fn : Callable
        JAX-differentiable function mapping ``params`` to a value or pytree.
        It must support ``jax.jvp``, and its primal value must not depend on
        the tangent direction. For field observables, compose ``sim.forward``
        with :func:`dft_field`, :func:`field_energy`, or :func:`field_softmax`.
    params : Any
        Nonempty pytree of design variables. With the identity basis, every
        leaf must be a floating-point scalar of shape ``()``. Array-valued
        leaves require explicit tangents.
    tangents : "identity" or pytree, optional
        Default ``"identity"`` supplies a one-hot direction per scalar leaf.
        An explicit batch must match the structure of ``params``; each leaf
        must have shape ``(n_t,) + params_leaf.shape``, with a common nonempty
        leading axis and a dtype compatible with ``jax.jvp``. For a flat
        parameter array, ``jnp.eye(n_p)`` supplies the full identity basis;
        a subset of its rows selects directions.
    batch_tangents : bool, optional
        Default ``True`` evaluates all directions with ``jax.vmap(jax.jvp)``.
        This shares the primal sweep and holds all tangent states at once.
        ``False`` evaluates directions sequentially, holding one tangent
        state at a time in exchange for more execution time. Neither path
        calls ``jax.jit``; the caller controls compilation. The paths may
        differ at float32 reassociation scale, about 1e-6 relative.

    Returns
    -------
    value : Any
        Single, unstacked primal value with the structure of ``sim_fn(params)``.
    jacobian : Any
        Pytree matching ``value``, with leaf shape ``(n_t,) + value_leaf.shape``.
        Row ``i`` is the directional derivative along tangent ``i``. For real
        input tangents and complex outputs, derivatives are unconjugated;
        conjugate one side when comparing with a reverse-mode convention
        that supplies the conjugate.

    Raises
    ------
    ValueError
        If a tangent string is not ``"identity"``; the identity basis has
        empty, non-scalar, or non-floating parameter leaves; or explicit
        tangents have a mismatched structure, shape, or empty or inconsistent
        leading axis. Also raised when the primal depends on the mapped
        tangent axis, or eager sequential primal values disagree. A
        ``checkpoint_segments`` setting that does not divide ``n_steps``
        can raise through ``forward``.
    TypeError
        If parameter and tangent dtypes are incompatible with ``jax.jvp``.
    NotImplementedError
        Propagated from ``forward`` for a nonzero ``n_warmup`` on the uniform
        mesh.
    jax.errors.ConcretizationTypeError
        If the simulation requires a concrete value from a traced parameter,
        such as a traced ``Box`` corner.

    Limits
    ------
    Supported on the uniform mesh on one device. ``checkpoint`` and
    ``checkpoint_segments`` are reverse-mode options and
    provide no forward-mode memory benefit; pass ``checkpoint=False`` and
    leave ``checkpoint_segments`` unset. A nonzero ``n_warmup`` is refused.
    The size or position of a shape drawn with ``Box`` cannot be
    differentiated (a traced corner raises, see above); differentiate a
    design region instead (a material density or
    ``pec_occupancy_override``). Tangents through a Kerr material have not
    been checked.
    """
    if isinstance(tangents, str):
        if tangents != "identity":
            raise ValueError(
                f"jacobian_fwd: tangents string must be 'identity', got "
                f"{tangents!r}."
            )
        tangent_batch, n_t = _identity_tangent_batch(params)
    else:
        tangent_batch, n_t = _explicit_tangent_batch(params, tangents)

    def _jvp_row(tangent_row):
        return jax.jvp(sim_fn, (params,), (tangent_row,))

    if batch_tangents:
        value, jacobian = jax.vmap(
            _jvp_row, in_axes=0, out_axes=(None, 0),
        )(tangent_batch)
    else:
        try:
            jax.eval_shape(
                lambda t: jax.vmap(_jvp_row, in_axes=0, out_axes=(None, 0))(t),
                tangent_batch,
            )
        except jax.errors.ConcretizationTypeError:
            pass
        values = []
        jac_rows = []
        for i in range(n_t):
            row = jax.tree_util.tree_map(lambda x: x[i], tangent_batch)
            v, j = jax.jvp(sim_fn, (params,), (row,))
            values.append(v)
            jac_rows.append(j)
        value = values[0]
        params_leaves = jax.tree_util.tree_leaves(params)
        if params_leaves and not any(
            is_tracer(leaf) for leaf in params_leaves
        ):
            value_leaves = jax.tree_util.tree_leaves(value)
            for i, v in enumerate(values[1:], start=1):
                mismatched = [
                    idx for idx, (a, b) in enumerate(
                        zip(value_leaves, jax.tree_util.tree_leaves(v))
                    )
                    if not bool(jnp.array_equal(a, b, equal_nan=True))
                ]
                if mismatched:
                    raise ValueError(
                        "jacobian_fwd(batch_tangents=False): the primal "
                        f"value from tangent row {i} differs from tangent "
                        f"row 0's at output leaf index/indices "
                        f"{mismatched} -- sim_fn's primal output depends "
                        "on which tangent direction is being evaluated, "
                        "violating jax.jvp's primal/tangent independence "
                        "contract, OR sim_fn has a cross-call side effect "
                        "(e.g. mutable state) that a single vmap trace "
                        "cannot see. batch_tangents=True MAY also raise "
                        "this (as a jax.vmap out_axes=(None, 0) error) "
                        "if the "
                        "cause is a genuine tangent dependency -- but NOT "
                        "if the cause is a cross-call side effect, which "
                        "vmap's single trace cannot observe (measured: "
                        "batch_tangents=True silently returns a value in "
                        "that case, it does not raise). Fix sim_fn to be "
                        "a pure function of params alone."
                    )
        jacobian = jax.tree_util.tree_map(
            lambda *rows: jnp.stack(rows, axis=0), *jac_rows,
        )
    return value, jacobian
