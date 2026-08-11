"""Differentiable observables built on registered DFT-plane probes (#579),
plus a batched forward-mode Jacobian wrapper (#577).

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
  ``jax``-differentiable). See its own docstring for the full contract,
  including why it is PACKAGING rather than new solver capability, and the
  scope limits that come with that.

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

E/H mixing hazard
------------------
``update_dft_plane_probe`` stamps EVERY component -- electric or magnetic --
at the same traced time ``t = state.step * dt``
(``rfx/probes/probes.py:452-460``). Physically, on the Yee leapfrog H lives
half a step behind E (``H`` at ``t - dt/2``, not ``t``). Magnitude-only
objectives (``field_energy``, ``field_softmax``, or any other function of
``|field|`` alone) are UNAFFECTED -- a uniform per-frequency phase rotation
does not change a magnitude. But composing an E x H* cross term (a Poynting-
flux-like quantity, or anything computing complex ``Zin = V/I`` from a
line-integrated E-derived V and H-derived I) needs the same correction the
production MSL V/I extractor applies: multiply every H-component plane by
``exp(+j * 2*pi*f * dt/2)`` before combining it with an E-component plane
(see ``rfx/api/_sparams.py:2975-2995`` for the worked derivation and the
"``Re(Zin) < 0``" artefact class this omission produces). ``dt`` is not
carried on ``ForwardResult`` or ``DFTPlaneProbe`` -- it comes from the
``Simulation`` / ``Grid`` the caller already built (``sim._build_grid().dt``
or ``result.grid.dt`` when the result carries one), so this module does not
attempt the correction itself and does not accept an ``h_phase_correction``
kwarg: it would need a value neither object supplies, and unit-modulus phase
is invisible to both built-in functionals, so an untested flag would ship.
Apply the ``exp(+j*omega*dt/2)`` factor to H-component planes yourself, one
line, if you build an E x H* objective on top of :func:`dft_field`.

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

    Auto-scaling (fixes a measured footgun -- read before raising *beta*
    "to be safe")
    -------------------------------------------------------------------
    Earlier versions of this function computed ``logsumexp(beta*vals)/beta``
    directly against the RAW ``|field|**2`` values, so *beta* had to be
    hand-matched to whatever physical units/magnitude the accumulator
    happened to carry (rfx's own DFT-plane accumulators commonly sit
    around ``|field|**2`` ~1e-10 to 1e-22 depending on
    geometry/normalization). At the then-default ``beta=1.0`` the
    objective sat at ~100.000002% of the design-independent constant
    ``log(count)/beta`` (measured, `#619`): both the value and its
    gradient were differentiating ROUNDING NOISE in that constant, not
    physics, and it could coincidentally still pass a single
    finite-difference check while failing at a different step size (a
    `feedback_gate_can_bind_artifact`-class trap).

    This function now computes ``beta_eff = beta / stop_gradient(max(vals))``
    and uses ``beta_eff`` in place of a raw *beta* everywhere above.
    Because ``vals``' own current max scales out of the product exactly,
    ``beta_eff * max(vals) == beta`` ALWAYS, regardless of the field's
    physical units -- the ``log(count)/beta`` swallow above cannot recur
    at any finite field magnitude, at any *beta*, including the default.
    ``stop_gradient`` on the normalizer is deliberate: without it, the
    returned scalar would differentiate the (physically meaningless)
    rescaling itself rather than approximating ``max``; with it, the
    gradient is the standard softmax-weighted average at the CURRENT
    call's ``beta_eff`` (a real, directionally-correct signal, recomputed
    fresh -- and re-normalized -- every call).

    What *beta* now controls (and does not control): a larger *beta*
    tightens the approximation -- the softmax value can exceed the true
    ``max(vals)`` by a factor of at most ``1 + log(count)/beta`` (default
    ``beta=1.0`` is therefore LOOSE for a field with many samples: e.g.
    ~14.8x at ``count=1e6``), so raise *beta* (5-50 is a reasonable range
    for most plane sizes) for a tighter approximation once you know you
    need one, exactly as before -- but the DEFAULT is now safe at any
    field magnitude, merely loose, never rounding-noise. A large *beta* no
    longer risks the old top-end overflow footgun either: ``beta_eff *
    vals`` is bounded by *beta* itself (``vals <= max(vals)`` by
    definition, so ``beta_eff * vals <= beta_eff * max(vals) == beta``),
    not by the field's unknown absolute scale, so any finite *beta* you'd
    plausibly choose (up to jax.nn.logsumexp's own internal
    max-subtraction headroom) is safe.
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
        # Guard the degenerate all-zero-field case (scale == 0) rather
        # than dividing by it; beta_eff = beta there reproduces the
        # pre-auto-scale behaviour, which is harmless because vals is
        # uniformly 0 too (logsumexp(0)/beta = log(count)/beta exactly
        # either way -- there is no signal to lose).
        scale_safe = jnp.where(scale > 0, scale, jnp.asarray(1.0, dtype=vals.dtype))
        beta_eff = beta / scale_safe
        return logsumexp(beta_eff * vals) / beta_eff

    return objective


def _identity_tangent_batch(params):
    """Build the ``tangents="identity"`` basis for :func:`jacobian_fwd`.

    Defined as per-ELEMENT identity over SCALAR leaves only -- deliberately
    NOT a per-leaf identity, and it fails loud rather than silently picking
    one of the two readings. Measured trap (#577 evidence): on a
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
    """Batched forward-mode Jacobian of ``sim_fn`` at ``params`` (issue #577).

    ``jax.jvp`` / ``jax.jacfwd`` / ``jax.vmap(jax.jvp)`` already run
    end-to-end through ``sim.forward(eps_override=...)`` ->
    :func:`dft_field` / :func:`field_energy` / :func:`field_softmax` today,
    with zero solver changes -- this function is a thin, gated PACKAGING
    layer over that stock-JAX capability (tangent-basis construction,
    dtype/shape validation, primal de-duplication, sequential-vs-batched
    dispatch), not a new AD engine. It computes

        ``value, jacobian = jacobian_fwd(sim_fn, params)``

    where ``value = sim_fn(params)`` (bit-identical to calling ``sim_fn``
    directly -- see the API trap note below) and ``jacobian`` has the SAME
    pytree structure as *value* (``sim_fn``'s OUTPUT, not *params* -- this
    is what ``jax.jvp``'s own tangent output mirrors, and this function
    does not reshape it into a params-keyed structure the way
    ``jax.jacfwd`` does), with every leaf gaining one new LEADING axis of
    size ``n_t``: ``jacobian`` (or ``jacobian[k]`` for an output pytree
    with leaf ``k``) has shape ``(n_t,) + value.shape``, row ``i`` holding
    the directional derivative of *value* along tangent direction ``i``.
    With the default ``tangents="identity"``, direction ``i`` corresponds
    to ``params``' ``i``-th flattened leaf (``jax.tree_util.tree_flatten``
    order) -- i.e. ``jacobian[i]`` is ``d(value)/d(params_leaf[i])``.

    Mechanism: ``jax.vmap`` over the TANGENT argument of ``jax.jvp`` (NOT
    ``jax.linearize``). Measured on the real uniform-lane FDTD scan: the
    ``vmap(jvp)`` jaxpr contains exactly ONE ``lax.scan`` (the FDTD time
    loop) whose carry holds one UNBATCHED copy of the field state (the
    primal sweep) plus ``n_t`` BATCHED tangent copies -- the primal sweep
    really is shared across every tangent direction, which is the premise
    #577 asked to verify. ``jax.linearize`` was measured to cost 24x more
    peak memory on the same fixture (8654 MB vs 355 MB at n_steps=600,
    n_t=10) because it materialises the whole scan's primal residuals --
    i.e. the reverse-mode tape -- which forward mode never needs.

    API TRAP (verify this, do not assume it): a naive
    ``jax.vmap(jax.jvp(...))`` ALSO stacks the primal, returning a
    ``(n_t,) + value.shape`` array of ``n_t`` copies of the same value.
    This function passes ``out_axes=(None, 0)`` to ``jax.vmap`` so the
    returned ``value`` is the single, unstacked primal -- but that only
    works because the primal computation genuinely does not depend on
    which tangent direction is being evaluated. If you change ``sim_fn``
    in a way that breaks that invariant, ``jax.vmap`` raises loudly
    (``out_axes=None`` on an output that DOES depend on the mapped axis is
    a vmap error, not a silent wrong answer). ``batch_tangents=False``
    (the sequential path) has NO ``jax.vmap`` of its own to inherit that
    check from -- it just calls ``jax.jvp`` ``n_t`` times in a plain
    Python loop and returns ``values[0]``, so without an explicit guard a
    ``sim_fn`` that violates the invariant would silently return one
    arbitrary tangent row's primal on that path instead of raising. This
    function closes that gap on the sequential path via two guards, so
    the memory-vs-speed knob is not also a safety-vs-speed knob: (1) a
    ``jax.eval_shape`` abstract trace of the SAME ``out_axes=(None, 0)``
    vmap the batched path actually runs -- shape-only, zero FLOPs, zero
    extra memory (verified: reproduces the identical
    ``ValueError`` a real violating vmap call raises, including when this
    whole call is itself wrapped in an outer ``jax.jit`` by the caller,
    since ``eval_shape`` is trace-time like ``vmap``'s own check); (2)
    when running eagerly (params are not themselves tracers, so the
    computed primal values are concrete), an exact pairwise comparison of
    all ``n_t`` sequentially-computed primal values, raising ``ValueError``
    on any mismatch -- a strictly stronger, numeric confirmation of the
    same invariant, skipped only when it is not expressible (under an
    enclosing ``jax.jit``, where guard (1) is the check in effect).

    Cost characterization: NOT "a few times one solve" -- see
    ``scripts/benchmark_jacobian_fwd.py`` (regenerable; run it for current
    numbers on your hardware/grid) and the CHANGELOG entry for this issue
    for measured wall-time/flops/memory ratios. The honest summary: wall
    time and flops scale roughly as ``(1 + n_t)`` plain solves and that
    ratio MOVES with grid size (do not trust one ratio across problem
    sizes); memory is the argument FOR this mode, not speed -- forward-mode
    peak is ``O(state) * (1 + n_t)`` and independent of ``n_steps``, while
    reverse mode without segmented checkpointing is tens of times one
    plain solve even for a single scalar output, and ``jax.jacrev`` refuses
    a complex-valued observable outright (raises ``TypeError`` without
    ``holomorphic=True``). Numbers are never published here -- see the
    anti-rot rule in the module/CHANGELOG instead of trusting a docstring
    number, which rots.

    Parameters
    ----------
    sim_fn : Callable
        ``params -> value``, JAX-differentiable (typically
        ``lambda p: dft_field(name)(sim.forward(eps_override=eps(p), ...))``
        or the same composed with :func:`field_energy` /
        :func:`field_softmax`). Called under ``jax.jvp``, so it must be
        traceable the same way any ``jax.grad``/``jax.jvp`` target is.
    params : Any
        A JAX pytree of the design variables to differentiate with respect
        to. With the default ``tangents="identity"``, every leaf must be a
        floating-dtype SCALAR (shape ``()``) -- see :func:`_identity_tangent_batch`.
        A flat ``(n_p,)`` array (one non-scalar leaf) needs the explicit
        ``tangents=`` escape hatch below.
    tangents : "identity" or a pytree, optional
        ``"identity"`` (default): build one one-hot tangent direction per
        scalar leaf of *params* (``n_t = number of leaves``); fails loud
        (``ValueError``) if any leaf is non-scalar or non-floating -- see
        :func:`_identity_tangent_batch`'s docstring for the measured trap
        this avoids. Otherwise: an explicit tangent BATCH -- a pytree with
        the same structure as *params*, every leaf carrying one extra
        LEADING axis of a common size ``n_t`` (e.g. ``tangents=jnp.eye(n_p)``
        when *params* is a flat ``(n_p,)`` array, for the full Jacobian, or
        any ``(n_t, n_p)`` slice of it for a subset of columns).
    batch_tangents : bool, optional
        ``True`` (default): run all ``n_t`` tangent directions as ONE
        ``jax.vmap(jax.jvp(...))`` call -- lower wall time (measured 2.3-3.1x
        faster than the sequential control at n_t in {4, 10}), higher peak
        memory (``O(state) * (1 + n_t)``, all ``n_t`` tangent copies live
        at once). ``False``: run the ``n_t`` directions as independent
        sequential ``jax.jvp`` calls in a plain Python loop -- the
        memory-vs-speed knob, trading the wall-time win for a peak that
        scales with ONE tangent copy at a time instead of ``n_t``
        (measured: primal-sharing still holds sequentially, at ``~1 + n_t``
        compute but with a lower peak than the batched path scales to as
        ``n_t`` grows). CORRECTION: this function does not call
        ``jax.jit`` itself on EITHER path -- there is no ``jax.jit``
        anywhere in this module (verify: ``grep jax.jit rfx/observables.py``
        returns nothing). ``batch_tangents=False`` runs the ``n_t``
        sequential ``jax.jvp`` calls exactly as written, at whatever JIT
        boundary the CALLER supplies (eager if the caller does not
        ``jax.jit`` anything; compiled as part of one program if the
        caller wraps its own outer function, as
        ``scripts/benchmark_jacobian_fwd.py`` and
        ``tests/test_jacobian_fwd.py`` both do for compiled/jaxpr
        measurement) -- an earlier version of this docstring claimed
        ``jax.jit`` wrapping was part of this function's own behaviour,
        which was never true. The two paths (batched vs. sequential) are
        DIFFERENT XLA programs and may disagree at float32 reassociation
        scale (~1e-6 relative); they are not bit-identical. See "Fail-loud
        fences" below for how the two paths' SAFETY differs, not just
        their performance.

    Returns
    -------
    (value, jacobian) : tuple
        ``value``: the single primal, ``sim_fn(params)`` bit-identical
        (verify this in your own test if you change *sim_fn*'s structure --
        see the API trap above). ``jacobian``: a pytree mirroring *value*
        (``sim_fn``'s output, NOT *params* -- see the paragraph above the
        signature), each leaf an array of shape ``(n_t,) + value.shape``.

    Complex-Jacobian convention
    ----------------------------
    For a complex-valued *value* (e.g. :func:`dft_field`'s output), the
    returned Jacobian is ``dy/dx`` UNCONJUGATED -- exactly what forward-mode
    AD produces for a real input tangent and a complex output. This
    DIFFERS BY A CONJUGATE from anything derived through ``jax.vjp`` /
    ``jax.grad`` on the same computation (measured: 0.02% relative error
    against an unconjugated central-difference reference, 198% against the
    conjugated one). If you compare this Jacobian against anything built
    from reverse-mode AD, conjugate one side first. This is also why
    ``jax.jacrev`` cannot be used as a drop-in alternative here: it refuses
    a complex-dtype output outright (``TypeError``, needs
    ``holomorphic=True``) -- forward mode is the only stock-JAX mode that
    hands back this Jacobian without a holomorphy declaration.

    Fail-loud fences
    -----------------
    Three configurations are unsafe to combine with ``jacobian_fwd`` on
    ``sim_fn`` bodies built on ``Simulation.forward(...)``. This function
    is intentionally generic over *sim_fn* (it never sees ``forward()``'s
    own keyword arguments -- they live inside your closure), so the first
    two are enforced by a RAISE that propagates up through ``jax.jvp``
    from ``forward()``'s own pre-existing checks; the remaining one has no
    raise to inherit and cannot be intercepted from outside the closure,
    so it is a DOCUMENTED trap instead. Read the RAISES/DOCS tag on each
    before assuming "it didn't raise" means "it's fine":

    - **[RAISES, inherited]** non-uniform + ``distributed=True`` with a
      registered DFT-plane probe: ``forward()`` itself raises
      ``NotImplementedError`` (the #619 fence, ``rfx/api/_execute.py``) --
      neither sharded runner accumulates DFT-plane fields. ``jacobian_fwd``
      scope is the uniform single-device lane only; a distributed
      ``sim_fn`` fails loud before this function's own machinery runs.
    - **[RAISES, inherited]** ``n_warmup``: ``forward()`` itself raises
      ``NotImplementedError`` for a nonzero ``n_warmup`` on the uniform
      lane (issue #626) -- so on ``jacobian_fwd``'s supported (uniform)
      lane, a ``n_warmup``-using ``sim_fn`` already fails loud. It used to
      be a measured SILENT NO-OP (bit-identical value AND tangent at
      ``n_warmup=0`` vs ``n_warmup=60`` of 80 steps) before the fence was
      added. Do not read this as "n_warmup is safe elsewhere": on the
      non-uniform lane where it IS implemented, ``n_warmup`` truncation
      error DEPENDS ON DISTANCE FROM THE SOURCE TO THE DESIGN REGION, not
      merely on ``n_warmup``'s value -- it is (near-)exact while
      ``n_warmup`` stays below the wavefront's arrival time at the design
      region (``K_safe ~= floor(min_distance(source, design_region) /
      (C0 * dt))``) and grows sharply beyond it, up to ~6-7% at half the
      pre-loss-window length, 58% at the loss-window boundary itself,
      exactly zero once far enough beyond it, FOR A NEAR-SOURCE
      placement specifically (issue #626 part 2 / addendum -- a
      far-from-source placement measured error <0.04% for every
      ``n_warmup <= K_safe``, see ``rfx/nonuniform.py``'s ``n_warmup
      split`` comment for the full measured curves and formula) -- it is
      not fenced there (truncated-BPTT is a legitimate, if
      placement-dependent, construction), and any future non-uniform
      extension of this function must account for ``K_safe``, not treat
      ``n_warmup`` as uniformly lossy OR uniformly free.
    - **[DOCS ONLY -- inert, not a raise]** ``checkpoint`` /
      ``checkpoint_segments``: measured EXACTLY NEUTRAL under forward mode
      (flops/temp bytes identical across ``checkpoint=False``,
      ``checkpoint=True`` (the ``forward()`` default), and
      ``checkpoint_segments=K``) -- remat only pays off under reverse-mode
      transposition, and forward mode builds no tape to transpose. A
      ``sim_fn`` built for use with ``jacobian_fwd`` should pass
      ``checkpoint=False`` explicitly (``forward()`` defaults to
      ``checkpoint=True``) to avoid inheriting dead HLO plus
      ``checkpoint_segments``' ``n_steps % K == 0`` divisibility raise for
      no forward-mode benefit. These are reverse-mode knobs; document them
      as such, do not reach for them here.

    Scope limits
    -------------
    - Uniform single-device lane only (see the fences above).
    - "Geometry parameter" is NOT a parametric dimension here. A traced
      ``Box`` corner raises ``ConcretizationTypeError``
      (``rfx/geometry/csg.py``) and ``forward()``'s material rasterisation
      is a binary ``jnp.where`` mask with zero gradient almost everywhere.
      The only continuous geometry design channels this package has are
      topology density (via :func:`rfx.topology.density_to_material_fields`),
      ``pec_occupancy_override``, and the non-uniform ``dz_profile``. If you
      want ``d/d(patch width in metres)``, this function cannot supply that
      column -- rfx does not have it to give, on any AD mode.
    - Nonlinear (Kerr chi3) tangents ran clean under ``vmap(jvp)`` in
      testing but were numerically indistinguishable from the linear
      baseline at the tested chi3 -- that is NOT a validation of the
      nonlinear tangent path. Treat it as unverified until measured at a
      chi3 large enough to move the Jacobian outside noise.
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
        # Guard 1 (trace-time, works eager or under an outer jax.jit):
        # abstractly trace the SAME out_axes=(None, 0) vmap the batched
        # path actually runs, via jax.eval_shape -- shape-only, zero
        # FLOPs, zero extra memory (that O(n_t) memory cost is exactly
        # what this branch exists to avoid). If sim_fn's primal genuinely
        # depends on the tangent direction, this raises the identical
        # ValueError a real vmap(..., out_axes=(None, 0)) call would, so
        # the sequential path inherits the batched path's fail-loud
        # invariant instead of silently returning one arbitrary row's
        # primal. See jacobian_fwd's "API TRAP" docstring section.
        jax.eval_shape(
            lambda t: jax.vmap(_jvp_row, in_axes=0, out_axes=(None, 0))(t),
            tangent_batch,
        )
        values = []
        jac_rows = []
        for i in range(n_t):
            row = jax.tree_util.tree_map(lambda x: x[i], tangent_batch)
            v, j = jax.jvp(sim_fn, (params,), (row,))
            values.append(v)
            jac_rows.append(j)
        value = values[0]
        # Guard 2 (eager only -- a strictly stronger numeric check on top
        # of guard 1, skipped under an outer jax.jit where the primal
        # values are still abstract tracers and cannot be compared as
        # concrete numbers): every sequentially-computed primal must
        # agree EXACTLY with value (they are, mathematically, n_t
        # independent evaluations of the same sim_fn(params)).
        params_leaves = jax.tree_util.tree_leaves(params)
        if params_leaves and not isinstance(params_leaves[0], jax.core.Tracer):
            value_leaves = jax.tree_util.tree_leaves(value)
            for i, v in enumerate(values[1:], start=1):
                mismatched = [
                    idx for idx, (a, b) in enumerate(
                        zip(value_leaves, jax.tree_util.tree_leaves(v))
                    )
                    if not bool(jnp.array_equal(a, b))
                ]
                if mismatched:
                    raise ValueError(
                        "jacobian_fwd(batch_tangents=False): the primal "
                        f"value from tangent row {i} differs from tangent "
                        f"row 0's at output leaf index/indices "
                        f"{mismatched} -- sim_fn's primal output depends "
                        "on which tangent direction is being evaluated, "
                        "violating jax.jvp's primal/tangent independence "
                        "contract. batch_tangents=True would raise this "
                        "as a jax.vmap out_axes=(None, 0) error (see "
                        "jacobian_fwd's docstring, 'API TRAP'); fix "
                        "sim_fn to be a pure function of params alone."
                    )
        jacobian = jax.tree_util.tree_map(
            lambda *rows: jnp.stack(rows, axis=0), *jac_rows,
        )
    return value, jacobian
