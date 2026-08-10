"""Differentiable observables built on registered DFT-plane probes (#579).

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

Both lanes, both entry points: ``result.dft_planes`` is a name-keyed dict on
the uniform AND non-uniform meshes, for both ``run()`` and ``forward()``
(``rfx/runners/uniform.py:859-866``, ``rfx/runners/nonuniform.py:1064-1068``,
``rfx/api/_execute.py:1579-1596``). Everything here is lane-agnostic: it
only reads ``result.dft_planes``, so it works unmodified on any of the four
combinations. The exceptions are the two DISTRIBUTED lanes, which are
fenced fail-loud (see "Fail-loud fences" below) because neither sharded
runner accumulates DFT-plane fields.

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

from typing import Callable, Sequence

import jax.numpy as jnp
from jax.nn import logsumexp

__all__ = ["dft_field", "field_energy", "field_softmax"]


def _select_planes(result, names, caller: str) -> dict:
    """Return ``{name: jnp array}`` for the requested plane name(s).

    Shared by every public function in this module. Raises ``ValueError``
    naming the producing call (``add_dft_plane_probe`` + ``run()``/
    ``forward()``) when the result carries no planes at all, and naming the
    specific missing name(s) when the result has planes but not all of the
    requested ones -- both in the ``optimize_objectives`` error style (name
    the field that's missing and the call that would populate it).
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
    return {n: jnp.asarray(planes[n].accumulator) for n in name_list}


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
        shapes differ, or when ``stack=False``, the accessor instead
        returns ``dict[name] -> array`` (per-name dict form) so callers with
        heterogeneous planes are never silently truncated or broadcast.

    Returns
    -------
    callable(result) -> jnp.ndarray | dict[str, jnp.ndarray]
        JAX-differentiable: the returned array(s) are views/stacks of the
        same accumulator that was carried through the FDTD scan, so
        ``jax.grad`` flows through them unchanged.

    Raises
    ------
    ValueError
        If *result* carries no ``dft_planes`` at all, or is missing one of
        the requested *names*, or (list form, ``stack=True``) the requested
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

    Uses the standard log-sum-exp soft-max:
    ``softmax = logsumexp(beta * |field|**2) / beta``, computed with
    ``jax.nn.logsumexp`` for numerical stability. As ``beta -> infinity``
    this converges to the true ``max(|field|**2)``; a smaller *beta* trades
    a looser max approximation for a smoother, better-conditioned gradient
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
        Soft-max sharpness (default 1.0). Must be positive.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    if beta <= 0.0:
        raise ValueError(f"field_softmax: beta must be positive, got {beta}")
    name_list = [names] if isinstance(names, str) else list(names)

    def objective(result) -> jnp.ndarray:
        arrays = _select_planes(result, name_list, "field_softmax")
        vals = jnp.concatenate(
            [jnp.abs(arrays[n]).reshape(-1) ** 2 for n in name_list]
        )
        return logsumexp(beta * vals) / beta

    return objective
