"""Dispersion-pole mask keying helpers (issue #274).

Lives in its own module — NOT in ``rfx/geometry/rasterize.py`` — so that
``rfx/api/_compile.py`` can import these helpers at ``import rfx`` time
without initializing the ``rfx.geometry.rasterize`` SUBMODULE. That
submodule's name collides with the public ``rasterize`` FUNCTION that
``rfx/geometry/__init__.py`` re-exports from ``csg.py``: initializing
the submodule setattr's the module object onto the package, clobbering
the function attribute, and ``from rfx.geometry import rasterize`` then
yields a non-callable module (broke the rcs_scattering tutorial).
The underlying name collision is a pre-existing landmine tracked
separately; this module keeps #274 out of its blast radius.
"""

from __future__ import annotations

import warnings

from rfx.core.jax_utils import is_tracer


def _pole_key(pole):
    """Mask-dict key for a dispersion pole (issue #274).

    Key by VALUE whenever value equality is decidable: equal-valued
    poles from different ``add_material`` calls merge into one
    ``(pole, mask)`` entry, so overlapping geometry cannot apply the
    same pole twice (``init_debye`` / ``init_lorentz`` sum
    contributions over entries — a duplicate entry would silently
    double delta_eps on overlap cells). Plain id-keying for all poles
    is the recorded do-not-repeat (PR #272 branch: overlap-cell beta
    ratio 2.000 vs 1.000).

    Key resolution order:

    1. Hashable pole (plain Python-float fields) — the pole itself.
       Byte-identical to the historical dict-key behaviour.
    2. Unhashable but EAGER fields (e.g. ``jnp.float32(1.5)``, an
       unhashable scalar ``jax.Array``) — a plain tuple of
       ``float()``-coerced fields. Value equality IS decidable here;
       a NamedTuple hashes/compares equal to the plain tuple of the
       same values, so eager-array and Python-float spellings of the
       same pole also merge with each other. CAVEAT: that cross-
       spelling merge holds only for values exactly representable in
       the array dtype — ``jnp.float32(0.1)`` ``float()``-coerces to
       0.10000000149011612, which differs from Python ``0.1``, so the
       two spellings key as TWO entries and overlapping geometry
       applies the pole twice (double delta_eps on the overlap
       cells). Spell a shared pole consistently, or use exactly
       representable values (0.5, 1.5, ...).
    3. Fields carrying JAX TRACERS — ``id(pole)``. Value equality is
       undecidable at trace time; identity is the only stable key
       (the documented differentiable-dispersion path, #273).
    4. Eager but not ``float()``-coercible — fields that are
       non-scalar arrays OR scalars ``float()`` rejects (e.g. a
       complex 0-d array) — ``id(pole)`` plus a loud ``UserWarning``:
       such poles will not dedupe against value-equal duplicates
       (fail-loud beats silent double-counting).
    """
    try:
        hash(pole)
        return pole
    except TypeError:
        pass
    if any(is_tracer(f) for f in pole):
        return id(pole)
    try:
        # NamedTuples hash/compare equal to plain tuples of the same
        # values, so this key merges with value-equal hashable poles.
        return tuple(float(f) for f in pole)
    except (TypeError, ValueError):
        warnings.warn(
            f"Dispersion pole {pole!r} has unhashable fields that "
            f"cannot be coerced to float (non-scalar arrays or "
            f"non-float-coercible scalars, e.g. a complex 0-d array); "
            f"keying its geometry mask by object identity. "
            f"Value-equal duplicates of this pole will NOT dedupe and "
            f"can double-apply delta_eps on overlapping geometry "
            f"(issue #274).",
            UserWarning,
            stacklevel=3,
        )
        return id(pole)


def _accumulate_pole_mask(masks_by_pole: dict, pole, mask) -> None:
    """Merge ``mask`` into the entry for ``pole`` (see ``_pole_key``).

    Values are ``(pole, mask)`` pairs so iteration yields the pole
    object regardless of key form (pole itself, coerced value tuple,
    or ``id(pole)``). The first-seen pole object is kept on merge,
    matching the historical dict-key behaviour byte-for-byte.
    """
    key = _pole_key(pole)
    prev = masks_by_pole.get(key)
    if prev is not None:
        masks_by_pole[key] = (prev[0], prev[1] | mask)
    else:
        masks_by_pole[key] = (pole, mask)


def _spec_from_pole_masks(masks_by_pole: dict):
    """Build a ``(poles, masks)`` spec tuple, or None when empty."""
    if not masks_by_pole:
        return None
    entries = list(masks_by_pole.values())
    return ([p for p, _ in entries], [m for _, m in entries])
