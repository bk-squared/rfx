"""Mesh-quality and non-uniform-lane preflight, moved verbatim out of
``rfx.api._preflight``.

Issue #980 Phase 3, leg 5. The mesh family: whether the grid resolves what is
drawn on it (cells per wavelength in every material, imported-CAD thin
features, the #743 per-body coarsest-cell rule and the Taflove Ch.4
phase-velocity estimate), whether a GRADED mesh is inside the validated
SPEC-01 WP6 envelope and what that envelope excludes, what lands ON a grading
transition (the #669 lossy sheet, the #672 current source, the #688 lumped /
wire port -- each reported on its Ampere-loop axes only, through the shared
``_graded_node_report``), how a Box rasterizes across a graded z column, and
which features the non-uniform and SBP-SAT subgridded lanes do not implement.
Everything here was relocated byte for byte out of ``rfx/api/_preflight.py``
-- same text, same order, same indentation, same docstrings, nothing renamed,
reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first. The extension was chosen by a CALL CENSUS, as legs 3 and 4 were, and it
came back leg 4's answer six times over: all twelve bodies below were entered
-- the family sits on ``_validate_simulation_config``'s unconditional spine,
so 46 to 50 of the 50 fixtures called each -- while SIX of them emitted
nothing. The three graded-node advisories each need a specific object standing
on a grading step; ``_validate_cfg_nonuniform_limitations`` needed two
fixtures because its TFSF finding raises and aborts the body before its
CPML-thickness finding is reached; ``_validate_cfg_subgrid_limitations``
needed a refinement region carrying a refused feature, a pair no test in the
repo had built; and ``_validate_cfg_floquet_nonuniform`` is reachable only
through AUTO-MESH, because ``add_floquet_port`` refuses a DECLARED
``dz_profile`` at registration. Seven fixtures now close all six. The lock
module's own docstring carries the measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

THE THREE CLASS-BODY CONSTANTS DO NOT MOVE. ``_MULTIBAND_RATIO_CAP``,
``_INPLANE_RATIO_CAP`` and ``_AXIS_OF_COMPONENT`` are assignments in the
``_PreflightMixin`` class body, read as ``self.<NAME>`` by four of the bodies
below and -- for the first two -- read off the CLASS by
``tests/unit/nonuniform/test_multiband_nu_envelope.py:329``, which asserts the
two caps are DIFFERENT numbers so that a single shared constant cannot
silently re-couple the in-plane lock to the z one. Moving them here would keep
every ``self.`` read working (the method travels with the constant) while
breaking that test, and no module-level namespace lock would see it happen,
because they are not module-level names. They stay in the facade, the moved
bodies keep reaching them through the composed ``Simulation`` MRO, and
``tests/locks/test_preflight_split_snapshot.py``'s
``test_preflight_mixin_keeps_its_class_body_constants`` pins exactly those
three. The long SPEC-01 WP6 provenance comment stays with the two caps it
documents, for the same reason.

ONE module-level name moves with this leg: ``_local_cell``, re-exported by
``rfx/api/_preflight.py`` so its module namespace stays exactly as wide. The
split inventory filed it as SHARED between ``_validate_mesh_quality`` and
``_CampaignStaticsContext``, i.e. as a leaf for ``_common``. Measured, it is
not shared -- it is not even the same function. ``_CampaignStaticsContext.
entry_realizations`` imports a DIFFERENT ``_local_cell`` from
``rfx.geometry.rasterize_grid`` function-locally, with a different signature
(``(nodes, d, pos)`` against this one's ``(profile, lo, hi, fallback)``), and
that local binding shadows the module global for every read inside it. An AST
scope walk over the whole facade finds four loads of the name: one inside
``entry_realizations``'s nested ``_tie``, resolving to its local import, and
three inside ``_validate_mesh_quality``, resolving to the module global. So
the #743 helper is mesh-family-local and comes here rather than to
``_common``. Nothing outside this file imports it from the facade.
"""

from __future__ import annotations


def _local_cell(profile, lo, hi, fallback):
    """Coarsest cell a body spans on one axis (#743).

    ``profile`` is a per-cell size array whose cumulative sum gives node
    positions from the padded array's origin; ``lo``/``hi`` are the body's
    physical bounds. Returns ``fallback`` when there is no profile or the
    span selects no cell, so callers keep their previous behaviour on a
    uniform axis.
    """
    if profile is None:
        return fallback
    import numpy as _np
    d = _np.asarray(profile, dtype=float)
    edges = _np.concatenate([[0.0], _np.cumsum(d)])
    inside = (edges[1:] > min(lo, hi)) & (edges[:-1] < max(lo, hi))
    if not inside.any():
        return fallback
    return float(d[inside].max())
