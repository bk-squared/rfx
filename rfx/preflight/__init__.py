"""Preflight validation split out of ``rfx.api._preflight`` (issue #980).

``rfx/api/_preflight.py`` was one 9 400-line module holding the module-level
report types and geometry leaves followed by the 7 800-line
``_PreflightMixin`` class that ``Simulation`` inherits. #980 Phase 3 takes it
apart in verbatim code-motion steps, on the recipe the ``rfx/api/_sparams.py``
-> ``rfx/sparams/`` split just finished, each step gated on the committed
advisory-text snapshot in
``tests/locks/test_preflight_split_snapshot.py``.

Modules:

* :mod:`rfx.preflight._common` — the report/result types
  (:class:`~rfx.preflight._common.PreflightWarning`,
  :class:`~rfx.preflight._common.PreflightErrorWarning`,
  :class:`~rfx.preflight._common.PreflightConfigError`,
  :class:`~rfx.preflight._common.PreflightIssue`,
  :class:`~rfx.preflight._common.PreflightReport`), the unit-adaptive
  formatting leaves, and the absorber membership/proximity leaves.
* :mod:`rfx.preflight.msl` — the MSL port-geometry family: the
  probe-clearance geometry helpers
  (:func:`~rfx.preflight.msl.msl_min_probe_clearance`,
  :func:`~rfx.preflight.msl.msl_source_near_field_standoff_cells`,
  :func:`~rfx.preflight.msl.msl_nearest_downstream_reflector`,
  :func:`~rfx.preflight.msl.msl_probe_clearance_for_port`,
  :func:`~rfx.preflight.msl.msl_absorber_compliant_offset_max`), their
  constants, and the five ``_PreflightMixin`` MSL check bodies.
* :mod:`rfx.preflight.pec_geometry` — the conductor-realization family: the
  five issue-#703 gate constants and the ten ``_PreflightMixin`` check
  bodies that read them (the #703 campaign-statics umbrella and its four
  checks, the #931 per-declaration realization findings, the #669
  Leontovich advisories and the conformal-fine-dx guard).

``_PreflightMixin`` itself STAYS in ``rfx/api/_preflight.py``: its
``_validate_simulation_config`` body is an ordered sequence of 38 calls and
that order IS the observable the snapshot lock pins, so the composition point
does not move. ``rfx.api._preflight`` also re-exports every name moved here,
so the existing ``from rfx.api._preflight import <name>`` sites and the
module-object monkeypatches keep resolving.

This package deliberately imports nothing at package-import time: the leg
modules are leaves, and ``rfx/api/__init__.py`` stays the sole composition
point.
"""
