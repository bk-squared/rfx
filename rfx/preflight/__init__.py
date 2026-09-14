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
