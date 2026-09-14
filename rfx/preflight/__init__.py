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
* :mod:`rfx.preflight.waveguide` — the waveguide-port family: the three
  module-level leaves
  (:func:`~rfx.preflight.waveguide._waveguide_skipped_note`,
  :data:`~rfx.preflight.waveguide.WAVEGUIDE_DEFAULT_NUM_PERIODS`,
  :func:`~rfx.preflight.waveguide.resolve_waveguide_port_freqs`) and the
  twelve ``_PreflightMixin`` bodies that make up the realized-aperture and
  cutoff checks (#150 / #737 / #738), the P2.8 reference-plane check, and
  the three post-v1.8 S-parameter setup audits with their shared builder.
* :mod:`rfx.preflight.absorber` — the absorber/boundary-configuration
  family: eleven ``_PreflightMixin`` bodies covering what the absorber is
  made of (the #636 dispersive-pole advisory, the lossless-Q
  anti-pattern), how thick it is (per-face CPML thickness, the #647/#742
  allocation-budget advisory and the shared
  :func:`~rfx.preflight.absorber._preflight_face_layers`), which boundary
  combinations are refused (``pec_faces`` vs finite PEC, UPML vs
  refinement, UPML vs a mesh profile) and what stands inside the pad
  (probe/source placement, geometry extending into it, and the P0.4
  PEC-boundary-on-an-open-structure advisory). It is the only leg so far
  that moves NO module-level name: its leaves went to ``_common`` in leg 0
  because their readers were never confined to this family.

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
