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
* :mod:`rfx.preflight.mesh` — the mesh-quality and non-uniform-lane
  family: the #743 coarsest-cell leaf
  (:func:`~rfx.preflight.mesh._local_cell`) and the twelve
  ``_PreflightMixin`` bodies covering resolution and numerical dispersion,
  the realized metal plane on a graded axis, the SPEC-01 WP6 multi-band
  grading envelope, the three #669/#672/#688 graded-node advisories with
  their shared ``_graded_node_report``, graded Box rasterization, and the
  features the non-uniform and SBP-SAT subgridded lanes refuse. Its three
  class-BODY constants (``_MULTIBAND_RATIO_CAP``, ``_INPLANE_RATIO_CAP``,
  ``_AXIS_OF_COMPONENT``) deliberately stay on ``_PreflightMixin``.
* :mod:`rfx.preflight.sources` — the source-configuration family: four
  ``_PreflightMixin`` bodies covering the #471 TFSF vacuum-boundary lane
  guard (the split's one remaining ``@staticmethod``, re-wrapped by the
  facade), the P1.6 source-on-a-PEC/PMC-face check, the P0.5 no-sources
  guard and the #386 unresolved-pulse advisory. It moves no module-level
  name.
* :mod:`rfx.preflight.ports` — the lumped/wire-port and coaxial-port
  family: six ``_PreflightMixin`` bodies covering the #589 coax
  junction-aperture short, the #425 TFSF-plus-lumped-RLC refusal, the two
  #313 reference-plane advisories, the #929/#931 port-frozen-by-realized-PEC
  check with its wire-port cell-centre helper, and the #71 floating
  single-cell port. Its one module-level leaf, ``_component_is_dead``,
  went to ``_common`` instead, because ``_RealizedPEC`` still reads it.
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
