"""Preflight validation split out of ``rfx.api._preflight`` (issue #980).

``rfx/api/_preflight.py`` was one 9 400-line module holding the module-level
report types and geometry leaves followed by the 7 800-line
``_PreflightMixin`` class that ``Simulation`` inherits. #980 Phase 3 took it
apart in verbatim code-motion steps, on the recipe the ``rfx/api/_sparams.py``
-> ``rfx/sparams/`` split just finished, each step gated on the committed
advisory-text snapshot in
``tests/locks/test_preflight_split_snapshot.py``.

Eight modules, and the motion is complete: every check family and the
realization layer they all read now live here, and the facade keeps the
EXECUTION family (``preflight``, ``preflight_sparameters``, the request
validators and routers, the ADI/x64/settling-witness configuration checks)
plus the re-export blocks and the class-body rebinds. Leg 8 then added a
ninth module that moves nothing: ``_registry``, which turns the composition
point itself into data.

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
* :mod:`rfx.preflight.ntff` — the near-to-far-field family: three
  ``_PreflightMixin`` bodies covering the #334 inverse-design umbrella
  (PEC overlap as an error, the lambda/4 near-field advisory) with the
  small-ground-plane check it calls, and the #500 NTFF-box-in-the-absorber
  check. Four at leg 6: the fourth was the minimum-steps hint, which emitted
  nothing and only wrote ``self._ntff_min_steps_hint``, and issue #1030
  deleted it once a census found no consumer for that attribute. It moves no
  module-level name.
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

* :mod:`rfx.preflight.realization` — the conductor-realization layer, which
  is not a check family at all: it emits nothing and is the DATA every other
  family reads under the #931 lattice ownership contract. Three module-level
  classes (:class:`~rfx.preflight.realization._RealizedPEC` for the whole
  model, :class:`~rfx.preflight.realization._EntryRealization` per
  declaration, :class:`~rfx.preflight.realization._CampaignStaticsContext`
  for the shared per-configuration state), the four numpy leaves only they
  read (``_shape_bounds``, ``_shift_back_np``, ``_wall_nodes_on_plane``,
  ``_realized_edges_np``) and the four ``_PreflightMixin`` accessors that
  hand them out — ``_assemble_realized``, ``_port_realized_edges``, its
  kept-name alias ``_port_pec_mask``, and the configuration-keyed cache
  ``_campaign_ctx``, which 64 of the 65 snapshot fixtures enter. Its two
  ``_common`` leaves, ``_sorted_box_corners`` and ``_component_is_dead``,
  stay there because each still has a reader in ``pec_geometry`` / ``ports``.

* :mod:`rfx.preflight._registry` — the composition point as DATA, and the one
  module here that moves no check body.
  :class:`~rfx.preflight._registry.ConfigCheckContext` carries the seven
  shared values ``_validate_simulation_config`` computes once per
  ``preflight()``; :class:`~rfx.preflight._registry.ConfigCheck` is one entry
  (method name, adapter, family); and
  :data:`~rfx.preflight._registry.CORE_CONFIG_CHECKS` holds the 36 of them in
  the exact order the hub used to call them, transcribed by an AST walk of
  that body rather than by hand.
  :func:`~rfx.preflight._registry.register_config_check` appends to
  :data:`~rfx.preflight._registry.EXTRA_CONFIG_CHECKS`, which is how a family
  module adds a check without the facade being edited at all. Extras run
  AFTER the core sequence, so registering one cannot move an existing
  advisory. It imports nothing from ``rfx.api`` and nothing from the family
  modules -- its adapters reach the bodies through the composed
  ``Simulation`` -- so it adds no edge to the import graph.

``_PreflightMixin`` itself STAYS in ``rfx/api/_preflight.py``: its
``_validate_simulation_config`` body was an ordered sequence of 37 calls, that
order IS the observable the snapshot lock pins, and leg 8 moved the sequence
into ``_registry`` without moving the composition point out of the facade or
changing the order. (Earlier revisions of this docstring said 38; the
thirty-eighth statement is ``_validate_cfg_compute_cpml_thickness``, which is
not a check -- it emits nothing and produces the context -- and it stays in
the hub.) ``rfx.api._preflight`` also re-exports every name moved here, so the
existing ``from rfx.api._preflight import <name>`` sites and the module-object
monkeypatches keep resolving.

This package deliberately imports nothing at package-import time: the leg
modules are leaves, and ``rfx/api/__init__.py`` stays the sole composition
point.
"""
