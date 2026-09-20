"""S-matrix calculators split out of ``rfx.api._sparams`` (issue #980).

``rfx/api/_sparams.py`` was one 8 647-line module holding both the
module-level S-matrix helpers and the ``_SparamMixin`` class that
``Simulation`` inherits. #980 Phase 2 takes it apart in verbatim
code-motion steps, each gated on bit identity of the extracted S arrays.

Modules:

* :mod:`rfx.sparams._common` — the module-level helpers: MSL modal voltage
  and wave solve, the settling / passivity / reciprocity witnesses, the
  ringdown and geometry advisories, the mixed and coaxial power-wave
  assemblers, and the shared constants.
* :mod:`rfx.sparams.coax` — ``compute_coaxial_s_matrix``,
  ``compute_coaxial_line_reflection``, ``compute_coaxial_two_port`` and
  ``compute_coax_msl_transition``.
* :mod:`rfx.sparams.waveguide` — ``compute_waveguide_s_matrix`` and its
  non-uniform-mesh lane ``_compute_waveguide_s_matrix_nu``.
* :mod:`rfx.sparams.mixed` — ``compute_mixed_s_matrix``, the lumped/wire +
  MSL two-family driver of issue #488.
* :mod:`rfx.sparams.msl` — ``compute_msl_s_matrix``.
* :mod:`rfx.sparams.dispatch` — ``compute_s_matrix`` and ``s_matrix_lane``,
  the #980 Phase 1 unified entry point that reads the port registrations and
  forwards to exactly one of the calculators above (or names the run lane for
  lumped/wire ports). New surface, not moved code; it invents no defaults and
  refuses to guess where a registration does not determine a lane.

Every calculator is a module-level function whose first parameter is still
``self`` (the ``Simulation``); ``rfx.api._sparams`` binds each back onto
``_SparamMixin`` at its original position, so ``sim.compute_*`` keeps its
name, signature, docstring and bound-method behaviour.

``rfx.api._sparams`` still holds ``_SparamMixin`` and re-exports every name
moved here, so existing imports and string monkeypatches keep resolving.
"""
