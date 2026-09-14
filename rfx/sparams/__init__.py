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

``rfx.api._sparams`` still holds ``_SparamMixin`` and re-exports every name
moved here, so existing imports and string monkeypatches keep resolving.
"""
