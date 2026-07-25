"""External-solver emitters — the *projection* profile of the design interop.

``rfx/interop/_design.py`` carries the round-trip-complete rfx→rfx document.
This subpackage carries the other profile named by decision **D3** of
``docs/design_notes/geometry_setup_interop.md``: a projection onto a foreign
solver's input language, **explicitly lossy**, shipping its own itemised list
of approximations.  The two are deliberately separate artifacts, because a
reader of a single "portable-ish" schema cannot tell which fields survived.

An emitter is text in, text out: it consumes a ``rfx-design-ir/v1`` document
and returns a plain-text script.  Generation requires no licence and no
solver — that is what makes a CST/HFSS emitter possible at all, and it is why
generation is testable in CI while *execution* is target-dependent.

openEMS is the first target because it is the only one whose generated script
can also be executed in this environment (``/usr/bin/openEMS``, v0.0.35), so
emitter output can be checked rather than reviewed.

Status: **provisional**.  See ``openems.py`` for the supported/refused fence.
"""

from __future__ import annotations

from rfx.interop.emitters.openems import (
    OPENEMS_EMITTER_VERSION,
    OpenEMSPlan,
    emit_openems_script,
    plan_openems_projection,
)

__all__ = [
    "OPENEMS_EMITTER_VERSION",
    "OpenEMSPlan",
    "emit_openems_script",
    "plan_openems_projection",
]
