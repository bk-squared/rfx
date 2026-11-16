"""Errors raised by the design-interop layer."""

from __future__ import annotations

from rfx.errors import RfxRefusal


class UnsupportedDesignFeature(RfxRefusal):
    """Raised when a design cannot be represented without silent loss.

    The interop layer refuses rather than approximating: a design description
    that quietly drops a shape parameter, a dispersion pole, or a mesh profile
    is worse than no description at all, because the resulting comparison looks
    valid while comparing a different structure.
    """


__all__ = ["UnsupportedDesignFeature"]


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

def _refuse(construct: str, reason: str) -> UnsupportedDesignFeature:
    """Refusal naming the construct first, then why it cannot be projected.

    Per decision D4 of the design note the emitter refuses rather than
    approximating, and per the task's refusal contract the message must name
    the construct so a reader knows what to remove or which target to use.
    """
    return UnsupportedDesignFeature(
        f"openEMS emitter cannot project {construct}: {reason}"
    )


def refuse_openems_floor(solver):
    """Refuse a pinned step that openEMS cannot project with its operands."""
    if solver.get("dt_pin") is not None:
        raise _refuse('conformal_pec="cut_cell_floor" / Simulation(dt=...)',
                      'openEMS export cannot carry the pinned step and cut-face area floor')
