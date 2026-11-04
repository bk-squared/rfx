"""Admission checks shared by the coaxial S-parameter calculators."""

from __future__ import annotations


def _require_absorption_on_every_axis(sim, cpml_axes: str | None, lane: str) -> None:
    """Both coax line lanes absorb on x, y and z.

    With absorbers on z only, the lateral pads were vacuum backed by the PEC
    grid faces: a closed metal can around the line. Every unshielded
    conductor end — the open termination, and the pin and shell ending one
    cell past the matched feed — lets field out of the annulus, and the can
    held it between the shell and its walls; the open end coupled it back,
    so the open's |S11| rose above 1 and grew with the record (1.018 at 12
    line traversals, 1.046 at 24, at 9 annulus cells). With the lateral pads
    absorbing, the same open reads at most 1.0015 and does not move with the
    record (both in rfx-archive, rfx/records/20260924-coax-closed-can/
    open_closed_can_arms.json).
    A face without an absorber is a PEC wall and closes the can again, so
    every face needs positive CPML thickness.
    """
    from rfx.boundaries.depths import has_positive_cpml_faces
    if not has_positive_cpml_faces(sim._boundary_spec, sim._cpml_layers):
        raise ValueError(
            f"{lane}() requires positive CPML thickness on all six faces: "
            "it absorbs on all three axes, and a face without an absorber "
            "is a PEC wall that closes a metal can around the line" "."
        )
    if cpml_axes is not None and set(cpml_axes) != set("xyz"):
        raise ValueError(
            f"{lane}() absorbs on all three axes and accepts only "
            f"cpml_axes='xyz' (got {cpml_axes!r}). With absorbers on z only "
            "the lateral pads are vacuum backed by the PEC grid faces, a "
            "closed metal can that holds the field leaving an unshielded "
            "conductor end; the open termination's |S11| then rose above 1 "
            "and grew with the record."
        )
