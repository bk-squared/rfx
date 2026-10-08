"""Shared admission and diagnostic wording for #1465."""

SUBGRID_WARNING = (
    "The subgridded lane is unstable and unverified: an empty lossless PEC "
    "cavity's first resonance is 8 % high at a 1.5 mm coarse mesh and its "
    "field grows exponentially (#1465). Use a graded mesh (dx/dy/dz profiles) "
    "for local resolution."
)

SUBGRID_NOTICE = (
    SUBGRID_WARNING + ' Or pass validation="research" to run it anyway.'
)


class ExperimentalSubgridWarning(UserWarning):
    """An explicit opt-in to the unstable, unverified subgridded lane."""


def require_experimental(sim, *, diagnostics=None):
    """Use lane admission to refuse production before entering the runner."""
    # Only a declared refinement selects the unstable lane; a direct runner
    # call without one is left to that runner's own refusals.
    refinement = getattr(sim, "_refinement", None)
    if refinement is not None and refinement.get("validation", "production") == "production":
        from rfx.runners._admission import admit
        admit(
            sim,
            "run_subgridded",
            diagnostics=diagnostics,
        )


def warn_experimental():
    import warnings
    import sys
    # Attribute both public and direct/internal entry calls to the first
    # caller outside rfx, including run() wrappers and S-matrix replays.
    stacklevel = 2
    frame = sys._getframe(1)
    try:
        while frame is not None and (
            frame.f_globals.get("__name__", "") == "rfx"
            or frame.f_globals.get("__name__", "").startswith("rfx.")
        ):
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(SUBGRID_WARNING, ExperimentalSubgridWarning, stacklevel=stacklevel)
