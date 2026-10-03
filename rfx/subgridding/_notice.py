"""Shared admission and diagnostic wording for #1465."""

SUBGRID_NOTICE = (
    "The subgridded lane is unstable and unverified: an empty lossless PEC "
    "cavity's first resonance is 8 % high at a 1.5 mm coarse mesh and its "
    "field grows exponentially (#1465). Use a graded mesh (dx/dy/dz profiles) "
    "for local resolution, or pass validation=\"research\" to run it anyway."
)


class ExperimentalSubgridWarning(UserWarning):
    """An explicit opt-in to the unstable, unverified subgridded lane."""


def require_experimental(sim):
    """Use lane admission to refuse production before entering the runner."""
    if sim._refinement.get("validation", "production") == "production":
        from rfx.runners._admission import admit
        admit(sim, "run_subgridded")


def warn_experimental():
    import warnings
    warnings.warn(SUBGRID_NOTICE, ExperimentalSubgridWarning, stacklevel=3)
