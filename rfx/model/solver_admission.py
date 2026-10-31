"""Declare the supported solver before realizing a mesh."""

ADI_REMOVED = (
    'ADI was removed in 2.1: its unmatched absorber, half-cell interface error '
    'and unstable interior PEC outweigh the median 1.22x time-step gain. '
    'Use solver="yee" on a graded mesh (dz_profile, dx_profile, dy_profile) '
    'for thin layers, or local refinement where it is supported.'
)


def admit_solver(solver, adi_cfl_factor=None):
    """Refuse removed declarations before any grid is built."""
    if adi_cfl_factor is not None:
        raise ValueError(ADI_REMOVED)
    if solver not in {"yee"}:
        if solver == "adi":
            raise ValueError(ADI_REMOVED)
        raise ValueError(f"solver must be 'yee' or 'adi', got {solver!r}")
