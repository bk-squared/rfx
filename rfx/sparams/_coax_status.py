"""Shared extraction status for coaxial calculators."""

import numpy as np


def coax_extraction_status(annulus_cells, recurrence_residual=None, *, traced: bool) -> str:
    """Classify geometry resolution and, on concrete paths, fit cleanliness.

    On a traced path the recurrence-residual check is not evaluated: a Python
    branch cannot depend on a tracer. ``"differentiable"`` marks a geometry-resolved
    AD result, not a fit-clean one; inspect ``recurrence_residual`` for contamination.
    """
    if annulus_cells < 3.5:
        return "under_resolved"
    if traced:
        return "differentiable"
    if float(np.max(recurrence_residual)) > 0.1:
        return "contaminated"
    return "passed"
