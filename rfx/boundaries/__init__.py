"""Boundary conditions: CPML, PEC, PMC."""

# Re-export bound here: later patches of the defining module do not replace it.
from rfx.boundaries.cpml import (  # noqa: F401
    apply_cpml_e,
    apply_cpml_h,
    CPMLParams,
    init_cpml,
)
from rfx.boundaries.model import (  # noqa: F401
    refresh_boundary_model,
)
from rfx.boundaries.pec import (  # noqa: F401
    apply_pec,
)

# The layer entries the five large files may call (scripts/ci/check_layer_structure.py rule I).
__all__ = [
    "refresh_boundary_model",
]
