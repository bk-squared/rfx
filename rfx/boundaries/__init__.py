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
    require_floquet_lateral_faces,
)
from rfx.boundaries.pec import (  # noqa: F401
    apply_pec,
    apply_pec_edges,
    apply_pec_faces,
)
from rfx.boundaries.pmc import (  # noqa: F401
    refuse_waveguide_pmc,
)

# The layer entries the five large files may call (scripts/ci/check_layer_structure.py rule I).
__all__ = [
    "apply_pec",
    "apply_pec_edges",
    "apply_pec_faces",
    "refresh_boundary_model",
    "refuse_waveguide_pmc",
    "require_floquet_lateral_faces",
]
