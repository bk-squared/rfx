"""Boundary conditions: CPML, PEC, PMC."""

# Re-export bound here: later patches of the defining module do not replace it.
from rfx.boundaries.cpml import (  # noqa: F401
    apply_cpml_e,
    apply_cpml_h,
    CPMLParams,
    init_cpml,
)
from rfx.boundaries.depths import (  # noqa: F401
    adi_uniform_faces,
)
from rfx.boundaries.model import (  # noqa: F401
    Features,
    refresh_boundary_model,
    require_floquet_lateral_faces,
    resolve_kinds,
)
from rfx.boundaries.pec import (  # noqa: F401
    apply_pec,
    apply_pec_edges,
    apply_pec_faces,
)
from rfx.boundaries.pmc import (  # noqa: F401
    refuse_waveguide_pmc,
)
from rfx.boundaries.serialization import (  # noqa: F401
    legacy_spec,
)
from rfx.boundaries.spec import (  # noqa: F401
    normalize_boundary,
)

# The layer entries the five large files may call (scripts/ci/check_layer_structure.py rule I).
__all__ = [
    "adi_uniform_faces",
    "apply_pec",
    "apply_pec_edges",
    "apply_pec_faces",
    "Features",
    "legacy_spec",
    "normalize_boundary",
    "refresh_boundary_model",
    "refuse_waveguide_pmc",
    "require_floquet_lateral_faces",
    "resolve_kinds",
]
