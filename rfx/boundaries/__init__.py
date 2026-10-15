"""Boundary conditions: CPML, PEC, PMC."""

# Re-export bound here: later patches of the defining module do not replace it.
from rfx.boundaries.cpml import CPMLParams, init_cpml, apply_cpml_h, apply_cpml_e  # noqa: F401
from rfx.boundaries.pec import apply_pec  # noqa: F401
from rfx.boundaries.model import refresh_boundary_model  # noqa: F401

# The layer entries the five large files may call (scripts/ci/check_layer_structure.py rule I).
__all__ = ["refresh_boundary_model"]
