"""Smoothing option conflicts and declaration-level advisories."""

import warnings
from weakref import WeakSet

from rfx._warning_origin import user_stacklevel
from rfx.core.jax_utils import is_tracer
from rfx.errors import ConfigurationWarning, RfxUnsupported
from rfx.geometry.csg import Cylinder, Sphere
from rfx.model.smoothing_admission import admit_full_tensor

_warned_kottke_pec = WeakSet()


def warn_kottke_pec_volume(sim, subpixel_smoothing):
    """Warn once per declaration, on the host even under an outer JAX trace."""
    if subpixel_smoothing != "kottke_pec" or sim in _warned_kottke_pec:
        return
    from rfx.geometry.rasterize_grid import _box_zero_axes
    from rfx.model.materials import assembled_as_pec

    message = None
    for entry in sim._geometry:
        if not assembled_as_pec(sim, entry):
            continue
        shape = entry.shape
        lo, hi = getattr(shape, "corner_lo", None), getattr(shape, "corner_hi", None)
        if lo is not None and hi is not None and _box_zero_axes(lo, hi):
            continue  # A zero-thickness Box declares a sheet, not a volume.
        if getattr(shape, "points", None) is not None and shape.radius == 0:
            continue  # A zero-radius PolylineWire declares a filament.
        message = (
            "subpixel_smoothing='kottke_pec' with a PEC volume is "
            "experimental, not recommended for metal; see known limitations, "
            '"A metal face resting on a dielectric between grid nodes"'
        )
        # The first closed curved PEC body takes precedence over other PEC volumes.
        if isinstance(shape, Sphere) or (
            isinstance(shape, Cylinder)
            and (is_tracer(shape.height) or float(shape.height) > 0)
            and (is_tracer(shape.radius) or float(shape.radius) > 0)
        ):
            kind = "Sphere" if isinstance(shape, Sphere) else "Cylinder"
            message = (
                f"subpixel_smoothing='kottke_pec' on the closed curved PEC {kind} "
                f"'{entry.material_name}': on a PEC sphere at ka 6 with cells of λ/20 "
                "the extinction missed Mie by 20–51 % (the staircase by 13–14 %), "
                "and a late field of 3–5 % of the peak stayed after the pulse. "
                "Use subpixel_smoothing=False (the staircase), or judge the result "
                "over three refinements; see known limitations, "
                '"Subpixel smoothing with "kottke_pec" leaves a late field around '
                'a closed curved metal body".'
            )
            break
    if message is not None:
        warnings.warn(message, ConfigurationWarning, stacklevel=user_stacklevel())
        _warned_kottke_pec.add(sim)


def refuse_forward_smoothing(sim, *, subpixel_smoothing,
                             eps_override, pec_occupancy, design_occupancy):
    """Admit smoothing only on forward lanes that carry the tensor."""
    from rfx.runners._admission import refuse_smoothing_overrides
    admit_full_tensor(sim, subpixel_smoothing)
    from rfx.model.tensor_operands import refuse_full_tensor_conformal
    faces = sim._boundary_spec.conformal_faces()
    refuse_full_tensor_conformal(subpixel_smoothing, bool(faces), faces)
    if not subpixel_smoothing:
        return
    if sim._uses_nonuniform_mesh and subpixel_smoothing == "kottke_pec":
        raise RfxUnsupported(
            "subpixel_smoothing='kottke_pec' (Stage 2 unified PEC) "
            "is not supported on the non-uniform mesh path. "
            "Drop the dx/dy/dz profile or use the legacy "
            "``Boundary(conformal=True)`` Stage 1 path on a "
            "uniform mesh. Tracked at "
            "https://github.com/bk-squared/rfx/issues.")
    refuse_smoothing_overrides(subpixel_smoothing=subpixel_smoothing,
        eps_override=eps_override, pec_occupancy=pec_occupancy,
        design_occupancy=design_occupancy)


def refuse_nonuniform_smoothing(sim, **options):
    """Apply the shared tensor admission and material-override fences."""
    from rfx.runners._admission import refuse_smoothing_overrides
    admit_full_tensor(sim, options['subpixel_smoothing'])
    refuse_smoothing_overrides(**options)
