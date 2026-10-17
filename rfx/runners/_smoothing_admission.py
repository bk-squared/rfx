"""Forward smoothing option conflicts."""
from rfx.model.smoothing_admission import admit_full_tensor


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
    if sim._solver == "adi":
        sim._refuse_unsupported_run_kwargs("ADI (solver='adi')", {
            "subpixel_smoothing": subpixel_smoothing},
            instead="use the default solver='yee'")
    if sim._uses_nonuniform_mesh and subpixel_smoothing == "kottke_pec":
        raise NotImplementedError(
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
    """Refuse full tensors before the graded diagonal builder is reached."""
    from rfx.runners._admission import refuse_smoothing_overrides
    admit_full_tensor(sim, options['subpixel_smoothing'], unsupported=True)
    refuse_smoothing_overrides(**options)
