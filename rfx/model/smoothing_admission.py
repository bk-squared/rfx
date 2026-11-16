"""Admission for the single-device full permittivity tensor."""

from rfx.errors import RfxUnsupported

FULL_TENSOR_MESSAGE = (
    "subpixel_smoothing='full_tensor' is supported on a single-device Yee mesh only; "
    "True gives the diagonal (first-order) smoothing on this path"
)


def _traced_permittivities(sim):
    """Names of declared materials, used by a geometry entry, whose eps_r is a JAX tracer."""
    from rfx.core.jax_utils import is_tracer
    used = {entry.material_name for entry in getattr(sim, "_geometry", ())}
    return [name for name, spec in getattr(sim, "_materials", {}).items()
            if name in used and is_tracer(spec.eps_r)]


def _traced_coordinates_beside_box(sim):
    """Materials of dielectric entries with a traced shape coordinate, when a dielectric Box is present."""
    from rfx.core.jax_utils import is_tracer
    from rfx.geometry.box_content import is_exact_box
    from rfx.geometry.smoothing import _shape_parameters
    from rfx.model.materials import assembled_as_pec
    entries = [e for e in getattr(sim, "_geometry", ()) if not assembled_as_pec(sim, e)]
    if not any(is_exact_box(e.shape) for e in entries):
        return []
    return list(dict.fromkeys(e.material_name for e in entries
                              if any(is_tracer(p) for p in _shape_parameters(e.shape))))


def admit_full_tensor(sim, value, *, distributed=False, devices=None, unsupported=False):
    """Refuse a consumer that cannot carry every node-tensor coupling.

    A traced permittivity is refused: under ``jax.jit`` / ``jax.grad``
    the node-tensor build classifies cell fractions in float32, so a face on
    grid nodes is solved as a different model than the concrete float64
    build. A traced shape coordinate is refused when a dielectric Box is in
    the scene: the exact box content is scoped on concrete coordinates.
    """
    if value != "full_tensor":
        return
    if (unsupported or distributed or (devices is not None and len(devices) > 1)
            or sim._solver != "yee"
            or sim._refinement is not None or sim._boundary == "upml"):
        raise NotImplementedError(FULL_TENSOR_MESSAGE)
    traced = _traced_permittivities(sim)
    if traced:
        raise RfxUnsupported(
            "subpixel_smoothing='full_tensor' with a traced permittivity (material(s) "
            + ", ".join(repr(n) for n in traced) + ") is refused: the full-tensor "
            "build under jax.jit / jax.grad classifies grid-aligned faces in float32 and "
            "solves a different model from the concrete run. Use subpixel_smoothing=True "
            "or 'kottke_pec' for a gradient over eps_r; see docs/guides/known_limitations.md, "
            "full-tensor smoothing at box edges.")
    traced = _traced_coordinates_beside_box(sim)
    if traced:
        from rfx.geometry.box_content import traced_coordinate_message
        raise RfxUnsupported(traced_coordinate_message(traced))
