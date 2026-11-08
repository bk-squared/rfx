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


def admit_full_tensor(sim, value, *, distributed=False, devices=None, unsupported=False):
    """Refuse a consumer that cannot carry every node-tensor coupling.

    A traced permittivity is refused: under ``jax.jit`` / ``jax.grad``
    the node-tensor build classifies cell fractions in float32, so a face on
    grid nodes is solved as a different model than the concrete float64
    build. A traced shape parameter is not refused here.
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
