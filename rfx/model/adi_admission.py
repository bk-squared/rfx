from __future__ import annotations

from rfx.errors import RfxUnsupported

from rfx.core.jax_utils import is_tracer


ADI_INTERIOR_PEC_MESSAGE = (
    "adi_interior_pec_unsupported: solver='adi' cannot safely carry interior "
    "PEC sheets, wires, or volumes in 3D or 2D TMz. The current internal "
    "PEC projection has measured growing solutions, including at "
    "adi_cfl_factor=1 and the former default 5; no general stable factor is "
    "established. Use solver='yee' with the declared conductors retained. "
    "Lowering or clamping adi_cfl_factor is not a supported remedy. "
    "Domain-boundary PEC without interior PEC remains supported."
)


def _validate_interior_pec(mask):
    """Refuse supplied PEC masks before a solve or JAX trace begins.

    Structural on purpose: masks may be tracers, and a contents-based host
    test would fail under jit/grad. Even an explicitly supplied all-false
    mask is unsupported; use None for the no-interior-conductor control.
    Numerical kernels below remain intact for controlled research.
    """
    if mask is not None:
        raise ValueError(ADI_INTERIOR_PEC_MESSAGE)


ADI_INTERFACE_MESSAGE = (
    "adi_material_interface_unsupported: solver='adi' refuses a dielectric "
    "or conductivity interface. The ADI update gives each E "
    "component its own cell's eps_r and sigma, where every Yee lane gives "
    "each E edge the mean of its four cells, so a material face sits half a "
    "cell from where the Yee lanes put it. Measured on a slab-loaded PEC "
    "cavity at 12 cells per loaded wavelength: the first resonance is 2.2 % "
    "above the uniform Yee lane's (-0.2 % with no slab). A homogeneous fill "
    "is carried. Use solver='yee'; see docs/guides/known_limitations.md."
)


def _concrete_varies(values) -> bool:
    """Whether a concrete array takes more than one value over its cells; a
    traced array cannot be read and is reported as not varying.

    The last index on each axis is the node layer past the last cell: the
    outer wall (both ADI boundary modes end in a PEC wall), where the E
    components that read it are tangential and held at zero or lie outside
    the grid. A domain-filling Box leaves it vacuum on a PEC box, so it is
    not read here; an axis one cell thick (2-D) is read whole.
    """
    if values is None or is_tracer(values):
        return False
    import numpy as np
    arr = np.asarray(values)
    arr = arr[tuple(slice(0, n - 1) if n > 1 else slice(None) for n in arr.shape)]
    return arr.size > 0 and bool(np.any(arr != arr.flat[0]))


def adi_material_interface_refusal(eps_r, sigma=None):
    """Why the ADI lane cannot solve these materials, or ``None`` (#1373).

    ``eps_r`` and ``sigma`` are the material arrays the ADI coefficients read
    per cell (:func:`_adi_coeffs_eps_r`). A value that varies over the grid is
    a material interface, which the ADI lane puts half a cell from the Yee
    lanes' edge mean (:func:`rfx.core.yee.edge_averaged_materials`). Only
    concrete arrays are read: a traced array (``forward(eps_override=...)``
    under ``jax.grad``) is not inspected. ``sigma`` must be the material
    conductivity, before an absorber's grading is added to it.
    """
    if _concrete_varies(eps_r) or _concrete_varies(sigma):
        return ADI_INTERFACE_MESSAGE
    return None


ADI_TRACED_OVERRIDE_MESSAGE = (
    "adi_traced_material_array_unsupported: solver='adi' refuses a traced "
    "eps_r/sigma array (from add_material with a traced value, or an "
    "eps_override / sigma_override array). A traced array cannot be "
    "read before the solve, and the ADI lane puts any interface it holds "
    "half a cell from where the Yee lanes put it (+2.2 % on a slab-loaded "
    "PEC cavity's first resonance at 12 cells per loaded wavelength). The "
    "supported homogeneous form is a scalar (0-d) eps_override / "
    "sigma_override; otherwise use solver='yee'. See "
    "docs/guides/known_limitations.md."
)


def adi_traced_override_refusal(values):
    """Why a traced material array cannot reach the ADI lane, or ``None``.

    A scalar (0-d) traced value is a homogeneous fill and is admitted; a
    traced array may hold an interface that cannot be inspected (#1373)."""
    if values is not None and is_tracer(values) and getattr(values, "ndim", 0) > 0:
        return ADI_TRACED_OVERRIDE_MESSAGE
    return None


def _refuse_material_interface(eps_r, sigma=None):
    reason = adi_material_interface_refusal(eps_r, sigma)
    if reason is not None:
        raise RfxUnsupported(reason)
