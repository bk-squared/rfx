"""Specific periodic refusals of the single-device graded runner."""
import numpy as np

from rfx.core.jax_utils import is_tracer


def admit_graded_periodic(sim):
    """Keep unsupported periodic declarations from reaching a time step."""
    axes = sim._periodic_axes or ""
    if not axes:
        return
    if sim._interface_eps == "dual_average":
        raise NotImplementedError(
            f"interface_eps='dual_average' with graded periodic axes {axes!r} "
            "is not implemented; use the default interface_eps on a periodic axis")
    if sim._tfsf is not None:
        raise NotImplementedError(
            f"A plane wave with periodic axes {axes!r} on the graded mesh "
            "is not implemented; use the uniform mesh")
    for axis in axes:
        profile = getattr(sim, f"_d{axis}_profile")
        if profile is not None and (is_tracer(profile) or not np.all(
                np.asarray(profile) == np.asarray(profile)[0])):
            raise NotImplementedError(
                f"Graded periodic axis {axis!r} requires a constant profile; "
                "use a constant spacing along a periodic axis")
