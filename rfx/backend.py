"""Backend classification shared by execution, reporting, and memory planning."""

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class BackendClass:
    """Static backend policy; optional memory stats still require a device query."""

    name: str
    accelerator: bool
    cpml_writeback: str
    device_memory: str
    host_callbacks: bool


def backend_class(platform: str | None = None) -> BackendClass:
    """Classify a platform without initializing JAX when one is supplied.

    ``memory_stats_if_available`` means query the device, then report an
    unknown limit if it supplies no usable value. MPS retains slab CPML.
    """
    name = jax.default_backend() if platform is None else platform
    if name in ("gpu", "cuda", "rocm"):
        return BackendClass(name, True, "padded", "memory_stats", True)
    if name == "mps":
        return BackendClass(name, True, "slab", "memory_stats_if_available", False)
    return BackendClass(name, False, "slab", "none" if name == "cpu" else "unknown", name == "cpu")
