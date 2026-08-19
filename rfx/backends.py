"""Internal helpers for classifying JAX compute backends.

JAX backend names are supplied by plugins and are not guaranteed to use a
consistent case.  Keep all platform comparisons in this module so a plugin
reporting ``"METAL"`` behaves the same as one reporting ``"metal"``.
"""

from __future__ import annotations

from typing import Any


_GPU_PLATFORMS = frozenset({"gpu", "cuda", "rocm", "metal"})
_ACCELERATOR_PLATFORMS = _GPU_PLATFORMS | frozenset({"tpu"})

# The baked-PEC kernel is a CUDA/ROCm optimisation.  In particular, a Metal
# device must use the generic update until that kernel has independent
# correctness and performance coverage on Apple's JAX plugin.
_BAKED_PEC_FAST_PATH_PLATFORMS = frozenset({"gpu", "cuda", "rocm"})


def normalize_platform(value: Any) -> str:
    """Return a case-normalized backend name for a string or JAX device.

    ``jax.default_backend()`` returns a string while callers inspecting
    ``jax.devices()`` naturally have device objects.  Accepting either keeps
    their classification identical.  Missing and blank values are represented
    as ``"unknown"`` rather than leaking ``"None"`` into diagnostics.
    """
    platform = getattr(value, "platform", value)
    if platform is None:
        return "unknown"
    normalized = str(platform).strip().lower()
    return normalized or "unknown"


def current_backend() -> str:
    """Return the active JAX backend using normalized platform spelling."""
    import jax

    return normalize_platform(jax.default_backend())


def is_metal_backend() -> bool:
    """Whether the active JAX backend is Apple's experimental Metal plugin."""
    return current_backend() == "metal"


def supports_baked_pec_fast_path(platform: Any | None = None) -> bool:
    """Whether *platform* is approved for the CUDA/ROCm baked-PEC kernel.

    When omitted, the active JAX backend is inspected.  The allowlist is
    deliberately narrow: unknown accelerators, TPUs, and Metal all stay on
    the portable Yee update path.
    """
    backend = (
        current_backend()
        if platform is None
        else normalize_platform(platform)
    )
    return backend in _BAKED_PEC_FAST_PATH_PLATFORMS


def is_gpu_platform(platform: Any) -> bool:
    """Whether *platform* denotes a CUDA, ROCm, or Metal GPU backend."""
    return normalize_platform(platform) in _GPU_PLATFORMS


def is_accelerator_platform(platform: Any) -> bool:
    """Whether *platform* denotes a known non-CPU accelerator backend."""
    return normalize_platform(platform) in _ACCELERATOR_PLATFORMS


def reject_unsupported_metal(context: str, feature: str) -> None:
    """Raise an actionable error when an unverified path enters on Metal."""
    if is_metal_backend():
        raise NotImplementedError(
            f"{context} is unavailable on rfx's experimental Apple Metal "
            f"research backend. {feature} is outside the verified uniform, "
            "single-device, real-float32 forward time-domain lane. Start a "
            "fresh CPU process with JAX_PLATFORMS=cpu for this workload."
        )


__all__ = [
    "normalize_platform",
    "current_backend",
    "is_metal_backend",
    "supports_baked_pec_fast_path",
    "is_gpu_platform",
    "is_accelerator_platform",
    "reject_unsupported_metal",
]
