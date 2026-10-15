"""Single-device order; rfx issue 1373 tracks the one order for all lanes."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import TypeVar


class HookPoint(str, Enum):
    """Attachment times relative to the physical field updates and drives."""

    AFTER_H = "after_h"
    AFTER_E_UPDATE = "after_e_update"
    BEFORE_SOURCES = "before_sources"
    AFTER_SOURCES = "after_sources"
    STEP_END = "step_end"


class Kernel(str, Enum):
    """Lane operations: curls, boundary corrections, constitutive laws, drives."""

    HE_UPDATE = "he_update"
    H_UPDATE = "h_update"
    H_BOUNDARY = "h_boundary"
    E_UPDATE = "e_update"
    E_BOUNDARY = "e_boundary"
    CONSTITUTIVE = "constitutive"
    SOURCES = "sources"
    PEC = "pec"


# H reaches n+1/2 before E reaches n+1. Incident-field and absorber
# corrections belong to the corresponding lane boundary operation.
# Constitutive sheets/RLC consume the E update before electric drives.
# Conductors precede sheets/RLC and sources in today's single-device order.
STEP_SEQUENCE: tuple[Kernel | HookPoint, ...] = (
    Kernel.H_UPDATE,
    Kernel.H_BOUNDARY,
    HookPoint.AFTER_H,
    Kernel.E_UPDATE,
    HookPoint.AFTER_E_UPDATE,
    Kernel.E_BOUNDARY,
    Kernel.PEC,
    Kernel.CONSTITUTIVE,
    HookPoint.BEFORE_SOURCES,
    Kernel.SOURCES,
    HookPoint.AFTER_SOURCES,
    HookPoint.STEP_END,
)

Frame = TypeVar("Frame")


def compose(
    kernels: Mapping[Kernel, Callable[[Frame], None]],
    attachments: Mapping[HookPoint, tuple[Callable[[Frame], None], ...]],
) -> Callable[[Frame], Frame]:
    """Compose lane kernels and ordered attachments into one physical step.

    A frame is transient Python bookkeeping holding array values. Operations
    update its attributes while tracing; only the resulting arrays enter the
    compiled program. Missing kernels and empty hooks add no operation, branch,
    array, JAX primitive or compilation boundary. Attachments at a hook run in
    their supplied order. Preparation and carry assembly remain lane-specific.
    """
    sequence = STEP_SEQUENCE
    if Kernel.HE_UPDATE in kernels:
        # The baked-PEC kernel spans both curls; neither internal hook exists.
        for point in (HookPoint.AFTER_H, HookPoint.AFTER_E_UPDATE):
            if attachments.get(point):
                raise ValueError(f"combined H/E fast path requires empty {point.value} attachments")
        covered = (Kernel.H_UPDATE, Kernel.H_BOUNDARY, Kernel.E_UPDATE)
        if any(phase in kernels for phase in covered):
            raise ValueError("combined H/E fast path cannot also supply separate H/E kernels")
        end = STEP_SEQUENCE.index(HookPoint.AFTER_E_UPDATE) + 1
        sequence = (Kernel.HE_UPDATE,) + STEP_SEQUENCE[end:]
    operations = tuple(
        operation
        for phase in sequence
        for operation in (
            attachments.get(phase, ()) if isinstance(phase, HookPoint)
            else (() if phase not in kernels else (kernels[phase],))
        )
    )

    def step(frame: Frame) -> Frame:
        for operation in operations:
            operation(frame)
        return frame

    return step
