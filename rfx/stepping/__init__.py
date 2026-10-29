"""Time stepping: the fixed physical step order, and placement of its inputs."""

from rfx.stepping.waveguide_source import (
    apply_waveguide_port_e,
    apply_waveguide_port_h,
    inject_waveguide_port,
)
from .graded import (
    GradedFrame,
    GradedStepContext,
    make_graded_step,
)
from .graded_loop import (
    run_nonuniform,
    run_nonuniform_until_decay,
)
from .progress_scan import (
    scan_with_progress,
)
from .remat import (
    plan_checkpoint_lengths,
    segmented_scan,
)
from .scan import (
    recorded_scan,
)
from .sequence import (
    compose,
    HookPoint,
    Kernel,
    STEP_SEQUENCE,
)
from .slab import (
    cut,
    FILL,
    Fill,
    Slab,
)
from .step_count import (
    steps_for_periods,
)
from .uniform_loop import (
    run_uniform_loop,
    run_uniform_until_decay,
)
from .uniform_setup import (
    core_step_invariants,
    make_core_step,
)
from .yee import (
    update_e,
    update_e_aniso,
    update_e_aniso_inv,
    update_e_box,
    update_e_fast,
    update_e_nu,
    update_e_nu_aniso,
    update_h,
    update_h_fast,
    update_h_nu,
)

__all__ = [
    "apply_waveguide_port_e",
    "apply_waveguide_port_h",
    "compose",
    "core_step_invariants",
    "cut",
    "FILL",
    "Fill",
    "GradedFrame",
    "GradedStepContext",
    "HookPoint",
    "inject_waveguide_port",
    "Kernel",
    "make_core_step",
    "make_graded_step",
    "plan_checkpoint_lengths",
    "recorded_scan",
    "run_nonuniform",
    "run_nonuniform_until_decay",
    "run_uniform_loop",
    "run_uniform_until_decay",
    "scan_with_progress",
    "segmented_scan",
    "Slab",
    "STEP_SEQUENCE",
    "steps_for_periods",
    "update_e",
    "update_e_aniso",
    "update_e_aniso_inv",
    "update_e_box",
    "update_e_fast",
    "update_e_nu",
    "update_e_nu_aniso",
    "update_h",
    "update_h_fast",
    "update_h_nu",
]
