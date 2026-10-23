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
from .remat import (
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
    "recorded_scan",
    "run_nonuniform",
    "run_nonuniform_until_decay",
    "run_uniform_loop",
    "run_uniform_until_decay",
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

# rfx.measurement → rfx.model → rfx.nonuniform → rfx.stepping loads this package
# before the uniform and graded runners are ready (rfx#1818):
# https://github.com/bk-squared/rfx/issues/1818
# Remove this lazy export when that import chain is removed.
_LAZY_EXPORTS = {
    "core_step_invariants": "rfx.stepping.uniform_setup",
    "make_core_step": "rfx.stepping.uniform_setup",
    "run_nonuniform": "rfx.stepping.graded_loop",
    "run_nonuniform_until_decay": "rfx.stepping.graded_loop",
    "run_uniform_loop": "rfx.stepping.uniform_loop",
    "run_uniform_until_decay": "rfx.stepping.uniform_loop",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
