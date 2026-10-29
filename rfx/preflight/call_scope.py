"""Lifetime of diagnostic assembly products during public model calls."""

from contextlib import contextmanager
from functools import wraps
from weakref import WeakKeyDictionary


PUBLIC_CALLS = (
    "run", "forward", "preflight", "realized_geometry", "fidelity_report",
    "compute_s_matrix", "compute_msl_s_matrix", "compute_waveguide_s_matrix",
    "compute_coaxial_line_reflection", "compute_coaxial_two_port",
    "compute_coax_msl_transition", "compute_mixed_s_matrix",
    "estimate_ad_memory", "plan_ad_memory", "preflight_sparameters",
    "explain_ad_memory", "ad_memory_preflight", "mesh_intelligence_report",
    "plan_mesh", "artifact_report", "export_artifact_bundle",
)

# Copies of a Simulation must not inherit an open scope.
_depths = WeakKeyDictionary()


def scope_open(sim):
    """Whether this particular model has an open public call."""
    return _depths.get(sim, 0) > 0


def release(sim):
    """Release diagnostic products, including an early release during a solve."""
    sim._pf_campaign_ctx = None
    sim._realized_geometry_record = None
    sim._coax_geometry = None


@contextmanager
def call_scope(sim):
    """Share diagnostic products until the outermost call returns or raises."""
    _depths[sim] = _depths.get(sim, 0) + 1
    try:
        yield
    finally:
        depth = _depths[sim] - 1
        if depth:
            _depths[sim] = depth
        else:
            del _depths[sim]
            release(sim)


def public_call(fn):
    """Wrap a public function receiving a model, including keyword arguments."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        sim = next((arg for arg in (*args, *kwargs.values())
                    if hasattr(arg, "_campaign_ctx")), None)
        if sim is None:
            # Let the original function diagnose missing or invalid arguments.
            return fn(*args, **kwargs)
        with call_scope(sim):
            return fn(*args, **kwargs)

    wrapped._rfx_public_call = True
    return wrapped


def __init_subclass__(cls, **kwargs):
    """Bound by the preflight mixin; install wrappers only on concrete classes."""
    owner = next(base for base in cls.__mro__
                 if getattr(vars(base).get("__init_subclass__"), "__func__", None)
                 is __init_subclass__)
    super(owner, cls).__init_subclass__(**kwargs)
    for name in PUBLIC_CALLS:
        fn = getattr(cls, name, None)
        if fn is not None and not getattr(fn, "_rfx_public_call", False):
            wrapped = public_call(fn)
            wrapped.__qualname__ = f"{cls.__qualname__}.{name}"
            setattr(cls, name, wrapped)
