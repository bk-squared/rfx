"""Keep concrete pulse samples independent of an enclosing user trace."""

from contextlib import nullcontext
from dataclasses import fields

import jax

from rfx.core.jax_utils import is_tracer


def concrete_waveform_eval(waveform, *time_parameters):
    """Evaluate known concrete waveforms eagerly, retaining traced inputs.

    The built-in waveform objects store every parameter in dataclass fields.
    Inspect those fields, not the object (an unregistered JAX pytree leaf).
    An arbitrary callable, including CustomWaveform, can hide traced state in
    a closure or globals; without a known parameter layout it keeps its
    existing evaluation path. Material coefficients belong outside here.
    """
    from rfx.sources.sources import CWSource, GaussianPulse, ModulatedGaussian

    if type(waveform) not in (CWSource, GaussianPulse, ModulatedGaussian):
        return nullcontext()
    parameters = tuple(getattr(waveform, field.name) for field in fields(waveform))
    if any(is_tracer(value) for value in
           jax.tree_util.tree_leaves((time_parameters, parameters))):
        return nullcontext()
    return jax.ensure_compile_time_eval()
