"""The concrete echo-window invariant at completion outputs."""
from dataclasses import replace
import math


def close_completion(plan, output, report=None, failure=None, *, grid=None, dt=None):
    """Preserve prior refusals; otherwise poison a completion with a short window.

    Reports retain all formed witnesses. Traced outputs use the same concrete
    condition and multiply by NaN so their derivatives carry the refusal too.
    """
    from rfx.measurement.record_end import ECHO_BOUND_FACTOR, echo_bound_reason, echo_delay_bound_on_grid
    from rfx.ringdown import RingdownWitness, _NotCompleted

    if failure is not None:
        return output, report, failure
    bound = (plan.echo_bound if grid is None else
             echo_delay_bound_on_grid(grid, plan.echo_bound))
    dt = plan.echo_dt if dt is None else dt
    window = (math.nan if dt is None else
              (int(plan.n_steps) - int(plan.n_start)) * float(dt))
    tau_max = math.nan if bound.get('reason') else float(bound['tau_max'])
    if not bound.get('reason') and window >= ECHO_BOUND_FACTOR * tau_max:
        return output, report, None
    reason = ('cannot judge: echo delay bound solver step is traced'
              if dt is None else echo_bound_reason(bound))
    if report is not None:
        value = math.nan if bound.get('reason') else window / tau_max
        witness = RingdownWitness('echo_bound', value, ECHO_BOUND_FACTOR, False, reason)
        report = replace(report, witnesses=report.witnesses + (witness,),
                         failure='echo_bound: ' + reason)
    failure = _NotCompleted('echo_bound', math.nan, ECHO_BOUND_FACTOR, reason)
    return None if output is None else output * math.nan, report, failure


def forward_closing_factor(plan, context, dt, cells, *, dtype):
    """Judge realized host widths at the completion's pole-mapping step.

    The callback reads the same live widths as the channels. Its decision is
    discrete and has no derivative; multiplying by NaN carries a refusal
    through both the completed value and its derivatives.
    """
    import jax
    import numpy as np
    from rfx.ringdown._jax import _callback_keywords

    if not cells:
        return close_completion(plan, 1., grid=context.grid, dt=dt)[0]
    template, axes = context.grid, context.traced_axes
    exact_of = {'dx_arr': 'dx_arr_f64', 'dy_arr': 'dy_arr_f64', 'dz': 'dz_f64'}

    def judge(*widths):
        fields = {}
        for (_, store), live in zip(axes, widths, strict=True):
            fields[store] = np.asarray(live)
            if exact_of[store] in template._fields:
                fields[exact_of[store]] = np.asarray(live, dtype=np.float64)
        grid = template._replace(**fields)
        return np.asarray(close_completion(plan, 1., grid=grid, dt=dt)[0], dtype=dtype)

    return jax.pure_callback(judge, jax.ShapeDtypeStruct((), dtype),
                             *(jax.lax.stop_gradient(c) for c in cells),
                             **_callback_keywords())
