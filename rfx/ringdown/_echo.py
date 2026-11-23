"""The realized echo-window invariant at completion outputs."""
from dataclasses import replace
import math


def close_completion(plan, output, report=None, failure=None, *, grid=None, dt=None):
    """Preserve prior refusals and judge the bound attached by the runner."""
    from rfx.measurement.record_end import ECHO_BOUND_FACTOR, echo_bound_reason
    from rfx.ringdown import RingdownWitness, _NotCompleted

    if failure is not None:
        return output, report, failure
    bound = plan.echo_bound
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


def forward_closing_factor(plan, dt, bound, *, dtype):
    """Judge on device; the stopped bound has no derivative or host callback."""
    import jax.numpy as jnp
    from rfx.measurement.record_end import ECHO_BOUND_FACTOR

    required = jnp.nextafter(jnp.asarray(ECHO_BOUND_FACTOR * bound['tau_max'], dtype),
                             jnp.asarray(jnp.inf, dtype))
    window = float(int(plan.n_steps) - int(plan.n_start)) * jnp.asarray(dt, dtype)
    window = jnp.nextafter(window, jnp.asarray(-jnp.inf, dtype))
    return jnp.where(jnp.isfinite(required) & (window >= required),
                     jnp.asarray(1., dtype), jnp.asarray(jnp.nan, dtype))
