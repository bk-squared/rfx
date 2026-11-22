"""The concrete echo-window invariant at completion outputs."""
from dataclasses import replace
import math


def close_completion(plan, output, report=None, failure=None):
    """Preserve prior refusals; otherwise poison a completion with a short window.

    Reports retain all formed witnesses. Traced outputs use the same concrete
    condition and multiply by NaN so their derivatives carry the refusal too.
    """
    from rfx.measurement.record_end import ECHO_BOUND_FACTOR, echo_bound_reason
    from rfx.ringdown import RingdownWitness, _NotCompleted

    if failure is not None:
        return output, report, failure
    bound = plan.echo_bound
    window = (math.nan if plan.echo_dt is None else
              (int(plan.n_steps) - int(plan.n_start)) * float(plan.echo_dt))
    tau_max = math.nan if bound.get('reason') else float(bound['tau_max'])
    if not bound.get('reason') and window >= ECHO_BOUND_FACTOR * tau_max:
        return output, report, None
    reason = echo_bound_reason(bound)
    if report is not None:
        value = math.nan if bound.get('reason') else window / tau_max
        witness = RingdownWitness('echo_bound', value, ECHO_BOUND_FACTOR, False, reason)
        report = replace(report, witnesses=report.witnesses + (witness,),
                         failure='echo_bound: ' + reason)
    failure = _NotCompleted('echo_bound', math.nan, ECHO_BOUND_FACTOR, reason)
    return None if output is None else output * math.nan, report, failure
