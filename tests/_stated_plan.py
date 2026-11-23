"""The measurement plan a direct call of the multi-device graded step states."""
from rfx.measurement.plan import MeasurementPlan, Node, Owner, TimeBase, field_channel


def stated_plan(grid, probes, *, n_steps, path='run_distributed_nu'):
    """Return a plan whose owners are the call's own probes, in call order.

    Each owner has the form ``rfx/measurement/plan.py:275-279`` gives a
    declared probe: id ``probe:<position in the list>``, one node at the
    probe's ``(component, i, j, k)`` with weight 1.0, and zero E and H time
    offsets. The indices are the ones the call passes; nothing is resolved
    here. The plan has no port owners; ``probes=[]`` gives no owners.
    ``grid`` supplies only the time step.
    """
    owners = tuple(Owner(f'probe:{i}', 'probe',
                         (field_channel(probe.component,
                                        (Node(probe.component, probe.i, probe.j, probe.k, 1.0),),
                                        e_time_offset=0., h_time_offset=0.),))
                   for i, probe in enumerate(probes))
    return MeasurementPlan(path, TimeBase(grid.dt, n_steps), owners)
