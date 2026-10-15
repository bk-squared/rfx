"""Explicit #1512 redraw for parameterized, single-Box trace fixtures.

Preflight fixtures vary port location/frequency. Preserve their outside-band
cases byte for byte; trim only an offending signal endpoint, never the ground.
The caller names the trace entry rather than guessing which PEC is the signal.
"""
from dataclasses import replace

from rfx import Box
from rfx.preflight.line_stub import line_stub_findings, read_band, resonant_odd_orders


def trim_resonant_trace(sim, trace_index):
    entry = sim._geometry[trace_index]
    assert isinstance(entry.shape, Box)
    lo, hi = list(entry.shape.corner_lo), list(entry.shape.corner_hi)
    for finding in line_stub_findings(sim):
        if finding.collection != '_msl_ports' or resonant_odd_orders(finding, read_band(sim)) is None:
            continue
        axis = 'xyz'.index(finding.axis)
        port = sim._msl_ports[finding.port_index]
        if port.direction.startswith('+'):
            lo[axis] = max(lo[axis], finding.port_node_m)
        else:
            hi[axis] = min(hi[axis], finding.port_node_m)
    if (tuple(lo), tuple(hi)) != (entry.shape.corner_lo, entry.shape.corner_hi):
        sim._geometry[trace_index] = replace(entry, shape=Box(tuple(lo), tuple(hi)))
    return sim
