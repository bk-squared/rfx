"""Distinct live device buffers and compiled temporary bytes at the record join."""
import gc
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx.runners import _record_join as join
from tests.unit.runners._record_join_scene import KINDS, layout_for
from tests.unit.runners.test_record_join import arrays


def buffers(arrays=None):
    """Count a physical allocation once per device, including aliasing views."""
    found = {}
    for array in jax.live_arrays() if arrays is None else arrays:
        for shard in array.addressable_shards:
            data = shard.data
            key = (shard.device.id, data.unsafe_buffer_pointer())
            found[key] = max(found.get(key, 0), data.nbytes)
    return found


@pytest.mark.parametrize('kind,size,single', [(k, s, False) for k in KINDS for s in (1, 2)] + [('planes', 1, True)])
def test_join_memory(kind, size, single, monkeypatch, tmp_path):
    layout = layout_for(kind, size=size, single_frequency=single)
    gc.collect()
    # Retain baseline allocations so a freed address cannot hide a new buffer.
    baseline_arrays = tuple(jax.live_arrays())
    baseline = buffers(baseline_arrays)
    carry = layout.initial()
    jax.block_until_ready(carry)
    own = [sum(a.addressable_shards[r].data.nbytes for a in jax.tree.leaves(carry)) for r in range(2)]
    peaks = own.copy()
    analyses = []
    original = join._join_program

    def instrument(*args):
        program = original(*args)
        shape, dtype, chunk = args[3:6]
        def measured(ranks, buffer):
            compiled = program.lower(ranks, buffer).compile()
            stats = compiled.memory_analysis()
            assert stats is not None, 'compiled join memory unavailable'
            analyses.append(dict(shape=shape, chunk=chunk, A=int(np.prod(chunk))*dtype.itemsize,
                                 temporary=stats.temp_size_in_bytes))
            return compiled(ranks, buffer)
        return measured
    monkeypatch.setattr(join, '_join_program', instrument)
    watched = {'_record_join.py', '_distributed_records.py', '_distributed_ntff.py', '_record_join_frozen.py'}

    def checkpoint(frame, event, arg):
        if event in ('line', 'return') and Path(frame.f_code.co_filename).name in watched:
            current = buffers()
            for rank, device in enumerate(layout.mesh.devices.flat):
                total = sum(size for (d, pointer), size in current.items()
                            if d == device.id and (d, pointer) not in baseline)
                peaks[rank] = max(peaks[rank], total)
        return checkpoint

    previous = sys.gettrace()
    try:
        sys.settrace(checkpoint)
        result = layout.assemble(carry)
        jax.block_until_ready(result)
    finally:
        sys.settrace(previous)
    record_size = sum(a.size*a.dtype.itemsize for a in arrays(kind, result))
    assert analyses, 'join programs not observed'
    largest = max(row['A'] for row in analyses)
    temporary = max(row['temporary'] for row in analyses)
    bar = 3*largest + .05*record_size
    report = dict(kind=kind, size=size, single_frequency=single, R=record_size, C=own, A=largest,
                  P=peaks, P_over_R=[p/record_size for p in peaks], temporary=temporary,
                  temporary_bar=bar, peak_bar=[c+record_size+bar for c in own], programs=analyses)
    (tmp_path / 'memory.json').write_text(json.dumps(report))
    print(json.dumps(report), flush=True)
    print(f'{kind}: one-device reference 1.0 R')
    assert largest <= .025*record_size, ('chunk bound', report)
    assert all(row['temporary'] <= 3*row['A'] + .05*record_size for row in analyses), ('compiled temporary bound', report)
    assert all(p <= c+record_size+bar for p, c in zip(peaks, own)), ('join distinct-buffer bound', report)
