"""Distinct live device buffers and compiled temporary bytes at the record join.

A <= max(0.025 R, one row along the two splittable axes). Compiled programs
allow 3 A + 0.05 R + 1024 bytes; live buffers allow C + R + 3 A + 0.05 R.
"""
import gc
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx.stepping import record_join as join
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
    post_call_peaks = [0] * len(own)
    analyses = []
    original = join._join_program
    original_chunk_shape = join.chunk_shape
    row_bytes = {}

    def plan(shape, dtype, record_size, x_axis):
        chunk = original_chunk_shape(shape, dtype, record_size, x_axis)
        axes = {0, next((a for a in range(1, len(shape)) if a != x_axis), 0)}
        row = int(np.prod([s for a, s in enumerate(shape) if a not in axes]))*dtype.itemsize
        row_bytes[(shape, dtype, chunk)] = row
        return chunk

    monkeypatch.setattr(join, 'chunk_shape', plan)

    def instrument(*args):
        program = original(*args)
        shape, dtype, chunk = args[3:6]
        def measured(ranks, buffer):
            compiled = program.lower(ranks, buffer).compile()
            stats = compiled.memory_analysis()
            assert stats is not None, 'compiled join memory unavailable'
            analyses.append(dict(shape=shape, chunk=chunk, A=int(np.prod(chunk))*dtype.itemsize,
                                 row_bytes=row_bytes.get((shape, dtype, chunk)), temporary=stats.temp_size_in_bytes))
            result = compiled(ranks, buffer)
            jax.block_until_ready(result)
            # Keep the compiled call's argument alive while sampling its result.
            current = sample()
            for rank, total in enumerate(current):
                post_call_peaks[rank] = max(post_call_peaks[rank], total)
            return result
        return measured
    monkeypatch.setattr(join, '_join_program', instrument)
    watched = {'record_join.py', '_distributed_records.py', '_distributed_ntff.py', '_record_join_frozen.py'}

    def sample():
        current = buffers()
        totals = []
        for rank, device in enumerate(layout.mesh.devices.flat):
            total = sum(size for (d, pointer), size in current.items()
                        if d == device.id and (d, pointer) not in baseline)
            peaks[rank] = max(peaks[rank], total)
            totals.append(total)
        return totals

    def checkpoint(frame, event, arg):
        if event in ('line', 'return') and Path(frame.f_code.co_filename).name in watched:
            sample()
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
    live_allowance = 3*largest + .05*record_size
    for row in analyses:
        row['temporary_bar'] = 3*row['A'] + .05*record_size + 1024
    report = dict(kind=kind, size=size, single_frequency=single, R=record_size, C=own, A=largest,
                  P=peaks, P_over_R=[p/record_size for p in peaks], temporary=temporary,
                  post_call_P=post_call_peaks, post_call_P_over_R=[p/record_size for p in post_call_peaks],
                  temporary_bar=live_allowance+1024, peak_bar=[c+record_size+live_allowance for c in own], programs=analyses)
    (tmp_path / 'memory.json').write_text(json.dumps(report))
    print(json.dumps(report), flush=True)
    print(f'{kind}: one-device reference 1.0 R')
    if single:
        assert largest < record_size, ('one-frequency chunk must be smaller than R', report)
    assert all(row['row_bytes'] is not None and row['A'] <= max(.025*record_size, row['row_bytes'])
               for row in analyses), ('chunk bound', report)
    assert all(row['temporary'] <= row['temporary_bar'] for row in analyses), ('compiled temporary bound', report)
    assert all(p <= c+record_size+live_allowance for p, c in zip(peaks, own)), ('join distinct-buffer bound', report)


@pytest.mark.slow
@pytest.mark.parametrize('mutation', ['M1', 'M3'])
def test_time_plane_memory_mutations(mutation, monkeypatch, tmp_path):
    from jax.sharding import NamedSharding, PartitionSpec as P
    from rfx.runners._distributed_records import TimePlaneRecords
    if mutation == 'M1':
        original = TimePlaneRecords.assemble
        def replicated(self, buffer, **kwargs):
            whole = jax.tree.map(lambda a: jax.device_put(a, NamedSharding(self.mesh, P())), buffer)
            result = original(self, whole, **kwargs)
            jax.block_until_ready(whole)
            return result
        monkeypatch.setattr(TimePlaneRecords, 'assemble', replicated)
        expected = 'join distinct-buffer bound'
    else:
        monkeypatch.setattr(join, 'chunk_shape', lambda shape, *args: tuple(shape))
        expected = 'chunk bound'
    with pytest.raises(AssertionError, match=expected) as error:
        test_join_memory('time_planes', 1, False, monkeypatch, tmp_path)
    print(f'time_planes {mutation} RED {error.value}', flush=True)
