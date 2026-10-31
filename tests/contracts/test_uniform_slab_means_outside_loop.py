"""Uniform slabs hoist material means only when CPML shares the consumers."""
import inspect
import re
from pathlib import Path

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core import yee
from rfx.model import electric_metrics
from rfx.stepping import slab_update


def scene(boundary):
    sim = Simulation(freq_max=15e9, domain=(12e-3, 9e-3, 7e-3), dx=1e-3,
                     boundary=boundary, cpml_layers=2)
    sim.add_material('dielectric', eps_r=4, sigma=.01)
    sim.add(Box((3e-3, 2e-3, 2e-3), (8e-3, 6e-3, 5e-3)), material='dielectric')
    sim.add_material('magnetic', mu_r=2)
    sim.add(Box((5e-3, 3e-3, 1e-3), (9e-3, 7e-3, 4e-3)), material='magnetic')
    sim.add_source((4e-3, 4e-3, 3e-3), 'ez', amplitude_kind='field')
    sim.add_probe((9e-3, 5e-3, 4e-3), 'ez')
    return sim


def _mean_sources():
    functions = [slab_update.magnetic_low_ghost, electric_metrics.slab_wall_view,
                 electric_metrics.slab_e_component_materials, yee.component_h_materials,
                 yee._material_wall_view, yee._material_bwd_neighbour,
                 yee._edge_mean_component, yee._edge_mean_four]
    return [(Path(inspect.getsourcefile(f)).name, first, first + len(lines), f.__name__)
            for f in functions for lines, first in [inspect.getsourcelines(f)]]


def means_inside_while(hlo):
    # Recent XLA stores source_file/source_line in an indexed metadata table.
    files = dict(re.findall(r'^(\d+) "([^"]+)"$', hlo.split('FunctionNames')[0], re.M))
    locations = {key: (files[file], line) for key, file, line in re.findall(
        r'^(\d+) \{file_name_id=(\d+) function_name_id=\d+ line=(\d+)', hlo, re.M)}
    frames = dict(re.findall(r'^(\d+) \{file_location_id=(\d+)', hlo, re.M))

    def source_metadata(match):
        file, line = locations[frames[match[1]]]
        return f'source_file="{file}" source_line={line}'

    hlo = re.sub(r'stack_frame_id=(\d+)', source_metadata, hlo)
    assert 'source_file=' in hlo, 'HLO must retain source metadata'
    computations = {}
    current = None
    for line in hlo.splitlines():
        match = re.match(r'^(?:ENTRY )?%?([\w.-]+)\s.*\{$', line)
        if match:
            current = match[1]
            computations[current] = []
        elif line == '}':
            current = None
        elif current is not None:
            computations[current].append(line)
    bodies = re.findall(r'\bbody=%?([\w.-]+)', hlo)
    assert bodies, 'no time-loop while body found'
    todo, seen, violations = list(bodies), set(), []
    sources = _mean_sources()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        assert name in computations, name
        for line in computations[name]:
            todo.extend(re.findall(r'\b(?:calls|to_apply|body|condition)=%?([\w.-]+)', line))
            source = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
            for filename, first, end, function in sources:
                if (source and Path(source[1]).name == filename and first <= int(source[2]) < end
                        or re.search(r'op_name="[^"]*\b' + function + r'\b', line)):
                    violations.append((name, function, line.strip()))
    return violations


@pytest.mark.parametrize('boundary', ['cpml', 'pec'])
def test_uniform_slab_mean_placement(boundary, monkeypatch, tmp_path):
    assert len(jax.devices('cpu')) == 2, 'run with two virtual CPU devices'
    original = jax.jit
    captured = []

    def jit(fun, *args, **kwargs):
        compiled = original(fun, *args, **kwargs)
        if fun.__module__ != 'rfx.stepping.slab_uniform_loop' or fun.__name__ not in ('_run_cpml', '_run_pec'):
            return compiled

        def run(*values, **named):
            lowered = compiled.lower(*values, **named)
            executable = lowered.compile()
            captured.append(lowered.compiler_ir('hlo').as_hlo_module().to_string())
            captured.append(executable.runtime_executable().hlo_modules()[0].to_string())
            return executable(*values, **named)
        return run

    monkeypatch.setattr(jax, 'jit', jit)
    result = scene(boundary).run(n_steps=6, devices=jax.devices('cpu'), skip_preflight=True)
    assert np.isfinite(np.asarray(result.time_series)).all()
    assert len(captured) == 2, 'public run must lower and compile the uniform slab driver'
    for stage, hlo in zip(('lowered', 'compiled'), captured):
        (tmp_path / f'{boundary}-{stage}.hlo').write_text(hlo)
        means = means_inside_while(hlo)
        if boundary == 'cpml':
            assert not means, f'material means inside while body: {means}'
        elif stage == 'lowered':
            # Pin the input program; CPU optimization may move invariant work.
            # Bandwidth: hoisting adds slab reads for update-only consumers (rfx-archive plans/absorber/20261026-cpml-cost-952 D1-L0 reading).
            assert means, 'PEC must form material means inside the while body'


def independent_means():
    """Cell-index reference: no rfx material or slab helpers build expected values."""
    nx, nx_per, ny, nz = 9, 5, 5, 4
    cells = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
    whole = (1 + (cells % 13) / np.float32(4),
             (cells % 7) / np.float32(100), 1 + (cells % 5) / np.float32(3))
    staged = [[], [], []]
    expected_e, expected_h, expected_view = [], [], []
    for rank in range(2):
        local = []
        for field, global_values in enumerate(whole):
            slab = np.full((nx_per + 2, ny, nz), 0 if field == 1 else 1, np.float32)
            for i in range(nx_per + 2):
                g = rank * nx_per - 1 + i
                if 0 <= g < nx:
                    slab[i] = global_values[g]
            staged[field].append(slab)
            local.append(slab)
        views = []
        for slab in local:
            view = slab.copy()
            for i in range(nx_per + 2):
                g = rank * nx_per - 1 + i
                source = 1 if rank == 0 and i == 0 else i
                if rank == 1 and g >= nx - 1:
                    source = nx - 2 - rank * nx_per + 1
                view[i] = slab[source]
            views.append(view)
        expected_view.append(views[2].copy())
        # All three high faces are PMC: high nodes read the last inside cell.
        for view in views:
            view[:, -1, :] = view[:, -2, :]
            view[:, :, -1] = view[:, :, -2]
        e_rank = np.empty((2, 3, nx_per + 2, ny, nz), np.float32)
        h_rank = np.empty((3, nx_per + 2, ny, nz), np.float32)
        for index in np.ndindex(nx_per + 2, ny, nz):
            for component in range(3):
                behind = list(index)
                behind[component] = max(0, behind[component] - 1)
                lo, hi = views[2][tuple(behind)], views[2][index]
                h_rank[(component, *index)] = hi if lo == hi else np.float32(2) / (np.float32(1) / lo + np.float32(1) / hi)
                axes = [a for a in range(3) if a != component]
                a, b, ab = list(index), list(index), list(index)
                a[axes[0]] = ab[axes[0]] = max(0, index[axes[0]] - 1)
                b[axes[1]] = ab[axes[1]] = max(0, index[axes[1]] - 1)
                for field in range(2):
                    value = local[field][index]
                    if 1 <= index[0] <= nx_per and rank * nx_per - 1 + index[0] < nx:
                        v = views[field]
                        value = ((v[index] + v[tuple(a)]) + (v[tuple(b)] + v[tuple(ab)])) * np.float32(.25)
                    e_rank[(field, component, *index)] = value
        expected_e.append(e_rank)
        expected_h.append(h_rank)
    return tuple(np.concatenate(s) for s in staged), (
        np.concatenate(expected_view), np.concatenate(expected_e, axis=2),
        np.concatenate(expected_h, axis=1))


def measure_means():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from rfx.stepping.rank import mesh_ranks
    staged, (view, electric, magnetic) = independent_means()
    mesh = Mesh(np.array(jax.devices('cpu')[:2]), ('x',))
    mat = yee.MaterialArrays(*(jax.device_put(a, NamedSharding(mesh, P('x'))) for a in staged))
    ranks = mesh_ranks(mesh)
    e = slab_update.slab_e_materials_shmap(mat, mesh, 5, 9, ranks=ranks, high_wall=(True,) * 3)
    h_view, h = slab_update.slab_h_materials_shmap(mat, mesh, 5, 9, ranks=ranks, high_wall=(True,) * 3)
    results = {}
    for name, actual, expected in (('wall_view', h_view, view), ('E', e, electric), ('H', h, magnetic)):
        actual = np.asarray(actual)
        results[name] = dict(equal=bool(np.array_equal(actual, expected)),
                             max_abs=float(np.max(np.abs(actual - expected))),
                             max_ulp=int(np.max(np.abs(actual.view(np.int32).astype(np.int64) - expected.view(np.int32)))))
    return results


def test_hoisted_means_match_independent_cells():
    results = measure_means()
    assert all(row['max_ulp'] <= 1 for row in results.values()), results
