"""Material realization and slab inputs for the canonical graded step."""
from contextlib import nullcontext
from functools import lru_cache
from types import SimpleNamespace
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.backend import host_cpu_device
from rfx.core.yee import FDTDState
from rfx.model.materials import (ComponentCells, realize_components, kernel_materials)
from rfx.stepping.slab import cut, zero_pole_state, place
from rfx.stepping.slab_view import (UniformSlabView, uniform_slab_rows, uniform_apply,
                                   uniform_from_cut, stack_and_place, product_rows, slab_views, device_rows)


def stage_x_widths(grid, rows, mesh):
    widths = (np.asarray(grid.dx_f64, dtype=grid.dx_arr.dtype)
              if isinstance(grid.dx_arr, jax.core.Tracer) and grid.dx_f64 is not None
              else np.asarray(grid.dx_arr))
    if widths.shape[0] == grid.shape[0] - 1:
        widths = np.concatenate((widths, widths[-1:]))
    return stack_and_place(widths, rows, mesh)


@lru_cache(maxsize=128)
def _drive_inputs_map(mesh, cells):
    """Copy only local edge scalars; retain the device axis, without gathering."""
    from rfx.stepping.rank import rank_shard_map
    def read(eps, sigma, keep, cb, *, rank):
        def take(arrays):
            return (None if arrays is None else
                    jnp.stack([arrays[axis][cell] for axis, cell in cells])[None])
        return tuple(take(arrays) for arrays in (eps, sigma, keep, cb))
    return jax.jit(rank_shard_map(read, mesh=mesh, in_specs=(P('x'),) * 4,
                                out_specs=P('x'), check_rep=False))


def slab_drive_columns(materials, debye, lorentz, sources, volumes, dt, layout, mesh, n_steps):
    """Read the local inputs, then run the model's eager source products."""
    from rfx.model.source_coefficients import drive_columns
    from rfx.stepping.rank import mesh_ranks
    from rfx import _realized
    if not sources:
        return place(jnp.zeros((n_steps, 0), jnp.float32), NamedSharding(mesh, P(None, 'x')))
    cells = tuple((('ex', 'ey', 'ez').index(s.component),
                   (layout.local_index(s.i), s.j, s.k)) for s in sources)
    owners = jnp.asarray([layout.owner(s.i) for s in sources])
    ranks = mesh_ranks(mesh)
    cb = None
    if debye is not None and lorentz is not None:
        from rfx.materials.lorentz import mixed_e_component_coeffs
        def coefficients(d, lorentz_coeffs):
            # Staging packs each device's pole-first leaves along axis zero.
            d, lorentz_coeffs = jax.tree.map(lambda a: a.reshape((mesh.size, -1) + a.shape[1:]), (d, lorentz_coeffs))
            values = tuple(jax.vmap(lambda d, lc: mixed_e_component_coeffs(d, lc, axis, dt)[1])(d, lorentz_coeffs)
                           for axis in range(3))
            return device_rows(values, mesh, unpack=False)
        cb = jax.checkpoint(coefficients)(debye[0], lorentz[0])
    elif debye is not None or lorentz is not None:
        cb = (debye if debye is not None else lorentz)[0].cb
    eps, sigma, keep, cb = _drive_inputs_map(mesh, cells)(ranks,
        materials.components.eps_update, materials.components.sigma_update, materials.edge_keep, cb)
    # The read map packs scalar edges as a tiny synthetic component array.
    # drive_columns uses the same indexing and products as its whole-grid call.
    scalar_cells = tuple((0, (n, 0, 0)) for n in range(len(sources)))
    samples = jnp.stack([jnp.asarray(s.waveform) for s in sources], axis=-1)
    def local(e, s, k, c, rank, samples=samples, volumes=volumes):
        def arrays(a):
            return None if a is None else (a[:, None, None],)
        return drive_columns(arrays(e), arrays(s), arrays(k), scalar_cells,
                             owners == rank, samples, volumes, dt, cb=c)
    if _realized.ACTIVE is None:
        table = jax.vmap(local)(eps, sigma, keep, cb, ranks)
    else:
        columns = tuple(n for n, v in enumerate(volumes) if v is not None)
        unit_volumes = tuple(None if v is None else (1., v[1]) if isinstance(v, tuple) else 1. for v in volumes)
        def scales(e, s, c, rank):
            return local(e, s, None, c, rank, samples=jnp.ones((1, len(sources)), samples.dtype), volumes=unit_volumes)[0]
        scales = jax.vmap(scales)(eps, sigma, cb, ranks)
        observed = _realized.runtime_drive(jnp.sum(scales[:, jnp.asarray(columns, dtype=jnp.int32)], axis=0), columns,
                                           volumes=tuple(volumes[n] for n in columns))
        # Publish Cb only to its owner; the local finisher retains each dV.
        factors = jnp.ones((len(sources),), samples.dtype).at[jnp.asarray(columns, dtype=jnp.int32)].set(observed)
        table = jax.vmap(lambda e, s, k, rank: local(e, s, k, factors, rank, volumes=volumes))(
            eps, sigma, keep, ranks)
    table = jnp.transpose(table, (1, 0, 2)).reshape(n_steps, layout.n_devices * len(sources))
    return place(table, NamedSharding(mesh, P(None, 'x')))


@lru_cache(maxsize=128)
def _owner_records_map(mesh, owner):
    from rfx.stepping.rank import rank_shard_map
    def read(values, *, rank):
        return jax.tree.map(lambda a: jax.lax.all_gather(a[0], 'x')[owner], values)
    return jax.jit(rank_shard_map(read, mesh=mesh, in_specs=P('x'), out_specs=P(), check_rep=False))


@lru_cache(maxsize=128)
def _drive_means_map(mesh, drives):
    from rfx.stepping.rank import rank_shard_map
    def read(eps, sigma, *, rank):
        return tuple(jnp.stack([a[('ex', 'ey', 'ez').index(component)][cell]
                               for _, _, cell, component, _, _ in drives])[None]
                     for a in (eps, sigma))
    return jax.jit(rank_shard_map(read, mesh=mesh, in_specs=(P('x'),) * 2,
                                out_specs=P('x'), check_rep=False))


@lru_cache(maxsize=128)
def _gather_drive_map(mesh, owners):
    def read(a):
        values = jax.lax.all_gather(a[0], 'x')
        return values[np.asarray(owners), np.arange(len(owners))]
    return jax.jit(jax.shard_map(read, mesh=mesh, in_specs=P('x'), out_specs=P(), check_vma=False))


def eager_drive_scales(eps, sigma, mesh, drives, dt, ranks, reduce_devices):
    """Scalar reads stay mapped; source arithmetic is eager over device rows."""
    from rfx.nonuniform import current_source_cb
    means = _drive_means_map(mesh, drives)(ranks, eps, sigma)
    def local(eps, sigma, rank):
        values = []
        for i, (owner, _, _, _, volume, pole) in enumerate(drives):
            cb = current_source_cb(jnp.where(rank == owner, eps[i], 1.),
                jnp.where(rank == owner, sigma[i] + pole, 0.), dt)
            values.append(jnp.where(rank == owner, cb, 0.))
        return jnp.stack(values)
    values = jax.vmap(local)(*means, ranks)
    return _gather_drive_map(mesh, tuple(d[0] for d in drives))(values) if reduce_devices else values


def input_materials(materials, layout, mesh):
    return jax.tree.map(lambda a: uniform_from_cut(a, layout, mesh),
                        materials._replace(components=None, edge_keep=None))


_warned_no_cpu = False


@jax.custom_jvp
def _custom_rule_probe(x):
    return x


_custom_rule_probe.defjvp(lambda primals, tangents: (primals[0], tangents[0]))


def _custom_rules_evaluate_concretely():
    """Inside jax.vmap a custom_jvp call returns a batch tracer even under
    ensure_compile_time_eval (the H-material harmonic mean is one), so the
    host producer could not read its products back."""
    with jax.ensure_compile_time_eval():
        return not isinstance(_custom_rule_probe(np.float32(0)), jax.core.Tracer)


def _material_host_device(*inputs, mesh=None):
    """One concreteness/backend decision for both material producers."""
    global _warned_no_cpu
    leaves = jax.tree.leaves(inputs)
    if ((mesh is not None and len({d.process_index for d in mesh.devices.flat}) > 1)
            or any(isinstance(a, jax.Array) and not isinstance(a, jax.core.Tracer)
                   and not a.is_fully_addressable for a in leaves)):
        return None, 'multi-process'
    if any(isinstance(a, jax.core.Tracer) for a in leaves):
        return None, 'traced inputs'
    if not _custom_rules_evaluate_concretely():
        return None, 'batching transform'
    try:
        return host_cpu_device(), None
    except RuntimeError:
        if not _warned_no_cpu:
            stacklevel = 2
            frame = sys._getframe(1)
            while frame is not None and frame.f_globals.get('__name__', '').startswith('rfx.'):
                stacklevel += 1
                frame = frame.f_back
            del frame
            warnings.warn(
                'no CPU backend in this JAX process (JAX_PLATFORMS does not list cpu), '
                'so the graded multi-device lane forms its material coefficients per slab; '
                'on a GPU they can differ from a one-device run in the last float32 bit '
                'at dielectric corner edges (within every accuracy bar)', RuntimeWarning, stacklevel=stacklevel)
            _warned_no_cpu = True
        return None, 'no CPU backend'


def _whole_material_rows(value, layout):
    """Read each real row from its owner, excluding input halos and padding."""
    def whole(a):
        if not hasattr(a, 'ndim'):
            return a
        if isinstance(a, jax.Array):
            # NumPy readback caches shard wrappers on its JAX source. Read a
            # producer-owned copy so caller-held mesh inputs retain none.
            with jax.ensure_compile_time_eval():
                a = a.copy()
        a = np.asarray(a)
        if a.ndim != 4:
            return a
        i = np.arange(layout.nx)
        rank = i // layout.nx_per_rank
        return a[rank, i - rank * layout.nx_per_rank + layout.ghost_width + 1]
    return jax.tree.map(whole, value)


def _host_products(producer, device, *inputs):
    """Release every whole-grid JAX product before placing any mesh shards."""
    with jax.default_device(device), jax.ensure_compile_time_eval():
        products = producer(*inputs)
        host = jax.tree.map(lambda a: np.asarray(a) if isinstance(a, jax.Array) else a, products)
        del products
    return host


def _host_arrays(value):
    """Concrete metric and pole leaves must not retain GPU placement."""
    return jax.tree.map(lambda a: np.asarray(a) if isinstance(a, jax.Array) else a, value)


@lru_cache(maxsize=128)
def _pole_first_map(mesh):
    """Reorder each cut shard from x/pole to the step's pole/x layout."""
    return jax.jit(jax.shard_map(lambda a: jnp.moveaxis(a, 1, 0), mesh=mesh,
                                in_specs=P('x'), out_specs=P('x'), check_vma=False))


def realize_materials(materials, grid, layout, mesh, occupancy=None, *, cpml=False,
                      debye=False, lorentz=False, _material_inputs=()):
    """Realize concrete materials on the whole CPU grid, then cut host products.

    Traced or non-addressable inputs and multi-process meshes retain the
    per-slab producer; a missing CPU backend also
    retains it, with one RuntimeWarning per process. Whole-grid host arrays
    live during formation. Fresh-process macOS measurements (ru_maxrss in
    bytes), with occupancy and CPML operands, including compilation and cut
    placement but no time steps: peak RSS increments were 2289.13 B/cell at
    44,361 cells and 661.85 B/cell at 255,915 cells. The two-size incremental
    slope was 320.63 B/cell, measured as (peak2 - peak1)/(cells2 - cells1).
    """
    from rfx.model.occupancy import with_edge_keep
    from rfx.stepping.absorber import graded_cpml_operands
    device, _ = _material_host_device(materials, occupancy, _material_inputs, mesh=mesh)
    with jax.ensure_compile_time_eval() if device is not None else nullcontext():
        inputs = input_materials(materials, layout, mesh)
        occ = uniform_from_cut(occupancy, layout, mesh, outside=0)
    model_grid = _host_arrays(grid) if device is not None else grid

    def local(cells, index, occupancy, widths):
        view = None if index is None else UniformSlabView(index, grid, {'x_widths': widths})
        components = realize_components(cells, model_grid, periodic=(False,) * 3, view=view)
        cells = cells._replace(components=components)
        cells = with_edge_keep(cells, occupancy, dtype=jnp.float32, view=view)
        inverse, loss = graded_cpml_operands(cells, None, use_cpml=cpml,
                                            use_debye=debye, use_lorentz=lorentz)
        return cells, inverse, loss
    if device is not None:
        staged, inverse, loss = _host_products(local, device,
            _whole_material_rows(inputs, layout), None, _whole_material_rows(occ, layout), None)
    else:
        rows = uniform_slab_rows(layout)
        indices = place(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
        widths = stage_x_widths(grid, rows, mesh)
        staged, inverse, loss = uniform_apply(local, inputs, indices, occ, widths,
                                              mesh=mesh, row_count=layout.nx_local + 1)
    def publish(a, kind):
        return (cut(a, layout, kind, mesh=mesh) if device is not None else
                product_rows(a, layout, mesh, kind))
    def finish(path, a):
        if not hasattr(a, 'ndim') or a.ndim < (3 if device is not None else 4):
            return a
        names = [str(getattr(p, 'name', getattr(p, 'key', ''))) for p in path]
        name = next((p for p in reversed(names) if p), '')
        kind = ('edge_keep' if 'edge_keep' in names else
                'eps_r' if 'eps' in name else 'sigma' if 'sigma' in name
                else 'mu_r' if 'mu' in name else 'lumped')
        return publish(a, kind)
    staged = jax.tree_util.tree_map_with_path(finish, staged)
    inverse = jax.tree.map(lambda a: publish(a, 'eps_r'), inverse)
    if loss is not None:
        loss = (jax.tree.map(lambda a: publish(a, 'eps_r'), loss[0]),
                jax.tree.map(lambda a: publish(a, 'sigma'), loss[1]))
    return staged, inverse, loss


def stage_dispersion(materials, dt, spec, grid, layout, mesh, family, *, debye_spec=None):
    """Judge owned cell reductions, then realize concrete ADE operands on CPU."""
    from rfx.model.pole_stability import (pole_set_extremes, combine_pole_extremes,
                                         judge_pole_sets, courant_budget)
    from rfx.materials.debye import debye_e_coeffs, DebyeState
    from rfx.materials.lorentz import lorentz_e_coeffs, LorentzState
    from rfx.core.yee import ade_state_dtype
    if spec is None:
        return None
    # Families decide separately: traced Lorentz masks need not trace Debye products.
    device, _ = _material_host_device(materials, spec, debye_spec, mesh=mesh)
    poles, masks = spec
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
    with jax.ensure_compile_time_eval() if device is not None else nullcontext():
        inputs = input_materials(materials, layout, mesh)
    rows = uniform_slab_rows(layout)
    indices = place(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
    staged_masks = (None if device is not None else
                    [None if m is None else stack_and_place(m, rows, mesh) for m in masks])
    if family == 'lorentz':
        # Static masks define each slab's pole patterns. Only their small
        # reduced records leave the shard_map; no staged slab is indexed here.
        from rfx.stepping.rank import mesh_ranks
        records = []
        for view in slab_views(grid, layout.n_devices):
            owned_view = view._replace(halo_lo=2,
                halo_hi=layout.nx_local + 1 - 2 - (view.x1 - view.x0))
            def local_masks(spec_masks):
                return [None if m is None else
                    (jnp if isinstance(m, jax.core.Tracer) else np).asarray(m)[
                        np.clip(rows.indices[view.rank], 0, layout.nx - 1)] for m in spec_masks]
            local_debye = None
            if debye_spec is not None:
                dp, dm = debye_spec
                dm = dm if isinstance(dm, (tuple, list)) else [dm] * len(dp)
                local_debye = (dp, local_masks(dm))
            lm = local_masks(masks)
            metadata = []
            def extrema(cells):
                found = pole_set_extremes(cells, (poles, lm), view=owned_view, debye_spec=local_debye)
                if not metadata:
                    metadata.extend(found)
                return tuple((r.eps_low, r.sigma_low, r.sigma_high, r.count, r.occupied) for r in found)
            values = jax.checkpoint(jax.vmap(extrema))(inputs)
            values = _owner_records_map(mesh, view.rank)(mesh_ranks(mesh), values)
            records.append(tuple(r._replace(eps_low=v[0], sigma_low=v[1], sigma_high=v[2], count=v[3], occupied=v[4])
                                 for r, v in zip(metadata, values)))
        judge_pole_sets(combine_pole_extremes(records), dt, courant_budget(grid),
                        grid_kind=type(grid).__name__, in_program=mesh.devices.size < 2)
    spatial = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
    model_grid = _host_arrays(grid) if device is not None else grid
    def local(cells, index, masks, widths):
        view = (None if index is None else
                UniformSlabView(index, grid, {'pole_sets_judged': True, 'x_widths': widths}))
        specs = dict(debye_spec=None, lorentz_spec=None)
        specs[family + '_spec'] = (poles, masks)
        c = realize_components(ComponentCells(cells, **specs), model_grid,
                                periodic=(False,) * 3, view=view)
        terms = getattr(c, family).coefficients
        coeffs = (debye_e_coeffs if family == 'debye' else lorentz_e_coeffs)(
            (c.eps_update, c.sigma_update), dt, *terms)
        return type(coeffs)(*(value if name in spatial else
            jax.tree.map(lambda a: jnp.moveaxis(a, 1, 0), value)
            for name, value in zip(coeffs._fields, coeffs)))
    if device is not None:
        poles, dt = _host_arrays((poles, dt))
        coeffs = _host_products(local, device, _whole_material_rows(inputs, layout),
                                None, _whole_material_rows(masks, layout), None)
        def publish(a, name):
            a = cut(a, layout, family + '_' + name, dt=dt, mesh=mesh)
            if name not in spatial:
                with jax.ensure_compile_time_eval():
                    a = _pole_first_map(mesh)(a)
            return a
        coeffs = type(coeffs)(*(jax.tree.map(lambda a: publish(a, name), value)
            for name, value in zip(coeffs._fields, coeffs)))
    else:
        widths = stage_x_widths(grid, rows, mesh)
        coeffs = uniform_apply(local, inputs, indices, staged_masks, widths, mesh=mesh, row_count=layout.nx_local + 1)
        coeffs = type(coeffs)(*(jax.tree.map(
            lambda a: product_rows(a, layout, mesh, family + '_' + name,
                dt=dt, pole_rows=name not in spatial), value)
            for name, value in zip(coeffs._fields, coeffs)))
    state = zero_pole_state(DebyeState if family == 'debye' else LorentzState,
        len(poles), layout, tuple(grid.shape[1:]), mesh=mesh, dtype=ade_state_dtype(), kind=family + '_state')
    return coeffs, state


def finish(invariants, carry, layout, mesh, *, waveforms, dt=None):
    """Publish already staged arrays and route the small source/probe tables."""
    from rfx.stepping.slab_wrap import cut_drives, cut_probe_table
    result = dict(invariants)
    result['drives'] = cut_drives(invariants['drives'], layout, mesh, waveforms=waveforms,
                                  per_rank=True)
    result['probe_table'] = cut_probe_table(invariants['probe_table'], layout, mesh)
    if 'port_table' in invariants:
        result['port_table'] = cut_probe_table(invariants['port_table'], layout, mesh)
    result['owns_x'] = {face: place(jnp.asarray([p[side] for p in layout.faces]),
        NamedSharding(mesh, P('x'))) for side, face in enumerate(('x_lo', 'x_hi'))}
    return result, carry


def build_setup(grid, materials, layout, mesh, n_steps, sources, probes, *,
                pec_mask=None, occupancy=None, debye=None, lorentz=None,
                cpml=False, cpml_state=None, emit_time_series=True, material_drive=None, port_plan=None):
    """Build the wrapped setup without a whole-domain _build_nu_scan."""
    from rfx.boundaries.setup import nonuniform_boundaries
    from rfx.nonuniform import resolve_cpml_axes
    from rfx.stepping.absorber import apply_cpml_h, apply_cpml_e_step
    from rfx.stepping.graded import GradedStepContext, make_graded_step, graded_probe_table
    from rfx.core.drives import drive_layout
    # Pole masks/parameters can be traced even when the cell arrays are fixed.
    pole_inputs = (None if debye is None else debye[0], None if lorentz is None else lorentz[0])
    device, reason = _material_host_device(materials, occupancy, pole_inputs, mesh=mesh)
    # Pole kernels drop their electric material components, just as one device
    # does; retain the absorber's operands separately in the native slab layout.
    staged, inverse, loss = realize_materials(materials, grid, layout, mesh, occupancy,
        cpml=cpml and (debye is not None or lorentz is not None),
        debye=debye is not None, lorentz=lorentz is not None, _material_inputs=pole_inputs)
    bundles = [debye, lorentz]
    for i, bundle in enumerate(bundles):
        if bundle is not None and bundle[0].ca is None:
            # Low-level compatibility for already staged fixed pole terms.
            from rfx.materials.debye import debye_e_coeffs
            from rfx.materials.lorentz import lorentz_e_coeffs
            coeffs, state = bundle
            fields = ('alpha', 'beta') if i == 0 else ('a', 'b', 'c')
            terms = [getattr(coeffs, name) for name in fields]
            def local(eps, sigma, *terms):
                return (debye_e_coeffs if i == 0 else lorentz_e_coeffs)((eps, sigma), grid.dt, *terms)
            def stage(eps, sigma, *terms):
                # Fixed pole terms use the same pole-first packing as the step.
                arrays = jax.tree.map(lambda a: a.reshape((mesh.size, -1) + a.shape[1:]),
                                      (eps, sigma, *terms))
                coeffs = jax.vmap(local)(*arrays)
                return jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), coeffs)
            coeffs = jax.checkpoint(stage)(staged.components.eps_update,
                                           staged.components.sigma_update, *terms)
            bundles[i] = (coeffs, state)
    debye, lorentz = bundles
    material_drive = (None,) * len(sources) if material_drive is None else material_drive
    if len(material_drive) != len(sources):
        raise ValueError('material_drive must contain one entry per source')
    waves = slab_drive_columns(staged, debye, lorentz, sources, material_drive,
                               grid.dt, layout, mesh, n_steps)
    faces = None if cpml else {f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')}
    (use_cpml, params, psi, cpml_grid, axes, curl_boundary, pec_faces, pmc_faces, use_pmc) = nonuniform_boundaries(
        grid, resolve_cpml_axes(grid, None if cpml else ''), faces, None)
    zeros = jnp.zeros((layout.n_devices * layout.nx_local,) + tuple(grid.shape[1:]),
                       dtype=jnp.float32, device=NamedSharding(mesh, P('x')))
    carry = dict(fdtd=FDTDState(*(zeros for _ in range(6)), step=jnp.asarray(0, dtype=jnp.int32)))
    if debye is not None:
        carry['debye'] = debye[1]
    if lorentz is not None:
        carry['lorentz'] = lorentz[1]
    if pec_mask is not None and not isinstance(pec_mask, tuple):
        from rfx.boundaries.pec import realized_pec_edge_masks
        cells = uniform_from_cut(pec_mask, layout, mesh, outside=False)
        masks = uniform_apply(lambda a: realized_pec_edge_masks(a), cells, mesh=mesh, row_count=layout.nx_local + 1)
        pec_mask = jax.tree.map(lambda a: product_rows(a, layout, mesh, 'pec_edge'), masks)
    inv = dict(inv_dx=cut(grid.inv_dx, layout, 'inv_dx', mesh=mesh),
        inv_dx_h=cut(grid.inv_dx_h, layout, 'inv_dx_h', mesh=mesh), dx_arr=cut(grid.dx_arr, layout, 'width', mesh=mesh),
        drives=drive_layout([(s.i, s.j, s.k, s.component) for s in sources], waves.dtype),
        probe_table=graded_probe_table([(p.i, p.j, p.k, p.component) for p in probes], grid.shape, dtype=zeros.dtype),
        materials=kernel_materials(staged, electric=not (debye or lorentz)), pec_edge_masks=pec_mask,
        pec_occupancy=(product_rows(occupancy, layout, mesh, 'pec_occupancy')
                       if occupancy is not None and occupancy.ndim == 4 else occupancy), pec_static_edge_masks=None, cpml_inv_eps_r=inverse, e_loss=loss,
        debye_coeffs=None if debye is None else debye[0], lorentz_coeffs=None if lorentz is None else lorentz[0],
        aniso_eps=None, sheet_impedance=None, sheet_coeffs=None, design_box_coeffs=None)
    ctx = GradedStepContext(use_lumped_rlc=False, rlc_metas=(), use_current_moments=False,
        _slab_e_snapshot=None, current_moments=None, dt=grid.dt, inv_dy_h=grid.inv_dy_h,
        inv_dz_h=grid.inv_dz_h, grid=grid, use_tfsf=False, tfsf_cfg=None, use_waveguide_ports=False,
        waveguide_meta=(), waveguide_port_metrics=(), use_cpml=use_cpml, apply_cpml_h=apply_cpml_h,
        cpml_params=params, cpml_grid=cpml_grid, cpml_axes_eff=axes, use_pmc_faces=use_pmc,
        _pmc_faces_frozen=pmc_faces, use_sheet_impedance=False, use_design_box=False,
        use_debye=debye is not None, use_lorentz=lorentz is not None, inv_dy=grid.inv_dy,
        inv_dz=grid.inv_dz, curl_boundary=curl_boundary, apply_cpml_e=apply_cpml_e_step,
        _pec_faces_frozen=pec_faces, use_pec_edges=pec_mask is not None, use_pec_occupancy=occupancy is not None,
        use_wire_ports=False, wp_meta=[], sp_freqs=None, use_dft_planes=False, dft_meta=(),
        use_flux_monitors=False, flux_meta=(), use_ntff=False, ntff_box=None, _accumulate_cm=None,
        emit_time_series=emit_time_series)
    from rfx.stepping.slab_ports import setup_ports
    ctx = setup_ports(ctx, carry, inv, port_plan, layout, mesh, n_steps)
    if cpml:
        # rfx #1402 item 5 (absorber)
        from rfx.stepping.slab_cpml import publish, import_state
        shell = SimpleNamespace(step_fn_inv=SimpleNamespace(context=ctx), invariants={}, carry_init={'cpml': psi})
        ctx, profiles, initial = publish(shell, layout, mesh)
        inv['cpml_x'] = profiles['cpml_x']
        # Native x faces retain the full one-device depth on every rank.
        # The y/z faces already contain the slab's x rows and ghosts.
        carry['cpml'] = initial['cpml']
        if cpml_state is not None:
            carry['cpml'] = import_state(cpml_state, psi, layout, native=True)
    return SimpleNamespace(invariants=inv, carry_init=carry, src_waveforms=waves,
                            step_fn_inv=make_graded_step(ctx),
                            materials_realized='whole_grid_host' if device is not None else 'per_slab',
                            materials_realized_reason=reason)


def stage_pec(cells, sheets, wires, grid, layout, mesh):
    """Concrete declaration views plus uniform, possibly traced cell masks."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.stepping.slab_view import SlabView
    if cells is None and not sheets and not wires:
        return None
    rows = uniform_slab_rows(layout)
    static = None
    if sheets or wires:
        per_rank = []
        for rank, indices in enumerate(rows.indices):
            lo, hi = max(0, int(indices[0])), min(layout.nx, int(indices[-1]) + 1)
            view = SlabView(rank, layout.n_devices, lo, hi, 0, 0, grid, {})
            masks = realized_pec_edge_masks(None, sheets, wires, view=view)
            per_rank.append(tuple(jnp.pad(a, ((lo - int(indices[0]), int(indices[-1]) + 1 - hi),
                                               (0, 0), (0, 0))) for a in masks))
        static = tuple(place(jnp.stack([rank[c] for rank in per_rank]),
                       NamedSharding(mesh, P('x'))) for c in range(3))
    if cells is None:
        masks = static
    else:
        cell_rows = stack_and_place(cells, rows, mesh, outside=False)
        indices = place(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
        def local(cells, index, fixed):
            view = UniformSlabView(index, grid, {} if fixed is None else {'pec_static_edge_masks': fixed})
            return realized_pec_edge_masks(cells, view=view)
        masks = uniform_apply(local, cell_rows, indices, static, mesh=mesh, row_count=layout.nx_local + 1)
    return jax.tree.map(lambda a: product_rows(a, layout, mesh, 'pec_edge'), masks)
