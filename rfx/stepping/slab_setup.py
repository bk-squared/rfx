"""Slab-local inputs for the canonical graded step (rfx #1402 item 6)."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.core.yee import FDTDState
from rfx.model.materials import (ComponentCells, realize_components, kernel_materials)
from rfx.stepping.slab import cut, zero_pole_state
from rfx.stepping.slab_view import (UniformSlabView, uniform_slab_rows, uniform_apply,
                                   uniform_from_cut, stack_and_place, product_rows, slab_views)


def input_materials(materials, layout, mesh):
    return jax.tree.map(lambda a: uniform_from_cut(a, layout, mesh),
                        materials._replace(components=None, edge_keep=None))


def realize_materials(materials, grid, layout, mesh, occupancy=None, *, cpml=False,
                      debye=False, lorentz=False):
    """Run the owners' producers eagerly, preserving their operation order."""
    from rfx.model.occupancy import with_edge_keep
    from rfx.stepping.absorber import graded_cpml_operands
    inputs = input_materials(materials, layout, mesh)
    rows = uniform_slab_rows(layout)
    indices = jax.device_put(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
    occ = uniform_from_cut(occupancy, layout, mesh, outside=0)

    def local(cells, index, occupancy):
        view = UniformSlabView(index, grid, {})
        components = realize_components(cells, grid, periodic=(False,) * 3, view=view)
        cells = cells._replace(components=components)
        cells = with_edge_keep(cells, occupancy, dtype=jnp.float32, view=view)
        inverse, loss = graded_cpml_operands(cells, None, use_cpml=cpml,
                                            use_debye=debye, use_lorentz=lorentz)
        return cells, inverse, loss
    staged, inverse, loss = uniform_apply(local, inputs, indices, occ,
                                          row_count=layout.nx_local + 1)
    def finish(path, a):
        if not hasattr(a, 'ndim') or a.ndim < 4:
            return a
        names = [str(getattr(p, 'name', getattr(p, 'key', ''))) for p in path]
        name = next((p for p in reversed(names) if p), '')
        kind = ('edge_keep' if 'edge_keep' in names else
                'eps_r' if 'eps' in name else 'sigma' if 'sigma' in name
                else 'mu_r' if 'mu' in name else 'lumped')
        return product_rows(a, layout, mesh, kind)
    staged = jax.tree_util.tree_map_with_path(finish, staged)
    inverse = jax.tree.map(lambda a: product_rows(a, layout, mesh, 'eps_r'), inverse)
    if loss is not None:
        loss = (jax.tree.map(lambda a: product_rows(a, layout, mesh, 'eps_r'), loss[0]),
                jax.tree.map(lambda a: product_rows(a, layout, mesh, 'sigma'), loss[1]))
    return staged, inverse, loss


def stage_dispersion(materials, dt, spec, grid, layout, mesh, family):
    """Judge owned cell reductions, then form ADE operands on uniform slabs."""
    from rfx.model.pole_stability import (pole_set_extremes, combine_pole_extremes,
                                         judge_pole_sets, courant_budget)
    from rfx.materials.debye import debye_e_coeffs, DebyeState
    from rfx.materials.lorentz import lorentz_e_coeffs, LorentzState
    from rfx.core.yee import ade_state_dtype
    if spec is None:
        return None
    poles, masks = spec
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * len(poles)
    if family == 'lorentz':
        records = []
        for view in slab_views(grid, layout.n_devices):
            # Reductions consume only owned cells; no whole-domain realization.
            start = view.rank * layout.nx_local
            uniform = materials.eps_r.ndim == 4
            local = jax.tree.map(lambda a: (a[view.rank] if uniform else a[start:start + layout.nx_local])
                if a is not None and np.ndim(a) else a, materials)
            halo = 2 if uniform else layout.ghost_width
            owned_view = view._replace(halo_lo=halo,
                halo_hi=local.eps_r.shape[0] - halo - (view.x1 - view.x0))
            indices = uniform_slab_rows(layout).indices[view.rank]
            local_masks = [None if m is None else
                (m[np.clip(indices, 0, layout.nx - 1)] if uniform else
                 cut(m, layout, 'pole_mask', host=not isinstance(m, jax.core.Tracer))[view.rank])
                for m in masks]
            records.append(pole_set_extremes(local, (poles, local_masks), view=owned_view))
        judge_pole_sets(combine_pole_extremes(records), dt, courant_budget(grid),
                        grid_kind=type(grid).__name__)
    inputs = input_materials(materials, layout, mesh)
    rows = uniform_slab_rows(layout)
    indices = jax.device_put(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
    staged_masks = [None if m is None else stack_and_place(m, rows, mesh) for m in masks]
    spatial = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
    def local(cells, index, masks):
        view = UniformSlabView(index, grid, {'pole_sets_judged': True})
        specs = dict(debye_spec=None, lorentz_spec=None)
        specs[family + '_spec'] = (poles, masks)
        c = realize_components(ComponentCells(cells, **specs), grid,
                                periodic=(False,) * 3, view=view)
        terms = getattr(c, family).coefficients
        coeffs = (debye_e_coeffs if family == 'debye' else lorentz_e_coeffs)(
            (c.eps_update, c.sigma_update), dt, *terms)
        return type(coeffs)(*(value if name in spatial else
            jax.tree.map(lambda a: jnp.moveaxis(a, 1, 0), value)
            for name, value in zip(coeffs._fields, coeffs)))
    coeffs = uniform_apply(local, inputs, indices, staged_masks, row_count=layout.nx_local + 1)
    coeffs = type(coeffs)(*(jax.tree.map(
        lambda a: product_rows(a if name in spatial else jnp.moveaxis(a, 2, 1),
            layout, mesh, family + '_' + name, row_axis=1 if name in spatial else 2, dt=dt), value)
        for name, value in zip(coeffs._fields, coeffs)))
    state = zero_pole_state(DebyeState if family == 'debye' else LorentzState,
        len(poles), layout, tuple(grid.shape[1:]), mesh=mesh, dtype=ade_state_dtype(), kind=family + '_state')
    return coeffs, state


def finish(invariants, carry, layout, mesh, *, waveforms, dt=None):
    """Publish already staged arrays and route the small source/probe tables."""
    from rfx.stepping.slab_wrap import cut_drives, cut_probe_table
    result = dict(invariants)
    result['drives'] = cut_drives(invariants['drives'], layout, mesh, waveforms=waveforms)
    result['probe_table'] = cut_probe_table(invariants['probe_table'], layout, mesh)
    result['owns_x'] = {face: jax.device_put(jnp.asarray([p[side] for p in layout.faces]),
        NamedSharding(mesh, P('x'))) for side, face in enumerate(('x_lo', 'x_hi'))}
    return result, carry


def build_setup(grid, materials, layout, mesh, n_steps, sources, probes, *,
                pec_mask=None, occupancy=None, debye=None, lorentz=None,
                cpml=False, cpml_state=None, emit_time_series=True, material_drive=None):
    """Build the wrapped setup without a whole-domain _build_nu_scan."""
    from rfx.boundaries.setup import nonuniform_boundaries
    from rfx.nonuniform import resolve_cpml_axes
    from rfx.stepping.absorber import apply_cpml_h, apply_cpml_e_step
    from rfx.stepping.graded import GradedStepContext, make_graded_step, graded_probe_table
    from rfx.core.drives import drive_layout
    from rfx.nonuniform import drive_table
    from rfx.model.materials import e_update_coefficient_at
    from rfx.model.source_coefficients import source_table
    from rfx import _realized
    staged, inverse, loss = realize_materials(materials, grid, layout, mesh, occupancy,
        cpml=cpml, debye=debye is not None, lorentz=lorentz is not None)
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
            from rfx.stepping.rank import rank_shard_map, mesh_ranks
            f = rank_shard_map(lambda *a, rank: local(*a), mesh=mesh,
                in_specs=(P('x'),) * (2 + len(terms)), out_specs=P('x'), check_rep=False)
            coeffs = jax.jit(f)(mesh_ranks(mesh), staged.components.eps_update,
                                  staged.components.sigma_update, *terms)
            bundles[i] = (coeffs, state)
    debye, lorentz = bundles
    source_rows = []
    material_drive = (None,) * len(sources) if material_drive is None else material_drive
    if len(material_drive) != len(sources):
        raise ValueError('material_drive must contain one entry per source')
    for source, volume in zip(sources, material_drive):
        samples = source.waveform
        # rfx #1402 item 4 (nu): drive_table slab view plugs in here
        owner, local = layout.owner(source.i), layout.local_index(source.i)
        cell = (owner * layout.nx_local + local, source.j, source.k)
        axis = ('ex', 'ey', 'ez').index(source.component)
        if volume is not None:
            volume = volume[0] if isinstance(volume, tuple) else volume
            if debye is not None and lorentz is not None:
                from rfx.materials.lorentz import mixed_e_component_coeffs
                # Reduce pole terms within each slab, never over packed ranks.
                def coeff_at(bundle):
                    c = bundle[0]
                    return c._replace(**{name: jax.tree.map(lambda a: a.reshape(
                        (layout.n_devices, -1, layout.nx_local) + a.shape[2:])[owner], value)
                        for name, value in zip(c._fields, c) if name not in ('ca', 'cb', 'cc')})
                d, l = coeff_at(debye), coeff_at(lorentz)
                # Spatial fields use the same owning slab as the pole terms.
                d = d._replace(ca=tuple(a[owner*layout.nx_local:(owner+1)*layout.nx_local] for a in d.ca),
                               cb=tuple(a[owner*layout.nx_local:(owner+1)*layout.nx_local] for a in d.cb),
                               cc=tuple(a.reshape((layout.n_devices, -1, layout.nx_local)+a.shape[2:])[owner] for a in d.cc))
                l = l._replace(**{name: tuple(a[owner*layout.nx_local:(owner+1)*layout.nx_local] for a in getattr(l,name)) for name in ('ca','cb','cc')})
                cb = mixed_e_component_coeffs(d, l, axis, grid.dt)[1][local, source.j, source.k]
            elif debye is not None or lorentz is not None:
                cb = (debye or lorentz)[0].cb[axis][cell]
            else:
                cb = e_update_coefficient_at(staged, cell, source.component, grid.dt, host=True)
            if _realized.ACTIVE is None:
                samples = source_table(samples, native='cb_over_dv', volume=volume, coefficient=lambda: cb)
            else:
                samples = _realized.runtime_drive(jnp.reshape(cb / volume, (1,)), (len(source_rows),))[0] * samples
        source_rows.append((source.i, source.j, source.k, source.component, samples))
    # drive_table reads only source nodes. Present the already staged owner
    # indices so occupancy weighting is unchanged without a whole keep array.
    source_meta = [(layout.owner(s.i)*layout.nx_local + layout.local_index(s.i), s.j, s.k, s.component) for s in sources]
    waves = drive_table([jnp.array(s[4]) for s in source_rows], source_meta, n_steps,
                        keep=staged.edge_keep)
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
        masks = uniform_apply(lambda a: realized_pec_edge_masks(a), cells, row_count=layout.nx_local + 1)
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
    if cpml:
        # rfx #1402 item 5 (absorber)
        from rfx.stepping.slab_cpml import publish, import_state
        if cpml_state is not None:
            psi = import_state(cpml_state, psi, layout)
        shell = SimpleNamespace(step_fn_inv=SimpleNamespace(context=ctx), invariants={}, carry_init={'cpml': psi})
        ctx, profiles, initial = publish(shell)
        inv['cpml_x'] = {name: cut(a, layout, 'width' if 'dx_x_' in name else
            'cpml_one' if name.rsplit('.', 1)[-1] in ('b', 'kappa') else 'cpml_zero', mesh=mesh)
            for name, a in profiles['cpml_x'].items()}
        carry['cpml'] = cut(initial['cpml'], layout, 'cpml_psi', mesh=mesh)
    return SimpleNamespace(invariants=inv, carry_init=carry, src_waveforms=waves,
                            step_fn_inv=make_graded_step(ctx))


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
        static = tuple(jax.device_put(jnp.stack([rank[c] for rank in per_rank]),
                       NamedSharding(mesh, P('x'))) for c in range(3))
    if cells is None:
        masks = static
    else:
        cell_rows = stack_and_place(cells, rows, mesh, outside=False)
        indices = jax.device_put(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))
        def local(cells, index, fixed):
            view = UniformSlabView(index, grid, {} if fixed is None else {'pec_static_edge_masks': fixed})
            return realized_pec_edge_masks(cells, view=view)
        masks = uniform_apply(local, cell_rows, indices, static, row_count=layout.nx_local + 1)
    return jax.tree.map(lambda a: product_rows(a, layout, mesh, 'pec_edge'), masks)
