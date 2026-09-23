"""Final-field dumps for only MSL s=0,N=8 and WR-90 s=0,N=4."""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
import time
import traceback
import types
import warnings
import sweep_driver as m
import extra_rigs

np = m.np


class DumpRecorder(m.Recorder):
    def __init__(self, out, sim, rig, dry=False):
        super().__init__(out, 0, dry)
        self.sim, self.rig = sim, rig

    def core(self, ctx):
        self.ctx = ctx
        return super().core(ctx)

    def run(self, grid, materials, n_steps, *args, **kwargs):
        result = super().run(grid, materials, n_steps, *args, **kwargs)
        state = result.state
        fields = {c: np.asarray(getattr(state, c)) for c in ('ex','ey','ez','hx','hy','hz')}
        eps, mu = np.asarray(materials.eps_r), np.asarray(materials.mu_r)
        coords = m.coords_from_uniform_grid(grid)
        interior = np.zeros(grid.shape, bool)
        interior[tuple(slice(getattr(grid,'pad_'+a+'_lo'), getattr(grid,'n'+a)-getattr(grid,'pad_'+a+'_hi')) for a in 'xyz')] = True
        under_port = np.zeros(grid.shape, bool)
        port_defs = []
        if self.rig == 'msl':
            for pe in self.sim._resolve_msl_probe_entries(grid):
                port = m.msl_port_from_entry(pe)
                span = m.msl_cross_section_span(grid, port)
                i = int(np.argmin(abs(np.asarray(coords.x)-port.feed_x)))
                j0,j1,k0,k1 = (int(span[x]) for x in ('w_lo','w_hi','n_lo','n_hi'))
                under_port[i,j0:j1+1,k0:k1+1] = True
                port_defs.append(dict(name=pe.name, x_index=i, width_nodes=[j0,j1], normal_nodes=[k0,k1]))
        else:
            for cfg in self.ctx.waveguide_meta:
                under_port[cfg.x_index,cfg.u_lo:cfg.u_hi+1,cfg.v_lo:cfg.v_hi+1] = True
                port_defs.append(dict(x_index=int(cfg.x_index),width_nodes=[int(cfg.u_lo),int(cfg.u_hi)],normal_nodes=[int(cfg.v_lo),int(cfg.v_hi)]))
        source_mask = np.zeros(grid.shape, bool)
        sources = kwargs.get('sources') or []
        for src in sources:
            source_mask[src.i,src.j,src.k] = True
        ef = .5*float(grid.dx)**3*m.EPS0*eps.astype(float)*sum(fields['e'+a].astype(float)**2 for a in 'xyz')
        hf = .5*float(grid.dx)**3*m.MU0*mu.astype(float)*sum(fields['h'+a].astype(float)**2 for a in 'xyz')
        total = ef+hf
        regions = {'interior_under_port':interior & under_port,
                   'interior_elsewhere':interior & ~under_port,
                   'absorber':~interior}
        sums = {}
        for name, mask in regions.items():
            sums[name] = dict(E_J=float(ef[mask].sum()), H_J=float(hf[mask].sum()), energy_J=float(total[mask].sum()), nodes=int(mask.sum()),
                              fraction_of_all_energy=float(total[mask].sum()/total.sum()))
        extra_masks = dict(interior=interior, source_nodes=source_mask)
        for axis,a in enumerate('xyz'):
            for side in ('lo','hi'):
                pad = getattr(grid,f'pad_{a}_{side}')
                mask = np.zeros(grid.shape,bool)
                if pad:
                    sl=[slice(None)]*3
                    sl[axis]=slice(0,pad) if side=='lo' else slice(grid.shape[axis]-pad,None)
                    mask[tuple(sl)]=True
                extra_masks[a+'_'+side] = mask
        extra = {name:dict(E_J=float(ef[mask].sum()),H_J=float(hf[mask].sum()),energy_J=float(total[mask].sum())) for name,mask in extra_masks.items()}
        maxima = {}
        for c,v in fields.items():
            ix=np.unravel_index(np.argmax(abs(v)),v.shape)
            maxima[c]=dict(value=float(v[ix]),ijk=list(ix),coordinate_m=[float(getattr(coords,a)[ix[k]]) for k,a in enumerate('xyz')],
                           region=next(name for name,mask in regions.items() if mask[ix]))
        arrays=dict(fields,eps_r=eps,mu_r=mu,interior_mask=interior,under_port_mask=under_port,source_mask=source_mask)
        if self.ctx.pec_edge_masks is not None:
            arrays.update({f'pec_{a}':np.asarray(v) for a,v in zip('xyz',self.ctx.pec_edge_masks)})
        for a in 'xyz':
            arrays[a+'_nodes_m']=np.asarray(getattr(coords,a))
        idx=len(self.calls)-1
        np.savez_compressed(self.out/f'fields_end_{idx:02d}.npz',**arrays)
        np.savez_compressed(self.out/f'energy_marginals_{idx:02d}.npz',**{a+'_J':total.sum(axis=tuple(k for k in range(3) if k!=i)) for i,a in enumerate('xyz')})
        diagnostic=dict(regions=sums,overlapping_measurements=extra,field_maxima=maxima,
                        port_masks=port_defs,definition='Same-step Yee field energy, 0.5*dx^3*(eps*E^2+mu*H^2); absorber excludes CPML auxiliary variables.',
                        port_mask_definition='One source/feed x node plane over the realized port width and ground-to-strip span (MSL), or entire modal aperture (WR-90).',
                        total_energy_J=float(total.sum()), interior_energy_J=float(total[interior].sum()),
                        energy_partition_sum_J=sum(v['energy_J'] for v in sums.values()),
                        field_dump=f'fields_end_{idx:02d}.npz')
        self.calls[-1]['final_field_diagnostic']=diagnostic
        m.write_json(self.out/f'field_diagnostic_{idx:02d}.json',diagnostic)
        print('FIELD_DIAGNOSTIC',json.dumps(m.serial(diagnostic)),flush=True)
        return result


def run_msl(out,dry):
    rows={}
    for label,single in [('two_port',False),('one_port',True)]:
        dest=out/label
        dest.mkdir()
        sim=m.build_msl(8,single,False)
        rec=DumpRecorder(dest,sim,'msl',dry)
        rec.install()
        row=dict(realized=m.msl_geometry(sim),preflight_text=m.preflight(sim,dest))
        try:
            res=sim.compute_msl_s_matrix(freqs=m.jnp.asarray(np.linspace(3e9,4.5e9,81)),num_periods=12,enforce_passivity=False,report_every=None)
            s=np.asarray(res.S)
            np.savez_compressed(dest/'sparams.npz',S=s,freqs_Hz=np.asarray(res.freqs))
            row.update(max_S11_db=float(20*np.log10(abs(s[0,0]).max())),settling_db=m.serial(res.settling_db))
        except m.DryCaptured:
            row['status']='build-only'
        row['solves']=rec.calls
        m.write_json(dest/'result.json',row)
        rows[label]=row
    return rows


def run_waveguide(out,dry):
    from rfx.sources.waveguide_port import extract_waveguide_s11,settling_db_from_port_records
    from rfx.sparams._common import _warn_if_nonpassive_smatrix
    sim=extra_rigs.build_waveguide(4)
    rec=DumpRecorder(out,sim,'waveguide',dry)
    rec.install()
    row=dict(preflight_text=m.preflight(sim,out))
    try:
        sim.run(num_periods=40,compute_s_params=False,skip_preflight=True)
        cfg=rec.last_result.waveguide_ports[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            s11=np.asarray(extract_waveguide_s11(cfg))
        row['lowlevel_extractor_warnings']=[str(w.message) for w in caught]
        with warnings.catch_warnings(record=True) as guard:
            warnings.simplefilter('always')
            _warn_if_nonpassive_smatrix(types.SimpleNamespace(s_params=s11[None,None,:],freqs=cfg.freqs,port_names=('port1',)),
                                       extractor='extract_waveguide_s11 (diagnostic replay through shared guard)',
                                       strict=False,passivity_tol=2.0)
        row['shared_guard_warnings']=[str(w.message) for w in guard]
        (out/'passivity_warning.txt').write_text('\n'.join(row['shared_guard_warnings'])+'\n')
        np.savez_compressed(out/'modal_records.npz',**{k:np.asarray(getattr(cfg,k)) for k in ('v_probe_t','v_ref_t','i_probe_t','i_ref_t','v_inc_t','e_inc_table','h_inc_table')})
        np.savez_compressed(out/'sparams.npz',S=s11[None,None,:],freqs_Hz=np.asarray(cfg.freqs))
        row.update(max_S11_db=float(20*np.log10(abs(s11).max())),max_column_power=float((abs(s11)**2).max()),
                   worst_frequency_Hz=float(np.asarray(cfg.freqs)[np.argmax(abs(s11))]),settling_db=settling_db_from_port_records([cfg]))
        print('PASSIVITY_TEXT',row['shared_guard_warnings'],flush=True)
    except m.DryCaptured:
        row['status']='build-only'
    row['solves']=rec.calls
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('rig',choices=['msl','waveguide'])
    ap.add_argument('--dry',action='store_true')
    args=ap.parse_args()
    m.jax.config.update('jax_enable_x64',False)
    if not args.dry:
        assert all(d.platform=='gpu' for d in m.jax.devices())
    m.cpml._cpml_profile=m.scaled_profile(0)
    out=m.ROOT/('dry_3' if args.dry else 'raw_3')/('diagnostic_'+args.rig)
    out.mkdir(parents=True)
    row=dict(rig=args.rig,scale=0,layers=8 if args.rig=='msl' else 4,
             commit=(m.SRC/'PROVENANCE.txt').read_text().strip(),dtype='float32',
             run_id=None if args.dry else (m.ROOT/'jobs_3'/args.rig/'run_id.txt').read_text().strip())
    assert row['commit']==m.EXPECTED_SHA
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        with contextlib.redirect_stdout(m.Tee(sys.stdout,log)),contextlib.redirect_stderr(m.Tee(sys.stderr,log)):
            try:
                row['measurements']=(run_msl if args.rig=='msl' else run_waveguide)(out,args.dry)
                row['status']='build-only' if args.dry else 'complete'
            except Exception as exc:
                row.update(status='STOP',exception=repr(exc),traceback=traceback.format_exc())
                print(row['traceback'],flush=True)
            row['wall_s']=time.monotonic()-start
            m.write_json(out/'result.json',row)
            if not args.dry:
                m.write_json(m.ROOT/'results_3'/('diagnostic_'+args.rig+'.json'),row)
            print('DIAGNOSTIC_FINAL',row['status'],args.rig,flush=True)
    return int(row['status']=='STOP')


if __name__=='__main__':
    raise SystemExit(main())
