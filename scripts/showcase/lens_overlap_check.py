"""One fixed two-plane overlap comparison; prints measurements, never a verdict."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def cosine(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(den) or den==0:raise ValueError('cosine undefined for zero/nonfinite map')
    return float(np.dot(a,b)/den)


def map_gradient(gradient,plane,u,v,offset):
    if gradient.shape!=(15,15,10):raise ValueError('expected 2,250 free pixels')
    U,V=np.meshgrid(u,v,indexing='ij');fixed=np.full_like(U,offset)
    X,Y,Z=(U,fixed,V) if plane=='xz' else (fixed,U,V)
    # Stored realized nodes, half-open printable cells, mirrored free-pixel map.
    full=[np.floor((c-origin)/.003+1e-5).astype(int) for c,origin in [(X,.018),(Y,.018),(Z,.072)]]
    inside=(full[0]>=0)&(full[0]<30)&(full[1]>=0)&(full[1]<30)&(full[2]>=0)&(full[2]<10)
    i,j,k=[f[inside] for f in full];i=np.where(i<15,14-i,i-15);j=np.where(j<15,14-j,j-15)
    return inside,gradient[i,j,k]


def compare(record,forward,adjoint,iterate,far_field_key=None):
    with np.load(record,allow_pickle=False) as z:
        if not np.array_equal(z['iteration'],np.arange(len(z['iteration']))):raise ValueError('record iteration indexing unsupported')
        grad=z['grad_eps'][iterate];freqs=z['freqs_hz'];fi=np.flatnonzero(np.isclose(freqs,1e10,rtol=0,atol=1))
        if len(fi)!=1:raise ValueError('unique 10 GHz bin required')
        key=far_field_key or ('boresight_E_x' if 'boresight_E_x' in z else None)
        ff=None
        if key:
            values=z[key]
            if not np.iscomplexobj(values) or values.shape!=(len(z['iteration']),len(freqs)):raise ValueError('far-field key must be complex [iterate,frequency] boresight Ex')
            ff=complex(values[iterate,fi[0]])
    maps={};rows=[];targets=[]
    with np.load(forward,allow_pickle=False) as f,np.load(adjoint,allow_pickle=False) as a:
        for data in (f,a):
            if int(data['iterate'])!=iterate or not np.allclose(data['freqs_hz'],[1e10],rtol=0,atol=1):raise ValueError('field iterate/frequency mismatch')
        for key in ('n_steps','dx_m'):
            if key in f and key in a and not np.array_equal(f[key],a[key]):raise ValueError('field record length/mesh mismatch')
        if 'TFSF' in str(f['drive']):raise ValueError('forward field must use feed drive')
        if 'empty' in a and bool(a['empty']):raise ValueError('empty witness cannot be used as lens adjoint')
        if 'reciprocal' not in str(a['drive']) and 'TFSF' not in str(a['drive']):raise ValueError('reciprocal drive required')
        for plane in ('xz','yz'):
            for suffix in ('axes','u_m','v_m','offset_m','eps'):
                if not np.array_equal(f[f'{plane}_{suffix}'],a[f'{plane}_{suffix}']):raise ValueError(f'field grid/material mismatch: {plane}_{suffix}')
            if str(f[f'{plane}_axes'])!=plane:raise ValueError('plane axis order mismatch')
            Ef,Ea=f[f'{plane}_E'],a[f'{plane}_E']
            if Ef.shape!=Ea.shape or Ef.shape[:2]!=(1,3):raise ValueError('field shape mismatch')
            h=np.sum(Ef[0].astype(np.complex128)*Ea[0].astype(np.complex128),axis=0)
            mask,g=map_gradient(grad,plane,f[f'{plane}_u_m'],f[f'{plane}_v_m'],float(f[f'{plane}_offset_m']))
            if not np.isfinite(h).all() or not np.isfinite(g).all():raise ValueError('nonfinite overlap inputs')
            rows.append(np.column_stack([h[mask].real,h[mask].imag]));targets.append(g)
            maps[plane]=dict(h=h,mask=mask,gradient=g,u=f[f'{plane}_u_m'],v=f[f'{plane}_v_m'])
    if ff is None:
        # One joint complex constant, not a separate fit/window for each plane.
        matrix=np.concatenate(rows);target=np.concatenate(targets)
        fitted,_,rank,_=np.linalg.lstsq(matrix,target,rcond=None)
        constant=complex(fitted[0],-fitted[1]);mode='fitted jointly to both complete block-plane maps'
    else:constant=np.conj(ff);mode='fixed conj(stored boresight Ex)';rank=None
    output={};similarities={}
    for plane,m in maps.items():
        overlap=np.real(constant*m['h']);ad=np.full(overlap.shape,np.nan);ad[m['mask']]=m['gradient']
        similarities[plane]=cosine(overlap[m['mask']],m['gradient'])
        overlap[~m['mask']]=np.nan
        output.update({f'{plane}_AD':ad,f'{plane}_g_ov':overlap,f'{plane}_mask':m['mask'],f'{plane}_u_m':m['u'],f'{plane}_v_m':m['v']})
    return dict(iterate=iterate,frequency_hz=1e10,constant_mode=mode,
      multiplying_constant=[constant.real,constant.imag],E_ff=None if ff is None else [ff.real,ff.imag],fit_rank=rank,
      cosine_similarity=similarities,far_field_key=key,
      convention='g_ov=Re[C*sum_components(E_fwd*E_adj)]; C=conj(E_ff), or one joint fitted complex constant; native Yee samples; raw stored AD objective gradient, no sign flip',
      limitation='Stored AD gradient is the three-frequency directivity objective; this 10 GHz overlap contains only the radiated-intensity contribution.'),output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('record','forward','adjoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterate',type=int,default=0);p.add_argument('--far-field-key')
    args=p.parse_args()
    if args.iterate<0:p.error('nonnegative iterate required')
    if (args.out/'overlap.json').exists():p.error('comparison output already exists; refusing overwrite')
    report,maps=compare(args.record,args.forward,args.adjoint,args.iterate,args.far_field_key)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'overlap_maps.npz',**maps)
    (args.out/'overlap.json').write_text(json.dumps(report,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o))+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(args.out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for plane in ('xz','yz'):
        ad,ov=maps[plane+'_AD'],maps[plane+'_g_ov'];limit=float(max(np.nanmax(abs(ad)),np.nanmax(abs(ov))))
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,array,title in zip(axes,(ad,ov),('stored AD gradient','reciprocal overlap')):
            image=ax.pcolormesh(maps[plane+'_u_m']*1e3,maps[plane+'_v_m']*1e3,array.T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
            ax.set(xlabel=plane[0]+' (mm)',ylabel='z (mm)',title=title);ax.set_aspect('equal')
        fig.colorbar(image,ax=axes);fig.savefig(args.out/f'overlap_{plane}.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=='__main__':main()
