"""Render final-field energy distributions; no time stepping."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_config')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=[]
for diag_file in sorted((ROOT/'raw_3').glob('diagnostic_*/**/field_diagnostic_*.json')):
    diag=json.loads(diag_file.read_text())
    path=diag_file.parent/diag['field_dump']
    with np.load(path) as z:
        coords=[z[a+'_nodes_m'].copy() for a in 'xyz']
        dx=float(coords[0][1]-coords[0][0])
        components={c:.5*dx**3*(8.8541878128e-12*z['eps_r'] if c.startswith('e') else 1.25663706212e-6*z['mu_r']).astype(float)*z[c].astype(float)**2 for c in ('ex','ey','ez','hx','hy','hz')}
        energy=sum(components.values())
        interior=z['interior_mask'].copy()
        x=np.sum(energy,axis=(1,2))
        x_inner=np.sum(energy*interior,axis=(1,2))
        quantiles=np.searchsorted(np.cumsum(x_inner),np.array([.025,.5,.975])*x_inner.sum())
        near={}
        ix=np.arange(energy.shape[0])
        for radius in (0,1,2,4,8,16):
            selected=np.zeros(energy.shape[0],bool)
            for port in diag['port_masks']:
                selected |= abs(ix-port['x_index'])<=radius
            near[str(radius)]={'half_width_m':radius*dx,'interior_fraction':float(x_inner[selected].sum()/x_inner.sum())}
        row=dict(field_dump=str(path.relative_to(ROOT)),component_energy_J={c:float(v.sum()) for c,v in components.items()},
                 interior_component_energy_J={c:float(v[interior].sum()) for c,v in components.items()},
                 interior_longitudinal_energy_quantiles_m=dict(zip(['2.5%','50%','97.5%'],[float(coords[0][q]) for q in quantiles])),
                 interior_energy_near_port_x_planes=near)
        rows.append(row)
        fig,axs=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
        for ax,remaining,summed,label in [(axs[0],1,2,'y'),(axs[1],2,1,'z')]:
            projected=energy.sum(axis=summed)
            db=10*np.log10(np.maximum(projected/projected.max(),1e-10))
            im=ax.pcolormesh(coords[0]*1e3,coords[remaining]*1e3,db.T,shading='auto',vmin=-80,vmax=0,cmap='magma')
            for p in diag['port_masks']:
                ax.axvline(coords[0][p['x_index']]*1e3,color='cyan',linewidth=.6)
            for q in (np.where(interior.any(axis=(1,2)))[0][0],np.where(interior.any(axis=(1,2)))[0][-1]):
                ax.axvline(coords[0][q]*1e3,color='white',linestyle='--',linewidth=.6)
            ax.set(xlabel='x (mm)',ylabel=label+' (mm)',title='Sum over '+('z' if summed==2 else 'y'))
        fig.colorbar(im,ax=axs[:2],label='Projected energy / peak (dB)',shrink=.85)
        axs[2].plot(coords[0]*1e3,10*np.log10(np.maximum(x/x.max(),1e-12)),label='all fields')
        axs[2].plot(coords[0]*1e3,10*np.log10(np.maximum(x_inner/x.max(),1e-12)),label='interior')
        axs[2].set(xlabel='x (mm)',ylabel='Slice energy / peak (dB)',ylim=(-120,5))
        axs[2].legend()
        fig.suptitle(str(path.parent.relative_to(ROOT))+'/'+path.stem)
        fig.savefig(path.with_suffix('.png'),dpi=160)
        plt.close(fig)
        print('FIELD_MAP',path.with_suffix('.png').relative_to(ROOT))
(ROOT/'FIELD_SPATIAL_DETAIL_3.json').write_text(json.dumps(rows,indent=2)+'\n')
