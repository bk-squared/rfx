"""Make factual Addendum 2 tables from retained numeric records."""
import csv
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/'results_3'/name).read_text())


def table(title,rows):
    return title+'\n\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'


scales=(0,1,3)
base=[['N \\ s',*scales],['---']*4]
for n in (4,8,16):
    base.append([n,*[f"{read(f'exact_2GHz_{s}_{n}.json')['reflectivity_db']:.6f}" for s in scales]])
(ROOT/'TABLE_plane.md').write_text(table('Committed point-source oracle: peak reference-subtracted difference / peak reference, dB. Source centre 2 GHz; original 400-step window and PEC reference declared as 200 mm (realized 200.860947 mm).',base))
frequency=[['Source centre (GHz) \\ s',*scales],['---']*4]
sizes=[['f0 (GHz)','Steps','Window end (ns)','Reference side (m)','Reference nodes','Nearest geometric wall echo (ns)'],['---']*6]
for f in (.5,1,2,4):
    arms=[read(f'frequency_{f:g}GHz_{s}_8.json') for s in scales]
    frequency.append([f,*[f"{r['reflectivity_db']:.6f}" if r['status']=='complete' else 'STOP' for r in arms]])
    r=arms[0]
    if r['status']=='complete':
        sizes.append([f,r['record_steps'],f"{r['window_s'][1]*1e9:.9f}",f"{r['reference_declared_side_m']:.11g}",'x'.join(map(str,r['reference']['shape'])),f"{r['reference']['earliest_geometric_wall_echo_s']*1e9:.9f}"])
    else:
        sizes.append([f,r['sizing']['n_steps'],f"{r['sizing']['window_ns']:.9f}",r['sizing']['reference_side_m'],'x'.join(map(str,r['sizing']['reference_shape'])),'not run'])
(ROOT/'TABLE_plane_frequency.md').write_text(table('Oracle method at N=8: peak time-domain ratio in dB, one source centre per run. Mesh remains dx=c/(20*5 GHz); CPML domain remains a realized 62.956416 mm cube (declared 60 mm).',frequency)+'\n'+table('Reference and record dimensions. The 0.5 GHz row contains build estimates only.',sizes))
spatial=[['Drive','Interior at port (J)','Other interior (J)','Absorber (J)','E / total (%)','Interior end / post-peak (dB)'],['---']*6]
field_rows=[]
for rig in ('msl','waveguide'):
    row=read('diagnostic_'+rig+'.json')
    ms=row['measurements']
    items=ms.items() if rig=='msl' else [('waveguide',ms)]
    for label,entry in items:
        for i,solve in enumerate(entry['solves']):
            diag=solve['final_field_diagnostic']
            reg=diag['regions']
            e=sum(r['E_J'] for r in reg.values())
            witness=solve['energy_witness']
            name=f'{rig}/{label}/drive{i+1}'
            spatial.append([name,*[f"{reg[k]['energy_J']:.9e}" for k in ('interior_under_port','interior_elsewhere','absorber')],f"{100*e/diag['total_energy_J']:.6f}",f"{witness['end_vs_post_peak_db']:.6f}" if 'end_vs_post_peak_db' in witness else 'source has not ended'])
            field_rows.append(dict(drive=name,regions=reg,energy_witness=witness,settling_db=entry['settling_db'],maxima=diag['field_maxima']))
(ROOT/'TABLE_fields_3.md').write_text(table('End-of-record field energy. The first three regions are disjoint; the absorber contains electromagnetic field energy only, without CPML auxiliary variables. Port means one realized feed/source plane across its aperture.',spatial))
(ROOT/'FIELD_SUMMARY_3.json').write_text(json.dumps(field_rows,indent=2)+'\n')

# Compare the permitted repeat measurements to their previous retained arrays.
comparisons=[]
for label in ('two_port','one_port'):
    old=ROOT/'raw/msl/0_8'/label/'sparams.npz'
    new=ROOT/'raw_3/diagnostic_msl'/label/'sparams.npz'
    with np.load(old) as a,np.load(new) as b:
        comparisons.append(dict(rig='msl',port_count=label,S_bit_identical=bool(a['S'].dtype==b['S'].dtype and a['S'].tobytes()==b['S'].tobytes()),max_abs_difference=float(abs(a['S']-b['S']).max())))
old=json.loads((ROOT/'raw/waveguide/0_4/result.json').read_text())['measurements']
old_s=np.array([complex(*z) for z in old['S11']])
with np.load(ROOT/'raw_3/diagnostic_waveguide/sparams.npz') as z:
    new_s=z['S'][0,0]
comparisons.append(dict(rig='waveguide',complex_values_equal_to_retained_JSON=bool(np.array_equal(old_s,new_s)),max_abs_difference=float(abs(old_s-new_s).max()),
                        max_abs_amplitude_difference_db=float(abs(20*np.log10(abs(new_s)/abs(old_s))).max()),
                        max_abs_phase_difference_deg=float(abs(np.angle(new_s/old_s,deg=True)).max())))
(ROOT/'repeat_comparison_3.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print('REDUCTION_COMPLETE: oracle tables, final-field table, repeat comparisons')
