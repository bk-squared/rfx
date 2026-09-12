"""Independent controlled-run V/I, consumed-plan, and spectral audit; no fields."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=ROOT/'gpu-369367260671/artifacts')
parser.add_argument('--previous-data',type=Path,default=ROOT/'gpu-369367260669/artifacts')
parser.add_argument('--output-prefix',type=Path,required=True)
parser.add_argument('--allow-partial',action='store_true')
args=parser.parse_args()
DATA=args.data
PREVIOUS=args.previous_data
LABELS = ('baseline','stub_1cell','stub_narrow')
BW_IDEAL = .210274  # Preserve the case's existing rounded r=1 reference.
REFERENCE_C0 = 2.998e8  # The case's declared reference constant, not CODATA substitution.


def vertex(frequency, magnitude):
    frequency=np.asarray(frequency,dtype=float);magnitude=np.asarray(magnitude,dtype=float)
    i=int(np.argmin(magnitude))
    h=frequency[i+1]-frequency[i] if i<len(frequency)-1 else frequency[i]-frequency[i-1]
    shift=0.
    if 0<i<len(frequency)-1:
        # Independent three-equation polynomial fit in BIN coordinates.
        # Production uses the right local spacing after float32 frequency rounding.
        ordinate=np.log(np.maximum(magnitude[i-1:i+2],1e-300))
        constant,linear,quadratic=np.linalg.solve(np.array([[1.,-1.,1.],[1.,0.,0.],[1.,1.,1.]]),ordinate)
        if quadratic>0:
            shift=float(np.clip(-linear/(2*quadratic),-1.,1.))
    return dict(index=i,bin_frequency_hz=float(frequency[i]),frequency_hz=float(frequency[i]+shift*h),
                shift_bins=shift,local_bin_hz=float(h))


def spectrum(frequency, magnitude, reference, *, frequency_gate=False):
    frequency=np.asarray(frequency,dtype=float);magnitude=np.asarray(magnitude,dtype=float)
    assert len(frequency)==100 and np.all(np.diff(frequency)>0)
    assert np.isfinite(magnitude).all() and np.all(magnitude>=0)
    feature=vertex(frequency,magnitude);i=feature['index']
    db=20*np.log10(np.maximum(magnitude,1e-300));level=-10.
    if db[i]>level:
        flo=fhi=0.;lo_bracket=hi_bracket=[];bw=0.
    else:
        left=np.flatnonzero(db[:i]>level)
        right=np.flatnonzero(db[i+1:]>level)
        if left.size:
            l=int(left[-1]);lo_bracket=[l,l+1]
            flo=float(frequency[l]+(level-db[l])/(db[l+1]-db[l])*(frequency[l+1]-frequency[l]))
        else:
            flo=float(frequency[0]);lo_bracket=[0]
        if right.size:
            h=i+1+int(right[0]);hi_bracket=[h-1,h]
            fhi=float(frequency[h-1]+(level-db[h-1])/(db[h]-db[h-1])*(frequency[h]-frequency[h-1]))
        else:
            fhi=float(frequency[-1]);hi_bracket=[len(frequency)-1]
        bw=(fhi-flo)/feature['frequency_hz']
    half=[vertex(frequency[p::2],magnitude[p::2]) for p in (0,1)]
    half_indices=[2*half[p]['index']+p for p in (0,1)]
    local_bin=abs(frequency[half_indices[1]]-frequency[half_indices[0]])/abs(half_indices[1]-half_indices[0])
    witness=abs(half[1]['frequency_hz']-half[0]['frequency_hz'])/local_bin
    error=abs(feature['frequency_hz']-reference)/reference*100
    ratio=bw/BW_IDEAL
    result=dict(**feature,notch_depth_db=float(db[i]),reference_deviation_pct=float(error),
                bandwidth_lo_hz=flo,bandwidth_hi_hz=fhi,bandwidth_fraction=float(bw),bandwidth_ratio=float(ratio),
                crossing_bin_indices=dict(left=lo_bracket,right=hi_bracket),half_grid_witness_bins=float(witness),
                G2=bool(.80<ratio<1.20),G3=bool(witness<1.),depth_witness=bool(db[i]<-10.))
    if frequency_gate:
        result['G1']=bool(error<4.)
    return result


def power_metrics(s):
    sf=s.transpose(2,0,1)
    gains=np.linalg.eigvalsh(sf.conj().transpose(0,2,1)@sf)
    return dict(max_coherent_gain=float(np.max(gains[:,-1])),max_entry=float(np.max(abs(s))),
                reciprocity_max_abs=float(np.max(abs(s[0,1]-s[1,0]))))


required={label:[DATA/f'{label}-raw-vi.npz',DATA/f'{label}-result.npz',DATA/f'cv06b_falsifier_{label}.json'] for label in LABELS}
available=tuple(label for label in LABELS if all(p.exists() for p in required[label]))
if not available:
    raise SystemExit('No complete arm artifacts are available yet.')
outcome_path=DATA/'outcome.json'
outcome=json.loads(outcome_path.read_text()) if outcome_path.exists() else None
complete=(available==LABELS and outcome is not None and outcome.get('complete') is True)
if not complete and not args.allow_partial:
    raise SystemExit('Controlled run is not complete; use --allow-partial for an explicitly partial audit.')
consumed=json.loads((DATA/'consumed-run-plans.json').read_text())
expected_calls=[(label,d) for label in LABELS for d in range(2)]
actual_calls=[(item['label'],item['drive']) for item in consumed]
assert actual_calls==expected_calls[:len(actual_calls)]
assert len(actual_calls)<=6
baseline_plans={}
baseline_observed={}
plan_rows=[]
for item in consumed:
    signature=item['signature']
    assert signature['schema_version']==1
    invariants=signature['invariants']
    digest=hashlib.sha256(json.dumps(invariants,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest==signature['invariant_sha256'], 'Stored signature hash does not match its full dictionary'
    d=item['drive']
    if item['label']=='baseline':
        baseline_plans[d]=invariants
        baseline_observed[d]=signature['observed']
    matches=invariants==baseline_plans[d]
    assert matches, (item['label'],d,'actual invariant dictionary differs')
    changed_pec=[name for name,current,previous in zip(('ex','ey','ez'),signature['observed']['pec_full'],baseline_observed[d]['pec_full']) if current!=previous]
    changed_sheets=[index for index,(current,previous) in enumerate(zip(signature['observed']['sheet_footprints_full'],baseline_observed[d]['sheet_footprints_full'])) if current!=previous]
    plan_rows.append(dict(label=item['label'],drive=d,recomputed_sha256=digest,all_invariants_equal_to_same_baseline_drive=matches,
                          changed_full_pec_components=changed_pec,changed_sheet_payload_indices=changed_sheets,
                          grid_shape=invariants['grid']['shape'],n_steps=invariants['n_steps'],
                          source_count=len(invariants['sources']),dft_plane_count=len(invariants['dft_planes']),
                          allowed_stub_box=invariants['allowed_stub_box']))
if complete:
    assert len(consumed)==6 and outcome.get('runner_plan_verified') is True
rows={};matrices={}
for label in available:
    with np.load(DATA/f'{label}-raw-vi.npz',allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata_json']))
        f=archive['freqs_hz'].astype(float)
        v=archive['raw_v'][:,:,0,:].astype(complex)
        current=archive['raw_i1'].astype(complex)
        order=archive['driven_port_indices']
        dumped=archive['production_smatrix']
    assert metadata['schema_version']==4 and metadata['s_wave_convention']=='power'
    assert metadata['production_smatrix_assembly']=='multi_drive_solve'
    np.testing.assert_array_equal(np.sort(order),[0,1])
    refs=np.array(metadata['s_reference_impedances_ohm'])
    assert refs.shape==(2,) and np.isfinite(refs).all() and np.all(refs>0)
    assert np.isfinite(v).all() and np.isfinite(current).all()
    # Stored currents retain the native port orientation: no additional sign flip.
    A=((v+current*refs[None,:,None])/(2*np.sqrt(refs)[None,:,None])).transpose(2,1,0)
    B=((v-current*refs[None,:,None])/(2*np.sqrt(refs)[None,:,None])).transpose(2,1,0)
    det=A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]
    assert np.all(abs(det)>0)
    inverse=np.stack([A[:,1,1],-A[:,0,1],-A[:,1,0],A[:,0,0]],axis=-1).reshape(-1,2,2)/det[:,None,None]
    rebuilt=(B@inverse).transpose(1,2,0)
    assert np.isfinite(rebuilt).all()
    with np.load(DATA/f'{label}-result.npz',allow_pickle=False) as archive:
        projected=archive['S'];raw=archive['S_raw'];z0=archive['Z0'];settling=archive['settling_db']
        reliable=archive['reliable'];rail=archive['beta_railed'];cond=archive['cond_a']
        correction=archive['passivity_correction']
        np.testing.assert_array_equal(archive['freqs_hz'].astype(float),f)
        np.testing.assert_array_equal(archive['reference_impedances'],refs)
    assert dumped.dtype==raw.dtype and dumped.tobytes()==raw.tobytes()
    assert np.isfinite(projected).all() and np.isfinite(raw).all()
    metrics_json=json.loads((DATA/f'cv06b_falsifier_{label}.json').read_text())
    ref=metrics_json['frequency_reference']
    eps=(ref['substrate_eps_r']+1)/2+(ref['substrate_eps_r']-1)/2/np.sqrt(1+12*ref['substrate_height_m']/ref['width_m'])
    f_ref=REFERENCE_C0/(4*ref['length_m']*np.sqrt(eps))
    np.testing.assert_allclose(f_ref,ref['frequency_hz'],rtol=1e-14)
    variants={
        'projected':spectrum(f,np.abs(projected[1,0]),f_ref,frequency_gate=(label=='baseline')),
        'raw_saved':spectrum(f,np.abs(raw[1,0]),f_ref,frequency_gate=(label=='baseline')),
        'raw_reconstructed':spectrum(f,np.abs(rebuilt[1,0]),f_ref,frequency_gate=(label=='baseline')),
    }
    i=variants['projected']['index']
    selected=sorted(k for k in set([i-1,i,i+1]+sum(variants['projected']['crossing_bin_indices'].values(),[])) if 0<=k<len(f))
    checks={str(k):dict(frequency_hz=float(f[k]),reliable_by_port=reliable[:,k].tolist(),beta_railed_by_port=rail[:,k].tolist()) for k in selected}
    producer_comparison={name:float(variants['projected'][ours]-metrics_json[name])
                         for name,ours in (('f_notch_refined','frequency_hz'),('notch_depth_db','notch_depth_db'),('bw_ratio','bandwidth_ratio'),('witness_bins','half_grid_witness_bins'))}
    rows[label]=dict(references_ohm=refs.tolist(),driven_port_indices=order.tolist(),frequency_reference=ref,
                    raw_reconstruction_max_abs_difference=float(np.max(abs(rebuilt-raw))),
                    raw_reconstruction_relative_peak_difference=float(np.max(abs(rebuilt-raw))/np.max(abs(raw))),
                    solve_relative_residual=float(np.max(abs(rebuilt.transpose(2,0,1)@A-B))/np.max(abs(B))),
                    saved_raw_equals_raw_dump_bytes=True,cond_a_max=float(np.max(np.linalg.cond(A))),
                    raw_power=power_metrics(rebuilt),projected_power=power_metrics(projected.astype(complex)),
                    variants=variants,producer_projected_metric_differences=producer_comparison,
                    z0_port0_median=float(np.median(z0[0].real)),G4_port0=bool(40<np.median(z0[0].real)<65),
                    settling_db=settling.tolist(),all_settling_pass=bool(np.isfinite(settling).all() and np.all(settling<=-40)),
                    reliability_false_counts_by_port=np.sum(~reliable,axis=1).tolist(),beta_railed_counts_by_port=np.sum(rail,axis=1).tolist(),
                    notch_and_crossing_flags=checks,projection_correction_max=float(np.max(correction)),
                    grid=metadata['grid'],port_definitions=metadata['port_definitions'],current_plane_stencils=metadata['current_plane_stencils'])
    matrices[label+'_raw_reconstructed']=rebuilt

baseline_raw_checks={}
if 'baseline' in rows:
    with np.load(DATA/'baseline-raw-vi.npz',allow_pickle=False) as current, np.load(PREVIOUS/'baseline-raw-vi.npz',allow_pickle=False) as previous:
        for key in ('freqs_hz','raw_v','raw_i1','raw_i1_left','raw_i1_same_index','production_smatrix'):
            a,b=current[key],previous[key]
            baseline_raw_checks[key]=dict(same_shape=a.shape==b.shape,same_dtype=a.dtype==b.dtype,byte_identical=a.tobytes()==b.tobytes())
changes_vs_previous={}
for label in available:
    with np.load(PREVIOUS/f'{label}-result.npz',allow_pickle=False) as previous:
        old_f=previous['freqs_hz'].astype(float)
        old_s={'projected':previous['S'],'raw_saved':previous['S_raw']}
    old_metrics=json.loads((PREVIOUS/f'cv06b_falsifier_{label}.json').read_text())
    old_reference=old_metrics['frequency_reference']['frequency_hz']
    changes_vs_previous[label]={}
    for version,old_matrix in old_s.items():
        before=spectrum(old_f,np.abs(old_matrix[1,0]),old_reference,frequency_gate=(label=='baseline'))
        now=rows[label]['variants'][version]
        changes_vs_previous[label][version]=dict(previous=before,current_minus_previous={
            key:now[key]-before[key] for key in ('frequency_hz','notch_depth_db','bandwidth_ratio','half_grid_witness_bins')})
conditions={}
if 'baseline' in rows and 'stub_1cell' in rows:
    prediction=(rows['stub_1cell']['frequency_reference']['frequency_hz']/rows['baseline']['frequency_reference']['frequency_hz']-1)*100
    for variant in ('projected','raw_saved','raw_reconstructed'):
        b=rows['baseline']['variants'][variant];one=rows['stub_1cell']['variants'][variant];narrow=rows['stub_narrow']['variants'][variant] if 'stub_narrow' in rows else None
        delta=(one['frequency_hz']/b['frequency_hz']-1)*100
        binned=(one['bin_frequency_hz']/b['bin_frequency_hz']-1)*100
        conditions[variant]=dict(baseline_G1=b['G1'],baseline_all_existing_gates=bool(b['G1'] and b['G2'] and b['G3'] and b['depth_witness'] and rows['baseline']['G4_port0']),
                                length_prediction_pct=float(prediction),one_cell_refined_shift_pct=float(delta),one_cell_binned_shift_pct=float(binned),
                                one_cell_visible=bool(abs(delta)>=.5*abs(prediction)),narrow_G2_fails=(not narrow['G2']) if narrow is not None else None,narrow_depth_pass=narrow['depth_witness'] if narrow is not None else None)
report=dict(scope='Independent controlled-run V/I reconstruction, consumed-plan dictionary/hash verification and spectral predicates; no production replay/estimator helper imports, no field evolution.',
            complete_artifacts=complete,available_arms=list(available),outcome=outcome,consumed_plan_verification=plan_rows,
            six_pre_step_signatures_verified=(len(plan_rows)==6),baseline_raw_byte_comparison_to_669=baseline_raw_checks,changes_vs_669=changes_vs_previous,
            reference_c0_m_per_s=REFERENCE_C0,bandwidth_reference_fraction=BW_IDEAL,
            estimator_convention='Natural-log three-point parabola in bin coordinates; right adjacent frequency spacing; +/-1 bin clamp; linear dB -10 crossings; sampled-bin depth.',
            float_rounding='Saved projected/raw metrics use NumPy complex64 abs then float64, as producer; reconstruction uses complex128 V/I and magnitude.',
            rows=rows,approved_conditions=conditions,
            geometry_and_observation_state={label:dict(grid=row['grid'],probe0_indices=[p['e_index'] for p in row['current_plane_stencils']],
                                                        offsets=[p['n_probe_offset'] for p in row['port_definitions']])for label,row in rows.items()},
            control_scope='Saved pre-step records verify identical consumed inputs per drive outside the allowed baseline stub sheet box; field results may differ.',
            limitations=['No calibrated MSL power claim; raw coherent gains remain above1.',
                         'Existing reliability and beta-rail flags retained without filtering.',
                         'Settling is the recorded local witness, not a global finite-window error bound.',
                         'No narrow approximate-quarter-wave accuracy gate and no unique historical frequency-shift attribution.'],
            raw_artifact_sha256={label:hashlib.sha256((DATA/f'{label}-raw-vi.npz').read_bytes()).hexdigest() for label in available})
prefix=args.output_prefix if complete else args.output_prefix.with_name(args.output_prefix.name+'-partial')
out=prefix.with_suffix('.json')
if out.exists() or out.with_suffix('.npz').exists():
    raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(out.with_suffix('.npz'),freqs_hz=f,**matrices)
print(json.dumps(report,indent=2))
