"""Independent retained-667 readout; node DTFT plus exact endpoint correction."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'docs/research_notes/issue726/power_budget/gpu-369367260667/artifacts'
plan = json.loads((DATA / 'plan.json').read_text())
dt = plan['dt_s']
norms = np.array([np.dot(p['volumes_m3'], np.square(p['profile_e'])) for p in plan['ports']])
sigma = np.array([p['added_sigma_s_per_m'] for p in plan['ports']])
refs = 1 / (sigma * norms)
sqrt_r = np.sqrt(refs)
np.testing.assert_allclose(refs, [p['effective_reference_ohm'] for p in plan['ports']], rtol=1e-15)


def node_dtft(series, frequencies):
    """Samples here are placed at E-node times (n+1)*dt, not midpoints."""
    t = (np.arange(series.shape[0], dtype=float) + 1) * dt
    output = np.empty((len(frequencies), series.shape[1]), dtype=complex)
    for start in range(0, len(frequencies), 8):
        fs = frequencies[start:start+8]
        kernel = np.exp((-2j*np.pi*fs[:,None]) * t[None,:])
        output[start:start+len(fs)] = (kernel @ series) * dt
    return output


records, quality = [], []
for drive in range(2):
    with np.load(DATA / f'drive-{drive}-source-work.npz', allow_pickle=False) as archive:
        current_freqs = archive['actual_freqs_hz'].astype(float)
        if drive == 0:
            freqs = current_freqs
        else:
            np.testing.assert_array_equal(current_freqs, freqs)
        u = archive['source_u'].astype(float)
        post_projection = []
        checks = []
        for p, definition in enumerate(plan['ports']):
            # Project integer-time E first. Do not build the author's midpoint array.
            post = archive[f'p{p}_ez_post'].astype(float)
            np.testing.assert_array_equal(post[-1], archive[f'p{p}_ez_end'])
            vector = np.array(definition['volumes_m3']) * np.array(definition['profile_e'])
            phi_post = np.einsum('nc,c->n', post, vector, optimize=False) / norms[p]
            post_projection.append(phi_post)
            phi_mid = (phi_post + np.r_[0., phi_post[:-1]]) / 2
            peak = np.max(abs(phi_mid))
            checks.append(dict(port=p,
                midpoint_last_10pct_rms_over_peak_db=float(10*np.log10(np.mean(phi_mid[-max(8,len(phi_mid)//10):]**2)/peak**2)),
                midpoint_final_over_peak_db=float(20*np.log10(abs(phi_mid[-1])/peak)),
                midpoint_last_1pct_peak_over_peak_db=float(20*np.log10(np.max(abs(phi_mid[-max(1,len(phi_mid)//100):]))/peak)),
                midpoint_last_5pct_peak_over_peak_db=float(20*np.log10(np.max(abs(phi_mid[-max(1,len(phi_mid)//20):]))/peak)),
                midpoint_last_10pct_peak_over_peak_db=float(20*np.log10(np.max(abs(phi_mid[-max(1,len(phi_mid)//10):]))/peak)),
                post_final_over_peak_db=float(20*np.log10(abs(phi_post[-1])/np.max(abs(phi_post))))))
        phi = np.column_stack(post_projection)
        n = len(phi)
        transformed = node_dtft(np.column_stack([phi, u]), freqs)
        theta = 2*np.pi*freqs*dt
        # E^0 = 0. This includes the exact finite-record last-sample term.
        end_term = (dt/2) * np.exp(-1j*theta*(n+.5))[:,None] * phi[-1][None,:]
        voltage = np.cos(theta/2)[:,None] * transformed[:,:2] - end_term
        source_current = transformed[:,2:] * np.exp(1j*theta/2)[:,None] * norms[None,:]
        net_current = source_current - voltage / refs[None,:]
        # Ordinary power-wave spelling, independent of the author's Norton simplification.
        a = (voltage + refs[None,:]*net_current) / (2*sqrt_r[None,:])
        b = (voltage - refs[None,:]*net_current) / (2*sqrt_r[None,:])
        records.append(dict(voltage=voltage, source_current=source_current, net_current=net_current, a=a, b=b))
        quality.append(dict(drive=drive, steps=n, source_point_checks=checks,
                            active_wave_min_relative_to_band_peak=float(np.min(abs(a[:,drive]))/np.max(abs(a[:,drive]))),
                            max_end_correction_relative_to_voltage_peak=float(np.max(abs(end_term))/np.max(abs(voltage)))))

A = np.stack([r['a'] for r in records],axis=-1)
B = np.stack([r['b'] for r in records],axis=-1)
S = np.stack([np.linalg.lstsq(aa.T, bb.T, rcond=None)[0].T for aa,bb in zip(A,B)])
solve_residual = np.max(abs(S @ A - B)) / np.max(abs(B))


def metrics(s):
    # Hermitian power eigensolve, independent of author's singular-value summary.
    power = s.conj().transpose(0,2,1) @ s
    gains = np.linalg.eigvalsh(power)
    peak_index = int(np.argmax(gains[:,-1]))
    return dict(max_coherent_gain=float(gains[peak_index,-1]), frequency_hz=float(freqs[peak_index]),
                max_column_power=float(np.max(np.real(np.diagonal(power,axis1=1,axis2=2)))),
                max_entry_magnitude=float(np.max(abs(s))), reciprocity_max_abs=float(np.max(abs(s[:,0,1]-s[:,1,0]))))


with np.load(DATA / 'raw-vi.npz', allow_pickle=False) as raw:
    raw_arrays = {name:raw[name] for name in ('raw_v','raw_i1','production_smatrix')}
    first_s = raw_arrays['production_smatrix'].astype(complex).transpose(2,0,1)
    np.testing.assert_array_equal(raw['freqs_hz'].astype(np.float32).astype(float), freqs)
with np.load(ROOT / 'docs/research_notes/issue726/power_budget/gpu-369367260633/artifacts/raw-vi.npz', allow_pickle=False) as previous:
    unchanged = {name:bool(array.dtype==previous[name].dtype and array.shape==previous[name].shape and array.tobytes()==previous[name].tobytes()) for name,array in raw_arrays.items()}

# Author output is used only after the independent reconstruction is complete.
with np.load(ROOT / 'docs/research_notes/issue726/power_budget/source-work-readout.npz', allow_pickle=False) as author:
    comparisons = dict(source_s_max_abs_difference=float(np.max(abs(S.transpose(1,2,0)-author['source_work_s']))))
    for key in ('voltage','source_current','a','b'):
        target = {'source_current':'source_current','voltage':'source_voltage','a':'source_a','b':'source_b'}[key]
        ours = np.stack([r[key].T for r in records])
        comparisons[key+'_relative_peak_difference'] = float(np.max(abs(ours-author[target])) / np.max(abs(author[target])))

report = dict(scope='Independent retained-667 node DTFT plus exact endpoint midpoint reconstruction, Inet and ordinary power waves; no author helper imports or field evolution.',
              source_reference_ohm=refs.tolist(), profile_norm_m=norms.tolist(),
              source_response=metrics(S), first_plane_response=metrics(first_s),
              source_a_condition_max=float(np.max(np.linalg.cond(A))), full_solve_relative_residual=float(solve_residual),
              author_settling_metric='10*log10(mean(midpoint_phi[last 10 percent]**2)/max(midpoint_phi**2)); RMS tail, not tail peak.',
              quality=quality, comparisons_to_author=comparisons, raw_msl_bit_identity_to_633=unchanged,
              artifact_sha256={f'drive-{d}-source-work.npz':hashlib.sha256((DATA/f'drive-{d}-source-work.npz').read_bytes()).hexdigest() for d in range(2)},
              finite_window_error_bound=None,
              limits=['Two fixed source-model channels only; no global CPML passivity theorem.',
                      'Not MSL first-plane calibration or a support promotion.',
                      'Does not uniquely locate side-inflow origin.',
                      'Ideal profile/base-u reconstruction retains float32 source-shape rounding uncertainty.'])
out=ROOT/'.git/issue726-independent-source-work.json'
out.write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(out.with_suffix('.npz'),freqs_hz=freqs,source_s=S.transpose(1,2,0),source_a=A,source_b=B,
                    source_voltage=np.stack([r['voltage'] for r in records]),source_current=np.stack([r['source_current'] for r in records]),references=refs)
print(json.dumps(report,indent=2))
