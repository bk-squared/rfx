"""Read retained 633 fields only; independent scalar quadrature/polarization audit."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parents[4]
ROOT = BASE / 'docs/research_notes/issue726/power_budget'
DATA = ROOT / 'gpu-369367260633/artifacts'
plan = json.loads((DATA / 'plan.json').read_text())
freqs = np.array(plan['field_freqs_hz'])
dx, dt = plan['dx_m'], plan['dt_s']
raw = []
for drive in range(2):
    with np.load(DATA / f'drive-{drive}-fields.npz') as f:
        raw.append({name: f[name].astype(np.complex128) for name in f.files})
with np.load(DATA / 'raw-vi.npz') as f:
    full_vi = {name: f[name] for name in ('metadata_json', 'freqs_hz', 'raw_v', 'raw_i1', 'production_smatrix')}
meta = json.loads(str(full_vi['metadata_json']))
selected = np.array(plan['selected_indices'])
nominal_freqs = full_vi['freqs_hz'][selected]
np.testing.assert_array_equal(nominal_freqs.astype(np.float32).astype(float), freqs)
V = full_vi['raw_v'][:, :, 0, :][:, :, selected].astype(complex)
I = full_vi['raw_i1'][:, :, selected].astype(complex)
R = np.array(meta['s_reference_impedances_ohm'])


def nodal_integral(values, axis):
    """Sum interior nodes plus half of each endpoint; spacing is external."""
    values = np.moveaxis(values, axis, -1)
    return np.sum(values[..., 1:-1], axis=-1) + (values[..., 0] + values[..., -1]) / 2


fields = {}
for face in plan['faces']:
    fs = {}
    for comp, keys in face['probes'].items():
        data = np.stack([[raw[d][key] for key in keys] for d in range(2)])
        if comp.startswith('h'):
            assert data.shape[1] == 2
            fs[comp] = (data[:, 0] + data[:, 1]) / 2 * np.exp(1j * np.pi * freqs * dt)[None, :, None, None]
        else:
            assert data.shape[1] == 1
            fs[comp] = data[:, 0]
    fields[face['name']] = fs


def scalar_face(face, f, coefficients):
    # First superpose actual complex fields; only then form pointwise power.
    a = {key: coefficients[0] * value[0, f] + coefficients[1] * value[1, f]
         for key, value in fields[face['name']].items()}
    if face.get('zero_flux_pec_boundary'):
        assert not np.any(a['ex']) and not np.any(a['ey'])
        return 0.0
    if face['axis'] == 0:  # arrays [y,z]
        pos = nodal_integral((a['ey'] * a['hz'].conj())[:-1, :], 1).sum()
        neg = nodal_integral((a['ez'] * a['hy'].conj())[:, :-1], 0).sum()
    elif face['axis'] == 1:  # arrays [x,z]
        pos = nodal_integral((a['ez'] * a['hx'].conj())[:, :-1], 0).sum()
        neg = nodal_integral((a['ex'] * a['hz'].conj())[:-1, :], 1).sum()
    else:  # arrays [x,y]
        pos = nodal_integral((a['ex'] * a['hy'].conj())[:-1, :], 1).sum()
        neg = nodal_integral((a['ey'] * a['hx'].conj())[:, :-1], 0).sum()
    return float(np.real(pos - neg) * dx**2)


def scalar_vi(f, coefficients):
    vc = coefficients @ V[:, :, f]
    ic = coefficients @ I[:, :, f]
    return float(np.real(np.dot(vc, ic.conj())))


def matrix_from_scalar(fun):
    q0, q1 = fun(np.array([1, 0])), fun(np.array([0, 1]))
    qreal = fun(np.array([1, 1]))
    qimag = fun(np.array([1, 1j]))
    cross = (qreal - q0 - q1) / 2 + 1j * (q0 + q1 - qimag) / 2
    return np.array([[q0, cross], [cross.conjugate(), q1]])


closed_matrices, vi_matrices, rows = [], [], []
face_matrices = {face['name']: [] for face in plan['faces']}
for f, frequency in enumerate(freqs):
    A = ((V[:, :, f] + I[:, :, f] * R[None, :]) / (2 * np.sqrt(R)[None, :])).T
    def original_drive(input_coefficients):
        return np.linalg.solve(A, input_coefficients)
    def vi_in_input(c):
        return scalar_vi(f, original_drive(c))
    qvi = matrix_from_scalar(vi_in_input)
    vi_matrices.append(qvi)
    values, vectors = np.linalg.eigh(qvi)
    c = original_drive(vectors[:, 0])
    assert abs(np.linalg.norm(A @ c) - 1) < 1e-12
    signed = {face['name']: face['outward'] * scalar_face(face, f, c)
              for face in plan['faces'] if face['outward']}
    xout = signed['x_station_0'] + signed['x_station_5']
    sideout = sum(value for key, value in signed.items() if not key.startswith('x_'))
    def closed_scalar(input_coefficients):
        original = original_drive(input_coefficients)
        return sum(face['outward'] * scalar_face(face, f, original)
                   for face in plan['faces'] if face['outward'])
    qclosed = matrix_from_scalar(closed_scalar)
    closed_matrices.append(qclosed)
    # Independently recover each face's original-drive Gram for comparison.
    for face in plan['faces']:
        face_matrices[face['name']].append(matrix_from_scalar(lambda z: scalar_face(face, f, z)))
    drive_rows = []
    for d in range(2):
        cd = np.eye(2)[d] / abs(A[d, d])
        drive_faces = {face['name']: scalar_face(face, f, cd) for face in plan['faces'] if face['outward']}
        drive_x = -drive_faces['x_station_0'] + drive_faces['x_station_5']
        drive_side = -drive_faces['y_lo'] + drive_faces['y_hi'] - drive_faces['z_lo'] + drive_faces['z_hi']
        drive_rows.append(dict(drive=d, positive_axis_faces=drive_faces, x_out=drive_x,
                               side_out=drive_side, closed_out=drive_x + drive_side,
                               normalization='squared own-port native VI incident wave'))
    rows.append(dict(frequency_hz=float(frequency), apparent_vi_excess=float(-values[0]),
                     x_out=xout, side_out=sideout, closed_out=xout + sideout,
                     signed_coherent_faces=signed, closed_gram_eigenvalues=np.linalg.eigvalsh(qclosed).tolist(),
                     vi_inward_eigenvalues=values.tolist(), drive_rows=drive_rows))

# Independent native V/I reconstruction from full face samples at both ports.
errors = []
for p, station in enumerate(('x_station_0', 'x_station_5')):
    face = next(face for face in plan['faces'] if face['name'] == station)
    fs = fields[station]
    span = plan['port_spans'][p]
    j0, j1, jc = [span[key] - face['region'][0] for key in ('w_lo', 'w_hi', 'w_centre')]
    k = span['n_hi'] - face['region'][2]
    kg = span['n_lo'] - face['region'][2]
    vre = sum(fs['ez'][:, :, jc, z] * dx for z in range(kg, k))
    contour = sum((fs['hy'][:, :, j, k - 1] - fs['hy'][:, :, j, k]) * dx for j in range(j0, j1 + 1))
    contour += (fs['hz'][:, :, j1, k] - fs['hz'][:, :, j0 - 1, k]) * dx
    ire = (-1 if p == 0 else 1) * contour
    errors.append(dict(port=p, max_v_relative=float(abs(vre - V[:, p]).max() / abs(V[:, p]).max()),
                       max_i_relative=float(abs(ire - I[:, p]).max() / abs(I[:, p]).max())))

with np.load(BASE / 'docs/research_notes/issue726/collocation/gpu-369367260605/artifacts/clean-phasors.npz') as old:
    equality = {key: dict(same_shape=full_vi[key].shape == old[key].shape,
                         same_dtype=full_vi[key].dtype == old[key].dtype,
                         same_bytes=full_vi[key].tobytes() == old[key].tobytes())
                for key in ('raw_v', 'raw_i1', 'production_smatrix')}

# Read author's outputs only AFTER all independent calculations are complete.
with np.load(ROOT / 'field-power-audit.npz') as author:
    comparisons = {name: float(np.max(abs(np.array(matrices) - author[name + '_gram'])))
                   for name, matrices in face_matrices.items()}
    closed_difference = float(np.max(abs(np.array(closed_matrices) - author['closed_outward_in_input_basis'])))
    vi_difference = float(np.max(abs(np.array(vi_matrices) - author['vi_inward_in_input_basis'])))

report = dict(scope='Independent scalar surface integration and polarization of retained 633 raw fields; no field evolution or production helper imports.',
              normalization='Original native VI A basis; not independent physical incident-power calibration.',
              nominal_vi_metadata_frequencies_hz=nominal_freqs.tolist(), actual_field_frequencies_hz=freqs.tolist(),
              raw_fields_sha256={f'drive-{d}-fields.npz': hashlib.sha256((DATA / f'drive-{d}-fields.npz').read_bytes()).hexdigest() for d in range(2)},
              rows=rows, reconstructed_native_vi_errors=errors, bit_identity_605=equality,
              max_closed_gram_eigenvalue_magnitude=float(np.max(abs(np.linalg.eigvalsh(np.array(closed_matrices))))),
              comparisons_to_author=dict(face_gram_max_absolute_differences=comparisons,
                                         closed_input_basis_max_difference=closed_difference,
                                         vi_input_basis_max_difference=vi_difference),
              limits=['Finite-window endpoint terms unbounded.', 'Source, finite-lead and boundary origin unresolved.',
                      'VI-power versus x-face power residual remains; no port calibration claim.'])
out = BASE / '.git/issue726-independent-field-budget.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
