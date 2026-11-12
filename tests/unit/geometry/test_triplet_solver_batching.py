"""A34: fixed 144-matrix solver inputs retain every bit across batch sizes.

The inputs and radius tangent were frozen from a separate eager jax.jvp build
at 56b9e4ce7: Grid((.005, .005, .003), dx=.001, freq_max=10e9,
cpml_axes=''), Cylinder((.0024, .0026, .0015), float32(.0017), .008,
'z'), eps_r=38. They are the first +++ safe_pi array, shape (6,6,4,3,3).
The test never rebuilds geometry; solver inputs are dynamic jit arguments.
"""
import base64
import hashlib
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry import triplet_tensor as triplet

_MATRIX = (
    'c${NkXs~BM0oeGXYDU$Jsu>PydfIs$E#BTd0OCpT|2$yy_NN1~iigXq9O$VgdgpZqw}`NVAU;ck(?Pa9uN;xplzhJ8NKZ8r'
    'w--6gPLKhMck=Cb*sil4MNOsjbGoYGz3Sx1ebop_95qgJ#C|x?Q_WyY&x1`({l+7XjT`qI0P%dhxevN{dpjVjm|nBmk)CRN'
    '-&}F*ncD$Q^B_7^-3=v;0duAkJ=OgGKiiR!VG39rM1%MPsfI1~ucMYJ56(0X%7gS&GuYDeU{m87pzE0SBijMQ3tZ^r`0d9='
    '2V@n;%x5~%Q_WyY&!9YqoCiU9av;?lm#=qR@$Jk3gcxd^g6dp)su^tQc`&P)x2w=0;O-0uWEEap9u1h9X}_2qPyE=9lE#hz'
    '^WZ?K*(jcOV1sxaikhjwI(MMdKmaNq)W#l6Y6g3nA5}A|W>gLFY5?&-n@a'
)
_MATRIX_SHA256 = '91458351a81f189969b09f47f1445caaf7c753851738ded2972785933868f958'

_TANGENT = (
    'c${Nk7zLwX6pR$Gs_Uj(c;vogAol96o5w;T_qid<^KveBqo*2VKKGvFV=*^a+>yo3*vYyNl$u?a<lK^eaG{7z4iKZOnwP0('
    '+@1k30}im~*EctMs=*~U3LwD6oX6eOoaY!W`5nJR-07)?c$be%kO<6kpuC4m9+U^^sfKu$k4&H!!tU<!&kJW5vcyQa(^C!c'
    'E+3g77Ffsf0P9*@@}N4Go@$79`6ze}Yzw?XYY*_fQF9w8HK4o*%9F@q=y{NyYCv@_s18OJL$7n`sRkQBZ(|R%no%$cM#1m_'
    '0H6@<od'
)
_TANGENT_SHA256 = 'f784d4dd2416a5d712d5f1449ec116cecc1604055076eeec3eae0b25017b61df'


def fixed_inputs(isolated=False):
    """Decode frozen little-endian float32 bits; isolate flat index 28 if asked."""
    arrays = []
    for encoded, digest in ((_MATRIX, _MATRIX_SHA256), (_TANGENT, _TANGENT_SHA256)):
        raw = zlib.decompress(base64.b85decode(encoded))
        assert hashlib.sha256(raw).hexdigest() == digest
        arrays.append(np.frombuffer(raw, dtype='<f4').reshape(6, 6, 4, 3, 3).copy())
    matrix, tangent = arrays
    if isolated:
        single = np.broadcast_to(np.eye(3, dtype=np.float32), matrix.shape).copy()
        direction = np.zeros_like(tangent)
        single[1, 1, 0] = matrix[1, 1, 0]
        direction[1, 1, 0] = tangent[1, 1, 0]
        matrix, tangent = single, direction
    return jnp.asarray(matrix), jnp.asarray(tangent)


def _assert_bits(actual, expected):
    for index, (a, b) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected))):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        different = np.count_nonzero(a.view(np.uint8) != b.view(np.uint8))
        print(f'array={index} entries={a.size} differing_bytes={different}')
        assert a.tobytes() == b.tobytes(), (index, int(different))


@pytest.mark.parametrize('isolated', [False, True], ids=['full', 'one-among-identities'])
@pytest.mark.parametrize('operation', ['inv', 'eigh', 'eigvalsh'])
def test_fixed_solver_batches_equal_bits(monkeypatch, isolated, operation):
    matrix, _ = fixed_inputs(isolated)
    if operation != 'inv':
        matrix = (matrix + jnp.swapaxes(matrix, -1, -2)) / 2
    outputs = []
    for batch in (2**20, 64):
        monkeypatch.setattr(triplet, 'MATRIX_BATCH_SIZE', batch)
        jax.clear_caches()
        outputs.append(jax.jit(lambda m: triplet._linalg(jnp, operation, m))(matrix))
    _assert_bits(outputs[1], outputs[0])
    jax.clear_caches()


@pytest.mark.parametrize('isolated', [False, True], ids=['full', 'one-among-identities'])
def test_fixed_inverse_jvp_batches_equal_bits(monkeypatch, isolated):
    matrix, tangent = fixed_inputs(isolated)
    outputs = []
    for batch in (2**20, 64):
        monkeypatch.setattr(triplet, 'MATRIX_BATCH_SIZE', batch)
        jax.clear_caches()
        inverse = jax.jit(lambda m: triplet._linalg(jnp, 'inv', m))
        outputs.append(jax.jvp(inverse, (matrix,), (tangent,)))
    _assert_bits(outputs[1], outputs[0])
    jax.clear_caches()


@pytest.mark.parametrize('isolated', [False, True], ids=['full', 'one-among-identities'])
@pytest.mark.parametrize('operation', ['inv', 'eigh', 'eigvalsh', 'inverse-jvp'])
def test_fixed_solver_serial_and_sliced_equal_bits(monkeypatch, isolated, operation):
    """The same batch-64 solves retain every bit with slices or serial map."""
    matrix, tangent = fixed_inputs(isolated)
    if operation in ('eigh', 'eigvalsh'):
        matrix = (matrix + jnp.swapaxes(matrix, -1, -2)) / 2
    monkeypatch.setattr(triplet, 'MATRIX_BATCH_SIZE', 64)
    outputs = []
    for serial in (False, True):
        monkeypatch.setattr(triplet, '_serial_pieces', lambda: serial)
        jax.clear_caches()
        solve = jax.jit(lambda m: triplet._linalg(
            jnp, 'inv' if operation == 'inverse-jvp' else operation, m))
        outputs.append(jax.jvp(solve, (matrix,), (tangent,))
                       if operation == 'inverse-jvp' else solve(matrix))
    _assert_bits(outputs[1], outputs[0])
    jax.clear_caches()
