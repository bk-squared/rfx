"""The two streams of a smoothed sample's normal line give the same bits
with either driver.

The merge stream (painted pieces into runs of one medium) and the own-run
stream (the own group's intervals into runs) each have one step function.
NumPy applies it in a Python loop; a traced build applies it in
``jax.lax.scan``, so the compiled program keeps one loop instead of an
unrolled chain that XLA recomputes inside every output. The step is selects,
comparisons and in-order sums, so the scan driver must equal the loop
driver bit for bit on the same inputs. Each column below is one sample's
line: several runs, an empty piece, an abutting pair, equal codes across a
gap, an empty interval and runs reaching past the support.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry.outside_content import _merge_stream, _own_stream

# Merge stream: sorted ends per column (low = ends[:-1], high = ends[1:]),
# one code and value per piece; code 0 is the background.
MERGE_ENDS = np.array([
    [-1., -.5, 0., .3125, 1.],        # four runs of alternating codes
    [-1., -.25, -.25, .375, 1.],      # an empty piece of code 2 inside a run
    [-1., 0., .5, .5, 1.],            # an abutting pair of one code, then an empty piece
    [-1., -.375, .25, .5, 1.],        # code 1 on both sides of a background gap
    [-1., -1., -1., .5, 1.],          # empty pieces first
    [-1., -.75, .125, .625, 1.],      # one medium throughout
], dtype=np.float32).T
MERGE_CODES = np.array([
    [0, 1, 2, 1],
    [0, 2, 1, 0],
    [1, 1, 3, 0],
    [1, 0, 1, 1],
    [2, 3, 1, 0],
    [1, 1, 1, 1],
], dtype=np.int32).T
MERGE_VALUES = np.array([
    [1., 4., 2., 4.],
    [1., 2., 4., 1.],
    [4., 4., 3., 1.],
    [4., 1., 4., 4.],
    [2., 3., 4., 1.],
    [4., 4., 4., 4.],
], dtype=np.float32).T
# The positive runs (lo, hi, eps) each column must merge to.
MERGE_RUNS = [
    [(-1., -.5, 1.), (-.5, 0., 4.), (0., .3125, 2.), (.3125, 1., 4.)],
    [(-1., -.25, 1.), (-.25, .375, 4.), (.375, 1., 1.)],
    [(-1., .5, 4.), (.5, 1., 1.)],
    [(-1., -.375, 4.), (-.375, .25, 1.), (.25, 1., 4.)],
    [(-1., .5, 4.), (.5, 1., 1.)],
    [(-1., 1., 4.)],
]

# Own-run stream: intervals sorted by lo (an empty interval sorts last at
# FAR), whether each is planar, support [-HALF, HALF].
FAR, HALF = 10., .5
OWN_LINES = [
    [(-.375, -.25, True), (-.125, 0., True), (.125, .25, True), (FAR, FAR, True)],    # three runs
    [(-.375, -.125, True), (-.125, .125, True), (FAR, FAR, True), (FAR, FAR, True)],  # abutting pair
    [(-1., -.25, True), (.25, .375, False), (FAR, FAR, False), (FAR, FAR, True)],     # past the support
    [(-.25, .125, True), (0., .25, False), (FAR, FAR, True), (FAR, FAR, True)],       # overlap, mixed
    [(0., .125, True), (FAR, FAR, True), (FAR, FAR, True), (FAR, FAR, True)],         # one interval
]
OWN_THIN = [True, True, False, False, True]
OWN_INSIDE = [.375, .5, .375, .5, .125]


def _merge_inputs():
    low, high = MERGE_ENDS[:-1], MERGE_ENDS[1:]
    return tuple(jnp.asarray(a) for a in (low, high, MERGE_VALUES, MERGE_CODES))


def _own_inputs():
    lo, hi, flat = (np.array([[line[i][k] for line in OWN_LINES] for i in range(4)])
                    for k in range(3))
    half = jnp.full(lo.shape[1:], HALF, dtype=jnp.float32)
    return (jnp.asarray(lo, dtype=jnp.float32), jnp.asarray(hi, dtype=jnp.float32),
            jnp.asarray(flat, dtype=bool), half)


def _assert_bitwise(a, b):
    a, b = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        assert (x.dtype, x.shape) == (y.dtype, y.shape)
        assert x.tobytes() == y.tobytes(), (x, y)


@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
def test_merge_scan_equals_loop_bitwise(jit):
    inputs = _merge_inputs()
    loop = _merge_stream(jnp, *inputs, scan=False)

    def scan(*a):
        return _merge_stream(jnp, *a, scan=True)

    _assert_bitwise((jax.jit(scan) if jit else scan)(*inputs), loop)
    lo, hi, eps = (np.asarray(a) for a in loop)
    assert lo.shape[0] == MERGE_ENDS.shape[0]  # one run per piece and the open run
    for column, runs in enumerate(MERGE_RUNS):
        positive = hi[:, column] > lo[:, column]
        found = list(zip(lo[positive, column], hi[positive, column], eps[positive, column]))
        assert found == runs, (column, found)


@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
def test_own_scan_equals_loop_bitwise(jit):
    lo, hi, flat, half = _own_inputs()
    shape = lo.shape[1:]
    loop = _own_stream(jnp, lo, hi, flat, half, shape, scan=False)

    def scan(*a):
        return _own_stream(jnp, *a, shape, scan=True)

    _assert_bitwise((jax.jit(scan) if jit else scan)(lo, hi, flat, half), loop)
    thin, inside, gaps = loop
    assert len(gaps) == lo.shape[0] + 1
    assert np.asarray(thin).tolist() == OWN_THIN
    assert np.asarray(inside).tolist() == OWN_INSIDE


def test_traced_build_keeps_each_stream_a_loop():
    """By default ``jax.numpy`` takes the scan driver: the program holds a loop."""
    merge = jax.make_jaxpr(lambda *a: _merge_stream(jnp, *a))(*_merge_inputs())
    lo, hi, flat, half = _own_inputs()
    own = jax.make_jaxpr(lambda *a: _own_stream(jnp, *a, lo.shape[1:]))(lo, hi, flat, half)
    for program in (merge, own):
        assert [e.primitive.name for e in program.eqns].count('scan') == 1
