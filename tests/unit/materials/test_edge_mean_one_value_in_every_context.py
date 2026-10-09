"""The graded E-edge mean has ONE float value in every compile context.

On a graded mesh the permittivity, conductivity and pole fraction of an E edge
are the cell-area weighted mean of its four cells, built from the pair mean
``lo + (hi - lo) * fraction`` on two axes. Written plainly, the compiler may
fuse the multiply into the add, and the same cells then give means one last
bit apart eager, under ``jax.jit`` and per device -- which is why a two-device
run and a one-device run could not agree bit for bit on a graded mesh. The
pair mean now adds exact part products, so fusing changes nothing.

The reference below is written separately (parts by frexp/ldexp, not by bit
masks) and is itself checked against the float64 mean.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import (
    MaterialArrays, component_e_materials, edge_mean_components,
)

SHAPE = (12, 9, 7)
HEAD_BITS = {np.float32: ((12,), (12,)), np.float64: ((26,), (18, 18))}


def _fields(dtype, seed=3):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(1.0, 12.0, SHAPE).astype(dtype)
    sizes = tuple(rng.uniform(0.5e-3, 1.5e-3, n).astype(dtype) for n in SHAPE)
    return arr, sizes


def _back(arr, axis, periodic):
    if periodic[axis]:
        return np.roll(arr, 1, axis=axis)
    first = np.take(arr, [0], axis=axis)
    return np.concatenate([first, np.take(arr, range(arr.shape[axis] - 1), axis=axis)], axis=axis)


def _parts(x, leading):
    out = []
    for keep in leading:
        mantissa, exponent = np.frexp(x)
        head = np.ldexp(np.trunc(np.ldexp(mantissa, keep)), exponent - keep).astype(x.dtype)
        out.append(head)
        x = (x - head).astype(x.dtype)
    return out + [x]


def _reference_pair(lo, hi, fraction):
    dtype = hi.dtype.type
    d_bits, f_bits = HEAD_BITS[dtype]
    d_parts = _parts((hi - lo).astype(dtype), d_bits)
    f_parts = _parts(fraction, f_bits)
    order = sorted(((i, j) for i in range(len(d_parts)) for j in range(len(f_parts))),
                   key=lambda ij: (ij[0] + ij[1], ij[0]))
    result = lo
    for i, j in order:
        result = (result + (d_parts[i] * f_parts[j]).astype(dtype)).astype(dtype)
    return result


def reference_mean(arr, sizes, periodic):
    out = []
    for c in range(3):
        result = arr
        for t in (t for t in range(3) if t != c):
            shape = [1, 1, 1]
            shape[t] = arr.shape[t]
            d = sizes[t].reshape(shape)
            fraction = np.broadcast_to((d / (_back(d, t, periodic) + d)).astype(arr.dtype), arr.shape)
            result = _reference_pair(_back(result, t, periodic), result, fraction)
        out.append(result)
    return out


def float64_mean(arr, sizes, periodic):
    out = []
    for c in range(3):
        result = arr.astype(np.float64)
        for t in (t for t in range(3) if t != c):
            shape = [1, 1, 1]
            shape[t] = arr.shape[t]
            d = sizes[t].reshape(shape)
            fraction = (d / (_back(d, t, periodic) + d)).astype(np.float64)   # the fraction the code forms
            lo = _back(result, t, periodic)
            result = lo + (result - lo) * fraction
        out.append(result)
    return out


def _contexts(fn, x):
    zero = jnp.zeros((), x.dtype)
    return {
        "eager": fn(x),
        "jit": jax.jit(fn)(x),
        "scan": tuple(v[-1] for v in jax.jit(lambda v: jax.lax.scan(
            lambda c, _: (c, fn(v + zero * c)), zero, None, length=2)[1])(x)),
    }


def _assert_equal_bits(got, want, label):
    got = np.asarray(got)
    assert got.dtype == want.dtype, (label, got.dtype)
    unequal = int((got.view(np.uint8) != want.view(np.uint8)).reshape(got.shape + (-1,)).any(-1).sum())
    assert unequal == 0, f"{label}: {unequal} of {got.size} entries differ in their bits"


PERIODIC = [(False, False, False), (False, True, False), (True, True, True)]


@pytest.mark.parametrize("periodic", PERIODIC)
def test_reference_is_the_float64_mean_to_rounding(periodic):
    arr, sizes = _fields(np.float32)
    ref, wide = reference_mean(arr, sizes, periodic), float64_mean(arr, sizes, periodic)
    for c in range(3):
        assert np.max(np.abs(ref[c] - wide[c]) / wide[c]) < 4 * np.finfo(np.float32).eps


@pytest.mark.parametrize("periodic", PERIODIC)
def test_graded_mean_has_one_value_eager_jit_scan_and_numpy(periodic):
    arr, sizes = _fields(np.float32)
    ref = reference_mean(arr, sizes, periodic)
    jsizes = tuple(jnp.asarray(s) for s in sizes)
    for label, got in _contexts(
            lambda v: edge_mean_components(v, periodic, cell_sizes=jsizes), jnp.asarray(arr)).items():
        for c in range(3):
            _assert_equal_bits(got[c], ref[c], f"{label} component {c}")
    host = edge_mean_components(arr, periodic, cell_sizes=sizes, array_module=np)
    for c in range(3):
        _assert_equal_bits(host[c], ref[c], f"numpy module component {c}")


def test_a_stamp_added_after_the_mean_does_not_change_its_bits_under_jit():
    """An edge-owned stamp (a port load, a film) is added after the mean; a
    mean ending in a multiply would be fused with that add."""
    periodic = (False, False, False)
    arr, sizes = _fields(np.float32)
    stamp = np.random.default_rng(9).uniform(0.5, 400.0, SHAPE).astype(np.float32)
    ref = reference_mean(arr, sizes, periodic)
    jsizes = tuple(jnp.asarray(s) for s in sizes)
    got = jax.jit(lambda v, p: tuple(
        m + p for m in edge_mean_components(v, periodic, cell_sizes=jsizes)))(jnp.asarray(arr), jnp.asarray(stamp))
    for c in range(3):
        _assert_equal_bits(got[c], (ref[c] + stamp).astype(np.float32), f"stamped component {c}")


@pytest.mark.parametrize("n_slabs", [2, 3])
def test_each_slab_forms_the_whole_domain_mean_on_its_own_rows(n_slabs):
    """A slab holding its own rows plus the row below (its ghost) computes, in
    its own compiled program, the bits of the whole-domain mean on its rows."""
    periodic = (False, False, False)
    arr, sizes = _fields(np.float32)
    ref = reference_mean(arr, sizes, periodic)
    edges = np.linspace(0, SHAPE[0], n_slabs + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        ghost = max(lo - 1, 0)
        local_sizes = (jnp.asarray(sizes[0][ghost:hi]), jnp.asarray(sizes[1]), jnp.asarray(sizes[2]))
        got = jax.jit(lambda v: edge_mean_components(v, periodic, cell_sizes=local_sizes))(jnp.asarray(arr[ghost:hi]))
        for c in range(3):
            _assert_equal_bits(np.asarray(got[c])[lo - ghost:], ref[c][lo:hi], f"slab {lo}:{hi} component {c}")


def test_lane_function_hands_the_step_the_reference_bits():
    """What the uniform and graded lanes read: component_e_materials inside a
    compiled program, for permittivity and conductivity."""
    periodic = (False, False, False)
    eps, sizes = _fields(np.float32)
    sigma, _ = _fields(np.float32, seed=11)
    ref_eps, ref_sig = reference_mean(eps, sizes, periodic), reference_mean(sigma, sizes, periodic)
    jsizes = tuple(jnp.asarray(s) for s in sizes)

    def lane(e, s):
        materials = MaterialArrays(eps_r=e, sigma=s, mu_r=jnp.ones_like(e))
        return component_e_materials(materials, periodic, cell_sizes=jsizes)

    got_eps, got_sig = jax.jit(lane)(jnp.asarray(eps), jnp.asarray(sigma))
    for c in range(3):
        _assert_equal_bits(got_eps[c], ref_eps[c], f"eps component {c}")
        _assert_equal_bits(got_sig[c], ref_sig[c], f"sigma component {c}")


def test_gradient_is_the_area_weight():
    periodic = (False, False, False)
    arr, sizes = _fields(np.float32)
    jsizes = tuple(jnp.asarray(s) for s in sizes)
    grad = np.asarray(jax.grad(lambda v: jnp.sum(
        edge_mean_components(v, periodic, cell_sizes=jsizes)[0]))(jnp.asarray(arr)))
    ones = np.ones(SHAPE, np.float64)
    # d(sum of means)/d(cell) = total weight the cell gets from the edges that read it.
    weight = np.zeros(SHAPE)
    f = []
    for t in (1, 2):
        shape = [1, 1, 1]
        shape[t] = SHAPE[t]
        d = sizes[t].astype(np.float64).reshape(shape)
        f.append(np.broadcast_to(d / (_back(d, t, periodic) + d), SHAPE))
    for s1 in (0, 1):
        for s2 in (0, 1):
            w = (f[0] if s1 == 0 else 1 - f[0]) * (f[1] if s2 == 0 else 1 - f[1])
            # weight w at edge (j, k) goes to cell (j - s1, k - s2), replicated at the wall
            src = np.indices(SHAPE)
            j = np.maximum(src[1] - s1, 0)
            k = np.maximum(src[2] - s2, 0)
            np.add.at(weight, (src[0], j, k), w * ones)
    assert np.max(np.abs(grad - weight)) < 1e-5


def test_float64_fields_have_one_value_too():
    with jax.enable_x64(True):
        periodic = (False, True, False)
        arr, sizes = _fields(np.float64)
        ref = reference_mean(arr, sizes, periodic)
        jsizes = tuple(jnp.asarray(s) for s in sizes)
        for label, got in _contexts(
                lambda v: edge_mean_components(v, periodic, cell_sizes=jsizes), jnp.asarray(arr)).items():
            for c in range(3):
                _assert_equal_bits(got[c], ref[c], f"float64 {label} component {c}")


def test_uniform_mesh_mean_is_untouched():
    arr, _ = _fields(np.float32)
    periodic = (False, False, False)
    a1, a2 = _back(arr, 1, periodic), _back(arr, 2, periodic)
    want = (((arr + a1) + (a2 + _back(a1, 2, periodic))) * np.float32(0.25)).astype(np.float32)
    got = jax.jit(lambda v: edge_mean_components(v, periodic))(jnp.asarray(arr))
    _assert_equal_bits(got[0], want, "uniform component 0")


@pytest.mark.gpu_gate
def test_graded_mean_has_one_value_on_this_backend():
    """The same comparison on whatever backend the suite runs on; the
    merge train's GPU gate runs it on a GPU."""
    periodic = (False, True, False)
    arr, sizes = _fields(np.float32)
    ref = reference_mean(arr, sizes, periodic)
    jsizes = tuple(jnp.asarray(s) for s in sizes)
    for label, got in _contexts(
            lambda v: edge_mean_components(v, periodic, cell_sizes=jsizes), jnp.asarray(arr)).items():
        for c in range(3):
            _assert_equal_bits(got[c], ref[c], f"{jax.default_backend()} {label} component {c}")
