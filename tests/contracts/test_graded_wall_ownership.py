"""W0 T2: wall rows belong to physical faces, including aligned x slabs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.pec import apply_pec_faces
from rfx.boundaries.pmc import apply_pmc_faces
from rfx.core.yee import init_state
from rfx.stepping.slab import FILL, Slab, cut


FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
FACES = frozenset(f"{a}_{s}" for a in "xyz" for s in ("lo", "hi"))


def _check_bits(actual, expected):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert got.tobytes() == want.tobytes()


def _live_check():
    with pytest.raises(AssertionError):
        _check_bits(jnp.array([1.]), jnp.array([0.]))


def _state(shape=(11, 4, 5)):
    state = init_state(shape)
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1
    return state._replace(**{c: (values + i).at[2, 1, 1].set(-0.)
                             for i, c in enumerate(FIELDS)})


def _operator(kind):
    if kind == "pec":
        return apply_pec_faces
    return lambda state, faces, **kwargs: apply_pmc_faces(state, faces, image=True, **kwargs)


@pytest.mark.parametrize('kind', ('pec', 'pmc'))
@pytest.mark.parametrize('subset', range(64))
def test_framed_all_face_subsets_equal_default_on_random_state(kind, subset):
    _live_check()
    rng = np.random.default_rng(20261012)
    state = init_state((11, 4, 5))
    state = state._replace(**{c: jnp.asarray(rng.standard_normal(state.ex.shape), dtype=state.ex.dtype)
                             for c in FIELDS})
    faces = frozenset(face for index, face in enumerate(sorted(FACES)) if subset & (1 << index))
    op = _operator(kind)
    call = jax.jit(lambda st, owners: op(st, faces, frame={'x': (0, st.ex.shape[0])}, owned=owners))
    _check_bits(call(state, {face: jnp.bool_(True) for face in faces}), op(state, faces))


@pytest.mark.parametrize("kind", ("pec", "pmc"))
def test_t2_face_ownership_and_static_frame(kind):
    _live_check()
    op = _operator(kind)
    target = "x_lo" if kind == "pec" else "x_hi"
    components = ("ey", "ez") if kind == "pec" else ("hy", "hz")
    for frame, row in ((None, 0 if kind == "pec" else 10),
                       ({"x": (1, 9)}, 1 if kind == "pec" else 8)):
        state = _state()
        # Other faces start satisfied; their ordinary writes stay enabled.
        if kind == "pec":
            hi = 10 if frame is None else 8
            state = state._replace(
                ex=state.ex.at[:, (0, -1), :].set(0.).at[:, :, (0, -1)].set(0.),
                ey=state.ey.at[hi].set(0.).at[:, :, (0, -1)].set(0.),
                ez=state.ez.at[hi].set(0.).at[:, (0, -1), :].set(0.).at[:, :, -1].set(0.))
        else:
            state = state._replace(
                hx=state.hx.at[:, -1, :].set(0.).at[:, :, -1].set(0.),
                hy=state.hy.at[:, :, -1].set(0.), hz=state.hz.at[:, -1, :].set(0.))
        # A -0.0 on the target row: a non-owner must keep it (select), not add a zero (+0.0).
        state = state._replace(**{c: getattr(state, c).at[row, 1, 1].set(-0.) for c in components})
        call = jax.jit(lambda st, own: op(st, FACES, frame=frame, owned={target: own}))
        _check_bits(call(state, jnp.bool_(False)), state)
        expected = state._replace(**{c: getattr(state, c).at[row].set(0.) for c in components})
        _check_bits(call(state, jnp.bool_(True)), expected)
        _check_bits(op(state, FACES, frame=frame, owned={target: False}), state)
    plain = str(jax.make_jaxpr(lambda st: op(st, FACES))(state))
    explicit = str(jax.make_jaxpr(lambda st: op(st, FACES, owned={f: True for f in FACES}))(state))
    assert plain == explicit
    if kind == "pmc":
        for owner in (False, True):
            _check_bits(jax.jit(lambda st, own: op(
                st, {"x_lo"}, frame={"x": (1, 9)}, owned={"x_lo": own}))(
                    state, jnp.bool_(owner)), state)


@pytest.mark.parametrize("kind", ("pec", "pmc"))
@pytest.mark.parametrize("devices", (1, 2, 3))
def test_t2_slab_rows_equal_global_operator(kind, devices):
    _live_check()
    op, state = _operator(kind), _state()
    layout = Slab(state.ex.shape[0], devices, ghost_width=1)
    if devices > 1:
        assert layout.pad_x > 0
    assert FILL["state"].alignment == FILL["state"].physical == 0.
    slabs = {c: cut(getattr(state, c), layout, "state") for c in FIELDS}
    global_result = op(state, FACES)
    expected_slabs = {c: cut(getattr(global_result, c), layout, 'state') for c in FIELDS}
    lo, hi = 1, 1 + layout.owned[-1][1] - layout.owned[-1][0]
    call = jax.jit(lambda st, low, high: op(
        st, FACES, frame={"x": (lo, hi)},
        owned={"x_lo": low, "x_hi": high, 'y_lo': True, 'y_hi': True, 'z_lo': True, 'z_hi': True}))
    for rank in range(devices):
        local = state._replace(**{c: slabs[c][rank] for c in FIELDS})
        expected = {c: expected_slabs[c][rank] for c in FIELDS}
        result = call(local, jnp.bool_(rank == 0), jnp.bool_(rank == devices - 1))
        _check_bits(result, local._replace(**expected))


@pytest.mark.parametrize("kind", ("pec", "pmc"))
@pytest.mark.parametrize("axis", ("y", "z"))
def test_t2_nondefault_transverse_frame_refuses(kind, axis):
    _live_check()
    with pytest.raises(ValueError, match="Only x"):
        _operator(kind)(_state(), FACES, frame={axis: (1, 3)})


@pytest.mark.parametrize("kwargs", ({"frame": {}}, {"owned": {}}))
def test_t2_legacy_pmc_refuses_new_arguments(kwargs):
    _live_check()
    with pytest.raises(ValueError, match="image=True"):
        apply_pmc_faces(_state(), FACES, **kwargs)
