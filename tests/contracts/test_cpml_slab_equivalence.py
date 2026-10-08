"""Slab CPML equals an independent whole-array stencil for explicit faces."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, init_cpml
from rfx.core.yee import EPS_0, MU_0, init_materials, init_state
from rfx.grid import Grid
from tests._x64_compat import enable_x64

FACES = tuple(f"{a}_{s}" for a in "xyz" for s in ("lo", "hi"))


def _whole_array(state, params, psi, grid, materials, faces, magnetic):
    """Independent full-field zero-extended differences and sequential scatters.

    No slab-neighbor, selection, write-back, or coefficient helper is shared
    with production. Each difference is constructed over the whole array.
    """
    kind, source = ("h", "e") if magnetic else ("e", "h")
    profiles = params.magnetic if magnetic else params
    fields = {c: getattr(state, kind + c) for c in "xyz"}
    coefficients = []
    for target in range(3):
        material = materials.eps_r
        if magnetic:
            # Uniform Yee H faces use the harmonic mean with the backward
            # cell on the component axis; the outer cell is edge-replicated.
            mu = materials.mu_r
            indices = jnp.maximum(jnp.arange(mu.shape[target]) - 1, 0)
            backward = jnp.take(mu, indices, axis=target)
            material = jnp.where(backward == mu, mu, 2 / (1 / backward + 1 / mu))
        coefficients.append(float(grid.dt) / (material * (MU_0 if magnetic else EPS_0)))
    changes = {}
    for axis, (a, pairs) in enumerate(zip("xyz", (((1, 2, -1), (2, 1, 1)),
                                                ((0, 2, 1), (2, 0, -1)),
                                                ((0, 1, -1), (1, 0, 1))))):
        for target, operand, sign in pairs:
            original = getattr(state, source + "xyz"[operand])
            neighbor = jnp.roll(original, -1 if magnetic else 1, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = -1 if magnetic else 0
            neighbor = neighbor.at[tuple(edge)].set(0)
            difference = neighbor - original if magnetic else original - neighbor
            for side in ("lo", "hi"):
                face = f"{a}_{side}"
                if face not in faces:
                    continue
                component = "xyz"[target]
                key = f"psi_{kind}{component}_{a}{side}"
                depth = getattr(psi, key).shape[axis]
                sl = [slice(None)] * 3
                sl[axis] = slice(None, depth) if side == "lo" else slice(-depth, None)
                sl = tuple(sl)
                shape = [1] * 3
                shape[axis] = depth
                profile = getattr(profiles, face)
                window = slice(None, depth) if side == "lo" else slice(-depth, None)
                b, c, k = (v[window].reshape(shape) for v in
                           (profile.b, profile.c, profile.kappa))
                curl = (difference / grid.boundary_cell(a, side))[sl]
                updated = b * getattr(psi, key) + c * curl
                changes[key] = updated
                coefficient = coefficients[target][sl]
                signed = (-sign if magnetic else sign) * coefficient
                field = fields[component].at[sl].add(signed * updated)
                # The x branch multiplies the coefficient first; the y/z
                # branches form kappa*curl before coefficient multiplication.
                correction = (signed * (1 / k - 1) * curl if axis == 0
                              else signed * ((1 / k - 1) * curl))
                fields[component] = field.at[sl].add(correction)
    return state._replace(**{kind + c: v for c, v in fields.items()}), psi._replace(**changes)


def _assert_equal(actual, expected):
    for value, reference in zip(actual[0], expected[0]):
        np.testing.assert_array_equal(value, reference)
    for value, reference in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(value, reference)


def _scene(wall="z_hi", varying_mu=False):
    h = 1 / 512
    depths = (2, 4, 3, 2, 4, 0) if wall == "z_hi" else (0, 2, 3, 4, 2, 3)
    grid = Grid(freq_max=5e9, domain=(9*h, 12*h, 15*h), dx=h,
                cpml_layers=4, face_layers=dict(zip(FACES, depths)),
                pec_faces=frozenset({wall}))
    assert grid.face_pads == depths
    params, psi = init_cpml(grid, kappa_max=3.0)
    rng = np.random.default_rng(952)
    state = init_state(grid.shape)
    state = state._replace(**{k: jnp.asarray(rng.normal(size=v.shape), jnp.float32)
                             for k, v in zip(state._fields, state)})
    psi = psi._replace(**{k: jnp.asarray(rng.normal(size=v.shape), jnp.float32)
                         for k, v in zip(psi._fields, psi)})
    materials = init_materials(grid.shape)
    # Eps, and optionally mu, vary along all three axes, including absorbers.
    eps = jnp.asarray(rng.uniform(1, 4, grid.shape), jnp.float32)
    mu = (jnp.asarray(rng.uniform(1, 3, grid.shape), jnp.float32)
          if varying_mu else materials.mu_r)
    materials = materials._replace(eps_r=eps, mu_r=mu)
    return grid, params, psi, state, materials


@pytest.mark.parametrize("wall", ("z_hi", "x_lo"), ids=("hi-wall", "lo-wall"))
@pytest.mark.parametrize("varying_mu", (False, True), ids=("constant-mu", "varying-mu"))
@pytest.mark.parametrize("magnetic", (False, True))
def test_slab_update_equals_whole_array_per_face_subset(magnetic, varying_mu, wall):
    grid, params, psi, state, materials = _scene(wall, varying_mu)
    apply = apply_cpml_h if magnetic else apply_cpml_e
    for enabled in itertools.product((False, True), repeat=6):
        faces = tuple(f for f, active in zip(FACES, enabled) if active)
        actual = apply(state, params, psi, grid, materials=materials, faces=faces)
        expected = _whole_array(state, params, psi, grid, materials, faces, magnetic)
        _assert_equal(actual, expected)


def test_equivalence_check_rejects_a_changed_field():
    """Removing the numerical check itself must make the contract red."""
    _, _, psi, state, _ = _scene()
    wrong = state._replace(ez=state.ez.at[1, 2, 3].add(1))
    with pytest.raises(AssertionError):
        _assert_equal((wrong, psi), (state, psi))


def test_float64_corrections_with_float32_fields_equal_whole_array():
    """Each wider slab sum rounds to the field work dtype, as scatter does."""
    with enable_x64():
        grid, params, psi, state, materials = _scene()
        materials = materials._replace(
            eps_r=materials.eps_r.astype(jnp.float64),
            mu_r=materials.mu_r.astype(jnp.float64))
        for magnetic, apply in ((False, apply_cpml_e), (True, apply_cpml_h)):
            actual = apply(state, params, psi, grid, materials=materials)
            expected = _whole_array(
                state, params, psi, grid, materials, FACES, magnetic)
            assert all(v.dtype == jnp.float32 for v in actual[0])
            _assert_equal(actual, expected)


def test_face_selection_intersects_axes_and_preserves_other_psi():
    grid, params, psi, state, materials = _scene()
    for apply in (apply_cpml_e, apply_cpml_h):
        actual = apply(state, params, psi, grid, axes="x", materials=materials,
                       faces=("x_hi", "y_lo"))
        expected = apply(state, params, psi, grid, materials=materials, faces=("x_hi",))
        _assert_equal(actual, expected)
        with pytest.raises(ValueError, match="unknown CPML faces"):
            apply(state, params, psi, grid, faces=("x_middle",))


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("mode", ("zero", "periodic", "pmc"))
def test_slab_h_neighbour_matches_shared_boundary_images(axis, mode):
    from rfx.boundaries._cpml_slab import slab_neighbor
    from rfx.core.yee import CurlBoundary, h_neighbor

    faces = frozenset({f"{'xyz'[axis]}_lo", f"{'xyz'[axis]}_hi"})
    boundary = CurlBoundary(
        pmc_faces=faces if mode == "pmc" else frozenset(),
        periodic=tuple(mode == "periodic" and a == axis for a in range(3)))
    for size in (1, 5):
        shape = [4, 6, 7]
        shape[axis] = size
        field = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
        whole = h_neighbor(field, axis, boundary=boundary)
        for depth in {1, (size + 1) // 2, size}:
            for lo in (False, True):
                sl = [slice(None)] * 3
                sl[axis] = slice(None, depth) if lo else slice(-depth, None)
                actual = slab_neighbor(field, axis, depth, lo, False, boundary)
                np.testing.assert_array_equal(actual, whole[tuple(sl)])
