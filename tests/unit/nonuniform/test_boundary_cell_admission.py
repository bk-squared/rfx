"""One declaration, one admission answer for a graded profile's end cells.

A graded x profile names its boundary cell twice: the scalar ``dx`` and the
profile's first and last cells. The time step stores cell widths as float32,
so a profile handed over in float32 realizes the same boundary cell as its
float64 ``dx`` although the two differ by up to half a float32 ULP (9.3e-12 m
at dx = 0.381 mm). The concrete builder refused that profile while the
traced builder, which cannot see values, accepted it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.nonuniform import make_nonuniform_grid

DX = 0.381e-3
DZ = np.full(8, DX)


def _profile(end):
    inner = np.linspace(DX, 0.2e-3, 6)
    return np.concatenate([[end], np.full(4, DX), inner, np.full(4, DX), [end]])


def _build(profile, length=None):
    if length is None:
        length = float(np.sum(np.asarray(profile, dtype=np.float64)))
    domain = (length, 20 * DX)
    return make_nonuniform_grid(domain, DZ, DX, cpml_layers=4, dx_profile=profile)


def test_float32_profile_admitted_on_both_paths_with_one_boundary_cell():
    prof = _profile(DX).astype(np.float32)
    # The float32 end cell is NOT the float64 dx: the case is the real one.
    assert abs(float(prof[0]) - DX) == pytest.approx(9.26e-12, rel=1e-2)
    concrete = _build(prof)
    length = float(np.sum(prof.astype(np.float64)))
    traced = jax.jit(lambda p: _build(p, length).boundary_cell("x", "lo"))(jnp.asarray(prof))
    for side in ("lo", "hi"):
        assert np.float32(concrete.boundary_cell("x", side)) == np.float32(traced)


@pytest.mark.parametrize("end", [DX * 1.01, DX + 3 * float(np.spacing(np.float32(DX)))],
                         ids=["one_percent", "three_ulp"])
def test_end_cell_off_by_more_than_one_float32_ulp_is_refused(end):
    with pytest.raises(ValueError, match="must equal boundary"):
        _build(_profile(end))


def test_y_end_cells_compared_within_one_float32_ulp():
    prof = _profile(DX).astype(np.float32)
    domain = (20 * DX, float(np.sum(prof.astype(np.float64))))
    make_nonuniform_grid(domain, DZ, DX, cpml_layers=4, dy_profile=prof)
    bad = prof.astype(np.float64)
    bad[-1] = DX * 1.01
    with pytest.raises(ValueError, match="boundary cells must match"):
        make_nonuniform_grid(domain, DZ, DX, cpml_layers=4, dy_profile=bad)
