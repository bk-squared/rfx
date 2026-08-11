"""Scan carries must follow the ambient precision, not a hard float32 pin.

Issue #646. Three ``lax.scan`` carry families allocated auxiliary state at a
hard-pinned float32/complex64 while the quantities driving them followed the
ambient dtype, so enabling JAX x64 anywhere in a pytest worker reddened them
with ``scan body function carry input and carry output must have equal types``:

    rfx/lumped.py       init_rlc_state()      inductor_current, capacitor_charge
    rfx/farfield.py     init_ntff_data()      {x,y,z}_{lo,hi}, c_{x,y,z}_{lo,hi}
    rfx/sources/tfsf_2d.py init_tfsf_2d()     ez_2d, hx_2d, hy_2d, psi_*

MECHANISM (measured, and NOT what the issue first proposed). The promoting
quantity is not the field dtype -- the fields stay float32 under x64. It is
``grid.dt``: ``Grid.__init__`` computes it with numpy, so it is an
``np.float64`` **scalar**, and numpy scalars are STRONGLY typed in JAX. Under
x64 they promote the phase/coefficient arithmetic (and hence the scan body's
output) a step above the dtype the carry was allocated at; with x64 off JAX
clamps float64 to float32 and the pin happens to agree. See
``test_dt_is_a_numpy_float64_scalar`` below, which pins that premise directly:
it is the reason a float32 pin is wrong, so if it ever stops holding, the
rest of this module is testing the wrong thing.

DTYPE POLICY. ``promote_types(field_dtype, float32)`` -- promote, never pin,
never a bare copy of the field dtype. Both failure modes are live:

  * a flat float32 pin breaks the oblique-Bloch path (#404), which needs a
    genuinely COMPLEX carry;
  * naively following the field dtype puts a recursive accumulator in float16
    under ``precision="mixed"``, contradicting what that mode promises
    ("float16 fields, float32 accumulation").

x64 is scoped with ``tests/_x64_compat.enable_x64()``. A bare
``jax.config.update("jax_enable_x64", True)`` is process-global, is forbidden
by this repo's CLAUDE.md, and reddened 13 unrelated tests on PR #645.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, init_materials
from rfx.geometry.csg import Box, rasterize
from rfx.farfield import ntff_accum_dtype, init_ntff_data, NTFFBox
from rfx.lumped import init_rlc_state
from rfx.rcs import compute_rcs
from rfx.simulation import run, ProbeSpec
from rfx.sources.tfsf import init_tfsf

from tests._x64_compat import enable_x64

PEC_SIGMA = 1e7


# ---------------------------------------------------------------------------
# The premise the whole issue rests on
# ---------------------------------------------------------------------------

def test_dt_is_a_numpy_float64_scalar():
    """``grid.dt`` is an np.float64 scalar -> strongly typed in JAX.

    This is WHY the carries must promote rather than pin: the promoting
    quantity is a numpy scalar coefficient, not the field storage. If this
    ever becomes a plain Python float (weakly typed), the mechanism recorded
    in this module's docstring no longer applies and the fix should be
    re-derived rather than trusted.
    """
    grid = Grid(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3)
    assert isinstance(grid.dt, np.floating), (
        f"grid.dt is {type(grid.dt).__name__}, expected a numpy scalar")
    assert jnp.dtype(np.asarray(grid.dt).dtype) == jnp.float64

    with enable_x64():
        # A numpy float64 scalar promotes a complex64 array; a Python float
        # (weakly typed) does not. That asymmetry is the whole defect.
        c = jnp.zeros(3, dtype=jnp.complex64)
        assert (c * np.float64(1e-12)).dtype == jnp.complex128
        assert (c * 1e-12).dtype == jnp.complex64


# ---------------------------------------------------------------------------
# The dtype policy itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_dtype,expected", [
    (jnp.float16, jnp.complex64),     # mixed precision -> float32 FLOOR
    (jnp.float32, jnp.complex64),     # default -> unchanged
    (jnp.complex64, jnp.complex64),   # oblique Bloch (#404) -> stays COMPLEX
])
def test_ntff_accum_dtype_promotes_never_pins(field_dtype, expected):
    """Holds identically with x64 off and on."""
    assert ntff_accum_dtype(field_dtype) == expected
    with enable_x64():
        assert ntff_accum_dtype(field_dtype) == expected


def test_ntff_accum_dtype_float64_needs_x64():
    """float64 fields -> complex128, but ONLY once x64 is enabled.

    With x64 off JAX has no float64, so ``result_type`` clamps the request to
    complex64. That is not a bug in this policy, it is the same documented
    footgun as ``precision="float64"`` without x64 (which silently gives
    float32 fields; ``preflight()`` warns ``precision_float64_without_x64``).
    Pinned here so the clamp is a recorded behaviour rather than a surprise.

    Keyed off the AMBIENT flag rather than assuming it is off: the documented
    way to reproduce #646 is to run this suite under ``JAX_ENABLE_X64=1``, so
    a test that hard-codes the x64-off branch fails in exactly the
    configuration the issue asks you to run.
    """
    ambient_x64 = bool(jax.config.jax_enable_x64)
    expected = jnp.complex128 if ambient_x64 else jnp.complex64
    assert ntff_accum_dtype(jnp.float64) == expected
    with enable_x64():
        assert ntff_accum_dtype(jnp.float64) == jnp.complex128


def test_ntff_float32_floor_holds_under_mixed_precision():
    """float16 fields must NOT give a float16 accumulator.

    The NTFF carry is a recursive DFT accumulator; float16's 11-bit mantissa
    would destroy it. Guards the "naively follow the field dtype" failure.
    """
    assert ntff_accum_dtype(jnp.float16) == jnp.complex64
    box = NTFFBox(i_lo=2, i_hi=6, j_lo=2, j_hi=6, k_lo=2, k_hi=6,
                  freqs=jnp.array([1e9], dtype=jnp.float32))
    data = init_ntff_data(box, field_dtype=jnp.float16)
    assert data.x_lo.dtype == jnp.complex64
    assert data.c_x_lo.dtype == jnp.complex64


def test_ntff_complex_carry_survives_the_oblique_path():
    """A flat float32 pin would destroy the #404 complex carry."""
    box = NTFFBox(i_lo=2, i_hi=6, j_lo=2, j_hi=6, k_lo=2, k_hi=6,
                  freqs=jnp.array([1e9], dtype=jnp.float32))
    data = init_ntff_data(box, field_dtype=jnp.complex64)
    assert jnp.issubdtype(data.x_lo.dtype, jnp.complexfloating)


def test_rlc_carry_follows_ambient_float_with_a_float32_floor():
    """The default carry dtype tracks the ambient float, never a float32 pin.

    Keyed off the ambient flag for the same reason as
    ``test_ntff_accum_dtype_float64_needs_x64`` — this suite has to stay green
    under ``JAX_ENABLE_X64=1``, which is how #646 is reproduced.
    """
    ambient = jnp.float64 if bool(jax.config.jax_enable_x64) else jnp.float32
    assert init_rlc_state().inductor_current.dtype == ambient
    with enable_x64():
        st = init_rlc_state()
        assert st.inductor_current.dtype == jnp.float64
        assert st.capacitor_charge.dtype == jnp.float64
    # flag restored -> back to the ambient default
    assert init_rlc_state().inductor_current.dtype == ambient


# ---------------------------------------------------------------------------
# The three scans must actually CLOSE under x64
# ---------------------------------------------------------------------------

def _lumped_sim():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=1e-3,
                     boundary="pec")
    sim.add_lumped_rlc(position=(0.005, 0.005, 0.005), component="ez",
                       R=50.0, L=1e-8, C=1e-12, topology="series")
    sim.add_source((0.004, 0.005, 0.005), "ez", amplitude_kind="field")
    sim.add_probe((0.006, 0.005, 0.005), "ez")
    return sim


def test_lumped_rlc_scan_closes_under_x64():
    with enable_x64():
        r = _lumped_sim().run(n_steps=20, skip_preflight=True)
    assert np.all(np.isfinite(np.asarray(r.time_series)))


def _rcs_args():
    f0 = 3e9
    dx = C0 / f0 / 10
    domain = 0.12
    grid = Grid(freq_max=f0 * 1.5, domain=(domain, domain, domain), dx=dx,
                cpml_layers=8)
    c = domain / 2
    ht, hs = dx / 2, 0.04 / 2
    plate = Box(corner_lo=(c - ht, c - hs, c - hs),
                corner_hi=(c + ht, c + hs, c + hs))
    eps_r, sigma = rasterize(grid, [(plate, 1.0, PEC_SIGMA)])
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    return grid, mats, f0


def test_ntff_rcs_scan_closes_under_x64():
    grid, mats, f0 = _rcs_args()
    with enable_x64():
        res = compute_rcs(grid, mats, 40, f0=f0, bandwidth=0.5,
                          theta_obs=np.linspace(0.01, np.pi - 0.01, 5),
                          phi_obs=np.array([0.0]))
    assert np.all(np.isfinite(np.asarray(res.monostatic_rcs)))


def test_oblique_tfsf_2d_scan_closes_under_x64():
    """The 2-D oblique aux grid — a THIRD site, absent from issue #646's table.

    It raised the same carry-type mismatch on
    ``carry['tfsf'].{ez_2d,hx_2d,hy_2d,psi_ez_x*,psi_hy_x*}`` and accounted
    for one of the nine reported failures.
    """
    grid = Grid(freq_max=5e9, domain=(0.06, 0.06, 0.006), dx=0.004,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    cfg, st0 = init_tfsf(grid.nx, grid.dx, grid.dt,
                         cpml_layers=grid.cpml_layers, tfsf_margin=3,
                         f0=5e9, bandwidth=0.2, amplitude=1.0,
                         polarization="ez", angle_deg=30.0, ny=grid.ny)
    probe = ProbeSpec(i=cfg.x_lo + 3, j=grid.ny // 2, k=grid.nz // 2,
                      component="ez")
    with enable_x64():
        r = run(grid, mats, 30, boundary="cpml", cpml_axes="x",
                periodic=(False, True, True), tfsf=(cfg, st0), probes=[probe])
    assert np.all(np.isfinite(np.asarray(r.time_series)))


# ---------------------------------------------------------------------------
# x64 must not change the default-precision numbers
# ---------------------------------------------------------------------------

def test_float32_ntff_accumulator_is_complex64_even_under_x64():
    """Turning x64 on must not silently widen (or narrow) the default lane.

    With float32 field storage the accumulator stays complex64 whether or not
    x64 is on -- so enabling x64 for an unrelated test does not double NTFF
    memory or perturb the validated float32 numbers.
    """
    box = NTFFBox(i_lo=2, i_hi=6, j_lo=2, j_hi=6, k_lo=2, k_hi=6,
                  freqs=jnp.array([1e9], dtype=jnp.float32))
    assert init_ntff_data(box, field_dtype=jnp.float32).x_lo.dtype == jnp.complex64
    with enable_x64():
        d = init_ntff_data(box, field_dtype=jnp.float32)
        assert d.x_lo.dtype == jnp.complex64
        assert d.c_x_lo.dtype == jnp.complex64


def test_x64_does_not_move_the_default_precision_lumped_result():
    """Same probe trace with x64 off and on, at precision="float32".

    The carry widens under x64 (that is the fix), but the float32 field lane
    it drives must be unaffected beyond float32 round-off.
    """
    ts_off = np.asarray(_lumped_sim().run(n_steps=20,
                                          skip_preflight=True).time_series)
    with enable_x64():
        ts_on = np.asarray(_lumped_sim().run(n_steps=20,
                                             skip_preflight=True).time_series)
    scale = max(float(np.max(np.abs(ts_off))), 1e-30)
    assert np.max(np.abs(ts_on - ts_off)) / scale < 1e-5


def test_enable_x64_context_restores_the_flag():
    """The guard in conftest depends on this; pin it explicitly."""
    before = bool(jax.config.jax_enable_x64)
    with enable_x64():
        assert bool(jax.config.jax_enable_x64) is True
    assert bool(jax.config.jax_enable_x64) is before
