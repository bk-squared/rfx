"""Source declarations: None means current; native helper contracts remain (#1373)."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.api._source_semantics import (
    needs_scale,
    source_amplitude_scale,
)
from rfx.core.yee import EPS_0, init_materials
from rfx.grid import Grid
from rfx.simulation import make_j_source, make_source

DX = 1e-3
NA, NB, NZ = 24, 20, 4
STEPS = 600
F0 = 5e9
DV = DX ** 3

_SRC = (NA * DX / 3, NB * DX / 3, 2 * DX)
_PRB = (2 * NA * DX / 3, 2 * NB * DX / 3, 2 * DX)


def _sim(boundary: str, nonuniform: bool) -> Simulation:
    profiles = (dict(dx_profile=np.full(NA, DX), dy_profile=np.full(NB, DX))
                if nonuniform else {})
    return Simulation(freq_max=1e10, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary=boundary,
                      cpml_layers=(0 if boundary == "pec" else 8), **profiles)


def _trace(boundary: str, nonuniform: bool, kind, amplitude: float = 1.0,
           omit_kwarg: bool = False) -> np.ndarray:
    sim = _sim(boundary, nonuniform)
    wf = GaussianPulse(f0=F0, bandwidth=0.8, amplitude=amplitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if omit_kwarg:
            sim.add_source(_SRC, "ez", waveform=wf)
        else:
            sim.add_source(_SRC, "ez", waveform=wf, amplitude_kind=kind)
    sim.add_probe(_PRB, "ez")
    result = sim.run(n_steps=STEPS, compute_s_params=False,
                     skip_preflight=True)
    return np.asarray(result.time_series, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Conversion-table unit contract (the ONE conversion site)
# ---------------------------------------------------------------------------

def test_conversion_table_and_dispatch():
    """Full target/native table + the Python-level needs_scale dispatch."""
    # no-op rows (bit-identity guarantee): None everywhere, kind == native
    for native in ("raw", "cb", "cb_over_dv"):
        assert needs_scale(None, native) is False
        assert source_amplitude_scale(None, native, cb=123.0, dV=456.0) == 1.0
    assert needs_scale("field", "raw") is False
    assert needs_scale("current", "cb_over_dv") is False
    # 'cb' realizes NEITHER kind (legacy open-uniform third contract)
    assert needs_scale("field", "cb") and needs_scale("current", "cb")
    # conversion rows
    assert source_amplitude_scale("field", "cb", cb=4.0, dV=None) == 0.25
    assert source_amplitude_scale("field", "cb_over_dv", cb=2.0, dV=8.0) == 4.0
    assert source_amplitude_scale("current", "raw", cb=2.0, dV=8.0) == 0.25
    assert source_amplitude_scale("current", "cb", cb=None, dV=2.0) == 0.5
    # validation
    with pytest.raises(ValueError, match="native"):
        needs_scale("field", "bogus")
    with pytest.raises(ValueError, match="amplitude_kind"):
        needs_scale("volts", "cb")


def test_dispatch_is_python_level_and_tracer_safe():
    """The dossier's ConcretizationTypeError fix: whether to scale is
    decided on the (kind, native) string pair, so a TRACED cb/dV never
    reaches a value comparison. Under jit this raises if anyone
    reintroduces ``if scale != 1.0``-style gating on the value."""
    @jax.jit
    def convert(cb, dv):
        w = jnp.ones(4, dtype=jnp.float32)
        if needs_scale("field", "cb"):
            w = source_amplitude_scale("field", "cb", cb=cb, dV=dv) * w
        if needs_scale("current", "raw"):
            w = source_amplitude_scale("current", "raw", cb=cb, dV=dv) * w
        return w
    out = convert(jnp.float32(2.0), jnp.float32(4.0))
    # (1/cb) * (cb/dV) = 1/dV = 0.25
    np.testing.assert_allclose(np.asarray(out), 0.25, rtol=1e-6)


def test_add_source_rejects_unknown_kind():
    sim = _sim("pec", False)
    with pytest.raises(ValueError, match="amplitude_kind"):
        sim.add_source(_SRC, "ez", amplitude_kind="volts")


# ---------------------------------------------------------------------------
# (a) Omitted and explicit None declarations agree; native helpers stay legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,nonuniform", [
    pytest.param("pec", False, id="closed-uniform"),
    pytest.param("cpml", False, id="open-uniform"),
    pytest.param("cpml", True, id="nonuniform"),
])
def test_kind_none_equals_omitting_the_kwarg_bitwise(boundary, nonuniform):
    """Full-trace byte identity: passing amplitude_kind=None explicitly is
    the documented spelling of today's default."""
    t_omit = _trace(boundary, nonuniform, None, omit_kwarg=True)
    t_none = _trace(boundary, nonuniform, None)
    assert np.abs(t_omit).max() > 0.0
    assert np.array_equal(t_omit, t_none)


def test_make_j_source_kind_none_bitwise_matches_pristine_formula():
    """Helper-level pin of the pristine open-uniform waveform: kind=None
    output of make_j_source is BIT-identical to the pre-#571 formula
    ``Cb * w(t)`` spelled inline here (same expression, same op order)."""
    grid = Grid(freq_max=1e10, domain=(NA * DX, NB * DX, NZ * DX),
                dx=DX, cpml_layers=8)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=F0, bandwidth=0.8)
    spec = make_j_source(grid, _SRC, "ez", pulse, 64, materials)
    # inline pristine formula
    i, j, k = grid.position_to_index(_SRC)
    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)
    times = jnp.arange(64, dtype=jnp.float32) * grid.dt
    pristine = cb * jax.vmap(pulse)(times)
    assert np.array_equal(np.asarray(spec.waveform), np.asarray(pristine))


def test_make_source_kind_none_bitwise_matches_pristine_formula():
    """Helper-level pin of the pristine closed-uniform waveform (raw add)."""
    grid = Grid(freq_max=1e10, domain=(NA * DX, NB * DX, NZ * DX),
                dx=DX, cpml_layers=0)
    pulse = GaussianPulse(f0=F0, bandwidth=0.8)
    spec = make_source(grid, _SRC, "ez", pulse, 64)
    times = jnp.arange(64, dtype=jnp.float32) * grid.dt
    pristine = jax.vmap(pulse)(times)
    assert np.array_equal(np.asarray(spec.waveform), np.asarray(pristine))


# ---------------------------------------------------------------------------
# (b) Declaration None == current on every path
# ---------------------------------------------------------------------------

def test_closed_uniform_none_equals_current_bitwise():
    t_none = _trace("pec", False, None)
    t_current = _trace("pec", False, "current")
    assert np.abs(t_none).max() > 0.0
    assert np.array_equal(t_none, t_current)


def test_nonuniform_none_equals_current_bitwise():
    """NU: legacy == amplitude_kind='current', bitwise (both take the
    no-multiply path through make_current_source)."""
    t_none = _trace("cpml", True, None)
    t_curr = _trace("cpml", True, "current")
    assert np.abs(t_none).max() > 0.0
    assert np.array_equal(t_none, t_curr)


def test_open_uniform_none_equals_current_bitwise():
    t_none = _trace("cpml", False, None)
    t_current = _trace("cpml", False, "current")
    assert np.abs(t_none).max() > 0.0
    assert np.array_equal(t_none, t_current)


# ---------------------------------------------------------------------------
# (c) The NEW invariant: explicit 'current' makes the two builders EQUAL
# ---------------------------------------------------------------------------

def test_current_kind_makes_uniform_and_nonuniform_traces_equal():
    """THE observable design win of #571 option 4: under
    amplitude_kind='current' the uniform and NU builders agree with NO
    factor table, open boundary (subpixel off — #582 is out of scope).

    Float tolerance, measured on this fixture: scale deviation 3e-8
    (gate rel 1e-5), residual 3.2e-5 of peak (gate 3e-4), 1-corr 5e-10
    (gate 1e-7)."""
    uni = _trace("cpml", False, "current")
    nu = _trace("cpml", True, "current")
    assert np.abs(uni).max() > 0.0
    assert np.corrcoef(uni, nu)[0, 1] > 1 - 1e-7
    scale = float(np.dot(uni, nu) / np.dot(uni, uni))
    assert scale == pytest.approx(1.0, rel=1e-5), (
        f"amplitude_kind='current' promises EQUAL traces on both builders; "
        f"measured cross-path scale {scale:.6g}")
    resid = float(np.abs(nu - uni).max() / np.abs(nu).max())
    assert resid < 3e-4


def test_field_kind_makes_uniform_and_nonuniform_traces_equal_pec():
    """Mirror witness for 'field' on the boundary where it is native on
    one side (uniform pec) and converted on the other (NU dV/Cb).
    Measured: scale deviation 1.4e-6 (gate rel 1e-5), residual 1.9e-4
    (gate 6e-4)."""
    uni = _trace("pec", False, "field")
    nu = _trace("pec", True, "field")
    assert np.abs(uni).max() > 0.0
    scale = float(np.dot(uni, nu) / np.dot(uni, uni))
    assert scale == pytest.approx(1.0, rel=1e-5)
    resid = float(np.abs(nu - uni).max() / np.abs(nu).max())
    assert resid < 6e-4


# ---------------------------------------------------------------------------
# (d) DeprecationWarning contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,nonuniform", [
    pytest.param("pec", False, id="closed-uniform"),
    pytest.param("cpml", False, id="open-uniform"),
    pytest.param("cpml", True, id="nonuniform"),
])
def test_kind_none_warns_once_per_sim_naming_the_concrete_meaning(
        boundary, nonuniform):
    sim = _sim(boundary, nonuniform)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim.add_source(_SRC, "ez")
        sim.add_source(_PRB, "ez")  # second bare call: NO second warning
    dep = [w for w in rec if issubclass(w.category, DeprecationWarning)
           and "amplitude_kind" in str(w.message)]
    assert len(dep) == 1, (
        f"expected exactly one amplitude_kind DeprecationWarning per "
        f"Simulation, got {len(dep)}")
    assert str(dep[0].message) == (
        "add_source(..., amplitude_kind=None) now means 'current' "
        "(E += Cb*I/dV, I is a current moment in A·m) on every path. "
        "Pass amplitude_kind explicitly to silence this warning."
    )
    # a NEW Simulation warns again (per-sim, not per-process)
    sim2 = _sim(boundary, nonuniform)
    with pytest.warns(DeprecationWarning, match="amplitude_kind"):
        sim2.add_source(_SRC, "ez")


def test_explicit_kind_is_warning_free():
    sim = _sim("cpml", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sim.add_source(_SRC, "ez", amplitude_kind="current")
        sim.add_source(_PRB, "ez", amplitude_kind="field")


# ---------------------------------------------------------------------------
# (e) AD witness: grad flows through explicit kinds on traced materials
# ---------------------------------------------------------------------------

def test_grad_flows_through_current_kind_on_traced_materials():
    """forward(eps_override=tracer) with amplitude_kind='current' (and the
    'field' conversion leg, whose scale 1/Cb IS a tracer): no
    ConcretizationTypeError, finite nonzero gradient."""
    for kind in ("current", "field"):
        sim = _sim("cpml", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            sim.add_source(_SRC, "ez",
                           waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                           amplitude_kind=kind)
        sim.add_probe(_PRB, "ez")
        grid = sim._build_grid()
        base = jnp.ones(grid.shape, dtype=jnp.float32)

        def loss(p):
            r = sim.forward(eps_override=base * p, n_steps=150,
                            skip_preflight=True)
            return jnp.sum(r.time_series ** 2)

        g = jax.grad(loss)(jnp.float32(2.0))
        g = float(g)
        assert np.isfinite(g), f"kind={kind}: non-finite grad {g}"
        assert g != 0.0, f"kind={kind}: gradient is exactly zero"
