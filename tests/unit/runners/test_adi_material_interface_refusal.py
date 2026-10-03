"""solver='adi' refuses a dielectric or conductivity interface (#1373, 2.0).

The ADI update gives each E component its own cell's eps_r and sigma; every
Yee lane gives each E edge the mean of its four cells. A dielectric face
therefore sits half a cell from where the Yee lanes put it: a slab-loaded PEC
cavity's first resonance came out 2.2 % above the uniform Yee lane at 12 cells
per loaded wavelength (-0.2 % with no slab). Until 2.1 the ADI lane refuses an
eps_r or sigma that varies over the grid, at four places: lane admission on
the declaration, the realized arrays inside ``_run_adi_from_materials`` (a
concrete override), a traced array there (it cannot be read), and the public
kernels ``run_adi_2d`` / ``run_adi_3d`` (eps_r only: their sigma carries the
absorber grading). A homogeneous fill, declared or as a scalar override,
still runs. preflight() reports the declared interface as an error.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

import rfx.adi
from rfx import Box, GaussianPulse, Simulation
from rfx.adi import ADI_INTERFACE_MESSAGE

WAVEFORM = GaussianPulse(f0=5e9, bandwidth=0.8)
ADI_TAG = "adi_material_interface_unsupported"


def mm(*v):
    return tuple(x * 1e-3 for x in v)


def _no_kernel(*_args, **_kwargs):
    raise AssertionError("the ADI kernel was reached")


# ---------------------------------------------------------------- ADI

def _adi(mode="3d", boundary="pec"):
    thick = 12 if mode == "3d" else 1
    sim = Simulation(freq_max=10e9, domain=mm(12, 12, thick), dx=1e-3, boundary=boundary,
                     solver="adi", mode=mode, adi_cfl_factor=1,
                     **({"cpml_layers": 4} if boundary == "cpml" else {}))
    z = 6e-3 if mode == "3d" else 0.0
    sim.add_source((4e-3, 6e-3, z), "ez", waveform=WAVEFORM, amplitude_kind="field")
    sim.add_probe((8e-3, 6e-3, z), "ez")
    return sim


def _slab(sim, **material):
    """A dielectric slab across part of the box: two material faces."""
    sim.add_material("slab", **material)
    hi_z = 12e-3 if sim._mode == "3d" else 1e-3
    sim.add(Box((5e-3, 0.0, 0.0), (7e-3, 12e-3, hi_z)), material="slab")
    return sim


def _fill(sim, **material):
    """The same material over the whole domain: no interface."""
    sim.add_material("fill", **material)
    hi_z = 12e-3 if sim._mode == "3d" else 1e-3
    sim.add(Box((0.0, 0.0, 0.0), (12e-3, 12e-3, hi_z)), material="fill")
    return sim


def _call(sim, entry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry == "run":
            return sim.run(n_steps=8, skip_preflight=True)
        return sim.forward(n_steps=8, skip_preflight=True, checkpoint=False)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("material", [{"eps_r": 4.0}, {"eps_r": 1.0, "sigma": 0.2}],
                         ids=["eps", "sigma"])
def test_adi_refuses_a_declared_interface_before_the_kernel(entry, mode, material):
    """Lane admission refuses it on the declaration, naming the measurement."""
    sim = _slab(_adi(mode), **material)
    kernel = "run_adi_3d" if mode == "3d" else "run_adi_2d"
    with patch.object(rfx.adi, kernel, _no_kernel), \
            pytest.raises(NotImplementedError) as exc:
        _call(sim, entry)
    message = str(exc.value)
    lane = "ADI run()" if entry == "run" else "ADI forward()"
    # Lane admission's refusal (not the later array check): its header.
    assert message.startswith(f"The {lane} lane would solve this Simulation"), message[:300]
    assert ADI_INTERFACE_MESSAGE in message


def _substrate(sim, cells, **material):
    """A layered substrate on the floor, ``cells`` cells thick: the board
    stack-up of a patch or a microstrip. In 3-D it is layered in z; the 2-D
    TMz box is one cell in z, so there it is layered in y. One cell thick it
    occupies only the first cell layer at the floor."""
    sim.add_material("substrate", **material)
    t = cells * 1e-3
    if sim._mode == "3d":
        sim.add(Box((0.0, 0.0, 0.0), (12e-3, 12e-3, t)), material="substrate")
    else:
        sim.add(Box((0.0, 0.0, 0.0), (12e-3, t, 1e-3)), material="substrate")
    return sim


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("cells", [1, 3], ids=["first-layer", "three-layers"])
@pytest.mark.parametrize("material", [{"eps_r": 4.0}, {"eps_r": 1.0, "sigma": 0.2}],
                         ids=["eps", "sigma"])
def test_adi_refuses_a_layered_substrate(entry, mode, cells, material):
    """The interface is normal to z (y in 2-D), not x; a one-cell substrate
    sits in the first cell layer only."""
    sim = _substrate(_adi(mode), cells, **material)
    kernel = "run_adi_3d" if mode == "3d" else "run_adi_2d"
    with patch.object(rfx.adi, kernel, _no_kernel), \
            pytest.raises(NotImplementedError) as exc:
        _call(sim, entry)
    assert ADI_INTERFACE_MESSAGE in str(exc.value)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_adi_carries_a_homogeneous_fill(entry, mode):
    """A whole-domain fill has no interface: it runs, and it is not vacuum."""
    filled = _call(_fill(_adi(mode), eps_r=2.2, sigma=0.05), entry)
    vacuum = _call(_adi(mode), entry)
    a, b = np.asarray(filled.time_series), np.asarray(vacuum.time_series)
    assert np.all(np.isfinite(a))
    assert float(np.max(np.abs(a - b))) > 1e-3 * float(np.max(np.abs(b)))


@pytest.mark.parametrize("override", ["eps_override", "sigma_override"])
def test_adi_refuses_a_concrete_override_that_makes_an_interface(override):
    """The declaration is vacuum, so admission passes; the arrays the lane
    would step are read after it. sigma is not checked by the kernel (its
    sigma carries the absorber), so this is the only check that sees it."""
    sim = _adi("3d")
    shape = sim._build_grid().shape
    base = 1.0 if override == "eps_override" else 0.0
    arr = jnp.full(shape, base, jnp.float32).at[5:7].set(base + 2.0)
    with patch.object(rfx.adi, "run_adi_3d", _no_kernel), \
            pytest.raises(NotImplementedError) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.forward(n_steps=8, skip_preflight=True, checkpoint=False, **{override: arr})
    assert str(exc.value) == ADI_INTERFACE_MESSAGE


def test_adi_kernels_refuse_a_concrete_eps_interface():
    """The public kernels read eps_r per cell too; a concrete one that varies
    is refused, a uniform one runs."""
    n = 8
    zeros3 = jnp.zeros((n, n, n), jnp.float32)
    eps3 = jnp.ones((n, n, n), jnp.float32)
    zeros2 = jnp.zeros((n, n), jnp.float32)
    eps2 = jnp.ones((n, n), jnp.float32)
    dt = 1e-12
    with pytest.raises(NotImplementedError, match=ADI_TAG):
        rfx.adi.run_adi_3d(*(zeros3,) * 6, eps3.at[3:5].set(4.0), zeros3,
                           dt, 1e-3, 1e-3, 1e-3, 2)
    with pytest.raises(NotImplementedError, match=ADI_TAG):
        rfx.adi.run_adi_2d(*(zeros2,) * 3, eps2.at[3:5].set(4.0), zeros2,
                           dt, 1e-3, 1e-3, 2)
    # A layer normal to z / y, and one in the first cell layer only.
    for layered in (eps3.at[:, :, 0].set(4.0), eps3.at[:, :, :3].set(4.0)):
        with pytest.raises(NotImplementedError, match=ADI_TAG):
            rfx.adi.run_adi_3d(*(zeros3,) * 6, layered, zeros3, dt, 1e-3, 1e-3, 1e-3, 2)
    with pytest.raises(NotImplementedError, match=ADI_TAG):
        rfx.adi.run_adi_2d(*(zeros2,) * 3, eps2.at[:, 0].set(4.0), zeros2,
                           dt, 1e-3, 1e-3, 2)
    out3 = rfx.adi.run_adi_3d(*(zeros3,) * 6, eps3 * 2.0, zeros3, dt, 1e-3, 1e-3, 1e-3, 2)
    out2 = rfx.adi.run_adi_2d(*(zeros2,) * 3, eps2 * 2.0, zeros2, dt, 1e-3, 1e-3, 2)
    assert np.all(np.isfinite(np.asarray(out3[2]))) and np.all(np.isfinite(np.asarray(out2[0])))


# ------------------------------------------------- traced overrides on ADI

@pytest.mark.parametrize("override", ["eps_override", "sigma_override"])
def test_adi_refuses_a_traced_array_override(override):
    """A traced array cannot be read before the solve, so whether it holds an
    interface is unknown: refused before the kernel (#1373)."""
    import jax
    from rfx.adi import ADI_TRACED_OVERRIDE_MESSAGE
    sim = _adi("3d")
    shape = sim._build_grid().shape
    base = 1.0 if override == "eps_override" else 0.0

    def loss(arr):
        ts = sim.forward(n_steps=8, skip_preflight=True, checkpoint=False,
                         **{override: arr}).time_series
        return jnp.sum(ts ** 2)

    with patch.object(rfx.adi, "run_adi_3d", _no_kernel), \
            pytest.raises(NotImplementedError) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax.grad(loss)(jnp.full(shape, base, jnp.float32))
    assert str(exc.value) == ADI_TRACED_OVERRIDE_MESSAGE


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_adi_carries_a_traced_scalar_override(mode):
    """A traced 0-d override is a homogeneous fill: it runs, its value is the
    uniform array's, and its derivative is finite and nonzero."""
    import jax
    sim = _adi(mode)
    shape = sim._build_grid().shape

    def energy(eps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = sim.forward(n_steps=8, skip_preflight=True, checkpoint=False,
                             eps_override=eps).time_series
        return jnp.sum(ts ** 2)

    scalar = float(energy(jnp.float32(2.0)))
    array = float(energy(jnp.full(shape, 2.0, jnp.float32)))
    assert scalar == array
    g = float(jax.grad(energy)(jnp.float32(2.0)))
    assert np.isfinite(g) and g != 0.0


def test_preflight_reports_an_adi_interface_as_an_error():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _substrate(_adi("3d"), 1, eps_r=4.0).preflight()
        filled = _fill(_adi("3d"), eps_r=2.2).preflight()
    assert any(ADI_INTERFACE_MESSAGE in str(i) and i.severity == "error"
               for i in report), list(report)
    assert not any(ADI_INTERFACE_MESSAGE in str(i) for i in filled)


def test_a_traced_material_from_geometry_names_add_material():
    """jax.grad with respect to add_material(eps_r=<tracer>) on a filling box
    gives a traced array: refused, naming add_material and the scalar form."""
    import jax

    def energy(e):
        sim = _adi("3d")
        sim.add_material("fill", eps_r=e)
        sim.add(Box((0.0, 0.0, 0.0), (12e-3, 12e-3, 12e-3)), material="fill")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = sim.forward(n_steps=8, skip_preflight=True, checkpoint=False).time_series
        return jnp.sum(ts ** 2)

    with patch.object(rfx.adi, "run_adi_3d", _no_kernel), \
            pytest.raises(NotImplementedError) as exc:
        jax.grad(energy)(jnp.float32(2.0))
    message = str(exc.value)
    assert "from add_material with a traced value" in message, message[:400]
    assert "scalar (0-d) eps_override" in message


def test_the_adi_admission_text_lists_no_material_row_and_no_subgridded_carrier():
    sim = _slab(_adi("3d"), eps_r=4.0)
    with pytest.raises(NotImplementedError) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _call(sim, "run")
    message = str(exc.value)
    assert "a dielectric (eps_r != 1) is not carried" not in message
    assert "subgridded run()" not in message
    assert ADI_INTERFACE_MESSAGE in message
