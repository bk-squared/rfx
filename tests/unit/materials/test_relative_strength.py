"""Declaration, non-field readers, and coefficient-path admission."""
import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPole, RelativeStrength, Simulation, drude_pole, lorentz_pole
from rfx.artifacts import build_scene_artifact
from rfx.core.yee import EPS_0
from rfx.errors import RfxUnsupported
from rfx.interop._materials import material_from_dict, material_to_dict
from rfx.materials.lorentz import pole_update_coeffs


STABILITY_N, STABILITY_DX = 12, 1e-3
UNSTABLE = "unstable at this model's time step"


def _stability_step():
    return float(Simulation(freq_max=10e9, domain=(STABILITY_N * STABILITY_DX,) * 3,
                            dx=STABILITY_DX, boundary="pec")._build_grid().dt)


def _stability_sim(kappa, graded):
    options = {"dz_profile": np.full(STABILITY_N, STABILITY_DX)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(STABILITY_N * STABILITY_DX,) * 3,
                     dx=STABILITY_DX, boundary="pec", **options)
    sim.add_material("m", eps_r=1., lorentz_poles=[LorentzPole(0., 0., kappa)])
    sim.add(Box((0, 0, 0), (STABILITY_N * STABILITY_DX,) * 3), material="m")
    sim.add_source((.005, .006, .006), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    sim.add_probe((.007, .005, .006), "ez")
    return sim


# omega_p dt = 0.45 is above the eps_inf = 1 limit (about 0.28), 0.2 below (test_pole_stability_admission).
# The unstable "rs" row puts the size in the reference (value 1e-6), so a reader of value alone sees a stable pole.
# The stable "rs" row (value 1e6) is a control: its value alone, read as kappa, is also below the limit.
STABILITY_CASES = {
    ("unstable", "si"): (0.45, None, True),
    ("unstable", "rs"): (0.45, 1e-6, True),
    ("stable", "si"): (0.2, None, False),
    ("stable", "rs"): (0.2, 1e6, False),
}


def _stability_declare(wdt, value, traced=None):
    kappa = (wdt / _stability_step()) ** 2
    if value is None:
        return kappa if traced is None else traced, kappa
    return RelativeStrength(kappa / value, value if traced is None else traced), value


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("route", ["run", "jit_grad"])
@pytest.mark.parametrize("case", list(STABILITY_CASES), ids=lambda c: "-".join(c))
def test_stability_reads_si(case, route, lane):
    wdt, value, expect_refusal = STABILITY_CASES[case]
    graded = lane == "graded"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if route == "run":
                kappa, _ = _stability_declare(wdt, value)
                ts = np.asarray(_stability_sim(kappa, graded).run(
                    n_steps=5, compute_s_params=False, skip_preflight=True).time_series)
            else:
                _, primal = _stability_declare(wdt, value)

                def loss(v):
                    kappa, _ = _stability_declare(wdt, value, traced=v)
                    return jnp.sum(_stability_sim(kappa, graded).forward(
                        n_steps=5, skip_preflight=True).time_series ** 2)

                ts = np.asarray(jax.block_until_ready(jax.jit(jax.grad(loss))(jnp.float32(primal))))
                jax.effects_barrier()
        outcome = f"ran finite={bool(np.all(np.isfinite(ts)))}"
    except Exception as exc:  # The in-program route surfaces the ValueError through XLA.
        outcome = f"refused {type(exc).__name__}: {UNSTABLE in str(exc)}"
        if UNSTABLE not in str(exc):
            outcome += " msg=" + str(exc)[:300]
    print(f"stability {case} {route} {lane}: {outcome}")
    if expect_refusal:
        assert outcome.startswith("refused") and outcome.endswith("True"), outcome
    else:
        assert outcome.startswith("ran finite=True"), outcome


@pytest.mark.parametrize("reference", [0., -1., float("inf"), float("nan"), 1, jnp.float32(1.)])
def test_invalid_reference(reference):
    with pytest.raises(ValueError, match="RelativeStrength reference"):
        RelativeStrength(reference, 1.)


def test_traced_reference_refused():
    with pytest.raises(ValueError, match="RelativeStrength reference"):
        jax.jit(lambda r: RelativeStrength(r, 1.).si)(1.)


def test_pytree_value_hashing_and_callable():
    from rfx.geometry._pole_keying import _pole_key
    a = RelativeStrength(np.float64(4.), np.array([1., .5]))
    b = RelativeStrength(4., jnp.array([1., .5]))
    assert a == b and hash(a) == hash(b)
    assert a != RelativeStrength(4., np.array([.5, 1.]))
    assert _pole_key(LorentzPole(jnp.float32(2.), jnp.float32(1.), a)) == _pole_key(LorentzPole(2., 1., b))
    leaves, tree = jax.tree.flatten(a)
    assert len(leaves) == 1
    assert jax.tree.unflatten(tree, leaves) == a
    assert RelativeStrength(4., lambda x, y, z: x + y + z).si(1., 2., 3.) == 24.
    def traced(q):
        x, y = RelativeStrength(4., q), RelativeStrength(4., q)
        assert x == x and x != y
        assert len({x, y}) == 2
        return x.si
    assert jax.jit(traced)(2.) == 8.


def test_helpers_and_si_coefficients():
    assert drude_pole(2., 4.) == LorentzPole(0., 2., 4.)
    assert drude_pole(2., 4., density=.5).kappa == RelativeStrength(4., .5)
    assert lorentz_pole(2., 3., 4.) == LorentzPole(3., 4., 18.)
    assert lorentz_pole(2., 3., 4., relative=True).kappa == RelativeStrength(9., 2.)
    with pytest.raises(ValueError, match="drude_pole"):
        lorentz_pole(2., 0., 4., relative=True)
    for k in (3e20, jnp.float32(3e20)):
        dt, d, w = np.float64(1e-12), 2e9, 3e9
        expected = ((2 - w**2 * dt**2) / (1 + d * dt), -(1 - d * dt) / (1 + d * dt),
                    EPS_0 * k * dt**2 / (1 + d * dt))
        for actual, old in zip(pole_update_coeffs(w, d, k, dt), expected):
            assert np.asarray(actual).tobytes() == np.asarray(old).tobytes()


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_relative_value(value):
    with pytest.raises(ValueError, match="RelativeStrength value"):
        pole_update_coeffs(0., 1., RelativeStrength(1., value), .1)


def test_scalar_refuses_callable_relative_strength():
    with pytest.raises(RfxUnsupported, match="RelativeStrength"):
        pole_update_coeffs(0., 1., RelativeStrength(1., lambda x, y, z: x), .1)


def test_non_field_readers_round_trip():
    from rfx.material_fit import eval_lorentz
    from rfx.differentiable_material_fit import _poles_to_params
    from rfx.interop.emitters.openems import _plan_materials
    from rfx.interop._errors import UnsupportedDesignFeature
    sim = Simulation(freq_max=6e9, domain=(.02, .01, .01), dx=.002, boundary="pec")
    pole = LorentzPole(2e10, 1e9, RelativeStrength(8e20, .5))
    sim.add_material("relative", eps_r=1., lorentz_poles=[pole])
    material = sim._materials["relative"]
    exported = material_to_dict(material)
    assert exported["lorentz_poles"][0]["kappa"] == pole.kappa.si
    restored_si = material_from_dict(exported)
    freqs = np.array([1e9, 2e9])
    np.testing.assert_array_equal(eval_lorentz(freqs, 1., [pole]), eval_lorentz(freqs, 1., restored_si.lorentz_poles))
    artifact = json.loads(json.dumps(build_scene_artifact(sim)))
    restored = material_from_dict(artifact["materials"]["relative"]["definition"])
    assert isinstance(restored.lorentz_poles[0].kappa, RelativeStrength)
    assert restored.lorentz_poles[0].kappa == pole.kappa
    with pytest.raises(RfxUnsupported, match="RelativeStrength"):
        _poles_to_params(1., [], [pole])
    with pytest.raises(UnsupportedDesignFeature, match="RelativeStrength"):
        _plan_materials({"materials": {"relative": exported}, "material_library": {}})


def test_relative_array_artifact_round_trip():
    sim = Simulation(freq_max=6e9, domain=(.02, .01, .01), dx=.002, boundary="pec")
    strength = RelativeStrength(8e20, np.array([[1., .5], [.25, 0.]]))
    sim.add_material("array", eps_r=1., lorentz_poles=[LorentzPole(2e10, 1e9, strength)])
    artifact = json.loads(json.dumps(build_scene_artifact(sim)))
    restored = material_from_dict(artifact["materials"]["array"]["definition"])
    assert restored.lorentz_poles[0].kappa == strength


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("entry", ["forward", "run"])
def test_distributed_relative_strength_uses_core(lane, entry, monkeypatch):
    from rfx.materials import lorentz
    if len(jax.devices("cpu")) < 2:
        pytest.skip("requires two CPU devices")
    calls = []
    original = lorentz.pole_update_coeffs
    def counted(w, d, k, *args, **kwargs):
        if not (w == 1. and d == 1. and k == 1.):  # Vacuum halo coefficients.
            calls.append(isinstance(k, RelativeStrength))
        return original(w, d, k, *args, **kwargs)
    monkeypatch.setattr(lorentz, "pole_update_coeffs", counted)
    results = []
    for relative in (True, False):
        options = {"dx_profile": np.r_[.002, np.linspace(.0018, .0022, 14), .002]} if lane == "graded" else {}
        sim = Simulation(freq_max=6e9, domain=(.032, .016, .016), dx=.002, boundary="pec", **options)
        strength = RelativeStrength(4e20, .5) if relative else 2e20
        sim.add_material("pole", eps_r=1., lorentz_poles=[LorentzPole(0., 2e9, strength)])
        sim.add(Box((.010, .004, .004), (.020, .012, .012)), material="pole")
        sim.add_source((.006, .008, .008), "ez", amplitude_kind="field")
        sim.add_probe((.008, .008, .008), "ez")
        options = {"distributed": True, "checkpoint": False} if entry == "forward" else {}
        result = getattr(sim, entry)(n_steps=12, devices=jax.devices("cpu")[:2],
                                     skip_preflight=True, **options)
        results.append(np.asarray(result.time_series))
        if relative:
            assert calls and all(calls)
            calls.clear()
    assert calls and not any(calls)
    assert np.any(results[0] != 0)
    np.testing.assert_allclose(*results, rtol=1e-6, atol=0.)
