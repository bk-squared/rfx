"""CPU reproduction and WR-90 control for the power-wave cutoff contract."""
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning

from rfx import Simulation
from rfx.measurement import waveguide_s as ws
from rfx.measurement.waveguide_power import WaveguideCutoffWarning


def scene(multimode):
    freqs = np.linspace(4.5e9, 8e9, 10) if multimode else np.linspace(5.5e9, 12e9, 14)
    dx = 299792458. / freqs[-1] / 20 if multimode else .00127
    domain = ((.12, int(np.ceil(.04/dx))*dx, int(np.ceil(.02/dx))*dx)
              if multimode else (.12, .02286, .01016))
    sim = Simulation(freq_max=float(freqs[-1]), domain=domain, dx=dx,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=10)
    f0 = float(freqs.mean())
    for name, x, direction in [("left", .01, "+x"), ("right", .09, "-x")]:
        sim.add_waveguide_port(x, direction=direction, mode=(1, 0), mode_type="TE",
                              freqs=jnp.asarray(freqs), f0=f0,
                              bandwidth=max(.2, min(.8, (freqs[-1]-freqs[0])/f0)),
                              waveform="modulated_gaussian", n_modes=2 if multimode else 1,
                              name=name, mode_profile="analytic" if multimode else "discrete")
    return sim


def capture_unmasked(monkeypatch):
    original = ws._s_matrix_port_waves
    waves = []

    def capture(cfg, **kw):
        a, b = original(cfg, **kw)
        waves.append((cfg, a, b))
        return a, b

    monkeypatch.setattr(ws, "_s_matrix_port_waves", capture)
    return waves


def unmasked_matrix(waves, channels):
    # The pre-fix assembly is written explicitly here, with no mask helper.
    columns = []
    for drive in range(channels):
        _, a, _ = waves[drive*(channels+1)]
        safe = jnp.where(jnp.abs(a) > 0, a, jnp.ones_like(a))
        columns.append(jnp.stack([waves[drive*(channels+1)+1+i][2] / safe
                                  for i in range(channels)]))
    return jnp.stack(columns, axis=1)


def test_e1_issue_scene_cpu(monkeypatch):
    waves = capture_unmasked(monkeypatch)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scene(True).compute_waveguide_s_matrix(normalize=False, num_periods=40, strict_passivity=True)
    s = np.asarray(result.s_params)
    elapsed = time.perf_counter()-start
    old = np.asarray(unmasked_matrix(waves, 4))
    defined = np.ones(s.shape, dtype=bool)
    defined[[1, 3], :, :8] = False
    defined[:, [1, 3], :8] = False
    assert np.isnan(s[~defined]).all()
    assert np.isfinite(s[defined]).all()
    assert s[defined].tobytes() == old[defined].tobytes()
    power = (np.abs(s[:, :, -2:])**2).sum(axis=0)
    assert power.max() <= 1.005
    assert_no_nonfinite_advisory(caught)
    notices = [w for w in caught if issubclass(w.category, WaveguideCutoffWarning)]
    assert len(notices) == 1
    message = str(notices[0].message)
    for port in (0, 1):
        assert f"port {port}, TE01" in message
    assert message.count("bins [0, 1, 2, 3, 4, 5, 6, 7]") == 2
    print(f"E1 wall_seconds={elapsed:.6f}; column_powers={power.tolist()}; "
          f"unmasked_max={np.abs(old).max():.12g}; NaNs={np.isnan(s).sum()}; unchanged=True")


def test_e2_wr90_cpu(monkeypatch):
    from rfx.sparams import waveguide as calculator

    finalized = []
    finalize = calculator._finalize_sparam_result

    def capture_finalizer(result, **kwargs):
        finalized.append((result, kwargs))
        return finalize(result, **kwargs)

    monkeypatch.setattr(calculator, "_finalize_sparam_result", capture_finalizer)
    waves = capture_unmasked(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scene(False).compute_waveguide_s_matrix(normalize=False, num_periods=40, strict_passivity=True)
    s = np.asarray(result.s_params)
    old = np.asarray(unmasked_matrix(waves, 2))
    assert np.isnan(s[:, :, :3]).all()
    assert np.isfinite(s[:, :, 3:]).all()
    assert s[:, :, 3:].tobytes() == old[:, :, 3:].tobytes()
    assert len([w for w in caught if issubclass(w.category, WaveguideCutoffWarning)]) == 1
    assert_no_nonfinite_advisory(caught)
    assert len(finalized) == 1
    measured, options = finalized[0]
    refused = np.zeros(s.shape, dtype=bool)
    refused[:, :, :3] = True
    np.testing.assert_array_equal(options["refused_positions"], refused)
    # The partial advisory has no returned record. A complete defined column
    # with power 2 * 1.3**2 = 3.38 must still exceed this lane's limit of 3.
    nonpassive = measured._replace(s_params=measured.s_params.at[:, 0, 3].set(1.3))
    with pytest.raises(ValueError, match="passivity_violation: max column power .*frequency index 3"):
        finalize(nonpassive, **options)
    # Refusals must not hide a numerical NaN in an otherwise defined entry.
    diverged = measured._replace(s_params=measured.s_params.at[0, 0, 3].set(jnp.nan))
    with pytest.raises(ValueError, match="nonfinite_sparams"):
        finalize(diverged, **options)
    print("P2 strict=True passed; cutoff_warnings=1; defined column power=3.38 flagged; unrelated NaN flagged")
    print(f"E2 cutoff={waves[0][0].f_cutoff:.12g}; unmasked_below={old[:,:,:3].tolist()}; "
          f"NaNs={np.isnan(s).sum()}; unchanged=True")


def test_e4_wr90_gradient(monkeypatch):
    waves = capture_unmasked(monkeypatch)
    sim = scene(False)
    grid = sim._build_grid()
    # A compact material perturbation, away from both admitted port sections.
    window = jnp.zeros(grid.shape).at[grid.index_of("x", .05):grid.index_of("x", .06), :, :].set(1.)

    def objective(eps, raw=False):
        waves.clear()
        result = sim.compute_waveguide_s_matrix(
            normalize=False, n_steps=128, eps_override=1.+eps*window)
        s = unmasked_matrix(waves, 2) if raw else result.s_params
        return jnp.real(s[:, :, 3:]).sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = jax.grad(lambda e: objective(e, raw=True))(jnp.float32(.2))
        after = jax.grad(objective)(jnp.float32(.2))
    assert np.isfinite(before) and np.isfinite(after)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=0.)
    print(f"E4 before={float(before):.12g}; after={float(after):.12g}; "
          f"relative_error={float(jnp.abs((after-before)/before)):.12g}; n_steps=128")


def assert_no_nonfinite_advisory(caught):
    bad = [str(w.message) for w in caught if issubclass(w.category, AccuracyWarning)
           and any(text in str(w.message).lower()
                   for text in ("unreliable", "nonfinite", "non-finite"))]
    assert not bad
