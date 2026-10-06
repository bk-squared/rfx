"""Shared port loads must precede drives on both sides of a device split."""

import os
from pathlib import Path
import subprocess
import sys


def test_shared_edge_ports_match_one_device():
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=2")
    env["PYTHONPATH"] = (str(Path(__file__).resolve().parents[3]) + os.pathsep
                         + env.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def _compare_shared_ports():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from rfx import Box, Simulation

    devices = jax.devices("cpu")
    assert len(devices) == 2
    for wire in (False, True):
        sim = Simulation(freq_max=8e9, domain=(.008, .008, .008), dx=.001,
                         boundary="pec", cpml_layers=0)
        sim.add_material("slab", eps_r=4.)
        sim.add(Box((.004, 0., 0.), (.008, .008, .008)), material="slab")
        for impedance in (50., 75.):
            sim.add_port((.004, .004, .004), "ez", impedance=impedance,
                         waveform=jnp.ones_like, **({"extent": .002} if wire else {}))
        sim.add_probe((.004, .004, .004), "ez")
        kw = dict(n_steps=12, skip_preflight=True, compute_s_params=False)
        reference = sim.run(**kw)
        actual = sim.run(devices=devices, **kw)
        # The comparison means nothing if devices= fell back to one device.
        assert len(actual.state.ez.devices()) == 2, actual.state.ez.devices()
        pairs = [(reference.time_series, actual.time_series)]
        pairs += [(getattr(reference.state, c), getattr(actual.state, c))
                  for c in ("ex", "ey", "ez", "hx", "hy", "hz")]
        errors = []
        for expected, observed in pairs:
            expected, observed = np.asarray(expected), np.asarray(observed)
            peak = np.max(np.abs(expected))
            error = np.max(np.abs(observed - expected))
            assert error <= 1e-4 * peak, (wire, error, peak)
            errors.append(float(error / peak) if peak else 0.)
        print(f"wire={wire} first_E={actual.time_series[0, 0]} "
              f"max_relative_peak_error={max(errors):.9g}")


if __name__ == "__main__":
    _compare_shared_ports()
