"""
rfx WP 2-A — real multi-device (multi-GPU) correctness witness.

Run on ANY box with >= 2 JAX devices (2+ GPUs, or a free Kaggle "GPU T4 x2"
notebook with Internet ON + `pip install git+https://github.com/bk-squared/rfx.git`).
It:

  1. prints the JAX backend + device list and ASSERTS jax.device_count() >= 2
     (the whole point of 2-A: rfx's distributed runner had never run on real
     multiple devices — the lab GPU suite runs on 1 GPU, where these checks skip),
  2. runs three small rfx sims (PEC, CPML, lumped-port) BOTH single-device and
     sharded across all available devices, and checks the field time-series match
     within the same tolerances the committed rfx test suite uses,
  3. prints a PASS/FAIL verdict per case + an overall verdict line.

Standalone form of the checks in rfx's tests/test_distributed.py
(TestDistributedRunner / TestDistributedCPML / TestDistributedLumpedPort), which
are device-count-adaptive: they EXECUTE (not skip) as soon as >= 2 real devices
are present, so CI on a single-GPU pod cannot exercise them (issue #162).

FIRST-EVER REAL MULTI-DEVICE RESULT (2026-07-09, Kaggle 2x NVIDIA T4, jax 0.7.2,
backend gpu, jax.device_count()==2) — distributed == single-device:
    [PASS] PEC          multi-vs-single rel_err = 1.03e-06  (gate 1e-04)
    [PASS] CPML         multi-vs-single rel_err = 1.10e-07  (gate 1e-03)
    [PASS] lumped-port  multi-vs-single rel_err = 2.89e-07  (gate 1e-02)
    OVERALL: PASS -- distributed == single on real multi-GPU
This is the first hardware confirmation that the distributed FDTD runner is
field-identical to the single-device path (~1e-6/1e-7) on genuine multiple GPUs.
The CPML case's tiny domain trips a source/probe-near-CPML advisory (the witness
checks distributed==single IDENTITY, not absolute physics, so it is harmless).
"""
import numpy as np
import jax

from rfx import Simulation, GaussianPulse


def _rel_err(ts_single, ts_multi):
    ts_single = np.asarray(ts_single)
    ts_multi = np.asarray(ts_multi)
    assert ts_single.shape == ts_multi.shape, (
        f"shape mismatch single={ts_single.shape} multi={ts_multi.shape}"
    )
    peak = np.max(np.abs(ts_single)) + 1e-30
    return float(np.max(np.abs(ts_single - ts_multi)) / peak)


def _pec_sim():
    sim = Simulation(freq_max=3e9, domain=(0.05, 0.02, 0.02), boundary="pec")
    sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
    sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")
    return sim, 100


def _cpml_sim():
    sim = Simulation(freq_max=3e9, domain=(0.13, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(0.065, 0.02, 0.02), component="ez",
                   waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9))
    sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
    return sim, 200


def _lumped_sim():
    sim = Simulation(freq_max=5e9, domain=(0.08, 0.024, 0.024), boundary="pec")
    sim.add_port(position=(0.04, 0.012, 0.012), component="ez", impedance=50,
                 waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9))
    sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")
    return sim, 100


# (name, sim-builder, tolerance) — same gates as the committed rfx tests.
CASES = [
    ("PEC",          _pec_sim,    1e-4),
    ("CPML",         _cpml_sim,   1e-3),
    ("lumped-port",  _lumped_sim, 1e-2),
]


def main():
    print("=" * 64)
    print("rfx WP 2-A — real multi-device correctness witness")
    print("=" * 64)
    print("jax", jax.__version__, "| backend:", jax.default_backend())
    devices = jax.devices()
    print(f"jax.device_count() = {len(devices)}")
    for d in devices:
        print("   ", d)

    if len(devices) < 2:
        print("\nRESULT: SKIPPED — need >= 2 devices for a real multi-device run.")
        print("On Kaggle set Accelerator = 'GPU T4 x2' (Settings panel), then re-run.")
        return

    import rfx
    print("\nrfx", getattr(rfx, "__version__", "?"), "| running",
          len(devices), "devices\n")

    all_pass = True
    for name, build, tol in CASES:
        sim, n_steps = build()
        ts_single = sim.run(n_steps=n_steps).time_series
        ts_multi = sim.run(n_steps=n_steps, devices=devices).time_series
        err = _rel_err(ts_single, ts_multi)
        ok = err < tol
        all_pass = all_pass and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} "
              f"multi-vs-single rel_err = {err:.2e}  (gate {tol:.0e})")

    print()
    print("OVERALL:", "PASS — distributed == single on real multi-GPU"
          if all_pass else "FAIL — see cases above")
    print(f"(devices={len(devices)}, backend={jax.default_backend()})")


if __name__ == "__main__":
    main()
