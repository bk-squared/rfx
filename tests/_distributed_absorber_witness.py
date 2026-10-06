"""Public one-/two-device TE10 reflection with an independent far reference.

The 1-D TEM witness requires a PMC transverse face, refused by the distributed
lane pending B4. This small PEC waveguide launches TE10: the symmetric pair of
oblique plane waves has one x-face reflection coefficient. A far reference
separates incident and echoed fields at the same modal probe. Constant NU
profiles isolate the absorber from mesh scattering, as in the PR1 witness.
"""
from functools import lru_cache, partial
import json
from pathlib import Path
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import C0
from tests._absorber_witness import _reflection_db

DX = 1e-3
DT = DX / (C0 * np.sqrt(3.0)) * 0.99
FREQS = np.array([12, 16, 20, 24, 28]) * 1e9
NY = 20
NEAR, FAR, REFERENCE = 30, 120, 260
STEPS = int(0.9 * (2 * FAR * DX / C0) / DT)


def pulse(t, scale=1.0):
    arg = (t - 60e-12) / 12e-12
    return scale * -arg * jnp.exp(-arg**2)


@lru_cache(maxsize=None)
def trace(nonuniform, distributed, interior, source):
    kwargs = dict(dz_profile=np.full(2, DX), dt=DT) if nonuniform else {}
    sim = Simulation(
        freq_max=30e9, domain=(interior * DX, NY * DX, 2 * DX), dx=DX,
        cpml_layers=16,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=16),
            y="pec", z="pec"), **kwargs)
    for y in range(1, NY):
        for z in range(3):
            sim.add_source((source * DX, y * DX, z * DX), component="ez",
                           waveform=partial(pulse, scale=np.sin(np.pi * y / NY)), amplitude_kind="field")
    sim.add_probe((source * DX, NY // 2 * DX, DX), component="ez")
    devices = jax.devices("cpu")[:2] if distributed else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=STEPS, devices=devices, skip_preflight=True)
    samples = np.asarray(result.time_series[:, 0])
    assert samples.dtype == np.float32, "reflection floor is calibrated in float32"
    return samples


def measure():
    assert len(jax.devices("cpu")) == 2, "set two host devices before importing JAX"
    returns = {}
    for nonuniform in (False, True):
        for distributed in (False, True):
            lane = ("graded" if nonuniform else "uniform") + ("_two" if distributed else "_one")
            reference = trace(nonuniform, distributed, 2 * REFERENCE, REFERENCE)
            assert np.max(np.abs(reference)) > 0.1, "vacuous incident plane wave"
            echoes = [trace(nonuniform, distributed, NEAR + FAR, source) for source in (NEAR, FAR)]
            if distributed:
                for source, samples in zip((NEAR, FAR), echoes):
                    single = trace(nonuniform, False, NEAR + FAR, source)
                    assert not np.array_equal(samples, single), "two-device echo trace silently used one device"
            returns[lane] = np.array([
                _reflection_db(samples, reference, DT, FREQS) for samples in echoes
            ])
    return returns


def main():
    started = time.monotonic()
    result = measure()
    payload = {"reflection_db": {key: value.tolist() for key, value in result.items()},
               "elapsed_seconds": time.monotonic() - started,
               "dt": DT, "steps": STEPS, "frequencies_hz": FREQS.tolist()}
    Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
