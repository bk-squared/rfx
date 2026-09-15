"""A node-aligned short with independently observable ports on both sides.

This is a 40 x 20 mm rectangular guide, not WR-90 (22.86 x 10.16 mm).
The 2 mm plate occupies x=84--86 mm, preserving the coarse mesh's previous
realized body while making the declaration exact. Both 2 mm and 1 mm meshes
divide every dimension and face; a machined 2 mm closing plate is buildable.
The coarse/fine comparison holds the physical source and measurement planes
fixed, instead of letting a ten-cell offset change the measured network.
"""

import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation

DOMAIN = (0.12, 0.04, 0.02)
SHORT_FACES = (0.084, 0.086)
SOURCE_PLANES = (0.010, 0.110)
REFERENCE_DISTANCE = 0.006
PROBE_DISTANCE = 0.020


def build(freqs, dx, cpml):
    freqs = np.asarray(freqs, float)
    f0 = float(freqs.mean())
    bw = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / f0))
    coordinates = (*DOMAIN, *SHORT_FACES, *SOURCE_PLANES,
                   REFERENCE_DISTANCE, PROBE_DISTANCE)
    np.testing.assert_allclose(np.asarray(coordinates) / dx,
                               np.rint(np.asarray(coordinates) / dx), atol=1e-10, rtol=0)
    sim = Simulation(freq_max=float(freqs[-1]), domain=DOMAIN,
                     boundary="cpml", cpml_layers=cpml, dx=dx)
    sim.freeze_mesh()
    sim.add(Box((SHORT_FACES[0], 0, 0),
                (SHORT_FACES[1], DOMAIN[1], DOMAIN[2])), material="pec")
    for source, direction, name in zip(SOURCE_PLANES, ("+x", "-x"), ("left", "right")):
        sim.add_waveguide_port(
            source, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(freqs), f0=f0, bandwidth=bw,
            ref_offset=round(REFERENCE_DISTANCE / dx),
            probe_offset=round(PROBE_DISTANCE / dx),
            waveform="modulated_gaussian", name=name,
        )
    return sim
