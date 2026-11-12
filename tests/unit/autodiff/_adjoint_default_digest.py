"""Identical subprocess program for main and HEAD default-forward digests."""
import hashlib
import json
from unittest.mock import patch

import jax
import numpy as np
from rfx import Simulation, GaussianPulse
from rfx.stepping import prepared


def digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        a = np.asarray(leaf)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


sim = Simulation(freq_max=16e9, domain=(.024, .020, .016), dx=.002,
                 boundary="cpml", cpml_layers=5, precision="float32")
sim.add_source((.004, .010, .008), "ez", amplitude_kind="field",
               waveform=GaussianPulse(f0=8e9, bandwidth=.8, cutoff=6))
sim.add_probe((.020, .010, .008), "ez")
sim.add_ntff_box((.004, .004, .004), (.020, .016, .012), freqs=[6.4e9, 8.4e9])
sim.add_dft_plane_probe(axis="x", coordinate=.018, component="hy",
                        freqs=[6.4e9, 8.4e9], name="h", region=(8, 10, 8, 10))
seen = []
original = prepared.advance


def advance(product, stepper, *args, **kwargs):
    def capture(*a, **k):
        result = stepper(*a, **{**k, "return_state": True})
        seen.append(result.state)
        return result
    return original(product, capture, *args, **kwargs)


with patch.object(prepared, "advance", advance):
    result = sim.forward(n_steps=128, skip_preflight=True)
assert len(seen) == 1
print(json.dumps(dict(fields=digest(seen[0]), probes=digest(result.time_series),
                      dft=digest(result.dft_planes["h"].accumulator),
                      ntff=digest(result.ntff_data))))
