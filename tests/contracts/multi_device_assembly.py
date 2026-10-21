"""S1–S4 from the step-A measured multi-device assembly ledger."""

import jax.numpy as jnp
import numpy as np

from rfx import GaussianPulse, Simulation

BINS = np.array([3e9, 5e9, 7e9], np.float32)
PROFILE = np.array(
    [1e-3] * 7
    + [0.9e-3, 1.1e-3, 0.95e-3, 1.05e-3, 0.9e-3, 1.1e-3, 1.05e-3, 0.95e-3]
    + [1e-3] * 9
)
N_STEPS = 60


def make_call(lane, scene, devices, *, forward_override=True):
    graded = lane.startswith("graded")
    sim = Simulation(
        freq_max=10e9,
        domain=(24e-3, 12e-3, 12e-3),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=6,
        **({"dx_profile": PROFILE.copy()} if graded else {}),
    )
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    if scene == "S1":
        sim.add_source(
            (6e-3, 6e-3, 6e-3), "ez", waveform=pulse, amplitude_kind="current"
        )
    else:
        sim.add_port(
            (6e-3, 6e-3, 6e-3),
            "ez",
            impedance=50.0,
            waveform=pulse,
            **({"extent": 2e-3, "direction": "+x"} if scene == "S3" else {}),
        )
    sim.add_probe((13e-3, 6e-3, 6e-3), "ez")
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    common = dict(n_steps=N_STEPS, devices=devices, skip_preflight=True)
    if lane.endswith("forward"):
        if forward_override:

            def call(e):
                return sim.forward(
                    **common,
                    distributed=True,
                    checkpoint=False,
                    eps_override=e,
                    **({"port_s11_freqs": BINS} if scene in ("S2", "S3") else {}),
                )
        else:

            def call():
                return sim.forward(
                    **common,
                    distributed=True,
                    checkpoint=False,
                    **({"port_s11_freqs": BINS} if scene in ("S2", "S3") else {}),
                )
    else:

        def call():
            return sim.run(
                **common,
                **(
                    {"compute_s_params": True, "s_param_freqs": BINS}
                    if scene in ("S2", "S3")
                    else {}
                ),
            )

    return call, eps, tuple(int(n) for n in grid.shape)


def outputs(result):
    arrays = {}
    for name in ("time_series", "s_params"):
        value = getattr(result, name, None)
        if value is not None:
            arrays[name] = value
    state = getattr(result, "state", None)
    if state is not None:
        for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
            value = getattr(state, name, None)
            if value is not None:
                arrays[name] = value
    return arrays
