"""Built-in point excitations leave no static field after the pulse (issue 1668).

Setup and metric: records/20261109-test-convert/measurements/static_residual_probe2.py,
CPML 16, record 1x. A13 excludes TFSF, Floquet and waveguide excitations: their
measured tails still depend on the record length or absorber in this box.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from rfx import Simulation

# records/20261109-test-convert/DESIGN.md, A13 (2026-11-09): the measured
# corrected floor 8.49e-8, times 1.5 and rounded up at quantum=1e7. A11
# requires a fixed bar so a broken waveform cannot raise its own threshold.
R_MAX = 2e-7


@pytest.mark.parametrize("row", [
    "point_default",
    "polarized_default",
    pytest.param(
        "polarized_circular",
        marks=pytest.mark.xfail(
            strict=True,
            reason="issue 1668: circular polarized source leaves a static field",
        ),
    ),
])
def test_builtin_source_leaves_no_static_field(row):
    f0, dx = 15e9, 0.001
    domain = (0.016, 0.016, 0.016)
    source, probe = (0.008, 0.008, 0.008), (0.009, 0.008, 0.008)
    sim = Simulation(freq_max=2 * f0, domain=domain, dx=dx,
                     boundary="cpml", cpml_layers=16)
    if row == "point_default":
        sim.add_source(source, "ez", amplitude_kind="field")
    elif row == "polarized_default":
        sim.add_polarized_source(source, amplitude_kind="field")
    else:
        sim.add_polarized_source(source, polarization="circular",
                                 amplitude_kind="field")
    for component in ("ex", "ey", "ez"):
        sim.add_probe(probe, component)

    grid = sim.freeze_mesh()
    # 16 interior cells plus 16 CPML cells on each side and the end node.
    assert grid.shape == (49, 49, 49)
    assert grid.position_to_index(source) == (24, 24, 24)
    assert grid.position_to_index(probe) == (25, 24, 24)
    # Closed forms, independent of the waveform implementation. Circular
    # Ex uses cutoff=5 and ends later than its inline Ey (cutoff=3).
    tau = 1.0 / (f0 * 0.8 * math.pi)
    t0 = (5 if row == "polarized_circular" else 3) * tau
    end = t0 + 6 * tau + 2 * max(domain) / 299792458.0
    steps = math.ceil(end / grid.dt) + 1  # Last sample is at (steps-1)*dt.
    assert steps == (211 if row == "polarized_circular" else 183)
    result = sim.run(n_steps=steps, compute_s_params=False)
    fields = np.asarray(result.time_series)

    assert fields.shape == (steps, 3)
    assert np.isfinite(fields).all(), f"{row}: nonfinite field record"
    # Same Yee probe indices, no interpolation; float64 norm as in the record.
    magnitude = np.linalg.norm(fields.astype(np.float64), axis=1)
    peak = float(magnitude.max())
    late = float(magnitude[math.floor(0.9 * steps):].max())
    assert peak > 0, f"{row}: zero peak field"
    residual = late / peak
    print(f"{row}: steps={steps} mesh={grid.shape} peak={peak:.12e} "
          f"late_max={late:.12e} r={residual:.12e} bar={R_MAX:.12e}",
          file=sys.stderr)
    assert residual <= R_MAX, (
        f"{row}: static residual r={residual:.12e} > R_MAX={R_MAX:.12e}; "
        f"steps={steps}, mesh={grid.shape}, peak={peak:.12e}, late_max={late:.12e}"
    )
