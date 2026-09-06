---
title: "Waveguide Ports"
sidebar:
  order: 10
---

rfx supports rectangular waveguide ports with analytical TE/TM mode profiles.
S-parameter claims are intentionally bounded: the documented full-matrix path is
`compute_waveguide_s_matrix(...)` under the rectangular-guide evidence envelope
in `docs/guides/sparameter_support_matrix.md`.

Current evidence level: Recommended for the documented WR-style rectangular-guide
cases. The main gates are in `tests/oracle/test_waveguide_port_validation_battery.py`
(empty-guide max `|S11| < 0.02`, passivity `< 1.02`, PEC-short
`0.99 <= min(|S11|)` and `max(|S11|) < 1.03`) and
`validation/crossval/11_waveguide_port_wr90.py` (analytic Airy/reference-plane
gates, with external references reported as available or skipped).

Waveguide ports do **not** use `run(compute_s_params=True)` for full
multi-port matrices. Use `compute_waveguide_s_matrix(...)` for the S-matrix;
`run(...)` exposes only single-port `result.waveguide_sparams` diagnostics.

## Single Port

```python
import jax.numpy as jnp
import numpy as np
from rfx import Simulation

sim = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

sim.add_waveguide_port(
    0.01,                    # x-position of port plane (meters)
    mode=(1, 0),             # TE10 dominant mode
    mode_type="TE",
    freqs=jnp.linspace(4.5e9, 8e9, 50),
    f0=6e9,                  # Center frequency for excitation pulse
    name="input",
)

result = sim.run(n_steps=500, compute_s_params=False)
# Access calibrated S-params
sp = result.waveguide_sparams["input"]
print(f"|S11| mean: {np.mean(np.abs(sp.s11)):.3f}")
print(f"|S21| mean: {np.mean(np.abs(sp.s21)):.3f}")
```

## Two-Port S-Matrix

For transmission measurements, use two ports with opposite directions:

```python
freqs = jnp.linspace(4.5e9, 8e9, 50)

# A fresh guide: the single-port run above keeps its own simulation.
sim = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

sim.add_waveguide_port(0.01, direction="+x", name="left",
                       mode=(1, 0), freqs=freqs, f0=6e9)
sim.add_waveguide_port(0.09, direction="-x", name="right",
                       mode=(1, 0), freqs=freqs, f0=6e9)

result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params  # (2, 2, n_freqs) complex

s11 = S[0, 0, :]  # Reflection at port 1
s21 = S[1, 0, :]  # Transmission port 1 → port 2
s12 = S[0, 1, :]  # Transmission port 2 → port 1 (reciprocal: S12 ≈ S21)
```

## Two-Run Normalization

For the documented empty-guide envelope, two-run normalization cancels Yee-grid
dispersion and should keep `|S21|` near unity:

```python
result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=True)
```

This runs a reference simulation (empty waveguide) to cancel Yee-grid numerical
dispersion. The single shared empty-guide reference is only correct when the
guide walls are the domain boundary. For a **branch / T-junction / septum**
(interior PEC), that reference strips the septum and radiates into free space,
so the incident power `P_inc` is mis-normalized and every `|S|` inflates
(a compact 3-port T-junction gives `normalize='flux'` max|S| ~ 9.8, |S11| ~ 1.9).

For `normalize=False`, a passive strong-reflector run can sit slightly above
unit column power because the single-run decomposition keeps a documented
near-cutoff/Yee-grid overshoot envelope. rfx emits a **soft advisory** when the
result rises above that envelope but remains below the hard unreliability limit.
Treat the advisory as a prompt to use `normalize="flux"`, increase settling or
mesh quality, or compare against a reference before promoting the number; it is
not a physics correction.

Junction S-matrices are measurable with `normalize='flux'` by passing per-port
matched-straight-guide references — one `Simulation` per driven port, each the
straight continuation of that port's guide with no junction:

```python
from rfx import Box

port_kwargs = dict(mode=(1, 0), mode_type="TE", f0=6e9,
                   freqs=jnp.linspace(4.5e9, 8e9, 6),
                   z_range=(0.0, 0.02), ref_offset=3, probe_offset=15)


def three_port_guide(walls):
    """Same domain, dx and boundary for the device and every reference —
    only the interior PEC walls differ."""
    s = Simulation(freq_max=10e9, domain=(0.12, 0.12, 0.02),
                   boundary="cpml", cpml_layers=10, dx=0.002)
    for corner_lo, corner_hi in walls:
        s.add(Box(corner_lo, corner_hi), material="pec")
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x",
                         name="left", **port_kwargs)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x",
                         name="right", **port_kwargs)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y",
                         name="top", **port_kwargs)
    return s


HORIZONTAL = [((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)),
              ((0.0, 0.08, 0.0), (0.12, 0.12, 0.02))]
VERTICAL = [((0.0, 0.0, 0.0), (0.04, 0.12, 0.02)),
            ((0.08, 0.0, 0.0), (0.12, 0.12, 0.02))]

# The device: main guide along x with one arm opening in +y. These arms are
# COMPACT — the paragraph below states what that costs.
sim = three_port_guide([((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)),
                        ((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)),
                        ((0.08, 0.08, 0.0), (0.12, 0.12, 0.02))])
ref_left = three_port_guide(HORIZONTAL)
ref_right = three_port_guide(HORIZONTAL)
ref_top = three_port_guide(VERTICAL)

result = sim.compute_waveguide_s_matrix(
    num_periods=30, normalize="flux",
    port_reference_sims=[ref_left, ref_right, ref_top],
)
```

This is correct **only under the far-port discipline**: place each probe plane
at least 5 evanescent decay lengths of the next higher mode from the junction,
use CPML at least ~0.5 guide wavelengths thick, and confirm a converged mesh.
On a far-port geometry the matched-reference path reaches passivity ~1.00,
reciprocity ~0.001 and ~0.087 vs MEEP. On **compact** geometry the reference
fixes |S11| (1.86 → 0.49) but the overall matrix stays non-physical (residual
max|S| ~ 3.9); `compute_waveguide_s_matrix` emits clearance / CPML advisories
and its passivity self-check fires. This does not make arbitrary compact
junctions valid — keep the far-port discipline.

## Multi-Axis Ports

Ports can be placed on any axis-normal boundary:

```python
sim = Simulation(freq_max=10e9, domain=(0.04, 0.12, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

# Y-normal ports for a y-directed waveguide
sim.add_waveguide_port(0.01, direction="+y", name="bottom")
sim.add_waveguide_port(0.09, direction="-y", name="top")
```

## Disjoint Aperture Ports (N-port)

Multiple ports on the same boundary for parallel-guide or branch networks:

```python
sim = Simulation(freq_max=10e9, domain=(0.12, 0.10, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

sim.add_waveguide_port(0.01, y_range=(0.0, 0.04), z_range=(0.0, 0.02),
                       direction="+x", name="left_lo")
sim.add_waveguide_port(0.01, y_range=(0.06, 0.10), z_range=(0.0, 0.02),
                       direction="+x", name="left_hi")
```

## Calibration Options

```python
sim = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

# Report S-params at the snapped measurement planes (default)
sim.add_waveguide_port(0.01, calibration_preset="measured")

# Report S11 at source plane, S21 at probe plane
sim.add_waveguide_port(0.01, calibration_preset="source_to_probe")

# Explicit reporting planes with de-embedding
sim.add_waveguide_port(0.01, reference_plane=0.012, probe_plane=0.034)
```

For reverse-mode AD or memory-heavy waveguide runs on the uniform Yee path,
`compute_waveguide_s_matrix(checkpoint_segments=K)` reuses the segmented
checkpointing machinery from the core runner. `K` must divide the timestep
count exactly; non-uniform waveguide extraction rejects this knob rather than
silently falling back to the linear-memory scan.
