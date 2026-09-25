---
title: "Waveguide Ports"
sidebar:
  order: 10
---

A waveguide port launches one TE or TM mode of a rectangular aperture and
splits the fields at its plane into incident and reflected mode amplitudes.
This page shows how to get a rectangular-guide S-matrix, which normalization
to pick, and how to handle junctions.

The port aperture defines the guide. For a straight guide you do not draw
side walls: the aperture edges act as the walls. Ports need
`boundary="cpml"` and `mode="3d"`, and cannot share a model with lumped ports
or a TFSF source.

## A two-port S-matrix

Place two ports facing each other and call `compute_waveguide_s_matrix()`. It
drives each port in turn and assembles the full matrix.

```python
import jax.numpy as jnp
import numpy as np
from rfx import Simulation

# A 40 x 20 mm guide: TE10 cuts off at 3.75 GHz, the next mode at 7.5 GHz.
freqs = jnp.linspace(5e9, 6.5e9, 7)

sim = Simulation(freq_max=10e9, domain=(0.10, 0.04, 0.02),
                 boundary="cpml", cpml_layers=16, dx=0.002)
sim.add_waveguide_port(0.01, direction="+x", mode=(1, 0), mode_type="TE",
                       freqs=freqs, f0=6e9, name="left")
sim.add_waveguide_port(0.09, direction="-x", mode=(1, 0), mode_type="TE",
                       freqs=freqs, f0=6e9, name="right")

result = sim.compute_waveguide_s_matrix(num_periods=40, normalize="flux")
S = result.s_params            # (n_ports, n_ports, n_freqs), S[receiver, driven]
print(np.abs(S[1, 0]).round(3))   # |S21| of the empty guide: 1.0 across the band
```

The first argument is the port plane's coordinate along its normal axis, the
axis named by `direction` (`"+x"`, `"-x"`, `"+y"`, `"-y"`, and so on). By
default the aperture spans the whole domain cross-section; restrict it with
`y_range=` / `z_range=` (or `x_range=` for a y-normal port).

The 2 mm mesh is coarse, chosen so the example runs in seconds. rfx warns
that the 16-layer absorber is thinner than it recommends at 5 GHz (see
[Absorber depth](#absorber-depth)). Keep the measurement band inside the
single-mode range: preflight warns when `freqs` reach toward the next mode's
cutoff. Place the two ports as mirror images in the domain, as here, so both
see the same grid.

## Choosing `normalize`

| `normalize` | Runs | What it corrects | Use it for |
|---|---|---|---|
| `False` (default) | one per port | nothing; magnitudes carry a few-percent Yee impedance error | \|S11\| of strong reflectors (shorts, high-Q loads) on a uniform mesh |
| `True` | two per port | one-way grid dispersion in transmission | S21 of a straight guide. Not for S11. |
| `"flux"` | two per port | magnitude from Poynting flux, phase from the mode | general use; required on a graded mesh |

`normalize=True` divides by an empty-guide reference run. That fixes
transmission but not reflection. `normalize=True` also cannot be
differentiated; `False` and `"flux"` can.

With `normalize=False`, a passive strong reflector can read slightly above
|S| = 1. When it rises beyond the expected range, rfx prints an advisory. Switch
to `"flux"` or refine the mesh rather than reporting that number.

rfx also checks every result for passivity and reciprocity and warns when a
column carries more power than it received. That warning almost always means
the extraction is wrong, not the physics.

## A single port with `run()`

A plain `run()` with one waveguide port gives that port's calibrated S11 and
S21 in `result.waveguide_sparams`. Use it as a quick diagnostic. It is not the
driven-in-turn matrix.

```python
single = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                    boundary="cpml", cpml_layers=10, dx=0.002)
single.add_waveguide_port(0.01, mode=(1, 0), freqs=freqs, f0=6e9, name="input")

res = single.run(n_steps=500)
sp = res.waveguide_sparams["input"]      # sp.freqs, sp.s11, sp.s21
```

## Reference planes

Fields are sampled on grid planes, so rfx snaps each port's planes to the
nearest one and reports where they landed. You choose where S is reported:

```python
cal = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

# Report at the snapped measurement planes (the default).
cal.add_waveguide_port(0.01, calibration_preset="measured", name="a")
# S11 at the source plane, S21 from source to probe plane.
cal.add_waveguide_port(0.01, calibration_preset="source_to_probe", name="b")
# Explicit planes, in metres along the port normal, with de-embedding.
cal.add_waveguide_port(0.01, reference_plane=0.012, probe_plane=0.034, name="c")
```

Three ports on one plane is only to show the options; a real model uses one.
The result's `reference_planes` and each port's `measured_reference_plane`
give the planes actually used.

## More port layouts

Ports can face along any axis, and several can share one boundary:

```python
# A y-directed guide.
ydir = Simulation(freq_max=10e9, domain=(0.04, 0.12, 0.02),
                  boundary="cpml", cpml_layers=10, dx=0.002)
ydir.add_waveguide_port(0.01, direction="+y", name="bottom")
ydir.add_waveguide_port(0.09, direction="-y", name="top")

# Two parallel guides entering through the same x face.
pair = Simulation(freq_max=10e9, domain=(0.12, 0.10, 0.02),
                  boundary="cpml", cpml_layers=10, dx=0.002)
pair.add_waveguide_port(0.01, y_range=(0.0, 0.04), direction="+x", name="lo")
pair.add_waveguide_port(0.01, y_range=(0.06, 0.10), direction="+x", name="hi")
```

`n_modes > 1` records several modes per port. Multimode results are assembled
outside the differentiable path.

## Junctions and interior walls

The default references assume the guide walls are the domain boundary. A
T-junction, branch or septum has interior PEC walls, and the empty reference
then radiates into open space: every |S| inflates badly (max |S| near 10 on a
compact T). For these, pass `normalize="flux"` with one reference simulation
per port: the straight continuation of that port's guide, with no junction.

```python
from rfx import Box

port_kwargs = dict(mode=(1, 0), mode_type="TE", f0=6e9,
                   freqs=jnp.linspace(4.5e9, 6.5e9, 4),
                   z_range=(0.0, 0.02), ref_offset=3, probe_offset=15)


def three_port(walls):
    """Same domain, mesh and ports each time; only the PEC walls change."""
    s = Simulation(freq_max=10e9, domain=(0.12, 0.12, 0.02),
                   boundary="cpml", cpml_layers=10, dx=0.002)
    for lo, hi in walls:
        s.add(Box(lo, hi), material="pec")
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x",
                         name="left", **port_kwargs)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x",
                         name="right", **port_kwargs)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y",
                         name="top", **port_kwargs)
    return s


horizontal = [((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)),
              ((0.0, 0.08, 0.0), (0.12, 0.12, 0.02))]
vertical = [((0.0, 0.0, 0.0), (0.04, 0.12, 0.02)),
            ((0.08, 0.0, 0.0), (0.12, 0.12, 0.02))]

tee = three_port([((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)),
                  ((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)),
                  ((0.08, 0.08, 0.0), (0.12, 0.12, 0.02))])

junction = tee.compute_waveguide_s_matrix(
    num_periods=20, normalize="flux",
    port_reference_sims=[three_port(horizontal), three_port(horizontal),
                         three_port(vertical)])
```

The references are necessary but not sufficient. This compact example still
gives a non-physical matrix, and rfx warns about probe clearance and absorber
depth. A physical junction S-matrix also needs:

- each probe plane at least 5 decay lengths of the next higher mode away from
  the junction;
- an absorber at least about 0.5 guide wavelengths deep at the lowest
  frequency;
- a converged mesh.

The walls here are PEC volumes, so the guide width the solver sees is the
width you drew. See
[How conductors land on the lattice](/rfx/guide/materials-geometry/#how-conductors-land-on-the-lattice).

## Absorber depth

A waveguide port's absorber must swallow the guided wave, and CPML does that
worse near cutoff, where the guide wavelength is long. rfx warns when the
absorber on a port's axis is thinner than 0.5 λ_g at the lowest measured
frequency. Treat that as a floor. For scale: a WR-90 guide that simply runs
into the CPML reflects about −13.9 dB with 8 layers and −22.3 dB with 16.

## Gradients and memory

`compute_waveguide_s_matrix` accepts `eps_override=` / `sigma_override=` for
differentiation with `normalize=False` or `"flux"`, single-mode only. For
long runs under `jax.grad`, `checkpoint_segments=K` trades compute for memory.
`K` must divide the number of time steps exactly.

## Limits

- **Validated scope** is uniform, single-mode, straight rectangular guides and
  same-guide junctions under the far-port conditions above. Phase is
  validated on fewer configurations than magnitude.
- **Ports with different cross-sections** are outside the validated scope.
  S is a ratio of each port's own modal waves, so it equals a power-wave S only
  when all ports share one cross-section and mode.
- **Compact junctions** give non-physical matrices even with
  `port_reference_sims`. Keep the far-port conditions.
- **Graded meshes** require `normalize="flux"`. Graded-mesh waveguide results
  outside the published WR-90 comparisons are experimental.
- **`port_reference_sims`** needs `normalize="flux"`, single-mode ports, a
  uniform mesh and no material override.
- Full scope, with the comparisons behind it, is in the
  [S-parameter support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/sparameter_support_matrix.md).
