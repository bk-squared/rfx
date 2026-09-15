# Graded farfield fixture repair: hold the radiator lattice fixed

Named run: **local-931-fixture-repair-farfield-600**, 2026-09-08, CPU,
`JAX_PLATFORMS=cpu PYTHONPATH=$PWD`. Source HEAD was
`949e1845b84b26826d4e5309363bea4c2ffa5920`, with the fixture repair uncommitted.
The actual `rfx` import was verified inside the requested worktree. The copied
[capture JSON](farfield-attribution.json) retains the original run fields and
adds an explicitly post-capture provenance block with fixture, diagnostic and
raw-artifact SHA256 hashes. Both source files' modification times preceded
the capture; the source HEAD alone does not identify the uncommitted repair.

The diagnostic SHA256 is
`6b5fb8a6a5575b6e820d39ecb67bfaa535014efccc4be03c2a6e1e92d2ab9b37`;
the fixture SHA256 is
`8e393f4e6c89f2f9c6fdd38a0fbb52c94af5c6eeecd33b7b978dbfdcd7949fcc`.

## Repair and declared falsifier

The common defect class is a historical fixture comparison whose locked
quantity no longer isolates its stated property. T8 established that the
old graded fixture changed the conductor stencil, source cell volume and
physical acquisition window as well as NTFF integration. Its old passing
2.1381% power metric already hid about 19% angular-shape disagreement.
The PI authorized physical fixture repair; the new comparison therefore
holds those local quantities fixed rather than changing the 5% envelope.

The domain remains 22 x 22 x 22 mm, with a 1.5 mm PEC cube bounded by
10.25 and 11.75 mm on all axes. A machined 1.5 mm cube is a realizable
solid, idealized here as PEC. The 250 um mesh gives six occupied cells
per axis and seven tangential wall planes including both declared faces.
The current source stays at (11, 11, 9.5) mm. NTFF faces stay at 4.5 and
17.5 mm; the source is at 30 GHz and the unchanged maximum frequency is
40 GHz. CPML remains six 250 um exterior layers.

Both meshes now use 250 um cells throughout the central 7--15 mm band.
Only the graded x/y shoulders at 4.5--7 and 15--17.5 mm use 312.5 um
cells; all exterior cells and all z cells use 250 um. Those dimensions
put every conductor face, source node and NTFF face on a node. The
shoulders contain 16 coarsened cells per in-plane axis inside the NTFF
span, so local NTFF weights and coordinates remain materially different
from the scalar-cell counterfactual. This is still a graded integration
test, with the radiator/source lattice held fixed.

The predeclared falsifier was local-cell discrepancy >=5% **or** local-cell
discrepancy >=half the scalar-cell discrepancy. Both original gates remain
unchanged. This was one repair qualification attempt, not a profile sweep.

## Measured results and independent witnesses

| Quantity | Uniform | Graded |
|---|---:|---:|
| Angular squared-field integral | 6.156803489220375e-14 | 6.178911931015615e-14 |
| NumPy integral | 6.156803765602908e-14 | 6.178912805885424e-14 |
| NumPy/JAX maximum intensity disagreement / peak | 9.01657375156514e-7 | 8.858689604539369e-7 |
| Occupied conductor cells | 216 | 216 |
| Ex / Ey / Ez conductor edges | 294 / 294 / 294 | 294 / 294 / 294 |
| Source dual volume, m3 | 1.5625e-11 | 1.5625002226443134e-11 |
| dt, ps | 0.4766437173827514 | 0.4766437173827411 |
| Acquisition duration, ps | 285.9862304296509 | 285.9862304296447 |

The local-cell discrepancy is **0.3590896125554137%**; the scalar-cell
counterfactual discrepancy is **4.523192821590266%**. Thus both unchanged
gates pass. The scalar-cell graded integral is 5.87831939575654e-14.
These are angular integrals of squared far-field amplitudes, not newly
calibrated watt measurements. No acceptance number is re-pinned.

All 300 angular bins per mesh, both complex polarization arrays, both
source records and all twelve complex NTFF face arrays were inspected for
finite/nonzero data. Source time and waveform arrays are exactly equal.
The small source-volume/dt differences shown above are floating-point
representation differences. NumPy and JAX provide independent extractor
implementations on the same fields, not independent FDTD physics oracles.

The [visually inspected angular figure](farfield-repaired-angular.png)
shows similar broad polar lobes, shared azimuthal modulation and small
structured angular differences, with no isolated corrupted bin or zero
record. The figure includes all 300 bins and two explicitly labelled
sampled cuts. The largest intensity difference relative to a common peak
is 0.020227694883942604. The maximum difference after separately normalizing
each pattern to its peak is 0.03170943260192871; power-normalized angular
L2 discrepancy is **0.010401696898043156 (1.0401696898043156%)**.
These are inspection metrics, not newly fitted gates.

[Compact traces](farfield-traces.npz) retain complex angular fields,
intensities, source waveforms/times and angle coordinates. Full NTFF arrays
remain in the local raw archive
`outputs/931-fixture-repair/farfield-600/farfield-attribution.npz`; its hash
is recorded in the copied JSON, but the large raw face arrays are not
duplicated into the committed evidence subset.

## Limits and next named run

The 600-step finite window has no field settling probe, no temporal
convergence witness and no mesh-convergence study. A zero source tail does
not establish scattered-field settling. The inherited coarse angular
quadrature is a common comparison metric, not an absolute radiation-power
qualification. The measured pass qualifies this fixed integration fixture
at its stated settings; it does not certify general graded-grid convergence.

Reproduction command:

```bash
PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/diagnostics/slow_931_farfield_attribution.py \
  --fixture repaired --operators current --steps 600 \
  --output-dir outputs/931-fixture-repair/farfield-600
```

For VESSL, launch the same current-operator pair with `--steps 1200` and
a new `farfield-1200` output directory as a named duration witness. Compare
both absolute integrals and full angular records against this 600-step
capture, retaining both original gates. No VESSL run was launched here.
