# Coarse-short fixture repair

The common defect class is an acceptance claim about an undeclared realized
network. A conductor's declaration, the mesh that realizes it, and the physical
quantity measured by the ports must describe the same network before historical
numbers can be meaningful. The former short witness violated all three: its
85--87 mm plate was off-lattice at 2 mm spacing, its right measurement planes
were across the plate from their source, and its warning helper accepted an
unrelated reciprocity advisory as evidence of the column-power advisory.

The repaired fixture is a 120 x 40 x 20 mm rectangular guide. The previous
WR-90 label was incorrect: WR-90 has a 22.86 x 10.16 mm cross-section. A 2 mm
closing plate now occupies x=84--86 mm. This preserves the current coarse
realized body, while making the drawn faces exact on both the 2 mm coarse and
1 mm fine grids. These meshes divide the plate thickness, all guide dimensions,
source coordinates and physical measurement offsets. `freeze_mesh` fixes the
grid before adding the plate; the declaration/resolution split is untouched.

Both grids use sources at x=10 and 110 mm, references at 16 and 104 mm, and
probes at 30 and 90 mm. Moving the right source from 90 to 110 mm puts its
reference and probe wholly to the right of the far face, in the same connected
vacuum region as that source. Specifying offsets as 6 mm and 20 mm divided by
dx keeps the measured network fixed across the two meshes. The short is a
buildable 2 mm plate closing the full 40 x 20 mm guide cross-section. This is
a geometry repair, not a search for a mesh or source position that reproduces
the old advisory number.

`tests/contracts/test_pec_short_advisory_geometry.py` checks the actual compiled
source/reference/probe coordinates and realized wall planes on both grids.
The slow witness checks finite S entries and nonzero reflected waves from both
drives before evaluating the unchanged `(2.25, 3]` interval. Its original fine
control `<= 2.25` is also unchanged. `_soft_fired` now identifies the column-power
advisory specifically, with a synthetic reciprocity false-positive check.

No numerical value is re-pinned here. If the repaired physical network misses
the historical interval, the miss remains visible and is evidence that this
network is unsuitable as a live over-unity advisory witness. The helper-level
tests in `test_sparam_passivity_guard.py` independently cover the precise
warn/silent behavior in the advisory interval, at its lower side, on the tight
tolerance path, and above the hard limit. Replacing the live witness with a
physical short/reflection test would preserve useful physical coverage, but
this repair does not remove or weaken its existing interval assertion.

Serial diagnostic commands (always local import, CPU, no other pytest running):

```bash
PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/diagnostics/slow_931_advisory_attribution.py --arm current --output-dir output/931-fixture-repair/short
PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/diagnostics/slow_931_advisory_attribution.py --arm current --fine --output-dir output/931-fixture-repair/short
```

The diagnostic records all complex S entries, per-column power, incident-wave
observability, settling, actual port positions and V/I time traces. Historical
attribution JSON files under `slow-consumer-evidence/` remain historical and
must not be confused with results from this repaired shared builder.
