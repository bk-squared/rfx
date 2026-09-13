# #947: repair the absorber and compare the measured field component

The red O3 tests on main `45481c6c` are caused by a malformed absorber in
the oracle fixture. A second, independent comparator defect applies the
Hy envelope prediction to Ez. This change fixes the fixture declaration
and predicts each component on its own Yee samples. The sheet operator,
port implementations and differentiation paths are unchanged.

## Material-array cause and introduction

The intended absorber has 120 one-cell slabs, from 130 to 190 mm, with
`sigma[i] = 2*((i+0.5)/120)**2`. The old declaration used
`lo = 0.130 + i*DX; hi = lo + DX`. On the canonical float64 node line,
layer 28's upper face is one ulp above its intended node. Its half-open
window contains two nodes; the one-cell Box branch therefore falls back
to the nearest midpoint and selects layer 29. The next eight declarations
also select the following node. Layer 37 overwrites the collision.

The assembled result contains **sigma=0 at x=144 mm**, where the intended
value is 0.1128125 S/m, and eight displaced values in the following cells.
Using `(260+i)*DX` and `(261+i)*DX` for the two faces realizes every intended
slab. An always-on test checks the full conductivity array, including the
zero regions, without running FDTD. No global Box selection rule changes.

Historical **source replay**, using the same grid and current JAX 0.6.2,
isolates the material-profile transition:

| Source revision | Profile defects |
|---|---:|
| `bb43d95f` (#700 comparator) | 0 |
| `92018513` (immediately before #834) | 0 |
| `635ab2e3` (#834 exact coordinates) | 9 |
| `df819523` (#931 first parent) | 9 |
| `45481c6c` (current main) | 9 |

This is **not an execution of the historical solver**. The replay executes
the recorded Box coordinate/mask source and retains its hashes. The defect
predates #931: the previous attribution to its added sheet rims was not
isolated. In particular, a y-PMC zeros tangential Hx/Hz, not the Hy that
enters the z-sheet Ex-current relation; the boundary label alone did not
establish zero dissipation at that rim.

## Controlled field comparison

Two retained runs use the same main revision, Python 3.10.12, JAX 0.6.2,
float32 CPU solver, grid, timestep, 4,000 steps, source, sheets, probes and
five frequency bins. The sole fixture edit is the absorber-bound arithmetic
(plus final-newline normalization). Compressed fixture sources match the
run-time SHA-256 receipts. All production code is the same.

| GHz | Hy relative fit RMS before | After | Ez alpha before | After |
|---:|---:|---:|---:|---:|
| 8 | 1.07698% | 0.56078% | 0.245818 | 0.231060 |
| 9 | 1.12549% | 0.26007% | 0.497386 | 0.463836 |
| 10 | 1.24241% | 0.32684% | 0.715650 | 0.698237 |
| 11 | 1.28333% | 0.33788% | 0.868713 | 0.873708 |
| 12 | 1.27499% | 0.37548% | 1.002854 | 1.014329 |

Alpha units are Np/m. The repaired values recover the printed #700 record;
the original historical complex profiles were not recovered, so this is
not a claim of historical field-array bit identity. Both current raw
complex Ez/Hy planes are retained, with an explicit 8 GHz profile CSV.

At 10 GHz, the endpoint-ratio alpha returns from 0.87333 to 0.72494797;
the original **0.72494** diagnostic pin is restored at its unchanged 5%
tolerance. The repaired settling witness is -71.02 dB. The 1% Hy fit gate
and the 9% alpha gate are unchanged. The new evidence replaces the former
re-pin's causal explanation; both previous and repaired records remain.

## Observable pairing and limits

For each forward TM mode in air, with `exp(+j omega t - j kx x)`, Ampere's
law gives `Ez_m = -eta0*(kx_m/k0)*Hy_m`. Different modes have different
complex factors. Consequently, the magnitude slopes of their Ez and Hy
sums need not be equal, even for an exact Maxwell solution.

The model fits only three complex amplitudes to the measured Hy plane.
Its eigenvalues and transverse profiles remain analytic. Those amplitudes
are referred to the first **physical Hy x sample**. Ez is predicted without
refitting at its actual x and z coordinates: here x_E is half a cell below
x_H, and the Ez plane registered at 3 mm samples z=3.25 mm. The existing
Hy alpha column samples z=2.75 mm. A common time-stagger phase does not
affect either magnitude slope.

A prescribed lossless/symmetric-lossy mixture with amplitude ratio 0.25j
falsifies the old shared-alpha rule by more than 20%, without selecting a
ratio from FDTD data. A separate TEM test anchors impedance and phase.
The new Ez alpha errors on the repaired run are 0.73%, 1.85%, 2.72%, 1.32%,
and 1.27%. The Hy errors remain 1.47–5.66%.

The Hy fit is a **model-adequacy check**, not an independent prediction of
the source launch. Ez is a predicted, unfitted observable. These tests
support this vacuum four-conductor guide, its fit window and 8–12 GHz
band. The API's real, frequency-flat Rs at f0 does not model the reactive
surface impedance. This repair does not validate arbitrary launches,
higher modes, other grids, or port power accounting.

## Visibility and retained evidence

The weekly CPU lane already selects `-m "not gpu and not highmem"`, which
includes these `slow_physics` tests. The actual
[September 7 scheduled job](https://github.com/bk-squared/rfx/actions/runs/34120118201/job/101736004116)
records both O3 failures. Thus the issue's "no battery sees them" premise
is stale. The marker description is corrected; no duplicate lane is added.

Evidence is in [issue947/](issue947/): both raw run archives, run-time
source hashes and compressed source snapshots, full logs, assembled sigma
arrays, historical mask replay and the scheduled job logs. `manifest.json`
records byte hashes. `executed_capture.py` and the history script preserve
the original workspace commands; their paths are provenance, not a portable
entry point. To replay the portable comparison without a field solve:

```sh
python docs/research_notes/issue947/replay_profiles.py
```

Validation: the complete oracle on Python 3.10/JAX 0.6.2 passes 16 tests
with the existing O4a guide-law strict xfail retained (41.23 seconds).
Python 3.11/JAX 0.10.2 also passes 16 with the same xfail (50.35 seconds).
Two additional fast tests exercise the actual recorder/model handoff with
synthetic returned fields, nonzero padding and nondefault Ez probe indices;
both pass in each environment. They check physical H coordinates, ghost
exclusion and the actual Ez index reaching the unfitted prediction.
The independent free-standing-sheet sigma discriminator remains green.
The original O4a guide comparator limitation is outside #947; no band or
expected-failure policy is changed to make this repair pass.
