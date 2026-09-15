# V173A repair: aligned composition sentinel and source-free measurement

V173A's unsupported antenna-accuracy/matching claim is retired. Its
replacement is the engineering review's explicitly synthetic composition
option: fixed geometry exercising source/material/sheet/CPML/Harminv
composition, with named numerical observables and unchanged drift
tolerances. It is not the review's finite-board 0.25 mm antenna proposal.

## Drawing and measurement choices

The fixed truncated domain remains **50 x 50 x 20 mm**, with an ideal
uniform laminate/infinite-ground model: lossless eps_r=4.3 substrate from
z=0 to **1.5 mm**, PEC domain ground at z=0, and a **28 x 36 mm** patch
footprint at x=11--39 mm, y=7--43 mm. Nominal 35 um copper is represented
as a zero-thickness PEC sheet on **z=1.5 mm**, not a 1 mm metal volume.
These are buildable laminate/etched-footprint dimensions with explicitly
idealized material, ground and excitation; no fabricated connector or
finite-board response is asserted.

Uniform **dx=0.5 mm** aligns every stated face and footprint endpoint,
resolves the laminate with three cells, and gives the patch 56 x 72
cells. The default PEC sheet's Ex/Ey footprints contain 56x73 / 57x72
edges; normal Ez remains live. The current source is at
**(25,18,0.5) mm**, and the Ez probe at **(25,18,1.0) mm** lies on the
remaining live gap edge below the foil. This also corrects the old
probe's relationship to the conductor/interface.

The source now explicitly uses a **1 A** Gaussian-pulse amplitude
parameter (`amplitude_kind="current"`, f0=2 GHz, bandwidth=0.8,
cutoff=3); it is not the retired implicit field-amplitude convention.
CPML physical padding is **4 mm**, whereas the retired 1 mm/eight-layer
fixture had **8 mm**. These intentional changes belong to the new
synthetic fixture and are not hidden as floating-point drift.
The record is 6000 samples at 0.9532874347655029 ps, preserving the old
3000-step/1 mm physical duration, approximately 5.720 ns.

The full-record Hann-windowed FFT minimum is named
`probe_spectrum_min_db`. It is an **unnormalised log-amplitude statistic**,
not S11. Doubling the trace raises it by 20 log10(2); the corresponding
fast test directly falsifies the retired reflection-coefficient name.
Its minimum remains at the upper band edge rather than an identified
matching resonance.

## Instrument repair and provenance

Initial whole-record Harminv fits failed the retained min_Q=5 criterion,
both with and without decimation. The full probe record is dominated by
the driven transient. The separately recorded parent-run
[offline instrument census](v173a-instrument-census.json) found whole-record
min_Q=0 poles near 2.07759 GHz / Q=1.3156 and 3.06112 GHz / Q=2.445.
That census is exploratory offline evidence, not a new solver run or
the final window-selection rule. Its exploratory sample-1000 window is
not used by the final instrument.

The final window is independently defined by the source: start at
**t0+6 tau=1.7904931097838228 ns**, rounded upward to **sample 1879**.
The analytic source bound there is **2.7834273962922828e-15 A**. The
remaining 4121 samples are DC-subtracted and fitted with min_Q=5,
`decimate=False`; this measures a source-free oscillatory record rather
than asking a pole fit to represent the driven transient. It does not
relax the mode Q threshold or choose a window by proximity to the old
frequency.

Named capture **local-931-v173a-source-free-reanalysis-20260908** is
**offline reanalysis**, using the unchanged FDTD trace from
`outputs/931-fixture-repair/v173a-instrument-repair/result.npz`
(exact raw archive: [v173a-source-record.npz](v173a-source-record.npz)).
The [copied result JSON](v173a-reanalysis.json) records source HEAD
`949e1845b84b26826d4e5309363bea4c2ffa5920`, CPU/JAX 0.6.2, x64 disabled,
the verified worktree import, full fixture/waveform manifest and harness
SHA256 `d881da9b5cbd249b77bf23e1fc88e00fa16e207805a028a8cc69065d44321240`.
The raw trace SHA256 is
`b0d3bf17ab221f35d4d31c8a0a8cec829fa15c1ee54f3c1fc407109206b1b09e`.
Its samples and dt were independently checked array-identical to the
original solver capture; changing the instrument did not change the
captured fields.

## Recorded values and inspected limits

| Observable | Recorded value |
|---|---:|
| Dominant source-free probe mode | **1,984,479,720.2163157 Hz** |
| Fitted mode Q, diagnostic only | 104.5664788462284 |
| Full-record FFT log-amplitude minimum | **190.48059426681422 dB** |
| FFT-minimum frequency | 3,496,671,844.996355 Hz |
| Raw full-trace tail / full-trace peak | 0.0003252121096011251 |
| DC-subtracted ring-down tail / ring-down peak | **0.768401026725769** |

These are confirmed numerical pins for the repaired composition fixture;
the frequency rtol=1e-6 and spectrum atol=0.1 dB remain unchanged. They
are not refreshes of an antenna-performance baseline. The selected mode
has no spatial mode label, so it must not be called an identified patch
fundamental.

The [visually inspected figure](v173a-inspection.png) shows the complete
probe trace, the final source-free/DC-subtracted record on its own scale,
and every full-record FFT bin in the measurement band. The driven peak
is 6.578120192e9, whereas the ring-down peak is 1.6958925e6. Thus a small
tail/full-record-peak ratio hides persistent ringing: the ring-down's
own tail remains **76.84%** of its peak. The FFT band descends toward its
upper-bound minimum; the plot supplies no S11 dip or matching claim.
All samples are finite and the trace is nonzero. The FFT value, minimum frequency and
ring-down tail ratio were independently recomputed exactly from the
saved trace, without rerunning FDTD or Harminv.

[Compact evidence](v173a-traces.npz) retains the original trace/dt,
DC-subtracted ring-down, window index and complete in-band FFT values.
There is **no settling, mesh/boundary convergence, steady-state, or
physical-Q qualification**. The fitted Q is retained only to expose the
instrument output. Existing mode-labelled coverage is in
`tests/locks/test_patch_edgefed_resonance_harminv.py`; finite-ground,
mode-labelled far-field/reference regression coverage is in
`tests/crossval/test_patch_canonical_farfield_e4.py`. Neither citation
turns this synthetic sentinel into calibrated matching coverage.

Independent FDTD confirmation **passed** in the named local run
`local-931-fixture-repair-final-selection-20260908`, node
`test_v173a_baseline_bit_identity`. It rebuilt the simulation and performed
a fresh6000-step solve, passing the unchanged1e-6 frequency and0.1 dB
spectrum drift tolerances against the new committed composition baseline.
See `selection-summary.json` and the retained selection log for the final
census. The offline reanalysis itself is not counted as that confirmation.

The independently solved frequency and FFT statistic both reproduced their
new baselines exactly (both deltas0); full payload is retained in
`v173a-independent-confirmation.json`.
