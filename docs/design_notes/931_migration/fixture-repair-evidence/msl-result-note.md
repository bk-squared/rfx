# MSL repair: aligned board, corrected reference length, source repair pending

**No MSL golden is re-pinned.** The completed capture uses fixture
**931-msl-254um-600um-aligned-v1**, before the explicit source-band repair.
Its corrected 4 mm reference-plane comparison passes the implemented
physical checks except the full-spectrum reliability screen: both ports
mark the 4.5 and 5 GHz bins unreliable. Final fixture **v2** uses a source
whose analytic spectrum covers those bins, but has **no completed FDTD
qualification** in this evidence. Its base/refined/longer-record runs
remain for VESSL.

## Physical drawing and mesh

The PI-authorized board is retained: lossless RO4350B-like eps_r=3.66,
**254 um laminate**, **600 um trace**, nominal 35 um copper represented
by a PEC sheet at **z=254 um**, and ground at z=0. The corrected domain
is **14 x 3.4 x 1.978 mm**. The trace spans x=0--14 mm and
y=1.4--2.0 mm; its center is y=1.7 mm. These are practical etched coupon
dimensions with idealized lossless material, PEC foil, and continued-line
absorbers, rather than a connector/hardware claim.

Uniform **50 um x/y cells** resolve the 600 um width with 12 cells and
place both footprint edges on nodes. A fixed z profile begins with two
50 um cells followed by 28 cells of **38.5 um**, and ends with sixteen
50 um cells. The substrate top is exactly two 50 um plus four 38.5 um
cells from z=0: **254 um**. All CPML boundary spacings are 50 um;
sixteen layers give 0.8 mm physical padding. Refinement bisects each
cell and doubles layer/probe-index counts to preserve physical geometry.
This replaces the 254/3 um uniform mesh's lateral width residual while
keeping the authorized laminate and 600 um drawing; it does not restore
the historical 80 um/320 um board.

Source/launch x positions remain **2 and 12 mm**, 10 mm apart. Each
five-plane extraction array begins 3 mm inward: left x=5,6,7,8,9 mm;
right x=9,8,7,6,5 mm. Production assembles S from probe-0 waves, so its
actual reference planes are **x=5 and 9 mm, 4 mm apart**. Comparing its
S21 phase to a 10 mm line was a comparator defect. Offline requalification
corrects that length without modifying any captured S entry.

## Completed v1 capture and offline requalification

Named CPU capture **local-931-msl-repaired-base-20260908** completed on
2026-09-09 UTC using 12 periods, JAX 0.6.2, x64 disabled and complex64
outputs. Its source HEAD is
`949e1845b84b26826d4e5309363bea4c2ffa5920`, with uncommitted repairs
identified by the source-file hashes in the
[unaltered original JSON](msl-v1-original.json). The import points to
the requested worktree. The **v1** designation matters: this capture
predates the v2 explicit waveform.

Named **local-931-msl-final-base-offline-4mm** is a comparator
requalification, explicitly `fresh_fdtd_run=false`, `golden_written=false`.
Its [initial unaltered JSON](msl-v1-requalified-4mm.json) embeds the original
capture, original failures, corrected failures and both sets of source
hashes. The [final endpoint-safe requalification](msl-v1-requalified-4mm-endpoint.json)
additionally includes the intended 4.5 GHz endpoint, represented in the
float32 capture as 4,500,000,256 Hz, using a one-float32-ULP boundary guard.
Its physical-band indices are explicitly **[5,6,7,8]**. This floating-point
selection correction leaves all physical budgets unchanged; reliability
remains the only failure after including that endpoint. The original report SHA256 is
`51e5d2ea7d0c0feabefcf517635c087171b1bc4e772ef31e351483d8c650a1ce`.
The corrected reference model is the drawing-based thin-strip quasi-TEM
approximation, with **Z0=47.89480046631184 ohm**, not an external full-wave
oracle. Source-fitted line impedance is about 46.5--46.9 ohm. The 10%
Z0/beta and 5 degree phase budgets are unchanged.

| Witness | Completed v1 result |
|---|---:|
| Left / right settling, dB | -103.18656728330424 / -103.40976882863211 |
| Drive condition-number range | 1.2065251628488811--1.3396130402015236 |
| Maximum raw singular value, recomputed in NumPy | 1.0001241823829228 |
| Maximum complex raw/projected entry difference | 6.847920674042203e-5 |
| Maximum raw complex reciprocity difference | 1.2982324018537504e-5 |
| Maximum phase error with correct 4 mm reference | 0.4034066482065368 degrees |
| Maximum phase error with incorrect 10 mm reference | 61.42696905682657 degrees |
| Beta-railed flags | False for both ports in all ten bins |
| Full-spectrum reliability | False at 4.5 and 5 GHz for both ports |

Thus settling, conditioning, raw reciprocity, raw singular-value and
drawing-model diagnostics do not erase the excitation failure. The
original lossless-material/Q advisory also remains visible; this line
comparison makes no resonator-Q claim.

## Source-band defect and v2 repair

The [independent analytic source analysis](msl-source-band-analysis.json)
uses the differentiated-Gaussian Fourier magnitude
`f * exp(-(f/(f0*bandwidth))**2)`, normalized by its median over the
declared ten bins. The implicit **f0=2.5 GHz, bandwidth=0.8** v1 pulse
has relative magnitudes **0.07248649196460143** at 4.5 GHz and
**0.024563478903906776** at 5 GHz. Both lie below the API's unchanged
**0.1** low-signal floor, at exactly the bins marked unreliable.

The implemented v2 source explicitly uses **f0=5 GHz, bandwidth=0.8**.
The same analytic calculation gives a minimum relative-band-median
amplitude of **0.35480668932926224** across all ten bins; its 4.5 and
5 GHz values are **0.9148843660582516** and **0.7554270338145136**.
This repairs the source coverage rather than relaxing a reliability
gate or discarding the upper frequency bins. Analytic source coverage
is evidence for the intervention, not qualification of final measured
port waves. The v2 live result remains unmeasured here.

## Full complex trace inspection and historical lock

The [full matrix figure](msl-v1-full-matrix.png) shows real and imaginary
parts of all four S entries at all ten frequencies, with raw and projected
v1 values and the unchanged historical golden. Raw/projected traces
nearly overlap; the historical reflection signs and transmission phase
differ substantially. The gray region identifies the two unreliable
upper bins, rather than omitting them. All 40 raw and projected complex
entries are finite.

The [qualification and source figure](msl-v1-qualification-and-source.png)
shows the corrected 4 mm phase comparison, the rejected 10 mm comparator,
raw/projected singular values, both analytic source spectra and all ten
phase errors with the unreliable bins marked. Both figures were visually
inspected. Full complex matrices, line parameters, flags and source-band
arrays are preserved in [compact NPZ](msl-v1-matrix.npz); derived checks
and hashes are in [inspection JSON](msl-v1-inspection.json).

An independent array comparison confirms **40 of 40 complex entries**
miss the historical golden at unchanged **rtol=0.005, atol=0.002**,
with maximum absolute difference **0.20305470527038097**. The golden's
SHA256 remains
`7203caac78f285e5dc3dfc73bd74dd71e4c2e33f0a92b095808d854cfeb6eddb`.
This changed-board comparison records the red; it does not authorize a
re-pin, and an old-board golden is not a physical oracle for the repaired
coupon. Frozen replay fixtures retain their separate historical geometry.

The final v2 source needs a named live base capture with every reliability
flag true, fixed-drawing refinement and longer-record witnesses, and an
independent confirmation before a golden is captured. No MSL re-pin,
new FDTD solve, pytest run or production edit was performed to create
these inspection artifacts.

The independent v1 FDTD test reproduced the raw S matrix **array-identically**
(maximum difference0); see `msl-v1-independent-confirmation.json`. Final v2
source/preflight checks passed in the18-case follow-up, without v2 time stepping.
The actual sampled drive spectra and preflight findings are preserved in
`msl-source-band-sampled.json`; the original analytic source report is unchanged.
