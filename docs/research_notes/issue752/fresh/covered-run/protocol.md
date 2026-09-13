# Issue 752 — fresh six-point comparison (coverage revision, before new fields)

This protocol replaces the unsafe old-impedance/current-geometry reanalysis.
The historical JSON remains immutable. This coverage revision follows VESSL
369367260698: its completed h3/h4 records failed the unchanged relative-signal
screen at4.379GHz. The run was stopped duringh5; completed and partial records
are retained as controls, not silently replaced. No threshold or geometry is
changed. This protocol and new driver are hashed before the replacement runs. All measurements below are fresh.
The implementation base is `cbdc0976bd67a7c5fc318f3bc1967cd947399495`.
Only diagnostics/evidence are added; the field, source, port and AD operators
are not changed by this campaign. An exact archive and driver SHA256 accompany
the run. Six cases run sequentially, with two sequential drives per case.

## Question and comparators

The historical hypothesis is **all six band means of Re(Z0[0]) deviate by
at most 0.4% from the repository's closed-form reference on the actual board**.
Use the original 30-bin 0.5–5 GHz frequency grid (float32), selecting bins
inclusively within 3–4.5 GHz, without rejecting inconvenient in-band bins.
Report both ports, complex Z0 spectra, frequency variation, beta, and raw S
power diagnostics as secondary results. A band-mean comparison is not an
every-frequency or arbitrary-structure accuracy certificate.

Source inspection found that `hammerstad_jensen_z0_eps_eff` is a simplified
piecewise formula using `(1+12/u)^(-1/2)`. It is not the full 1980 HJ formula.
Keep it as the **repo simplified reference** for the historical hypothesis.
Also calculate a separate **HJ1980 quasi-static reference**, following
[Qucs equations 11.4–11.18](https://qucs.github.io/tech/node75.html).
Neither reference is a Maxwell ground truth; both assume a zero-thickness,
quasi-static microstrip. The independent HJ1980 reference never enters the
production source, beta search or S normalization. No 0.4% absolute physical
accuracy claim follows from agreement with either approximate model.

## Geometry fixed before fields

Retain the six nominal mesh spacings: 254µm/3, /4, /5, /6, 80µm, 60µm.
For each, independently choose the nearest integer interval counts to the
254µm height and 600µm width, with half ties toward the lower count. Declare
the corresponding **node-aligned** board. A zero-thickness PEC foil, dielectric
top and port top share one plane; ground is the z=0 PEC domain face. This
avoids an embedded finite-thickness strip being compared with a thin-strip
formula. Desired dimensions stay separate from actual dimensions.

The dielectric is lossless, isotropic eps_r=3.66. Both source/load impedances
are 50 ohms. Use the public default static-Laplace port and uniform Yee solver.
There are five voltage probes per port; the first is ceil(6mm/dx) cells from
its feed and subsequent probes are ceil(3mm/dx) cells apart. The x domain is
ceil(40mm/dx) intervals; each feed is ceil(2mm/dx) cells from its adjacent
boundary. This supplies about 36mm between feeds and 12mm fit aperture, with
both ladders far from either feed. Side margins and air above the foil are
at least 8 actual substrate heights. CPML has eight external cells on x/y
and upper z; lower z is PEC. These clearances are fixed before the result,
not adjusted to obtain a desired impedance.

These are **six different actual boards**, not a fixed-geometry convergence
ladder or an aligned-versus-misaligned experiment. The 80/60µm points now have
240µm electrical gaps; the old 320/300µm material extents are not their height.
Build checks read actual tangential PEC edges, actual dielectric samples,
and validated port gaps. Runtime interception checks the consumed dielectric
and complete PEC edge arrays against that case's pre-solve plan. Source/load
sigma may legitimately differ from the passive geometry plan and is recorded.

## Excitation coverage correction

The GaussianPulse implementation is a differentiated Gaussian, not a
carrier-modulated Gaussian. Its infinite-record spectral magnitude is
proportional to f*exp[-(f/(f0*bandwidth))^2], peaking at
f0*bandwidth/sqrt(2). At the old default f0=2.5GHz, bandwidth0.8, the amplitude
at4.5GHz is only3.32% of that peak. Changing the parameter to3.75GHz increases
that fraction to36.86%; the peak is2.121GHz, not3.75GHz. This analytic calculation
sets the new excitation before fields, without inspecting a Z0 pass/fail target.
All six cases use this one explicit pulse; amplitude1 and cutoff3 are unchanged.
The completed old-pulse points provide a same-geometry control: report any
Z0 shift, never replace the production fitted value with a corrected value.

## Record and quality assessment

Set num_periods=20 per drive (grid frequency 5GHz, approximately 4ns;
not 20 cycles of the differentiated Gaussian with parameter f0=3.75GHz, bandwidth=0.8), 30 frequencies, `enforce_passivity=False`.
Record software/backend/field precision and the extractor's explicit
complex64/float32 casts. Save every raw N-probe voltage and loop current,
result flags, both bracketing-current alternatives, point time records,
DFT plane arrays, per-drive consumed plans, warnings, and elapsed times.
No passivity projection, clipping, empirical impedance correction or new
normalization is permitted. The existing unresolved #726 raw S-power issue
is separate and any observed excess is reported, not repaired in this task.

Before calling a point a usable comparison, require finite in-band results,
both drive settling witnesses <= -40 dB, no own-drive beta scan rails,
and the existing relative-signal screen to pass throughout the band. Record
layout recommendations separately. Independently reconstruct the two-wave
voltage model, reporting relative L2 residual and design-matrix condition;
use the existing 2% model-residual screening level as a rejection screen,
not a 0.4% uncertainty bound. Report the independent float64 best-fit beta/Z0
and its discrepancy from the production scan, without replacing the latter.
A good voltage residual alone does not certify the loop-current scale.

If settling alone fails, retain the first record and rerun that same case
with num_periods=40 once; all other inputs and analysis stay fixed. Any unresolved
validity failure is reported as an inconclusive point, not a pass. Any valid
point beyond 0.4% refutes the all-six numerical agreement hypothesis; no
threshold is relaxed. A negative six-point assessment still completes the
measurement obligation. It does not obligate a new #726 port redesign here.
