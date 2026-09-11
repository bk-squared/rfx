# AD fixture repair: substrate planes and source coverage

Plan recorded before the new live solve, 2026-09-11. This is a repair of
the existing uniform-runner AD acceptance fixture, not a new RF accuracy
claim or a change to its accepted derivative error.

The legacy fixture declares a 254 um port interval but places its trace at
320 um. Its default f0=2.5 GHz Gaussian pulse also leaves the 5 GHz objective
bin below the existing 10%-of-band-median low-signal screen. Live returned
records marked that bin unreliable. The old objective still averaged it.

The repaired fixture keeps the 254 um nominal laminate, 600 um nominal
trace, 6 mm launch separation, 2 mm margins, 5 GHz maximum frequency, eight
objective frequencies, 20 periods and 50 ohm source/load parameters. It
uses dx=254/3 um and a zero-thickness trace at 254 um, so the source's three
normal edges reach the actual trace. Width rasterization remains a coarse
grid approximation; this AD test is not an absolute line-impedance oracle.

An explicit GaussianPulse(f0=5 GHz, bandwidth=0.8) covers the full existing
frequency band, following the already-qualified excitation precondition
in `test_msl_sparam_ad.py`. A sampled finite-record spectrum test rejects
the old pulse and accepts the new one against the same 10% screen. That
input check alone does not certify resulting port data.

Before differentiation, the actual forward result must have every fixed
objective bin reliable at both ports, both drive records settled below
-40 dB, and the existing nonzero/plausible-S checks satisfied. Both FD legs
must meet the same reliability and settling prerequisites. No bins are
selected or discarded as a function of the design parameter.

The original design variable is retained: eps_override is a spatially
uniform alpha array, evaluated around alpha=1, while the static launch and
reference model uses the registered materials. This follows the existing
#483 frozen-launch contract; it does not claim the overridden DUT is the
nominal RO4350B board. Field precision is float32 for AD and explicitly
float64 for FD. The initial comparison keeps h=1e-3 and the 3% gate.
Should the reference require a smaller step, its convergence must be
demonstrated independently before changing the fixed test setting.

Pre-solve checks: six fast tests passed, including actual E/H dtype capture,
realized trace plane agreement, and sampled source coverage. Live acceptance
is pending at this checkpoint. Legacy geometry, weak-source and step-size
records remain in the adjacent run archives; they do not qualify this rig.

## Measurement after the predeclaration

Run 369367260436 (source b4b0893e) passed with the original h=1e-3 and
3% gate: AD–FD difference about 0.058%, all objective bins above the
wave-split low-signal screen, and both drives settled. See
[the measured diagnosis](ad-referee-diagnosis.md#repaired-fixture-measurement)
and the retained run directory. The same run passed the existing NU patch
gate afterward; no tolerance was widened.
