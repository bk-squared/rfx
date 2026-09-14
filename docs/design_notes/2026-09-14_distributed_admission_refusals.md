# Distributed admission refusals — closing five silently-wrong paths (B0)

2026-09-14. Branch `agent/distributed-admission-refusals`, on `origin/main`
d56f68eb.

Direction note:
`rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md`
§3–§4, with the simulator survey it rests on
(`decomposition-survey.md` §C1). **That note is the plan; this change is its
first layer** — the position-independent admission check, plus the one
position-dependent slab check. Lane B0 in the note's §6 table, deliberately
placed before lane B: until the silently-wrong paths are closed, every
measurement taken after B is only as honest as the configuration it happened
to use.

## 1. The rule this applies

No warn-and-drop. A feature that the distributed lane cannot realize is
refused, and the message names the feature and says how to proceed (drop the
feature, or omit `devices=...` and run single-device, which supports all
five). The survey's §B3 records what the alternative looks like in practice:
openEMS turns a PML that does not fit a rank into PEC, deactivates a split
probe, and deletes TFSF, each with one line on `cerr` that on rank > 0 is
redirected into a per-rank file nobody reads. That is the precedent this
change refuses to follow.

## 2. What each class did silently, with the numbers

All five were measured on `origin/main` d56f68eb, on 2 virtual CPU devices
(the root `conftest.py` default), with the harness that is now
`tests/unit/runners/test_distributed_admission_refusals.py` — `_build()` for
classes 1–4 and `_asym()` for class 5, **both of which now carry the source
position and the probe row explicitly**, because none of the digits below
survives a change to either. Each one is reachable through the public
`sim.run(devices=[d0, d1])` — not only through the internal runner. **None of
the five raised. None of them warned.**

Two conventions, so the numbers are re-derivable and not just quotable:

- "energy" means `np.sum(trace.astype(np.float64) ** 2)` over the whole probe
  trace. Summing in float32 moves the 7th digit (9.953717e-04 instead of
  9.953718e-04); dropping the second probe moves the 4th (9.951558e-04 and
  1.982875e-03 instead of 9.953718e-04 and 1.983971e-03) — which is why
  `_build` pins the two-probe row at x = 12 and 22 mm.
- no test in this repository re-measures a RED number. They are the
  *provenance* of the refusals, taken out-of-tree on a pristine d56f68eb
  checkout with these fixtures; the GREEN assertions in the test file are the
  refusals themselves. `tests/contracts/test_evidence_numeric_provenance.py`
  classifies this note accordingly.

### 2.1 Periodic / Bloch boundaries — silently non-periodic

`set_periodic_axes('y')` (and `BoundarySpec(y=Boundary('periodic',
'periodic'))`, which sets the same attribute), 24×12×12 mm PEC box at
dx = 1 mm, 40 steps, `amplitude_kind='field'` source at (6, 6, 6) mm, probes
at x = 12 / 22 mm (y = z = 6 mm) — i.e. `_build(periodic='y')` exactly:

| quantity | value |
|---|---|
| `max abs(Ez_distributed − Ez_native)` | 1.963798e-04 |
| native peak `abs(Ez)` | 1.090241e-03 |
| relative | **18.0 % of peak** |
| warnings / errors | 0 / 0 |

Through `sim.run(devices=...)` with the source at the domain centre the same
disagreement is 3.261566e-04 on a 7.734966e-01 peak.

Cause. The distributed local kernels are unconditionally non-periodic —
`rfx/runners/distributed.py:351` ("non-periodic (ghost cells handle
inter-device coupling)"), and the same sentence again at `:380`, `:415`,
`:440`. And `sim._periodic_axes` is read **zero** times in
`rfx/runners/distributed_v2.py`: the string does not occur in the file
(verified by grep on main, 2026-09-14; the survey's §C1-1 recorded two uses
at 1021/1201, which are not on d56f68eb). A periodic axis therefore became an
open, ghost-coupled axis with nothing said. The test
`test_the_runner_still_never_reads_periodic_axes` checks that premise rather
than trusting it, so the refusal has to be re-derived if the runner ever
learns about periodicity.

Bloch. **A placeholder, and labelled as one.** `run()` has no `bloch=`
parameter, and `sim._bloch` is not a `Simulation` attribute at all — the only
`_bloch` in the package is a local variable inside `rfx/simulation.py` (`:890`,
assigned at `:942` from `bloch_phase_tuple`, handed to the step context at
`:1117`), reachable only from oblique TFSF, which falls back to a single
device before the gate runs. So `getattr(sim, "_bloch", None)` in the gate
always yields `None` today and the periodic-axes half is the only reachable
one. `refuse_unsupported_distributed_features(..., bloch=...)` keeps the
parameter and the read so that an explicit phase cannot slip in *behind* the
refusal on the day one is added; no test can exercise this half without
monkeypatching, and the refusal claims **no measured coverage** for it. The
phrase "periodic / Bloch" in this note therefore means "periodic, measured;
Bloch, fenced but unreachable".

### 2.2 Extended lumped port — no source, no termination

`add_port(position=..., impedance=50.0, extent=3e-3)`:

| quantity | value |
|---|---|
| distributed probe energy `sum(Ez²)` | **0.000000e+00** (`max abs(Ez)` = 0.0) |
| native probe energy `sum(Ez²)` | 9.953718e-04 (`max abs(Ez)` = 1.412484e-02) |
| warnings | 0 |

Cause. `rfx/runners/distributed_v2.py` forks on
`pe.impedance > 0.0 and pe.extent is None`, then `elif pe.impedance == 0.0`
(the source/termination fork; its line numbers moved when this change added
the admission gate above it, so it is cited by its code).
A port with a positive impedance **and** an extent satisfies neither, so it
gets no source, no resistive termination and no error.
`rfx/runners/distributed.py:1414-1422` (the pmap runner, which is also the
`n_devices == 1` delegate) has the identical fork.

### 2.3 `excite=False` port — excited anyway

`add_port(impedance=50.0, excite=False, waveform=GaussianPulse(...))`:

| quantity | value |
|---|---|
| distributed energy `sum(Ez²)` | 1.983971e-03 (`max abs(Ez)` = 1.910045e-02) |
| the same port with `excite=True`, distributed | 1.983971e-03 — **bit-identical** |
| native energy `sum(Ez²)` | **0.000000e+00** (`max abs(Ez)` = 0.0) |
| warnings | 0 |

With the documented `waveform=None` default (`add_port` fills a waveform only
when `excite` is true) the lane instead died inside `make_port_source` with
`TypeError: Expected a callable value, got None` — a crash that names no
feature and points at no remedy.

Cause. No port branch in `distributed_v2.py` or `distributed.py` reads
`pe.excite` — on d56f68eb the name `excite` did not occur in either file, and
on this branch its occurrences in `distributed_v2.py` are all inside the
admission gate, which is why the runtime message says "outside this admission
gate" rather than "at all"; `rfx/runners/uniform.py:421,440` honours it (`if pe.excite:` guards both
the wire-port and the lumped-port source). Passive matched loads are how
multi-port S-parameters are extracted, so "every port is excited" is not a
small deviation.

### 2.4 Flux monitors / NTFF box — dropped

| quantity | distributed | native |
|---|---|---|
| `result.flux_monitors` | `None` | 1 monitor, `'flux_x_0'` |
| `result.ntff_data` | `None` | `NTFFData` |
| `result.ntff_box` | `None` | set |
| warnings | 0 | — |

Cause. Neither runner has a flux or NTFF surface-DFT accumulator and neither
has a cross-rank reduce for one (outside the admission gate, `flux` and
`ntff` do not occur in either file). This is the same class as the #579 DFT-plane refusal that the dispatch
block already carries, so it gets the same treatment and says so in the
message.

### 2.5 x-absorber spanning ranks — the CPML window slides into the halo

The distributed CPML windows on a per-rank slab of length
`nx_local = nx_per + 2·ghost` are (`rfx/runners/distributed.py:807-808`):

```
x-lo, rank 0     [ghost, ghost + n)
x-hi, rank N-1   [nx_per + ghost - pad_x - n, nx_per + ghost - pad_x)
```

`pad_x` is the #623 alignment pad, which sits *past* the real x-hi face on the
last rank. A rank owns cells `[ghost, ghost + nx_per)` of its own slab, of
which the **last rank's** trailing `pad_x` are those alignment cells — so its
real cells are `[ghost, ghost + nx_per - pad_x)`. The windows stay inside real
owned cells exactly while

```
n <= nx_per            (x-lo face)
n <= nx_per - pad_x    (x-hi face)
```

That is the condition implemented, per face — not `cpml_layers > nx_per -
ghost`: `ghost` shifts both ends of the x-lo window together and cancels, and
it is `pad_x`, not `ghost`, that eats into the last rank's usable depth.

Measured with the harness's `_asym(8, 6)` — `x_lo='cpml'` / `x_hi='pec'` with
y/z CPML, 6×8×8 mm at dx = 1 mm, `cpml_layers=8`, 2 devices, 60 steps,
`amplitude_kind='field'` Ez source at **(3, 4, 4) mm**, Ez probe row at
**x = 1, 3, 5, 6 mm** (y = z = 4 mm) → `nx=15`, `pad_x=1`, `nx_per=8`, so the
x-hi window wants cells `[0, 8)` of a slab whose real owned cells are
`[1, 8)`:

| quantity | value |
|---|---|
| `max abs(Ez_distributed − Ez_native)` | 2.207244e+00 |
| native peak | 4.416633e+00 (probe x = 3 mm) |
| relative | **49.98 % of peak** |
| probe at x = 5 mm, one cell inside the x-hi PEC face | 1.646758e-01 wrong on its own 1.646768e-01 peak — **99.9994 %** |
| warnings | 0 |
| the same fixture at `nx_per=17` (`_asym(8, 24)`, 24 mm domain) | 1.005828e-06 on a 4.422700e+00 peak = **2.274240e-07** of peak — parity |

Both the source position and the probe row are part of the measurement, which
is why the fixture now states them:

- the source is at x = 3 mm, **not** at `_build`'s default of 6 mm. On a 6 mm
  domain, 6 mm *is* the x-hi PEC plane, where the preflight reports the source
  as silently discarded; that variant is a different and smaller
  silent-wrong (native peak 5.554087e-03 against a distributed peak of
  2.499657e-08, rel 1.000003), and its 24 mm control sits at 3.05e-03 of peak
  — see §5 on the face-pad gap;
- the probe **on** the x-hi face (x = 6 mm) reads exactly 0.0 on both lanes,
  so the "99.9994 %" probe is the one a cell inside it, at x = 5 mm. An
  earlier draft of this note called it "the x-hi-face probe", which it is not,
  and quoted the control as `9.7e-05` of peak, which belongs to no
  reproducible fixture; both are corrected above.

One cell of overflow keeps the window's *length* at `n`, so every array shape
still matches and nothing raises: the absorber simply updates a halo cell
instead of the owned cell it should. Two or more cells of overflow run past
the slab end, the clipped window is shorter than the psi arrays, and XLA
raises a broadcasting error that names no feature. That quote needs its own
fixture, because the 8-layer / 6 mm fixture above cannot produce it: with the
same boundary spec but `cpml_layers=20` and a **7×12×12 mm** domain at
dx = 1 mm (`nx=28`, `pad_x=0`, `nx_per=14`, `ny=nz=53`) main dies with `mul
got incompatible shapes for broadcasting: (20, 1, 1), (15, 53, 53)` — the
`(20, 1, 1)` is the 20-layer psi profile and the `15` is the clipped window
`nx_local − (ghost + pad_x)`. The 8-layer fixtures give `(8, 1, 1), (5, 25,
25)` (6 mm) and `(8, 1, 1), (6, 25, 25)` (8 mm) instead. Both ends of that
band are now refused with one message.

Reachability. With a **symmetric** absorber the violation is unreachable at
2 devices: the absorber is padded *outside* the requested domain
(`rfx/grid.py:153-154`), so `nx = interior + 2·n` and
`nx_per = interior/2 + n > n` holds by construction. It becomes reachable
through per-face composition (an absorber on one x face and a reflector on the
other, which `rfx/grid.py:106-111` supports on purpose) and through higher
device counts, where `nx_per ≈ (interior + 2n)/N`. That is why the fixture is
asymmetric, and why the condition is checked per face.

## 3. The refusals

Two new entry points in `rfx/runners/distributed_v2.py`:

- `refuse_unsupported_distributed_features(sim, *, lane, bloch=None)` —
  classes 1–4, position-independent, `NotImplementedError`. Called from the
  `run()` distributed dispatch block in `rfx/api/_execute.py` (next to the
  #579 DFT refusal, **before** `_warn_unsupported_run_kwargs`, so a refusal
  precedes any drop warning) and again from `run_distributed()` itself, so a
  direct runner call is gated too.
- `check_x_absorber_fits_ranks(*, nx, n_devices, nx_per, pad_x, ghost,
  cpml_layers, pad_x_lo=None, pad_x_hi=None, lane=...)` — class 5,
  position-dependent, `ValueError`. Called inside `run_distributed()` at the
  point where `nx_per` / `pad_x` / `ghost` are known and before the state is
  split into slabs. The message carries `nx`, `n_devices`, `nx_per`, `pad_x`,
  `cpml_layers`, which face overflows and by how much, and — when a smaller
  **multi-device** count exists — the largest one that does fit at this `nx`
  (searched exactly rather than estimated as `nx // cpml_layers`, because
  `pad_x` is not monotone in `N`). The search starts at `N = 2`: `N = 1` fits
  whenever `cpml_layers <= nx`, i.e. for every real grid, so recommending it
  would just be "omit `devices=`" said twice. At `N = 2` the message names the
  depth that would fit instead.

### 3.1 The two call sites must agree, and the order is load-bearing

Placement inside `run_distributed()` is deliberate on both sides:

- **after** the TFSF and waveguide-port fallbacks — those run the whole model
  on one device, which is a right answer rather than a silently wrong one, and
  that single-device lane honours all five features;
- **before** the `n_devices == 1` delegation to the pmap runner — that runner
  carries the same four gaps.

The API-level call therefore has to be skipped for exactly the models the
runner will hand back to `sim.run()`, or it refuses a working call before the
fallback can happen. It is guarded on `self._tfsf is None and not
self._waveguide_ports`.

A first draft of this change had the API call unguarded, and that was a
**regression**, caught in review and measured: with TFSF + a flux monitor
(0.13 × 0.04 × 0.04 m CPML box, `add_tfsf_source(f0=2.5e9, bandwidth=0.5)`,
one centre Ez probe, `add_flux_monitor(axis='x', coordinate=0.09, n_freqs=3)`,
30 steps, 2 virtual CPU devices) `run(devices=...)` on d56f68eb fell back with
one warning and returned `flux_monitors == ['flux_x_0']` at probe peak
1.401931e-12, identical to the native run — while the unguarded gate raised
`NotImplementedError` and `run_distributed()` called directly on the *same*
model still ran. Two public entry points disagreeing about the same
simulation is worse than either answer. That is the standard RCS /
transmission setup (TFSF or a waveguide port plus a flux monitor or an NTFF
box), so it is pinned three ways now:
`test_a_fallback_model_with_a_monitor_still_falls_back_through_run`,
`test_both_entry_points_agree_on_a_fallback_model_with_a_monitor` (both
parametrised over TFSF+flux, TFSF+NTFF, waveguide+flux), and
`test_a_sharding_model_with_a_monitor_is_still_refused_at_the_door`, which
holds the refusal for models the runner really will shard. The three shipped
fallback tests in `tests/unit/runners/test_distributed.py` carry point probes
only, which is why they could not see it.

**Known inconsistency, left alone.** The pre-existing #579 DFT-plane refusal
at the top of the same dispatch block has the same ordering: a TFSF model
carrying a DFT plane is refused at the door even though the fallback would
have answered it. That is pre-existing `main` behaviour for a *different*
issue's refusal, it is an over-refusal rather than a silent wrong, and
changing it is #579's decision, not B0's. Recorded here so the asymmetry is
on purpose rather than forgotten.

Existing refusals are untouched: #579 DFT planes, `interface_eps='dual_average'`,
the #931 PEC-geometry refusal, `refuse_f0_sheets`, `boundary='upml'`, and the
NU-lane Phase-B refusals. Two of them are re-pinned here
(`test_the_579_dft_plane_refusal_is_untouched`,
`test_the_dual_average_refusal_is_untouched`) so this change cannot shadow
them.

## 4. One ghost-width formula

`rfx/api/_execute.py` (the NU-forward distributed preflight, check 3) computed
`ghost_width = floor(exchange_interval / 2) + 1`, while
`rfx/runners/distributed_nu.build_sharded_nu_grid` shards with
`ghost = exchange_interval`:

| K | `floor(K/2)+1` | builder `g = K` | difference |
|---|---|---|---|
| 1 | 1 | 1 | 0 |
| 2 | 2 | 2 | 0 |
| 3 | 2 | 3 | **−1** |
| 4 | 3 | 4 | **−1** |

The preflight was short by one cell from K = 3 up, i.e. it cleared
configurations the builder cannot shard. Measured on main with `nx=4` over
2 ranks (`nx_per_rank=2`): at K = 3 check 3 computed `ghost_width=2`, `2 > 2`
is false, the check **passed**, and the call fell through to the builder's own
"exchange_interval > 1 is reserved for Phase 2E"; at K = 4 it raised but named
3 where the builder needs 4.

Both sites now read `rfx.runners.distributed_nu.nu_ghost_width(K)` — one
source of truth, with the derivation in its docstring (a halo of `g` cells
keeps an owned slab correct for `K` steps between exchanges, so `g = K`; the
survey reaches the same `g = K` from the stencil side). The agreement is
tested for K = 1..4, the builder's own `ghost_width` is compared against the
helper at K = 1 (the only interval it accepts today), and a guard asserts the
preflight has not grown a second local formula.

## 5. What is deliberately NOT changed

- **`exchange_interval > 1` warning path.** `run_distributed()` still warns
  (rather than refusing) that ghost cells are stale for `K−1` steps and the
  boundary error is O(K·dt). That is a *quantified accuracy statement about a
  knob the caller turned on purpose*, not a dropped feature, and K > 1 needs
  an algorithm change (a full-state K-cell overlap) before a refusal would be
  the right shape. The direction note puts it last (§5, "알고리즘 변경이
  필요한 것") and the survey measures the upside at 1.03–1.15× — low priority.
  On the NU lane `build_sharded_nu_grid` already refuses K > 1 outright, and
  that refusal stays.
- **TFSF and waveguide-port fallbacks to a single device.** These are the
  third option in the survey's §B3 taxonomy (prohibit / move the cut /
  gather), and the one rfx already chose: the model runs, on one device, with
  a warning that says so. They are pinned by
  `test_the_tfsf_and_waveguide_fallbacks_are_deliberately_unchanged` and by
  the three monitor-carrying tests of §3.1, so the admission gate cannot
  quietly swallow them through either entry point. Whether they should become
  refusals is a separate decision with a separate cost (they work today).
- **The cut-plane census and automatic cut-plane moving** (§4 (2)(3) of the
  direction note, lane B2) and **the grade map** (§4 (4)). Nothing here
  inspects *where* the cut falls relative to a dielectric interface, a PEC
  cell, a dispersive pole or a source index. B0 is position-independent by
  construction, with class 5 the single exception, because the absorber's
  position is fixed by the frame.
- **The x-hi CPML window on a PEC x-hi face (the S5 gap).** The runner applies
  both x-face windows whenever `sim._boundary == "cpml"` and
  `grid.cpml_layers > 0`, without consulting `grid.face_pads` — so the
  asymmetric fixture in §2.5 gets an x-hi absorber correction at a reflector
  face. That is a *different* gap, it is not in the direction note's §3 list,
  and closing it changes numbers rather than closing a silent path. Left for
  lane B, where the distributed CPML outer termination is the subject. Two
  things for lane B to carry over:

  - **How big it is.** The fitting control for the asymmetric model is
    fixture-sensitive, and larger than a first draft of this note claimed. At
    `_asym(8, 24)` with the measuring fixture of §2.5 it is 2.274240e-07 of
    peak — parity. But with `_build`'s default centre source at x = 6 mm the
    same 24 mm control sits at 9.653624e-06 on a 3.165971e-03 peak =
    **3.05e-03 of peak**, i.e. *above* the shipped CPML parity tolerance of
    1e-3 (`tests/unit/runners/test_distributed.py:291`). Lane B should size
    the fix against 3e-03, not against 1e-04.
  - **It is why the class-5 condition is not narrowed.** `x='pec'` with y/z
    CPML declares no x absorber at all, yet `check_x_absorber_fits_ranks()`
    still fires — and it *should*: on pristine d56f68eb that configuration died
    inside XLA (`mul got incompatible shapes for broadcasting: (8, 1, 1),
    (5, 25, 25)` at 6×8×8 mm; `(8, 1, 1), (6, 25, 25)` at 8×8×8 mm), because
    the runner drives an 8-layer x window into a slab with no x pad. So the
    refusal is not an over-refusal of a working case. What *was* wrong is the
    message: it told a caller who asked for no x absorber to shrink one. It
    now reads `grid.pad_x_lo` / `grid.pad_x_hi` for that purpose only and says
    which x faces the caller left non-absorbing and that the lane adds the
    windows anyway (`test_pec_x_faces_are_still_refused_but_the_message_says_why`).
    Once lane B gates the windows on `face_pads`, this condition should be
    narrowed to the faces that really are CPML.

## 6. Tests

`tests/unit/runners/test_distributed_admission_refusals.py`, 40 tests. For each
of the five classes: the refusal through `sim.run(devices=...)`, the refusal
inside `run_distributed()`, and an assertion on the message content (the
feature name, the cause, the way out). Class 5 additionally gets a
simulation-free unit test of the exact window boundary — `n == nx_per - pad_x`
admitted, `n + 1` refused, for `pad_x ∈ {0, 1, 3}`.

Negative controls, with the fixtures and tolerances of the shipped parity
tests copied unchanged (`tests/unit/runners/test_distributed.py:149` PEC,
rel < 1e-4; `:291` CPML, rel < 1e-3): the default configuration still runs
distributed and still matches the single-device lane; a plain single-cell
excited port still injects; a fitting x-absorber is still admitted; the
admission gate is a no-op on a model that declares none of the five. No
tolerance anywhere in the repository was weakened for this change.

## 7. Where this sits

This is layer (1) of the direction note's §4 — "입장 검사 (위치 무관)" — and
nothing more. Layers (2) cut-plane census, (3) automatic cut-plane moving and
(4) the grade map are lanes B2 and C. The note's PI decisions E1 (B0 before B)
and E2 (all five as errors, no warn-and-drop) are what this change assumes;
E3 (automatic cut-plane moving vs always refusing) is untouched here because
nothing here moves a cut.
