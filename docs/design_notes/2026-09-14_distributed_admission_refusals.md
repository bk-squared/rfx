# Distributed admission refusals — closing seven silently-wrong paths (B0)

2026-09-14. Branch `agent/distributed-admission-refusals`, on `origin/main`
d56f68eb.

Direction note:
`rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md`
§3–§4, with the simulator survey it rests on
(`decomposition-survey.md` §C1). **That note is the plan; this change is its
first layer** — the position-independent admission check, plus the two
position-dependent slab checks.

**Round 2 (2026-09-14, after review).** The note's §3 list has five entries;
this change now refuses **seven** silently-wrong paths, because measurement
found two more while checking the five:

- **class 6**, the *phantom x CPML window*: the lane drives both x-face
  absorber windows whenever `boundary='cpml'` and `cpml_layers > 0`, without
  reading `grid.face_pads`, so an x face declared `pec`/`pmc` gets absorbed
  at. The first draft of this note filed that as "S5", a numbers-only gap
  for lane B, and sized it at 3.05e-03 of peak. Both were wrong: it is
  5.09e-02 of peak on the committed fixture, up to **100 % of a probe's own
  peak** next to the face, and it is a silent path with 0 warnings — the
  same class as the five. §2.6.
- **class 7**, the *ungated exported v1 runner*: `rfx.runners.run_distributed`
  is the pmap runner in `rfx/runners/distributed.py`, not `distributed_v2`,
  and the first round gated only `distributed_v2` and the `run(devices=...)`
  dispatch. All five classes rode straight through the exported name. §2.7.

This is lane B0 in the direction note's §6 table, deliberately placed before
lane B: until the silently-wrong paths are closed, every measurement taken
after B is only as honest as the configuration it happened to use.

## 1. The rule this applies

No warn-and-drop. A feature that the distributed lane cannot realize is
refused, and the message names the feature and says how to proceed (drop the
feature, or omit `devices=...` and run single-device, which supports all
seven). The survey's §B3 records what the alternative looks like in practice:
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
| the same fixture at `nx_per=17` (`_asym(8, 24)`, 24 mm domain) | 1.005828e-06 on a 4.422700e+00 peak = **2.274240e-07** of peak — parity at this probe row |

**The window arithmetic is necessary, not sufficient — and it is not what
produces the 49.98 %.** This is the round-2 correction, and it is a
measurement, not a re-reading. Re-measured on pristine d56f68eb, 2 CPU
devices, 60 steps:

- the case that satisfies the bound **exactly**, `n == nx_per - pad_x` — the
  same ASYM spec at 7×8×8 mm, `cpml_layers=8` (nx = 16, pad_x = 0,
  nx_per = 8), same source and probe row — is **46.16 % wrong** at the
  source probe (native 4.420052e+00 vs distributed 6.460177e+00, max|dEz|
  2.040125e+00), **99.97 %** wrong at x = 5 mm (1.800406e-01 vs
  5.905863e-05) and **100.0 %** at x = 6 mm (4.441055e-02 vs 1.295477e-07),
  with 0 warnings. The same order as the refused 6 mm case;
- the `_asym(8, 24)` "parity" control reads 2.27e-07 only because its probe
  row sits 17+ cells from the x-hi face and the diff is normalised by the
  source-probe peak. Adding probes at x = 20 / 22 / 23 mm to the *same*
  fixture: native 1.937526e-04 / 8.516839e-05 / 4.505432e-05 against
  distributed 1.139809e-04 / 5.447935e-07 / 1.325814e-09 — **41.2 % /
  99.4 % / 100.0 %** wrong on their own peaks, 0 warnings. With the source
  moved to x = 20 mm the probes at 3 mm and 12 mm are 185.5 % and 164.0 %
  off;
- the decisive one: the 24 mm asymmetric model is wrong by 1.610351e-04 on
  a 3.165971e-03 peak (5.086e-02) at `n_devices=2` **and by the identical
  1.610351e-04 / 5.086e-02 at `n_devices=1`**, where there is no slab to
  overflow. The defect does not need a decomposition.

So the dominant mechanism is not ghost-halo overflow. It is the phantom
x-hi window: `ce_xhi = dt / (eps_r * EPS_0)` is non-zero at a PEC face, so
the lane absorbs at the reflector (`rfx/runners/distributed.py`, the
`ce_xhi` and `xhi` slices). The class-5 refusal message no longer attributes
the 49.98 % to the overflow alone, the check's docstring states the bound as
necessary-only, and the phantom window is refused separately as class 6
(§2.6). `test_the_x_absorber_condition_is_the_window_arithmetic` says
"necessary condition only" in its docstring, and
`test_a_fitting_x_absorber_is_not_refused` no longer uses `_asym(8, 24)` as
its control — a symmetric absorber is the only configuration that really is
admitted, and it is at 1.836e-06 of peak.

Both the source position and the probe row are part of the measurement, which
is why the fixture now states them:

- the source is at x = 3 mm, **not** at `_build`'s default of 6 mm. On a 6 mm
  domain, 6 mm *is* the x-hi PEC plane, where the preflight reports the source
  as silently discarded; that variant is a different silent-wrong (native
  peak 5.554087e-03 against a distributed peak of 2.499657e-08, rel
  1.000003). Its 24 mm counterpart — same ASYM spec, `cpml_layers=8`,
  24×8×8 mm, `_build`'s default source (6, 6, 6) mm and default probe row
  x = 12/22 mm — is **1.610351e-04 on a 3.165971e-03 row peak = 5.086e-02 of
  peak**, with the x = 22 mm probe 99.46 % wrong on its own peak; an earlier
  draft of this note quoted that case as 3.05e-03, which is the x = 12 mm
  probe alone. This is class 6, not a tolerance band — §2.6;
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

### 2.6 Phantom x CPML window at a non-absorbing x face — absorbing at a reflector

The runner applies **both** x-face CPML windows whenever
`sim._boundary == "cpml"` and `grid.cpml_layers > 0`. It never consults
`grid.face_pads`, and its per-face E coefficient is
`ce_xhi = dt / (eps_r * EPS_0)` — non-zero at a PEC or PMC face. So a face
whose `BoundarySpec` token is `pec`/`pmc` (where `rfx/grid.py`'s `_face_pad`
gives `pad = 0`, i.e. no absorber was padded outside the domain), or an x
axis left out of `cpml_axes`, gets a `cpml_layers`-deep absorber driven into
it. The reflection the caller asked for is absorbed instead.

Measured on pristine d56f68eb, 2 virtual CPU devices, **0 warnings every
time**:

| fixture | steps | max abs(dEz) | native row peak | relative |
|---|---|---|---|---|
| `x=(cpml,pec)`, y/z cpml, `cpml_layers=8`, 24×8×8 mm, source (6,6,6) mm, probes x = 12/22 mm | 60 | 1.610351e-04 | 3.165971e-03 | **5.086e-02** |
| the same, x = 22 mm probe on its **own** peak | 60 | 1.610351e-04 | 1.619078e-04 | **99.46 %** |
| the same fixture | 100 | 3.955191e-04 | 5.013014e-03 | **7.890e-02** |
| the same fixture at `n_devices=1` | 60 | 1.610351e-04 | 3.165971e-03 | **5.086e-02** (identical) |
| `x=(pec,pec)`, y/z cpml, `cpml_layers=8`, nx = 16 / 20 / 40 | 60 | — | — | 2.74 % / 3.06 % / 3.07 % at the source, **99.2–100.0 %** at every probe inside the phantom window |
| `x=(pmc,cpml)`, y/z cpml, dx = 5 mm, 16×8×8 cells | 30 | 1.350925e-01 | 2.498208e-01 | **54.08 %** |
| the same | 80 | 1.021969e+00 | 1.093201e+00 | **93.48 %** |

Negative control, both x faces absorbing — `boundary='cpml'`,
`cpml_layers=8`, 24×8×8 mm, probes 12/22 mm, 60 steps: 5.820766e-09 on a
3.170117e-03 peak = **1.836e-06** of peak at 2 devices, 1.395e-06 at 1
device. Parity, three orders inside the shipped CPML tolerance of 1e-3.
**The refusal costs the symmetric absorber nothing** — and symmetric is the
common case, because `boundary='cpml'` pads both x faces outside the
requested domain by construction.

Two things follow.

- **The `n_devices=1` row is the proof of mechanism.** The wrongness is
  bit-for-bit identical with and without a decomposition, so this is the
  window and not the sharding. It is why class 6 is a separate refusal and
  not a tightening of class 5, and why the v1 runner's copy of the check
  fires at one device too (§2.7).
- **One shipped test was running on it.**
  `tests/unit/boundaries/test_boundary_pmc_distributed.py::test_pmc_distributed_v2_x_lo_owner_and_non_owner`
  used `x=Boundary(lo='pmc', hi='cpml')` with y/z CPML at 30 steps — the
  54.08 % row above. It stayed green because it asserts a zero-**pattern**
  on the PMC face (`hy[0]`, `hz[0]`) and a non-zero off it, never a value.
  Those assertions are equally valid on a reflector-only composition, so the
  test now composes its PMC face with PEC (`x=(pmc,pec)`, `y=z='pec'`): the
  x-slab decomposition, the owning / non-owning rank split and the
  `_apply_pmc_local` hook under test are unchanged, and the configuration is
  one this lane can actually realize. Nothing was deleted and no tolerance
  moved.

**This is an admission refusal, not the fix.** The fix is to gate the x
windows on `grid.face_pads`, which changes physics rather than admission and
is lane B's subject (the distributed CPML outer termination). Until it
lands, refusing is the honest answer; when it lands,
`check_x_absorber_faces_are_absorbing` should be deleted and class 5
narrowed to the faces that really are CPML.

### 2.7 The exported v1 pmap runner was ungated for all of 1–6

`rfx/runners/__init__.py` re-exports `run_distributed` from
**`rfx.runners.distributed`** — the pmap runner — not from `distributed_v2`.
The first round of this change put the gate in `distributed_v2.run_distributed`
and in the `run(devices=...)` dispatch, and `rfx.runners.run_distributed`
kept running every class. The runner's only new-ish guard was its
`nx % n_devices != 0` ValueError, which is the accident that made the 24 mm
fixtures bounce and the gap look closed.

Measured on the first-round tree with the committed `_build` fixture at
**23×12×12 mm** (nx = 24, evenly divisible, so the divisibility error cannot
bounce the call), 40 steps, 2 virtual CPU devices, **0 warnings every
time**:

| class | pmap runner | native |
|---|---|---|
| extended lumped port | probe energy 0.000000e+00 | 9.951533e-04 |
| `excite=False` port | 1.982867e-03 | 0.0 |
| flux monitor | `result.flux_monitors is None` | one monitor |
| periodic `'y'` | max abs(dEz) 1.963800e-04 on a 1.090239e-03 peak — 18 % | — |
| x absorber spanning ranks (`x=(cpml,pec)`, `cpml_layers=8`, 5×8×8 mm, nx = 14, nx_per = 7) | 2.207114e+00 on a 4.416774e+00 peak — 50 % | — |

The v2 docstring already acknowledged that "the pmap runner carries the same
four gaps", but it only used that to justify gating *before* the
`n_devices == 1` delegation — the package-level name still ran them. Fixed
by calling the same two entry points from `rfx/runners/distributed.py`:
`refuse_unsupported_distributed_features(sim, lane='distributed (v1) pmap
runner')` after the TFSF / waveguide fallbacks and after `_refuse_f0`, and
`check_x_absorber_fits_ranks(...)` plus
`check_x_absorber_faces_are_absorbing(...)` once `use_cpml` is known
(`pad_x=0` literally, because this runner requires `nx % n_devices == 0`).
Pinned by nine tests, including one that asserts the export really is the
pmap module — if that export ever moves to `distributed_v2` the other tests
would still pass while testing nothing.

## 3. The refusals

Three new entry points in `rfx/runners/distributed_v2.py`, called from
**three** lanes — the `run(devices=...)` dispatch, `distributed_v2.run_distributed()`
and (round 2) `distributed.run_distributed()`, the exported pmap runner:

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

  The bound is **necessary, not sufficient** — see §2.5. It stays as the
  arithmetic guard it is (past it the window slides into the halo, or two
  cells past it XLA dies on a broadcast that names no feature), and the
  docstring says so rather than implying that passing it means the run is
  right.
- `check_x_absorber_faces_are_absorbing(*, cpml_layers, pad_x_lo, pad_x_hi,
  n_devices, lane=...)` — **class 6**, `ValueError`, round 2. Fires when
  `grid.pad_x_lo == 0` or `grid.pad_x_hi == 0` while the lane is building
  CPML windows. Called immediately after `check_x_absorber_fits_ranks` in
  both runners, deliberately **after** it: when both apply (the asymmetric
  fixture at a too-small `nx_per`) the arithmetic message is the more
  specific one and it is the one the class-5 tests pin. The message names
  the face, the depth, both pads, and three ways forward — make both x faces
  absorbing, drop the absorber entirely, or omit `devices=...`. It fires at
  `n_devices == 1` too, and must: §2.6's `n_devices=1` row is identical to
  its 2-device row.

Round 2 also added the gate to `rfx/runners/distributed.py` (§2.7), which is
what `rfx.runners.run_distributed` actually names: the feature gate after the
two fallbacks and after `_refuse_f0`, the two slab checks once `use_cpml` is
known, with `pad_x=0` passed literally because that runner refuses an nx that
is not evenly divisible.

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
- **The FIX for the phantom x window (what was "S5").** Refusing it is
  round 2's answer (§2.6, class 6); making it *work* is not. Gating the x
  windows on `grid.face_pads` so that a `pec`/`pmc` x face gets no absorber
  correction is a physics change in the distributed CPML outer termination,
  i.e. lane B's subject, and it is what lets class 6 be deleted and class 5
  narrowed to the faces that really are CPML. What lane B needs to carry
  over is the **size**, and the first draft of this note got it wrong twice
  — both corrected here by measurement:

  - the draft said "with `_build`'s default centre source at x = 6 mm the
    same 24 mm control sits at 9.653624e-06 on a 3.165971e-03 peak =
    **3.05e-03** of peak … Lane B should size the fix against 3e-03". The
    digit is real but it is the **wrong probe**: re-measured on pristine
    d56f68eb with the committed fixture (`_build(boundary=ASYM_SPEC,
    cpml_layers=8, domain=(24e-3, 8e-3, 8e-3))`, 2 devices, 60 steps), the
    x = 12 mm probe's diff is 9.653624e-06 — that is the draft's number —
    but the x = 22 mm probe's diff is **1.610351e-04 on its own peak of
    1.619078e-04 = 99.46 %**, so the max over `_build`'s default probe row
    is 1.610351e-04 on the 3.165971e-03 row peak = **5.086e-02 of peak**.
    9.653624e-06 / 3.049e-03 reproduces **only** with `probes=(12e-3,)`, a
    single probe, which is not `_build`'s default for a 24 mm domain. Under
    this note's own max-over-the-probe-row convention — the one every other
    RED number uses, including the 2.207244 class-5 figure — the gap is
    **17× larger than stated**, and it grows with time: 2.62e-02 at 40
    steps, 5.09e-02 at 60, 7.89e-02 at 100;
  - and "closing it changes numbers rather than closing a silent path" was
    false. It is a silent path, with 0 warnings and ~100 % error at the
    probe one absorber-window inside the reflector face. **Lane B should
    size the fix against 5e-02, not 3e-03**, and treat the face-adjacent
    cells as totally corrupted rather than as a tolerance band. The
    provenance comment in
    `tests/contracts/test_evidence_numeric_provenance.py` is corrected to
    match: "re-derivable by running those fixtures" now means the fixtures
    *as committed*, and the note states the probe row next to any number
    that depends on it.

  This is the same fixture-sensitivity failure mode round 1 flagged for the
  `9.7e-05` digit, caught a second time on a number that had already been
  through one correction. The lesson is in §6: a quoted relative figure
  needs its probe row printed next to it, not just its fixture name.

## 6. Tests

`tests/unit/runners/test_distributed_admission_refusals.py`, 64 tests (40
after round 1). For each of the five original classes: the refusal through
`sim.run(devices=...)`, the refusal inside `run_distributed()`, and an
assertion on the message content (the feature name, the cause, the way out).
Class 5 additionally gets a simulation-free unit test of the exact window
boundary — `n == nx_per - pad_x` admitted, `n + 1` refused, for
`pad_x ∈ {0, 1, 3}` — whose docstring now records that the bound is
necessary only, with the 46.16 % measurement of the admitted case.

Round 2 added, and each one is a fact the first round left unpinned:

- **class 6** (§2.6): the refusal through both entry points, the message
  content (face, both pads, the three ways out), a PMC variant, a
  simulation-free unit test of the `pad == 0` boundary including the
  "different per-face thickness, both absorbing" admit, and the proof that
  `boundary='pec'` never reaches the check;
- **the v1 pmap runner** (§2.7): five parametrised feature refusals at
  nx = 24, a lane-name assertion, the two slab checks, the class-6 refusal
  at `n_devices == 1`, a symmetric-absorber parity control through the same
  call at the shipped 1e-3, and one test asserting that
  `rfx.runners.run_distributed.__module__` really is `rfx.runners.distributed`
  — without it the other eight would keep passing while testing the wrong
  runner if the export ever moved;
- **the DEFAULT preflight path.** Every other `_run_api` call in the file
  passes `skip_preflight=True`, which pinned nothing about
  `sim.run(devices=...)` as a user calls it. Measured on pristine d56f68eb
  through the default path: all seven classes RAN — periodic peak
  8.938612e-04 with 0 warnings, extent port peak exactly 0.0 with 0
  warnings, `excite=False` peak 1.910045e-02 with 0 warnings, flux and NTFF
  returning `None` (the NTFF case with one advisory about PEC + far-field,
  which says nothing about the drop), `_asym(8, 6)` with one advisory about
  a probe near the absorber. Now a parametrised test over all seven runs
  without `skip_preflight`;
- **the fractional `exchange_interval`**: `nu_ghost_width(2.5)` used to
  truncate to 2 — a halo half a cell short of the interval it is sized for,
  consistent across both call sites and therefore invisible to any
  disagreement test. Refused now.

Negative controls, with the fixtures and tolerances of the shipped parity
tests copied unchanged (`tests/unit/runners/test_distributed.py` PEC,
rel < 1e-4; CPML, rel < 1e-3): the default configuration still runs
distributed and still matches the single-device lane; a plain single-cell
excited port still injects; a **symmetric** x-absorber is still admitted and
at parity on both runners; the admission gate is a no-op on a model that
declares none of the classes. The class-5 control changed fixture in round 2
— `_asym(8, 24)` "fits the slab" but is 5.09e-02 of peak wrong, so it is now
a class-6 RED case and the control is a symmetric absorber, which is the
only configuration that really is admitted. **No tolerance anywhere in the
repository was weakened for this change**, in either round.

One process note, because it is the second time the same thing happened. A
relative figure quoted without its probe row is not reproducible even when
the fixture name is given: `9.7e-05` (caught in round 1) and `3.05e-03`
(caught in round 2) were both max-over-one-probe numbers presented against a
row-max convention. Every number in this note that depends on a probe row
now prints the row.

## 7. Where this sits

This is layer (1) of the direction note's §4 — "입장 검사 (위치 무관)" — and
nothing more. Round 2 did not change that: classes 6 and 7 are the same
layer, found by measuring the five rather than by moving to layer (2).
Class 6 is grid-dependent (it reads `grid.face_pads`) but not cut-dependent;
class 7 is the same gate on a lane that had been missed. Layers (2) cut-plane census, (3) automatic cut-plane moving and
(4) the grade map are lanes B2 and C. The note's PI decisions E1 (B0 before B)
and E2 (all five as errors, no warn-and-drop) are what this change assumes;
E3 (automatic cut-plane moving vs always refusing) is untouched here because
nothing here moves a cut.

## 8. Round-2 review record — checked and NOT changed

Recorded so that "unmentioned" is not read as "unexamined".

- **The ghost-width unification (§4) is not a silent-wrong, and nothing here
  says it is.** Reproduced on d56f68eb, nx = 4 over 2 ranks: at K = 3 the
  old `floor(K/2)+1` cleared the check and the call fell through to
  `build_sharded_nu_grid`'s own `exchange_interval > 1 is reserved for
  Phase 2E` `NotImplementedError` — nothing ran wrong; at K = 4 it raised
  `ghost_width=3 exceeds nx_per_rank=2`. After the change K = 3 becomes a
  `ValueError` from check 3 instead. It is a consistency fix that removes a
  second formula, and the `rfx/api/_execute.py` comment beside it is
  accurate as written.
- **Check 3 is NOT made to defer to the builder's K > 1 refusal.** The
  suggestion was that the two messages now differ by `nx` (K = 2, 3 reach
  the Phase-2E message at small `nx`, K = 4 reaches check 3's), so check 3
  could hand the caller the reason that actually blocks them. Declined:
  `build_sharded_nu_grid`'s K > 1 refusal is a *temporary* Phase-2E
  placeholder, and check 3 is the only ghost-width guard that survives its
  removal. Making the gate defer to a refusal that is scheduled to go away
  would weaken it the day K > 1 lands, which is exactly when the ghost width
  starts mattering. Both paths raise today; no working configuration is
  lost either way.
- **The `ghost > 1` branch of the class-5 inequality has no simulation
  coverage.** `run_distributed` hardcodes `ghost = 1` (and so does the
  kernel), so the inequality is verified end to end only there. It *is*
  verified by brute force against the literal window slices — `nx_per`
  1..13, `ghost ∈ {1, 2}`, `pad_x` 0..3, `n` 1..19: 0 mismatches, one-cell
  overflow keeps length `n`, two-cell clips to `nx_per + ghost - pad_x`
  (e.g. 15 at nx = 28) — and the call sites already forward `ghost`, so
  there is nothing to change now. Recorded in the check's docstring so the
  gap is not rediscovered as a surprise.
- **S-parameter sweeps cannot collide with the passive-port refusal.**
  `rfx/api/_sparams.py` rewrites `self._ports` with `excite=False` on
  undriven ports during the multi-port drive loop, which looks like it would
  trip class 3. It cannot: that code path takes no `devices=` argument
  (`grep -c devices rfx/api/_sparams.py` → 0), so it never reaches a
  distributed lane.
- **`run(devices=[one_device])` is unchanged.** `rfx/api/_execute.py`
  dispatches distributed only for `len(devices) > 1`, so the single-device
  public path still runs the uniform lane and still returns
  `flux_monitors == ['flux_x_0']`. The single-device *runner* call
  (`rfx.runners.run_distributed(sim, devices=[d0])`) is now refused for a
  flux monitor, where on d56f68eb it ran and returned `flux_monitors=None`.
  That is a correct refusal of a genuinely silent drop, and it is reachable
  only through the runner name.
- **The NU-forward preflight's check 4 is left alone.**
  `cpml_layers*2 >= nx_local_real` is a different condition from
  `check_x_absorber_fits_ranks` — stricter for a rank owning both outer
  faces, and silent about a face that declares no absorber at all. Changing
  it is the NU lane's decision, not B0's; the two lanes therefore state the
  x-absorber fit differently, which is now a comment at the check and an
  open item here.

### 8.1 Still open — a class outside B0's scope, measured

**A source whose x cell is the first real cell of rank 1** (global index
`== nx_per`). Reproduced on pristine main and unchanged on this branch —
symmetric `boundary='cpml'`, `cpml_layers=8`, 2 devices, 60 steps, **0
warnings**:

| fixture | probe | native | distributed | rel |
|---|---|---|---|---|
| 4×8×8 mm (nx = 21, nx_per = 11), source x = 3 mm = global 11 | source | 4.422122e+00 | 4.557471e+00 | 3.073e-02 |
| the same | one cell before | 8.203998e-01 | 6.845230e-01 | — |
| 24×8×8 mm (nx = 41, nx_per = 21), source x = 13 mm = global 21 | source | 4.422516e+00 | 4.558136e+00 | 3.067e-02 |

Moving the source **one cell earlier**, onto the last cell of rank 0,
restores 1.08e-07 parity. That is 30× the shipped CPML tolerance of 1e-3
(`tests/unit/runners/test_distributed.py`). It is position-dependent **in
the source**, not in the absorber, so it belongs to the cut-plane census
(lane B2) and not to an admission check — B0 is position-independent by
construction, with the two absorber checks the stated exceptions. It is
recorded here and in the test module's docstring so that B0's
"position-independent" scope statement is not read as "no other silent class
remains". It does not.
