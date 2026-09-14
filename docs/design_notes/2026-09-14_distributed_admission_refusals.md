# Distributed admission refusals — closing seven silently-wrong paths (B0)

2026-09-14. Branch `agent/distributed-admission-refusals`, rebased onto
`origin/main` **7b511591**. The round-1 and round-2 RED numbers were measured
on d56f68eb, which is identical to 7b511591 in every runtime file this lane
touches; round-3 numbers are measured on 7b511591 and say so beside each
figure.

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

**Round 3 (2026-09-14, after review).** One of the seven was **over-refusing**
and is narrowed by measurement: class 5 refused a one-cell window overflow
that `origin/main` runs at parity (byte-identically to the fitting run), so
its bound moves out by one cell on each face. §2.5 carries the measurement and
the mechanism. Three message/attribution defects found in the same round are
fixed with it: the class-5 message no longer asserts "died inside XLA" for a
configuration that ran, no longer says "an absorber on neither" when one x
face is CPML, and the port-fork citation is by symbol instead of a line number
this very branch moved by 62 lines.

### What this branch actually ships

Named here because a reader of the PR description should not have to diff to
find it:

- `refuse_unsupported_distributed_features()` (classes 1–4) at the two runner
  entry points and the `run(devices=...)` dispatch;
- `check_x_absorber_fits_ranks()` (class 5) and
  `check_x_absorber_faces_are_absorbing()` (class 6) inside both runners;
- the same six-class gate in the **v1 pmap runner** `rfx/runners/distributed.py`
  (class 7), which is the exported `rfx.runners.run_distributed`;
- one ghost-width formula, `rfx.runners.distributed_nu.nu_ghost_width` (§4);
- **a rewrite of one pre-existing test.**
  `tests/unit/boundaries/test_boundary_pmc_distributed.py`'s
  `test_pmc_distributed_v2_x_lo_owner_and_non_owner` moved from
  `x=Boundary(lo='pmc', hi='cpml')` with `y=z='cpml'` to
  `x=Boundary(lo='pmc', hi='pec')` with `y=z='pec'`, because class 6 refuses
  the configuration it used to run. It was running on a 54.1 %-wrong
  configuration and asserting only a zero-pattern, so nothing it claimed is
  lost — but a changed shipped test must be declared, not discovered. §2.6.

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
classes 1–4, `_asym()` for class 6 and `_asym_deep()` for class 5, **all of
which carry the source position and the probe row explicitly**, because none of the digits below
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
at 1021/1201, which are not on d56f68eb). A periodic axis was therefore
solved with the **declared** non-periodic boundary instead — PEC in this
fixture, CPML under `boundary='cpml'` — with nothing said.

Round 3 corrected the earlier "open, ghost-coupled axis" wording here and in
the refusal message, by measurement: y and z are not sharded at all (the
decomposition is 1-D along x), and on 7b511591 the distributed
`periodic_axes='y'` probe trace is **byte-identical** to the distributed PEC
trace of the same model (`tobytes()` equal, `max|d| = 0.0`, and that
distributed PEC trace is 2.910383e-10 from native PEC). Which is why the RED
1.963798e-04 above equals native-periodic-minus-native-PEC to every digit:
the lane reflected the axis, it did not leave it open. The test
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
`rfx/runners/distributed.py`'s `if pe.impedance > 0.0 and pe.extent is None:`
/ `elif pe.impedance == 0.0:` pair (the pmap runner, which is also the
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

### 2.5 x-absorber overflowing a rank's slab by more than one cell

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
owned cells exactly while `n <= nx_per` (x-lo) and `n <= nx_per - pad_x`
(x-hi).

**That is not the bound this check enforces, and round 3 is why.** Rounds 1
and 2 refused at exactly those two inequalities. One cell of overflow is
provably and measurably a **no-op**, so that bound refused configurations
`origin/main` runs at parity. The enforced bound is one cell looser on each
face:

```
n <= nx_per + 1            (x-lo face)
n <= nx_per - pad_x + 1    (x-hi face)
```

**Why one cell is free.** `_cpml_profile` (`rfx/boundaries/cpml.py`) grades
`rho = 1 - arange(n)/(n-1)`, so the innermost layer — profile index `n-1`,
or index `0` of the flipped x-hi profile — has `rho = 0`, hence `sigma = 0`,
`kappa = 1`, and `c = sigma*(b-1)/denom = 0`. Its `psi` therefore stays at its
zero initial value forever, and the correction it applies,
`-ce*psi - ce*(1/kappa - 1)*curl`, is **identically zero** — not small,
exactly zero. Both windows are *anchored* (x-lo at `ghost`, x-hi at
`-(ghost + pad_x)`), so one cell of overflow moves exactly that no-op layer
into the halo and leaves layers `0..n-2` (x-hi: `1..n-1`) on the very cells
they occupy when the absorber fits.

Measured on pristine `origin/main` 7b511591 with a **symmetric** absorber, so
class 6 is silent and the overflow is the only variable — `boundary='cpml'`,
`cpml_layers=8`, dx = 1 mm, y/z 8 mm, `amplitude_kind='field'` Ez source at
x = 3 mm, Ez probe row x = 1/3/5/6 mm, 120 steps,
`XLA_FLAGS=--xla_force_host_platform_device_count=4`:

| fixture | nx | devices | pad_x | nx_per | overflow | result |
|---|---|---|---|---|---|---|
| 11×8×8 mm | 28 | 4 | 0 | 7 | **both faces by 1** | 9.536743e-07 on a 9.238437e+00 peak = **1.032290e-07**, 0 warnings, trace **byte-identical** to the fitting 2-device run (`nx_per=14`) |
| 8×8×8 mm | 25 | 3 | 2 | 9 | **x-hi by 1** | 1.430511e-06 on 9.238317e+00 = **1.548455e-07**, 0 warnings, same digits as the fitting 2-device run |

A 25-row sweep on the same tree (domains 6 / 8 / 11 mm, `cpml_layers`
8 / 12 / 16, `n_devices` 2…6) separates the two bands exactly: **every**
configuration the narrowed bound admits ran at 5.4e-08…1.1e-07 of peak with 0
warnings, and **every** configuration it refuses died with `TypeError: mul got
incompatible shapes for broadcasting` — 0 mismatches in either direction.

**Two cells of overflow** move a layer with `sigma > 0` into the halo. At
`ghost == 1` — what `run_distributed` hardcodes — that is also where the
window runs past the slab end, the clipped window is shorter than the psi
arrays, and XLA raises a broadcasting error that names no feature. The general
statement is **per `ghost`, not per two cells**: from the literal slices the
window clips when `n > nx_per + ghost` (x-lo) / `n > nx_per + ghost - pad_x`
(x-hi), so at `ghost > 1` an overflow of 2…`ghost` cells would be *silent*
rather than fatal. That is why the bound is `> 1` and not the clip, and why it
does not go stale the day `ghost` stops being 1. The check still receives
`ghost` and uses it to say which band the caller is in — the message asserts
"died inside XLA" only when the window really clips.

The class-5 fixture is therefore `_asym_deep()`: `BoundarySpec(x=(cpml, pec),
y=cpml, z=cpml)`, `cpml_layers=8`, 4×8×8 mm at dx = 1 mm (`nx=13`, `pad_x=1`,
`nx_per=7`), field source at (2, 4, 4) mm, probes x = 1/2/3 mm. On 7b511591 it
does not run at all: `TypeError: mul got incompatible shapes for broadcasting:
(8, 1, 1), (7, 25, 25)`.

#### What `_asym(8, 6)` really was

`_asym(8, 6)` — `x_lo='cpml'`/`x_hi='pec'` with y/z CPML, 6×8×8 mm at
dx = 1 mm, `cpml_layers=8`, 2 devices, 60 steps, `amplitude_kind='field'` Ez
source at **(3, 4, 4) mm**, Ez probe row at **x = 1, 3, 5, 6 mm** →
`nx=15`, `pad_x=1`, `nx_per=8` — overflows x-hi by exactly one cell, and on
7b511591 it **runs**:

| quantity | value |
|---|---|
| `max abs(Ez_distributed − Ez_native)` | 2.207244e+00 |
| native peak | 4.416633e+00 (probe x = 3 mm) |
| relative | **49.98 % of peak** |
| probe at x = 5 mm, one cell inside the x-hi PEC face | 1.646758e-01 wrong on its own 1.646768e-01 peak — **99.9994 %** |
| errors | 0 |
| warnings | 0 |

That 49.98 % is **100 % class 6**, not class 5. Two measurements say so, not a
reading:

- on 7b511591 the *overflowing* 2-device trace and the *no-slab* 1-device
  trace of this fixture differ by 7.152557e-07 (float32 noise) while **both**
  are 2.207244e+00 from native — the wrongness does not need a decomposition;
- the case that satisfies the old bound **exactly**, `n == nx_per - pad_x` —
  same ASYM spec at 7×8×8 mm, `cpml_layers=8` (nx = 16, pad_x = 0,
  nx_per = 8), same source and probe row — is **46.16 % wrong** at the source
  probe (native 4.420052e+00 vs distributed 6.460177e+00, max|dEz|
  2.040125e+00), **99.97 %** wrong at x = 5 mm (1.800406e-01 vs
  5.905863e-05) and **100.0 %** at x = 6 mm (4.441055e-02 vs 1.295477e-07),
  with 0 warnings.

So the bound is **necessary, not sufficient**, and what makes these
configurations wrong is the phantom x-hi window: `ce_xhi = dt / (eps_r *
EPS_0)` is non-zero at a PEC face, so the lane absorbs at the reflector
(`rfx/runners/distributed.py`, the `ce_xhi` and `xhi` slices). The decisive
one: the 24 mm asymmetric model is wrong by 1.610351e-04 on a 3.165971e-03
peak (5.086e-02) at `n_devices=2` **and by the identical
1.610351e-04 / 5.086e-02 at `n_devices=1`**, where there is no slab to
overflow. `_asym(8, 6)` and `_asym(8, 24)` are both class-6 fixtures now; §2.6.

Both the source position and the probe row are part of every number above,
which is why the fixtures state them:

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

#### The broadcasting shapes belong to a different fixture

Round 3's third attribution miss in this note, and the reason the check's
message and docstring now name the spec next to every shape pair. Measured on
pristine main with the committed `_build`, 2 devices, 60 steps:

| spec | `cpml_layers` | domain | nx / pad_x / nx_per | result |
|---|---|---|---|---|
| `x=(cpml,pec)`, y/z cpml (ASYM) | 8 | 6×8×8 mm | 15 / 1 / 8 | **runs**, rel 4.997572e-01, 0 warnings |
| `x=(cpml,pec)`, y/z cpml (ASYM) | 8 | 8×8×8 mm | 17 / 1 / 9 | **runs**, max\|dEz\| 1.309694e+00 on 4.421298e+00 = **2.962240e-01**, 0 warnings |
| `x=(pec,pec)`, y/z cpml | 8 | 6×8×8 mm | 7 / 1 / 4 | dies: `(8, 1, 1), (5, 25, 25)` |
| `x=(pec,pec)`, y/z cpml | 8 | 8×8×8 mm | 9 / 1 / 5 | dies: `(8, 1, 1), (6, 25, 25)` |
| `x=(cpml,pec)`, y/z cpml (ASYM) | 20 | 7×12×12 mm | 28 / 0 / 14 | dies: `(20, 1, 1), (15, 53, 53)` |

Earlier drafts attached `(5, 25, 25)` / `(6, 25, 25)` to "the 8-layer
fixtures" *with the ASYM spec*; they belong to `x=(pec,pec)`, which is what
the test file's `test_pec_x_faces_are_still_refused_but_the_message_says_why`
already said. The row that attribution hid is the second one: **the 8-layer
ASYM model at 8 mm is a 29.6 %-of-peak silent case that class 5 admits** (nx =
17, pad_x = 1, nx_per = 9, n = 8 ≤ 9) and only class 6 catches. It is recorded
in §2.6's table with the rest of class 6.

Reachability. With a **symmetric** absorber the class-5 violation is
unreachable at 2 devices: the absorber is padded *outside* the requested
domain (`rfx/grid.py:153-154`), so `nx = interior + 2·n` and
`nx_per = interior/2 + n > n + 1` holds by construction (confirmed at
`cpml_layers=16` on a 2 mm interior — nx = 35, which runs). Every asymmetric
route is refused earlier-in-effect by class 6. So what class 5 actually
catches on the 2-device CPU lane is `n_devices >= 3` with a deep absorber, and
a `pec`/`pmc`-composed x face at any device count — exactly the band that dies
in XLA naming no feature. It earns its place as the check that turns that
`TypeError` into a message with a feature, a face and a remedy in it, and the
numbers that are not its own live in this note rather than in the message.

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
| `x=(cpml,pec)`, y/z cpml, `cpml_layers=8`, **8×8×8 mm**, source (3,4,4) mm, probes x = 1/3/5/6 mm | 60 | 1.309694e+00 | 4.421298e+00 | **2.962240e-01** — the row §2.5's shape-pair misattribution hid; class 5 **admits** it (nx = 17, pad_x = 1, nx_per = 9, n = 8 ≤ 9) |
| `x=(pec,pec)`, y/z cpml, `cpml_layers=8`, 15 mm (nx=16), source x = 7 mm, probes x = 2/7/13 mm | 60 | 7.786655e-02 | 4.421772e+00 | 1.760981e-02 at the row peak; **99.9986 %** / **99.9319 %** on the x = 2 / 13 mm probes' own peaks |
| the same at 19 mm (nx=20), source x = 9 mm, probes 2/9/17 mm | 60 | 2.789631e-03 | 4.422243e+00 | 6.308181e-04 at the row peak; **99.9917 %** / **99.8366 %** on the face probes |
| the same at 39 mm (nx=40), source x = 19 mm, probes 2/19/37 mm | 60 | 1.462868e-04 | 4.422517e+00 | 3.307773e-05 at the row peak; **99.9424 %** / **99.4168 %** on the face probes |
| `x=(pmc,cpml)`, y/z cpml, dx = 5 mm, 16×8×8 cells | 30 | 1.350925e-01 | 2.498208e-01 | **54.08 %** |
| the same | 80 | 1.021969e+00 | 1.093201e+00 | **93.48 %** |

Negative control, both x faces absorbing — `boundary='cpml'`,
`cpml_layers=8`, 24×8×8 mm, probes 12/22 mm, 60 steps: 5.820766e-09 on a
3.170117e-03 peak = **1.836e-06** of peak at 2 devices, 1.395e-06 at 1
device. Parity, three orders inside the shipped CPML tolerance of 1e-3.
**The refusal costs the symmetric absorber nothing** — and symmetric is the
common case, because `boundary='cpml'` pads both x faces outside the
requested domain by construction.

**Round 3, on the three `x=(pec,pec)` rows.** An earlier draft gave them as
"2.74 % / 3.06 % / 3.07 % at the source" with **no source position and no
probe row**, in violation of this note's own §6 rule, and they are not
re-derivable as written: with the fixture stated above the source-probe figure
is 1.76 % / 0.063 % / 0.0033 % — it *decays* with domain length, because the
source moves away from the phantom window while the window itself stays
`cpml_layers` deep. The figure that does not depend on where the source sits
is the one on the **face probes**, 99.4–100 % of their own peak in all three
domains, and that is what lane B has to size against. The row is replaced by
the three fully-stated rows above.

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

**Newly refused configurations, named.** Class 6 fires on *any* distributed
model with `boundary='cpml'`, `cpml_layers > 0` and `grid.pad_x_lo == 0` or
`grid.pad_x_hi == 0`. That is wider than the `pec`/`pmc`-composed faces the
measurements above use: `rfx/grid.py:109-110` also sets both x pads to 0 when
the x axis is simply **left out of `cpml_axes`**, so such a model is refused
too. Nothing in-tree does it (the whole suite is green, and the api-level
`Simulation` has no `cpml_axes` parameter — the route exists for callers that
construct a `Grid` or call a low-level entry point themselves), but it is a
behaviour change and it belongs on this list rather than in a diff.

**A per-face gap class 6 does *not* close.** Both slab checks read
`n = grid.cpml_layers` for both faces, which matches the kernel — the
distributed CPML windows also ignore `grid.face_layers`
(`rfx/grid.py:111`). So a model with `face_layers` `x_lo=6` / `x_hi=10` is
**admitted** by class 6 (both pads are non-zero; pinned simulation-free at
`test_the_phantom_window_check_is_a_no_op_when_both_faces_absorb`) while the
lane drives `cpml_layers` deep at both faces regardless. That is a further
silent per-face gap, unmeasured here, and it hands to lane B alongside the
phantom window: the same `grid.face_pads` gating that fixes one should read
`face_layers` for the depth.

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
configurations the builder cannot shard. Measured on main with the exact
construction the test uses — `Simulation(domain=(3e-3, 4e-3, 4e-3),
dx=1e-3, boundary='pec')` with a **3-cell** `_dx_profile`, which realizes
`grid.nx = 4`, so 2 ranks give `nx_per_rank = 2` (stated in full because
`nx_per_rank` is what the check compares against, and a nominally similar
"nx=4" construction can realize a different one: `_make_nu_sim_small(nx=4)`
realizes nx = 5, nx_padded = 6, `nx_per_rank = 3`, where K = 2/3/4 all fall
through to the builder's Phase-2E message and behaviour is unchanged by this
lane): at K = 3 check 3 computed `ghost_width=2`, `2 > 2` is false, the check
**passed**, and the call fell through to the builder's own "exchange_interval
> 1 is reserved for Phase 2E"; at K = 4 it raised but named 3 where the
builder needs 4.

Both sites now read `rfx.runners.distributed_nu.nu_ghost_width(K)` — one
source of truth, with the derivation in its docstring (a halo of `g` cells
keeps an owned slab correct for `K` steps between exchanges, so `g = K`; the
survey reaches the same `g = K` from the stencil side). The agreement is
tested for K = 1..4, the builder's own `ghost_width` is compared against the
helper at K = 1 (the only interval it accepts today), and a guard asserts the
preflight has not grown a second local formula.

One precision, so the agreement is not overclaimed: what the two sites now
share is **the same formula**, not the same allocation. `build_sharded_nu_grid`
raises `NotImplementedError("exchange_interval > 1 is reserved for Phase 2E")`
*before* it reads `nu_ghost_width` (`rfx/runners/distributed_nu.py:429` vs
`:435`), so for K = 2..4 the builder never allocates `g = K` at all — the
preflight and the builder agree on a line the builder cannot reach. That is
exactly the point of unifying them before K > 1 lands, but "the same number
the builder allocates" would be false today and "the same formula" is what is
true.

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
  through one correction. Round 3 caught it a third and a fourth time — the
  `(5, 25, 25)` / `(6, 25, 25)` shape pairs attributed to the ASYM spec
  (§2.5) and the `2.74 / 3.06 / 3.07 %` row with no fixture at all (§2.6).
  The lesson is in §6: a quoted relative figure needs its **source position
  and probe row** printed next to it, not just its fixture name, and a quoted
  *error string* needs the spec that produced it.
- **The NU-forward distributed lane is not gated by classes 1–4.**
  `rfx/api/_execute.py`'s NU-forward branch (~:2318–2340) refuses flux
  monitors and DFT planes with its own checks but never calls
  `refuse_unsupported_distributed_features`, so periodic axes, extended ports
  and `excite=False` ports are **not** gated on that lane. B0's declared scope
  is the uniform `run(devices=...)` lane and the two runners behind it; this is
  stated here so the gate is not read as covering the NU lane too. It is the
  smallest remaining piece of class-1–4 surface and should be picked up with
  §8's other open items.

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
- **`run(devices=[one_device])` is unchanged; the new single-device refusals
  reach *direct runner callers* only.** `rfx/api/_execute.py:3669` dispatches
  distributed only for `len(devices) > 1`, so the public single-device path
  still runs the uniform lane and still returns
  `flux_monitors == ['flux_x_0']`. Two gates in `rfx/runners/distributed.py`
  do fire at `n_devices == 1` — the classes 1–4 gate at `:1364` and **class 6
  at `:1463`** — and both are reachable only through
  `rfx.runners.run_distributed(sim, devices=[d0])` or as `distributed_v2`'s
  `n_devices == 1` delegate, never through `sim.run(devices=[d])`. Both are
  correct refusals of genuinely silent paths rather than working paths taken
  away: the flux monitor returned `flux_monitors=None` on d56f68eb, and
  `x=('pec','pec')` with y/z CPML through this runner at **one** device
  (`cpml_layers=8`, 15×8×8 mm at dx = 1 mm, field source x = 7 mm, probes
  x = 2/7/13 mm, 60 steps, `origin/main` 7b511591) is
  max|dEz| 7.786655e-02 on a 4.421772e+00 row peak, with the two probes
  inside the phantom windows **99.9986 %** and **99.9319 %** wrong on their
  own peaks — bit-for-bit the same figures as the 2-device run, 0 warnings
  both times. Phrased
  as "direct runner callers" here rather than "the single-device path", which
  would imply the public API changed.
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

**The centred source is the default way to hit this**, which round 3 adds so
lane B2 does not have to rediscover it. With a symmetric absorber, a source at
x = L/2 lands exactly on the first cut whenever `nx` is even — i.e. whenever
the interior cell count `L/dx` is odd, since `nx = L/dx + 1 + 2n` then makes
`pad_x = 0` and `nx_per = n + L/(2·dx)`, which is precisely the source's
global index. Measured on `origin/main` 7b511591, symmetric
`boundary='cpml'`, `cpml_layers=8`, dx = 1 mm, source and probe both at
x = L/2, 60 steps, **0 warnings** every time:

| domain | nx | global source index | 2 devices | 4 devices |
|---|---|---|---|---|
| 11 mm | 28 | 14 = `nx_per` | rel **3.065794e-02** | rel **3.065794e-02** |
| 15 mm | 32 | 16 = `nx_per` | rel **3.066272e-02** | rel **3.066272e-02** |
| 19 mm | 36 | 18 = `nx_per` | rel **3.066518e-02** | rel **3.066518e-02** |
| 4 mm | 21 | 10 (`nx_per` = 11) | rel 1.078176e-07 | — |
| 24 mm | 41 | 20 (`nx_per` = 21) | rel 1.078203e-07 | rel 1.078203e-07 |

So it is not an exotic placement: it is the one a user reaching for "put the
source in the middle" writes, on any odd-cell-count domain, at both 2 and 4
devices, and the magnitude is the same 3.07e-02 each time. On an **even**
cell count the centred source is one cell off the cut and the same models are
at 1.08e-07 parity, which is why the defect is easy to miss. Moving the
source **one cell earlier**, onto the last cell of rank 0,
restores 1.08e-07 parity. That is 30× the shipped CPML tolerance of 1e-3
(`tests/unit/runners/test_distributed.py`). It is position-dependent **in
the source**, not in the absorber, so it belongs to the cut-plane census
(lane B2) and not to an admission check — B0 is position-independent by
construction, with the two absorber checks the stated exceptions. It is
recorded here and in the test module's docstring so that B0's
"position-independent" scope statement is not read as "no other silent class
remains". It does not.

### 8.2 Round-3 review record — what changed and what did not

Changed, each by measurement (§2.5, §2.6, §4, §8.1 above): the class-5 bound
narrowed by one cell per face; the class-5 message's "died inside XLA" and
"an absorber on neither" clauses made conditional on what actually happened;
the port-fork citation moved from `distributed.py:1414-1422` (its position on
`origin/main`; this branch inserts 62 lines above it, so on HEAD it is
`:1476`/`:1484`) to the symbol pair itself; the periodic refusal's "open
ghost-coupled axis" replaced by "solved with the declared non-periodic
boundary"; §2.5's shape-pair attribution and §2.6's unfixtured
`2.74 / 3.06 / 3.07 %` row; the `g = K` agreement stated as a shared formula
rather than a shared allocation; and the base commit, the newly refused
`cpml_axes` configuration, the `face_layers` gap, the ungated NU-forward lane
and the centred-source trigger all recorded.

Checked and **not** changed:

- **Every class-1–4, 6 and 7 number in this note reproduced to the printed
  digit on `origin/main`** — class 1 (1.963798e-04 / 1.090241e-03, centred
  3.261566e-04 / 7.734966e-01), class 2 (0.0 vs 9.953718e-04, native max
  1.412484e-02), class 3 (1.983971e-03 bit-identical to `excite=True` vs
  native 0.0; `waveform=None` → `TypeError: Expected a callable value, got
  None`), class 4 (`flux_monitors=None` vs `['flux_x_0']`; `ntff_data=None` vs
  an `NTFFData`), class 6 (1.610351e-04 / 3.165971e-03 at 2 devices, 1 device
  and v1; x = 22 mm probe 99.46 %; symmetric control 1.836136e-06 /
  1.395464e-06), class 7 (v1 at nx = 24, including asym 5 mm at
  2.207114e+00 / 4.416774e+00), and the ghost-width fall-through at K = 2/3
  against the `ghost_width=3 exceeds nx_per_rank=2` raise at K = 4. All ran
  through `sim.run(devices=...)` **with** the default preflight ("All checks
  passed"), so none was already refused on main.
- **The class-5 message keeps its own measured numbers and gives up the
  others'.** Round 3 shortened it: the 49.98 % and the 2.207244 / 4.416633
  pair have moved out of the message into §2.5, because they belong to a
  configuration class 5 no longer refuses. What stays in the message is the
  arithmetic, which band the caller is in, the remedy, and a one-line
  "necessary, not sufficient" pointer carrying the 46.16 % figure and a link
  to this note.
- **`check_x_absorber_fits_ranks` is not deleted** even though a symmetric
  absorber at 2 devices cannot reach it (§2.5, Reachability). It is what turns
  the `TypeError: mul got incompatible shapes` band into a named refusal, and
  that band is reachable at `n_devices >= 3` and through any
  `pec`/`pmc`-composed x face at any device count.
