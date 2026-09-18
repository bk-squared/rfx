# Distributed admission refusals — closing seven silently-wrong paths (B0)

2026-09-14, last revised 2026-09-18 (round 5: rebase onto 8bc6c084).
Branch `agent/distributed-admission-refusals`. **The tree this PR lands on
is `origin/main` 8bc6c084** (`git rebase origin/main`, 2026-09-18).

Rounds 1–4 measured on, in order, d56f68eb → 7b511591 → 883615c6, and each
RED figure below still says beside itself which of those trees produced it.
Those figures are kept as history, with their trees named, and are **no
longer claimed to reproduce on the landing tree**: unlike the 7b511591 →
883615c6 step (which was one line in one file across every file this lane
touches — `if not issues:` → `if not len(issues):` in
`rfx/api/_execute.py`), 88651a1c → 8bc6c084 moves the distributed runners
themselves. #1038 legs 1–6 hoisted the shared sharding / CPML / shard_map
helpers into `rfx/runners/_distributed_common.py` and retired the
package-level `rfx.runners.run_distributed`; #1041/#1055 moved source
injection to **before** the E ghost exchange in all three distributed
runners; #1053 taught `distributed_v2` to realize declared PEC volumes.
A re-derivation of the RED table on 8bc6c084 is therefore owed and is NOT
done here — what the rebase does verify is that every refusal, every
message and every parity control in
`tests/unit/runners/test_distributed_admission_refusals.py` is still GREEN
on 8bc6c084, i.e. the classes are still refused and the admitted
configurations still match single-device. The one class whose *reachability
argument* main changed is class 7; §2.7 says how.

Direction note:
`rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md`
§3–§4, with the simulator survey it rests on
(`decomposition-survey.md` §C1). **That note is the plan; this change is its
first layer** — the position-independent admission check, plus the two
position-dependent slab checks.

**Round 2 (2026-09-14, after review).** The note's §3 list has five entries;
this change now refuses **seven** silently-wrong paths, because measurement
found two more while checking the five:

- **class 6**, the *phantom CPML window*: the lane drives every face's
  absorber window whenever `boundary='cpml'` and `cpml_layers > 0`, without
  reading `grid.face_pads`, so a face declared `pec`/`pmc` gets absorbed at.
  Round 2 found and refused this on the two x faces; round 4 measured it on
  the y and z faces too and widened the check (§2.6.1). The first draft of this note filed that as "S5", a numbers-only gap
  for lane B, and sized it at 3.05e-03 of peak. Both were wrong: it is
  5.09e-02 of peak on the committed fixture, up to **100 % of a probe's own
  peak** next to the face, and it is a silent path with 0 warnings — the
  same class as the five. §2.6.
- **class 7**, the *ungated v1 pmap runner* in
  `rfx/runners/distributed.py`: the first round gated only `distributed_v2`
  and the `run(devices=...)` dispatch, and all five classes rode straight
  through the pmap runner — at the time reachable as the package-level
  `rfx.runners.run_distributed`, and since #1038 leg 6 by full module path
  and as `distributed_v2`'s one-device delegate. §2.7.

**Round 4 (2026-09-15, after review).** Class 6 was **under-refusing**, and
one shipped number was not re-derivable:

- **class 6 was x-only while the defect is on all six faces.**
  `_init_cpml_distributed` (`rfx/runners/distributed.py`) builds ONE scalar
  `_cpml_profile` and the shmap kernel applies it at y-lo/y-hi/z-lo/z-hi
  unconditionally, so a `pec`/`pmc` y or z face composed with
  `boundary='cpml'` carries the identical phantom window — and was
  **ADMITTED**. Measured 90.58 % wrong at the source probe. Class 6 now
  reads all six entries of `grid.face_pads`. §2.6.1.
- **the `x=('pec','pec')` "2.7–3.1 % at the source" row is replaced by the
  face-probe figures**, which do not depend on where the source sits, in the
  message, the docstring and this note. §2.6.
- the class-6 remedy text no longer offers an x-only way forward: a caller
  with y/z reflectors who followed "make BOTH x faces absorbing" landed in
  the 90.58 % gap with 0 warnings. §2.6.1.

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
  `check_absorber_faces_are_absorbing()` (class 6, all six faces since
  round 4) inside both runners;
- the same six-class gate in the **v1 pmap runner** `rfx/runners/distributed.py`
  (class 7), which since #1038 leg 6 is reached as
  `rfx.runners.distributed.run_distributed` and as `distributed_v2`'s
  `n_devices == 1` delegate;
- one ghost-width formula, `rfx.runners.distributed_nu.nu_ghost_width` (§4);
- a `face_pads` property on `rfx.nonuniform.NonUniformGrid` (round 4), beside
  the `axis_pads` one it already had. `distributed_v2` reaches the class-6
  check with an **NU** grid — the `is_nu and use_cpml` NotImplementedError
  below it is a documented backstop, not a guard that runs first — so once
  the check reads `grid.face_pads` the NU grid has to answer it, or a direct
  NU + CPML + `devices=` runner call raises `AttributeError` where it used to
  raise a named refusal. The six fields were already on the dataclass; this
  only gives them the name `Grid` uses. Pinned by
  `test_the_nu_grid_answers_face_pads_so_the_check_can_read_it`, which also
  checks the tuple order against a real `Grid` rather than against the
  attribute names, and re-asserts the Phase-C refusal end to end;
- **a rewrite of THREE pre-existing tests**, each of them a false green on
  a configuration class 6 refuses. A changed shipped test must be declared,
  not discovered:
  - `tests/unit/boundaries/test_boundary_pmc_distributed.py`'s
    `test_pmc_distributed_v2_x_lo_owner_and_non_owner` moved from
    `x=Boundary(lo='pmc', hi='cpml')` with `y=z='cpml'` to
    `x=Boundary(lo='pmc', hi='pec')` with `y=z='pec'` (round 2). It was
    running on a 54.08 %-wrong configuration and asserting only a
    zero-pattern, so nothing it claimed is lost. §2.6.
  - `tests/unit/boundaries/test_boundary_pmc_composition.py`'s
    `test_oq9_distributed_v2_cpml_path_enforces_pec_face_via_cpml_init`
    split in two (round 4). Its `x='cpml'`, `y='cpml'`,
    `z=Boundary(lo='pec', hi='cpml')` fixture was **90.58 %** wrong at
    `devices=devices[:2]` and stayed green for the same reason — a
    zero-pattern assertion and never a value. Its structural claim (PEC
    enforced through the per-face CPML profile, no scan-body hook) is true
    on the lane that implements it, so it now runs single-device with its
    assertions unchanged, and a sibling test pins what the distributed lane
    owes the fixture: a named refusal. §2.6.1.
  - `tests/unit/boundaries/test_boundary_pmc_distributed.py`'s
    `test_pmc_distributed_legacy_mixed_z` moved from `x="cpml", y="cpml"`
    to `x="pec", y="pec"` (round 4). With x/y CPML both z pads are 0, so
    the lane drove its 16-layer window at the PMC z_lo and the PEC z_hi:
    **6.10 %** of peak wrong at the test's own 30 steps and 15.49 % at 80,
    while all four of its zero-pattern assertions held. A liveness
    assertion was added with the fixture change — four "is zero" claims
    with no energised-interior check also pass on a dead grid, and this
    test had none. §2.6.1.

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
`rfx/runners/distributed.py:330` ("non-periodic (ghost cells handle
inter-device coupling)"), and the same sentence again at `:359`, `:394`,
`:419`. And `sim._periodic_axes` is read **zero** times in
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
gate" rather than "at all"; `rfx/runners/uniform.py:429,448` honours it (`if pe.excite:` guards both
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
`nx_local = nx_per + 2·ghost` are (`rfx/runners/distributed.py:786-787`):

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

### 2.6 Phantom CPML window at a non-absorbing face — absorbing at a reflector (the x faces; y/z in §2.6.1)

This section is the **x** faces, where the class was found in round 2; §2.6.1
is the same class on the y and z faces, where round 4 found it admitted. The
runner applies **both** x-face CPML windows whenever
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

**Round 3, corrected in round 4, on the three `x=(pec,pec)` rows.** An
earlier draft gave them as "2.74 % / 3.06 % / 3.07 % at the source" with **no
source position and no probe row**, in violation of this note's own §6 rule,
and they are not re-derivable as written. Round 3 replaced them with the three
fully-stated rows above but mislabelled its own replacement, saying "the
source-probe figure is 1.76 % / 0.063 % / 0.0033 %". Only the first of those
three is a source-probe figure. The other two are **row figures** —
max abs(d) over the whole probe row divided by the row peak — and on those two
domains the max sits on the **x = 2 mm face probe**, not on the source probe.
Measured on 883615c6 with the fixture stated above, both quantities, so the
labels can never be crossed again:

| domain | source probe, on its own peak | row max abs(d) / row peak | where the row max sits |
|---|---|---|---|
| 15 mm (nx=16), src x=7 mm | 7.786655e-02 of 4.421772e+00 = **1.7610 %** | 1.760981e-02 | the source probe (x=7 mm) |
| 19 mm (nx=20), src x=9 mm | 9.970665e-04 of 4.422243e+00 = **0.0225 %** | 6.308181e-04 | the **x=2 mm face probe** |
| 39 mm (nx=40), src x=19 mm | 9.536743e-07 of 4.422517e+00 = **0.00002 %** | 3.307773e-05 | the **x=2 mm face probe** |

The source-probe figure *decays* with domain length, because the source moves
away from the phantom window while the window itself stays `cpml_layers`
deep — which is exactly why it is the wrong quantity to ship. The figure that
does not depend on where the source sits is the one on the **face probes**,
99.4–100 % of their own peak in all three domains, and that is what lane B has
to size against and what the shipped message and docstring now quote.

**Reachability — why this is a refusal and not a warning.** A caller who
probes only far from the faces cannot see this class at any tolerance the
suite uses, so no threshold would have caught it. Measured on 883615c6,
`x=(pec,pec)` with y/z CPML, `cpml_layers=8`, 39 mm (nx=40), source x = 19 mm,
**the centre probe alone and 200 steps** — the shape of the shipped
`tests/unit/runners/test_distributed.py` parity tests: 8.356664e-04 on a
9.237972e+00 peak = **9.045994e-05 of peak**, which is *inside* the shipped
1e-3 CPML tolerance, while the face probes on the same run are 99.4–99.9 %
wrong. A warn-and-continue would read as "passes" to exactly the fixtures most
likely to be written. That is the argument for refusing rather than warning,
and it is why the probe row is part of every figure above.

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
model with `boundary='cpml'`, `cpml_layers > 0` and **any** entry of
`grid.face_pads` equal to 0 (round 4; it was the two x pads before — §2.6.1).
That is wider than the `pec`/`pmc`-composed faces the measurements above use:
`rfx/grid.py:110-111` also sets an axis's pads to 0 when the axis is simply
**left out of `cpml_axes`**, so such a model is refused too. For a 3-D model
nothing in-tree does it (the api-level `Simulation` has no `cpml_axes`
parameter — the route exists for callers that construct a `Grid` or call a
low-level entry point themselves). For a **2-D** model that route is taken by
the grid builder itself, which drops `z` from `cpml_axes`, so every 2-D CPML
distributed model is now refused — and it died inside XLA before, so nothing
that worked is taken away (§2.6.1). Both are behaviour changes and belong on
this list rather than in a diff.

**A per-face gap class 6 does *not* close.** Both slab checks read
`n = grid.cpml_layers` for every face, which matches the kernel — the
distributed CPML windows also ignore `grid.face_layers`
(`rfx/grid.py:112`). So a model with `face_layers` `x_lo=6` / `x_hi=10` (or
the y/z equivalent) is **admitted** by class 6 (both pads are non-zero;
pinned simulation-free at
`test_the_phantom_window_check_reads_all_six_pads`, whose
`face_pads=(6, 10, 8, 8, 7, 9)` row is exactly this case) while the lane
drives `cpml_layers` deep at every face regardless. That is a further silent
per-face gap, unmeasured here, and it hands to lane B alongside the phantom
window: the same `grid.face_pads` gating that fixes one should read
`face_layers` for the depth.

**This is an admission refusal, not the fix.** The fix is to gate **every**
face window on `grid.face_pads`, which changes physics rather than admission
and is lane B's subject (the distributed CPML outer termination). Until it
lands, refusing is the honest answer; when it lands,
`check_absorber_faces_are_absorbing` should be deleted and class 5
narrowed to the faces that really are CPML.

### 2.6.1 The same phantom window is on the y and z faces — round 4

Class 6 shipped as an **x-face** refusal for two rounds, and both this note's
§5 and the check's own docstring described the phantom window as an x-face
problem. The consistency lens found that wrong by reading the kernel and then
measuring it. `_init_cpml_distributed` (`rfx/runners/distributed.py`) builds
**one** scalar profile, `_cpml_profile(grid.cpml_layers, grid.dt, grid.dx,
...)`, and the shmap step body applies it at y-lo / y-hi / z-lo / z-hi
**unconditionally** — it consults `grid.face_pads` on no axis, not just on x.
Neither runner reads `grid.pad_y_*` or `grid.pad_z_*` anywhere (pinned by
`test_the_runners_still_never_read_the_y_z_face_pads`). The single-device lane
does not share the defect: its `init_cpml` clamps each face's profile to that
face's allocated pad (`rfx/boundaries/cpml.py`), which is the mechanism
`tests/unit/boundaries/test_boundary_pmc_composition.py`'s OQ9 docstring
describes — and describes only for that lane.

MEASURED on `origin/main` 883615c6, with **identical digits** on this
branch's HEAD 461cfe53 and on the base 7b511591 (so this is a pre-existing
class, not a regression this branch introduced): 24×8×8 mm at dx = 1 mm,
`cpml_layers=8`, `amplitude_kind='field'` Ez source at (6, 4, 4) mm, Ez probes
at x = 6 / 12 / 20 mm, 60 steps, 2 virtual CPU devices, **0 warnings every
time**, and every row ADMITTED on HEAD before this round:

| spec | source probe x=6 mm | probe x=12 mm | probe x=20 mm |
|---|---|---|---|
| `x='cpml'`, `y='cpml'`, `z=Boundary(lo='pec', hi='cpml')` | **90.5782 %** of own peak (max abs(dEz) 4.003862e+00 on 4.420338e+00) | **409.4824 %** of own peak | **281.3395 %** of own peak |
| the same at `n_devices=1` through the v1 pmap runner | 90.5782 % | 409.4823 % | 281.3395 % — bit-for-bit |
| `z=Boundary(lo='pec', hi='pec')` | 81.2831 % | max abs(dEz) **2.615769e+00** on a 5.171041e-03 probe peak | 1.967369e+00 on 8.379044e-04 |
| `y=Boundary(lo='pmc', hi='cpml')` | 48.6214 % | **521.9300 %** of own peak (3.988669e-02 on 7.642152e-03) | 368.9100 % |
| `y=Boundary(lo='pec', hi='pec')` | **373.3991 %** of own peak (1.649596e+01 on 4.417783e+00) | 1.144e+05 % | 6.830e+06 % |
| SYMMETRIC control — all six faces absorbing | **1.078195e-07 of peak** (4.768372e-07 on 4.422548e+00) | — | — |

**A second shipped test was running on it.** The first row is the exact
boundary composition of
`tests/unit/boundaries/test_boundary_pmc_composition.py::test_oq9_distributed_v2_cpml_path_enforces_pec_face_via_cpml_init`,
which ran at `devices=devices[:2]` and stayed green on a 90.58 %-wrong run,
because it asserts a zero-**pattern** on the PEC face (`ex[:,:,0]`,
`ey[:,:,0]`) and never a value. That is the identical false green the x face
had in `test_boundary_pmc_distributed.py` (§2.6). Its structural claim — PEC
enforced through the per-face CPML profile, no scan-body hook — is true, on
the lane that implements it, so the test now makes it on the single-device
lane (`test_oq9_uniform_cpml_path_enforces_pec_face_via_cpml_init`,
assertions unchanged) and a sibling
(`test_oq9_distributed_v2_refuses_the_pec_face_composition`) pins what the
distributed lane owes the same fixture: a named refusal. Nothing was deleted
and no tolerance moved.

**And a third, found only because the re-run list was widened.**
`tests/unit/boundaries/test_boundary_pmc_distributed.py::test_pmc_distributed_legacy_mixed_z`
uses `x="cpml", y="cpml", z=Boundary(lo="pmc", hi="pec")` at dx = 5 mm,
16×8×24 cells, default `cpml_layers` (16), and calls the **v1 pmap runner
directly at one device** to exercise its `_apply_pmc_local` hook. Both z
pads are 0 (`grid.face_pads == (16, 16, 16, 16, 0, 0)`), so the lane drives
a 16-layer window at the PMC z_lo *and* at the PEC z_hi. MEASURED on
883615c6 with that fixture exactly: the Ex probe trace is **6.1030 %** of
peak wrong at the test's own 30 steps (max abs(dEx) 2.869174e-02 on a
4.701230e-01 peak) and **15.4906 %** at 80 steps, 0 warnings — and all four
of its zero-pattern assertions (`hx[:,:,0]`, `hy[:,:,0]`, `ex[:,:,-1]`,
`ey[:,:,-1]`) held on that run. Same remedy as the other two: it now
composes x and y with `pec`, where the four zeros are exact, the hooks and
the pmap scan body under test are untouched, and the model runs at parity
with the single-device lane (rel 1.361770e-07). A liveness assertion was
added at the same time, because four "is zero" claims with no
energised-interior check pass on a dead grid too.

That this test was found by *widening the re-run list* and not by reading
the diff is the argument for MATERIAL 8 being a replacement rather than an
addition: two of the three false greens in this class live in
`tests/unit/boundaries`, which round 3's sweep never ran.

**Why widen rather than hand it to lane B.** Round 2 already made that
mistake once on the x face: the first draft of this note filed the phantom
window as "S5", a numbers-only gap for lane B, and it was in fact a silent
path with 0 warnings and up to 100 % of a probe's own peak — the same class
as the five. The y/z faces are the same defect by the same mechanism at the
same magnitude, and leaving them admitted while refusing x would ship a gate
that is fail-closed on one axis and fail-open on two. So class 6 reads all
six pads and both runners pass `grid.face_pads`.

**The remedy text was unsafe, and that is the sharper half of this.** The
round-3 message offered "make BOTH x faces absorbing (`BoundarySpec(x='cpml')`
...)" as a way forward. A caller with y/z reflectors who followed it — say
`x='cpml'`, `y='cpml'`, `z=Boundary(lo='pec', hi='cpml')` — got the
90.58 %-wrong run in the table above, admitted, with 0 warnings. The gate
would have *routed* people into the gap. The message now says **all six
faces absorbing**, spells out that an x-only fix is the 90.58 % row, and names
omitting `devices=` as the way out for a model that cannot make all six
absorbing.

**2-D models are refused, and gain by it.** `rfx/grid.py` drops `z` from
`cpml_axes` when `mode` starts with `2d`, so every 2-D CPML model has
`pad_z_lo == pad_z_hi == 0` and lands here. Nothing is taken away: measured on
883615c6, `mode='2d_tmz'` and `'2d_tez'` at 24×8×8 mm, dx = 1 mm,
`boundary='cpml'`, `cpml_layers=8`, 2 devices, the run **died inside XLA**
with `ValueError: Incompatible types for broadcasting: input
type=float32[23,25,8] and requested type=float32[23,25,1]` — an error that
names no feature, no face and no remedy. The refusal replaces that with a
named one. Nothing in-tree combines 2-D with `boundary='cpml'` and
`devices=` (`tests/unit/api/test_api.py`'s 2-D distributed cases use `upml`,
already refused, and `pec`).

**Still not closed, and now stated on all six faces rather than one.** The
per-face `grid.face_layers` depth is still ignored by the kernel and by this
check: a model with `face_layers` `x_lo=6` / `x_hi=10` (or the y/z
equivalent) is ADMITTED because both pads are non-zero, while the lane drives
`cpml_layers` deep everywhere. Unmeasured, and it hands to lane B with the
phantom window — the same `grid.face_pads` gating that fixes one should read
`face_layers` for the depth.

### 2.7 The v1 pmap runner was ungated for all of 1–6

The first round of this change put the gate in
`distributed_v2.run_distributed` and in the `run(devices=...)` dispatch, and
the pmap runner in `rfx.runners.distributed` kept running every class. The
runner's only new-ish guard was its `nx % n_devices != 0` ValueError, which
is the accident that made the 24 mm fixtures bounce and the gap look closed.

**Round 5 (the 8bc6c084 rebase) changed how this runner is reached, not
whether it needs the gate.** When rounds 1–4 measured it,
`rfx/runners/__init__.py` re-exported `run_distributed` from
**`rfx.runners.distributed`** — the pmap runner — and not from
`distributed_v2`, so the package-level name itself was the open door. #1038
leg 6 retired that re-export: `rfx.runners.run_distributed` no longer
resolves at all, and `distributed_v2` is the trunk. Two live routes into the
pmap runner remain, which is why all of §2.7 stands:

1. the full module path `from rfx.runners.distributed import run_distributed`,
   which `rfx/runners/__init__.py` documents as the migration path;
2. `distributed_v2.run_distributed`'s `n_devices == 1` fast path, which
   delegates to it verbatim — so every direct one-device distributed call
   lands here.

Both are pinned by
`test_the_pmap_runner_is_still_reachable_and_still_gated`, which replaces
round 2's export-module assertion: it now asserts the package-level name is
**gone** (so a silent reinstatement fails the suite), that the module path
still resolves to the pmap runner, and that v2 still delegates to it at one
device. If (2) ever goes away the v1 gates become reachable only by a direct
module-path caller, and this section should say so rather than imply more.

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
`check_absorber_faces_are_absorbing(...)` once `use_cpml` is known
(`pad_x=0` literally, because this runner requires `nx % n_devices == 0`).
Pinned by nine tests, including one that asserts the pmap runner is still
reachable by both routes above — without it the other eight would still
pass while testing nothing, or testing v2 twice.

## 3. The refusals

Three new entry points in `rfx/runners/distributed_v2.py`, called from
**three** lanes — the `run(devices=...)` dispatch, `distributed_v2.run_distributed()`
and (round 2) `distributed.run_distributed()`, the v1 pmap runner:

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
- `check_absorber_faces_are_absorbing(*, cpml_layers, face_pads, n_devices,
  lane=...)` — **class 6**, `ValueError`, round 2, **widened to all six faces
  in round 4** (§2.6.1). Fires when ANY entry of `grid.face_pads` is 0 while
  the lane is building CPML windows. `face_pads` — the six-tuple — is the
  parameter rather than the two x pads because `grid.face_pads` is precisely
  the attribute the runner fails to read; taking the two x pads is what made
  the y/z gap invisible for two rounds. Called immediately after
  `check_x_absorber_fits_ranks` in both runners, deliberately **after** it:
  when both apply (the asymmetric fixture at a too-small `nx_per`) the
  arithmetic message is the more specific one and it is the one the class-5
  tests pin. The message names the face(s), the depth, all six pads, and the
  ways forward — make **all six** faces absorbing, drop the absorber
  entirely, or omit `devices=...`. It fires at `n_devices == 1` too, and
  must: §2.6's and §2.6.1's `n_devices=1` rows are identical to their
  2-device rows.

  Round 4 renamed it from `check_x_absorber_faces_are_absorbing`: the old
  name asserted the x-only scope that was the defect. The symbol is new on
  this branch, so nothing outside it refers to the old name.

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
*before* it reads `nu_ghost_width` (`rfx/runners/distributed_nu.py:401` vs
`:406`), so for K = 2..4 the builder never allocates `g = K` at all — the
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
- **The FIX for the phantom window on ALL SIX faces (what was "S5").**
  Refusing it is round 2's answer for x and round 4's for y and z (§2.6,
  §2.6.1, class 6); making it *work* is not. Gating **every** face window on
  `grid.face_pads`, and reading `grid.face_layers` for the depth, so that a
  `pec`/`pmc` face gets no absorber correction on any axis, is a physics
  change in the distributed CPML outer termination — lane B's subject — and
  it is what lets class 6 be deleted and class 5 narrowed to the faces that
  really are CPML. Round 4 re-sized this item: it is not an x-face item.
  `_init_cpml_distributed` builds one scalar profile and the kernel drives it
  at y-lo/y-hi/z-lo/z-hi unconditionally, so lane B has to fix four more
  faces than the first three rounds of this note said, and the y/z magnitude
  is **larger** than the x one it was sized against — 90.58 % of the source
  probe's own peak and 409 % / 281 % downstream (§2.6.1), against 5.09e-02 of
  the row peak on x. **Lane B should size the fix against 9e-01, not 5e-02**,
  and it should not assume the x face is the worst one. What lane B needs to
  carry over is the **size**, and the first draft of this note got it wrong
  twice on x alone — both corrected here by measurement:

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
  Round 4 caught it a **fifth** time, in round 3's own replacement for that
  row: "the source-probe figure is 1.76 % / 0.063 % / 0.0033 %" mixed one
  source-probe figure with two row figures whose maxima sit on a face probe
  (§2.6). The lesson is in §6: a quoted relative figure needs its **source
  position, probe row AND which probe the max sits on** printed next to it,
  not just its fixture name, and a quoted *error string* needs the spec that
  produced it. Round 4's rule, added because relabelling was not enough
  twice: when a row figure and a per-probe figure are both interesting,
  print **both**, in a table with a column saying where the max sits.
- **The NU-forward distributed lane is not gated by classes 1–4.**
  `rfx/api/_execute.py`'s NU-forward branch
  (`_forward_distributed_nonuniform_from_materials`, ~:2365–2378) refuses
  flux monitors and DFT planes with its own checks but never calls
  `refuse_unsupported_distributed_features`, so periodic axes, extended ports
  and `excite=False` ports are **not** gated on that lane. B0's declared scope
  is the uniform `run(devices=...)` lane and the two runners behind it; this is
  stated here so the gate is not read as covering the NU lane too. It is the
  smallest remaining piece of class-1–4 surface and should be picked up with
  §8's other open items.

## 6. Tests

`tests/unit/runners/test_distributed_admission_refusals.py`, **83 tests**
(40 after round 1; 65 after round 3 — round 3's note said 64, and
`pytest --collect-only` said 65, so that digit was wrong too; 82 after
round 4). Counted with `pytest --collect-only -q` on the tree this note
describes, not from memory — and the round-3 digit was checked the same way,
against `git show HEAD~1:...`, which is how the off-by-one was confirmed
rather than assumed. For each of the five original classes: the refusal through
`sim.run(devices=...)`, the refusal inside `run_distributed()`, and an
assertion on the message content (the feature name, the cause, the way out).
Class 5 additionally gets a simulation-free unit test of the exact window
boundary. Round 3 moved that boundary out by one cell on each face and round 4
fixed this sentence, which had kept the pre-round-3 wording: the last depth
the check **admits** is `n == nx_per - pad_x + 1` (x-hi) / `nx_per + 1`
(x-lo), and one deeper is the first it **refuses**, for `pad_x ∈ {0, 1, 3}` —
which is what the test body has pinned since round 3. Its docstring also
records that the bound is necessary only, with the 46.16 % measurement of a
case one cell inside it.

Round 2 added, and each one is a fact the first round left unpinned:

- **class 6** (§2.6): the refusal through both entry points, the message
  content (face, all six pads, the ways out), a PMC variant, a
  simulation-free unit test of the `pad == 0` boundary including the
  "different per-face thickness, both absorbing" admit, and the proof that
  `boundary='pec'` never reaches the check;
- **the v1 pmap runner** (§2.7): five parametrised feature refusals at
  nx = 24, a lane-name assertion, the two slab checks, the class-6 refusal
  at `n_devices == 1`, a symmetric-absorber parity control through the same
  call at the shipped 1e-3, and one test asserting that the pmap runner is
  still reachable — `rfx.runners` no longer exports `run_distributed`
  (#1038 leg 6), `rfx.runners.distributed.run_distributed` still resolves to
  it, and `distributed_v2` still delegates to it at one device — without
  which the other eight would keep passing while testing the wrong runner;
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

Round 4 added (§2.6.1), all on the widened class 6:

- the y/z refusal through the runner, through `sim.run(devices=...)` and
  through the v1 pmap runner, parametrised over four RED specs —
  `z=(pec,cpml)`, `z=(pec,pec)`, `y=(pmc,cpml)`, `y=(pec,pec)` — each
  asserting the faces it names;
- the y/z refusal at `n_devices == 1` through the v1 runner, which is where
  the bit-for-bit 1-device measurement was taken;
- a simulation-free unit test that walks all six entries of
  `grid.face_pads` one at a time, plus the all-six-off message and the
  rejection of a two-tuple (so a caller cannot pass the old `pad_x_lo,
  pad_x_hi` shape and be silently half-checked);
- `test_the_remedy_never_points_at_the_x_only_composition`, which asserts the
  message contains no `BoundarySpec(x='cpml')` and no "BOTH x faces" — the
  remedy that routed callers into the gap;
- `test_a_2d_cpml_model_gets_a_named_refusal_instead_of_an_xla_crash`, on
  both 2-D modes;
- `test_the_runners_still_never_read_the_y_z_face_pads`, the refusal's
  premise checked rather than trusted: if either runner grows a
  `grid.pad_y_*` / `grid.pad_z_*` / per-face `face_layers` read, or stops
  building the single scalar `_cpml_profile`, this fails and class 6 must be
  re-derived;
- a six-face symmetric parity control at the shipped 1e-3, unweakened;
- and in `tests/unit/boundaries/`, the two test rewrites described in
  §2.6.1: the OQ9 split in `test_boundary_pmc_composition.py` (the
  structural claim on the uniform lane, assertions unchanged, plus the
  distributed lane's named refusal beside it) and
  `test_pmc_distributed_legacy_mixed_z` moved to a reflector-only x/y
  composition in `test_boundary_pmc_distributed.py`, with a liveness
  assertion added — its four "is zero" claims had no energised-interior
  check, and four such claims pass on a dead grid.

**The pre-PR re-run list.** Round 3's sweep ran `tests/unit/runners`,
`tests/contracts`, `tests/locks` and ruff, and that is not enough for a
change to a **shared runner**. Round 4's lens found seven more files with
`devices=` / `distributed=True` callers that the sweep never ran —
`tests/unit/boundaries` (beyond the one pmc file), `tests/unit/grid/
test_precision_lane_guard.py`, `tests/unit/autodiff/
test_observables_dft_field.py`, `tests/unit/materials/
test_sheet_impedance.py`, `tests/contracts/
test_declared_solver_and_monitor_domain.py`, `tests/unit/autodiff/
test_jacobian_fwd.py`, `tests/unit/autodiff/test_progressive_optimize.py` —
and `tests/unit/boundaries` is where the second false green (§2.6.1) was
hiding. So the list is now: **the whole fast suite the CI runs** (the entire
`tests/` tree under the default `addopts`, i.e. `-m 'not gpu and not slow and
not slow_physics'`), not a directory sample. `tests/unit/boundaries` and
`tests/unit/materials/test_sheet_impedance.py` are named explicitly because
they carry distributed callers and are easy to miss. The count from that run
is recorded in §8.3.

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
  reach *direct runner callers* only.** `rfx/api/_execute.py:3713` dispatches
  distributed only for `len(devices) > 1`, so the public single-device path
  still runs the uniform lane and still returns
  `flux_monitors == ['flux_x_0']`. Two gates in `rfx/runners/distributed.py`
  do fire at `n_devices == 1` — the classes 1–4 gate at `:1352` and **class 6
  at `:1462`** — and both are reachable only through
  `rfx.runners.distributed.run_distributed(sim, devices=[d0])` or as `distributed_v2`'s
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
the port-fork citation moved from `distributed.py:1414-1422` (its position
on the round-3 `origin/main`; this branch inserts lines above it, so it does
not stay put — on the 8bc6c084 rebase the pmap fork is `:1393`/`:1401` on
main and `:1473`/`:1481` on HEAD) to the symbol pair itself; the periodic refusal's "open
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

### 8.3 Round-4 review record — what changed and what did not

Measured on `origin/main` 883615c6 unless a row says otherwise. Every RED
figure below was re-derived on that tree, to the printed digit, before
anything was edited.

**Changed.**

- **Class 6 widened from two faces to six** (§2.6.1), in both runners, with
  the parameter changed from `pad_x_lo, pad_x_hi` to `face_pads` and the
  function renamed `check_x_absorber_faces_are_absorbing` →
  `check_absorber_faces_are_absorbing`, because the old name asserted the
  scope that was the defect. RED: `x='cpml'`, `y='cpml'`,
  `z=Boundary(lo='pec', hi='cpml')` at 24×8×8 mm, dx = 1 mm,
  `cpml_layers=8`, field Ez source (6, 4, 4) mm, probes x = 6/12/20 mm, 60
  steps, 2 CPU devices, 0 warnings, ADMITTED on HEAD 461cfe53 and on the base
  7b511591 with identical digits: **90.5782 %** of the source probe's own
  peak (max abs(dEz) 4.003862e+00 on 4.420338e+00), **409.4824 %** and
  **281.3395 %** at x = 12/20 mm. Symmetric control on the same fixture
  1.078195e-07 of peak.
- **The `x=('pec','pec')` "2.7–3.1 % wrong at the source" row is gone from
  the shipped message and docstring**, replaced by the face-probe figures —
  99.9986/99.9319 %, 99.9917/99.8366 %, 99.9424/99.4168 % — which do not
  depend on where the source sits. The measured source-probe figures for
  those three domains are **1.7610 % / 0.0225 % / 0.00002 %**, so
  "2.7–3.1 %" was not re-derivable with the fixture the branch itself
  states. The refusal was always right; only the number in it was wrong.
- **§2.6's own round-3 relabelling was itself wrong** and is now a table with
  a "where the max sits" column: `0.063 %` and `0.0033 %` were row figures,
  not source-probe figures, and their maxima sit on the x = 2 mm **face**
  probe.
- **The class-6 remedy text no longer offers an x-only way forward.** It said
  "make BOTH x faces absorbing (`BoundarySpec(x='cpml')` ...)"; a caller with
  y/z reflectors who followed it got the 90.58 %-wrong run above, admitted,
  with 0 warnings. It now says "all six faces absorbing", names the x-only
  composition as the 90.58 % row, and names omitting `devices=` as the only
  route for a model that cannot make all six absorbing.
- **The class-5 message's "exact arithmetic limit" is corrected** (§2.5). The
  46.16 % fixture (`x_lo='cpml'`/`x_hi='pec'`, `cpml_layers=8`, 7×8×8 mm,
  nx = 16, pad_x = 0, nx_per = 8, n = 8) sits **one cell inside** the bound
  round 3 installed (x-hi limit `nx_per - pad_x + 1` = 9). "Exact limit" was
  true of the round-1/2 bound `n <= nx_per - pad_x` and went stale with the
  narrowing. The docstring at the same check already phrased it correctly.
- **`check_x_absorber_fits_ranks`' 3-device row no longer implies
  byte-identity** (§2.5). It said the 8×8×8 mm / 3-device case gave "the same
  digits as the fitting 2-device run". Correct on rel — 1.548455e-07 both —
  but, unlike the 11 mm / 4-device row, the traces are **not** byte-identical:
  3-dev vs 2-dev max abs(d) = **9.536743e-07**, one float32 last bit on a
  9.238317e+00 peak. Stated as "same rel to 7 digits" now, so the docstring
  carries one byte-identity claim and not two.
- **The stale "`n == nx_per - pad_x` admitted, `n + 1` refused" sentence** is
  fixed in both places it survived round 3 — this note's §6 and
  `test_the_x_absorber_condition_is_the_window_arithmetic`'s docstring, whose
  next paragraph already said the opposite and whose body pins `n + 1`
  admitted / `n + 2` refused.
- **Line-number citations re-anchored to HEAD** (they had drifted, in one case
  since before round 3): the v1 runner's gates at
  `rfx/runners/distributed.py:1373` (classes 1–4) and `:1483` (class 6), not
  `:1364`/`:1463`; `rfx/api/_execute.py:3671` for
  `_distributed_run = devices is not None and len(devices) > 1`, not `:3669`;
  `rfx/grid.py:110-111` for the `cpml_axes` → `return 0` pair and `:112` for
  the `face_layers` read, not `:109-110`/`:111`. Round 5 re-anchored them
  again onto 8bc6c084 — see §8.5.
- **The note's header names the tree the PR lands on**, 883615c6, with the
  per-round provenance of every figure kept beside it.
- **§2.6 gained the reachability argument** (below) and §8's lane-B sizing
  item was re-sized from 5e-02 to 9e-01 and from one face to six.
- **The pre-PR re-run list is the whole fast suite**, not a directory sample
  (§6). Counts in §8.4.

**Checked and not changed.**

- **Class 6's reachability argument, which is why it refuses instead of
  warning.** At 39 mm with only the centre probe and 200 steps — the shape of
  the shipped `tests/unit/runners/test_distributed.py` parity tests — the
  `x=('pec','pec')` phantom-window run is 8.356664e-04 on a 9.237972e+00 peak
  = **9.045994e-05 of peak**, i.e. *inside* the shipped 1e-3 CPML tolerance,
  while the face probes on that same run are 99.4–99.9 % wrong. No threshold
  the suite uses would see it. Recorded in §2.6 and in the check's docstring
  as the argument for fail-closed.
- **`check_x_absorber_fits_ranks`' "died inside XLA" band sentence is
  correct.** Verified at `ghost=1` on the base 7b511591 for every refused row:
  4 devices at nx = 24/25/22/26/23 all raise `TypeError: mul got incompatible
  shapes for broadcasting`, and 3 devices at nx = 22 likewise; and every
  ADMITTED one-cell-overflow row (nx = 28/4dev, nx = 25/3dev, nx = 23/3dev)
  ran at rel 1.078e-07 vs native on both trees. The round-3 conditional
  (`clips`) asserts the claim only where it is true, which is what makes it
  safe at `ghost > 1`. Re-checked on 883615c6 as well, through the public
  `sim.run(devices=...)` with `x=('pec','pec')` and y/z CPML at
  `cpml_layers=8`: nx = 24/25/22/26/23 at 4 devices and nx = 22 at 3 all
  raise `TypeError: mul got incompatible shapes for broadcasting: (8, 1, 1),
  (7, 25, 25)` / `(5, 25, 25)` / `(6, 25, 25)`. No change; recorded so the
  reviewer's numbers sit beside the lane's.
- **The y/z gap is not a regression.** Identical digits on HEAD 461cfe53, on
  the base 7b511591 and on 883615c6. It is refused rather than handed to lane
  B because round 2 already made that mistake on the x face, and because a
  gate that is fail-closed on one axis and fail-open on two is worse than
  either.
- **The 2-D refusal takes nothing away.** 2-D + `boundary='cpml'` +
  `devices=` died inside XLA on main (`Incompatible types for broadcasting:
  float32[23,25,8]` vs `float32[23,25,1]`), so the refusal replaces an
  unnamed crash with a named one. Nothing in-tree combines the three.
- **The `face_layers` depth gap stays open** and is now stated for all six
  faces rather than for x alone. Unmeasured, hands to lane B.

**Found while doing it, not asked for.** A **third** shipped false green of
this class,
`tests/unit/boundaries/test_boundary_pmc_distributed.py::test_pmc_distributed_legacy_mixed_z`
— `x/y='cpml'` with `z=(pmc,pec)`, 6.1030 % of peak wrong at its own 30
steps and 15.4906 % at 80, all four zero-pattern assertions holding, 0
warnings (§2.6.1). It surfaced only because MATERIAL 8 replaced the
directory sample with the whole fast suite; reading the diff would not have
shown it, and neither would re-running `tests/unit/runners`. Fixed the same
way as the other two, with a liveness assertion added.

### 8.4 Round-4 suite run — counts

Round 3's sweep was `tests/unit/runners`, `tests/contracts`, `tests/locks`
and ruff. That is not a sufficient gate for a change to a **shared runner**,
and the proof is that the second false green (§2.6.1) was in
`tests/unit/boundaries`, a directory the sweep never ran. Round 4 ran the
whole fast suite the CI runs — the entire `tests/` tree under the repo's
default `addopts`, `-m 'not gpu and not slow and not slow_physics'` — on the
committed tree, plus the CI ruff gate
(`ruff check rfx/ tests/ validation/ --select E,F,W --ignore
E501,F401,E741,E731,E701,E702,E402`).

Whole `tests/` tree, 2026-09-15, on this worktree (macOS arm64, CPU only,
2 virtual devices from the root `conftest.py`):

```
5 failed, 9023 passed, 52 skipped, 413 deselected, 32 xfailed
in 4157.83s (1:09:17)
```

**All five failures are pre-existing on `origin/main` 883615c6** and none of
them touches the distributed lane. Checked by running the same five in a
pristine 883615c6 worktree with none of this branch in it — `5 failed, 12
passed`, the identical five:

| test | why it fails on main too |
|---|---|
| `tests/crossval/test_cv11_normalization_evidence.py::test_historical_tables_do_not_imply_flux_or_ad_coverage` | committed cv11 table vs replay |
| `tests/crossval/test_slab_family_code_motion_identity.py::test_records_and_masks_are_bit_identical[cv22]` | float64 last digits in a committed artifact — e.g. `mean_mag_abs_diff` replays as `0.0009904761904761554` against a committed `0.000990476190476166` |
| the same `[cv23]` | as above |
| `...::test_committed_artifacts_replay_to_the_same_verdicts[cv22]` | as above |
| the same `[cv23]` | as above |

They are a crossval artifact/platform-arithmetic item for that lane, not a
B0 regression, and they are recorded here rather than left for the next
reader to rediscover. Directory counts on the same tree, for the four the
task list names plus the two round 4 added:

| selection | result |
|---|---|
| `tests/unit/runners` + `tests/contracts` + `tests/locks`, one invocation | **2522 passed, 14 skipped, 71 deselected** (10:35) |
| `tests/unit/boundaries` | **201 passed, 2 skipped, 8 deselected** (3:10) |
| `tests/unit/runners/test_distributed_admission_refusals.py` alone | **83 passed** |
| ruff, the CI gate (`--select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402` over `rfx/ tests/ validation/`) | **All checks passed** |

One caveat stated rather than hidden: the whole-tree run above was started
before this §8.4 paragraph and one docstring line in
`tests/unit/boundaries/test_boundary_pmc_distributed.py` were written (a
renamed-symbol reference, no behaviour). `tests/unit/boundaries`,
`tests/unit/runners`, `tests/contracts` and `tests/locks` were re-run after
both edits, on the tree exactly as committed; their counts are the ones in
the table.

### 8.5 Round-5 record — the 8bc6c084 rebase

`git rebase origin/main` (88651a1c → 8bc6c084, 27 commits). One textual
conflict, in `tests/contracts/test_evidence_numeric_provenance.py`: main
added #928 item 2's `slab_family_per_arm_lattice_window_predeclaration.md`
row to `CLASSIFICATION` at the same place this lane adds its own note's row.
Both rows are kept; no other file conflicted.

A clean text merge is not the same as a correct one, and here it was not.
Two things main changed needed a real answer:

- **`rfx.runners.run_distributed` is gone** (#1038 leg 6). Class 7 (§2.7)
  was written when the package-level name was itself the open door, and two
  tests imported it: `test_the_exported_runner_is_the_pmap_one_and_it_is_gated`
  and `test_the_y_z_refusal_fires_in_the_v1_pmap_runner_too` (× 4 specs).
  Those five were the only failures the rebase produced. Re-pointed to
  main's truth rather than around it: the y/z test reaches the pmap runner
  by full module path, and the export test became
  `test_the_pmap_runner_is_still_reachable_and_still_gated`, which asserts
  the package-level name is gone, that `rfx.runners.distributed.run_distributed`
  still resolves to the pmap runner, and that `distributed_v2` still
  delegates to it at `n_devices == 1`. No refusal changed.
- **Line-number citations re-anchored onto 8bc6c084**, in the refusal
  messages, the gate docstrings, the test module docstring and this note.
  `rfx/runners/distributed.py` "non-periodic" `:351/:380/:415/:440` →
  `:330/:359/:394/:419`; its `xlo`/`xhi` slices `:807-808` → `:786-787`;
  its v1 gates `:1373`/`:1483` → `:1352`/`:1462`;
  `rfx/runners/uniform.py` `setup_wire_port` `:419-420` → `:427-428` and
  `if pe.excite:` `:421,440` → `:429,448`; `rfx/api/_execute.py:3671` →
  `:3713` and its NU-forward branch `~:2318–2340` →
  `_forward_distributed_nonuniform_from_materials`, `~:2365–2378`;
  `rfx/runners/distributed_nu.py` `:429`/`:435` → `:401`/`:406`; the port
  fork `:1414/:1422` → `:1393/:1401` on main and `:1473/:1481` on HEAD (v2's
  own fork: `:733/:740` → `:1599/:1606`). Verified unchanged and left as
  they were: `rfx/grid.py:110-111`, `:112`, `:153-154`,
  `rfx/simulation.py:890`, `:942`, `:1117`, and
  `tests/unit/runners/test_distributed.py:149`, `:291`.

**Nothing main added opens a new way into any of the seven classes.**
Checked rather than assumed: `BOUNDARY_TOKENS` is unchanged
(`cpml`, `upml`, `pec`, `pmc`, `periodic` — no new token, no Bloch token);
no new port kind or port field, and no new monitor kind, is added to
`Simulation`; `_cpml_profile`'s grading — which class 5's one-free-cell
argument rests on — is untouched by #1047 (that change is per-component
`ce`, not the `rho` ramp); #1053's declared-PEC-volume admission is about
volumes, not about the six face pads class 6 reads; and the one lane the
note already records as ungated for classes 1–3, `forward()`'s
`fwd_distributed_nu`, is still reached only from `forward()` and was not
widened.

What is **owed and not done here**: a re-derivation of the RED table on
8bc6c084. #1041/#1055 moved source injection before the E ghost exchange in
all three distributed runners, so the printed digits of the
before-the-refusal measurements are unlikely to reproduce on this tree. The
header says so; the figures keep the trees that produced them.

Counts on the rebased tree (macOS arm64, CPU only, 2 virtual devices):

| selection | result |
|---|---|
| `tests/unit/runners` + `tests/unit/boundaries` + `tests/contracts` + `tests/locks`, one invocation | **3597 passed, 31 skipped, 81 deselected** (22:55) |
| ruff on every file this branch changes | 3 pre-existing `F401` re-export findings, **identical on a pristine 8bc6c084 worktree** (`_update_e_local_nu`, `_update_h_local_nu`, `gather_array_x`), none from this lane |
| ruff, the CI gate (`--select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402` over `rfx/ tests/`) | **All checks passed** |

Zero failures, so no pre-existing-on-main exemption is claimed this round —
round 4's five crossval failures (§8.4) are in `tests/crossval`, outside this
selection. `tests/unit/boundaries` is in the selection deliberately: §8.4
records that leaving it out is how round 3's second false green survived.
