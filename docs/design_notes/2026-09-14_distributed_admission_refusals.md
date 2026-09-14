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
`tests/unit/runners/test_distributed_admission_refusals.py`. Each one is
reachable through the public `sim.run(devices=[d0, d1])` — not only through
the internal runner. **None of the five raised. None of them warned.**

### 2.1 Periodic / Bloch boundaries — silently non-periodic

`set_periodic_axes('y')` (and `BoundarySpec(y=Boundary('periodic',
'periodic'))`, which sets the same attribute), 24×12×12 mm PEC box at
dx = 1 mm, 40 steps, source at x = 6 mm, probes at x = 12 / 22 mm:

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

Bloch. `run()` on main has no `bloch=` parameter: `bloch` appears only in
`rfx/simulation.py` and `rfx/core/yee.py`, derived from oblique TFSF, and TFSF
on this lane already falls back to a single device. So the periodic-axes half
is the reachable one today; `refuse_unsupported_distributed_features(...,
bloch=...)` takes an explicit phase as well, so a future one cannot slip in
behind the refusal.

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

Cause. `excite` does not occur in `distributed_v2.py` or `distributed.py` at
all; `rfx/runners/uniform.py:421,440` honours it (`if pe.excite:` guards both
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
has a cross-rank reduce for one (`flux` and `ntff` do not occur in either
file). This is the same class as the #579 DFT-plane refusal that the dispatch
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
last rank. A rank owns the real cells `[ghost, ghost + nx_per)` of its own
slab, so the windows stay inside owned cells exactly while

```
n <= nx_per            (x-lo face)
n <= nx_per - pad_x    (x-hi face)
```

That is the condition implemented, per face — not `cpml_layers > nx_per -
ghost`: `ghost` shifts both ends of the x-lo window together and cancels, and
it is `pad_x`, not `ghost`, that eats into the last rank's usable depth.

Measured with `x_lo='cpml'` / `x_hi='pec'`, 6×8×8 mm at dx = 1 mm,
`cpml_layers=8`, 2 devices, 60 steps → `nx=15`, `pad_x=1`, `nx_per=8`, so the
x-hi window wants cells `[0, 8)` of a slab whose owned cells are `[1, 9)`:

| quantity | value |
|---|---|
| `max abs(Ez_distributed − Ez_native)` | 2.207244 |
| native peak | 4.416633 |
| relative | **50.0 % of peak** |
| x-hi-face probe alone | **99.9994 %** of its own peak |
| warnings | 0 |
| the same model at `nx_per=17` (24 mm domain) | 9.7e-05 of peak — parity |

One cell of overflow keeps the window's *length* at `n`, so every array shape
still matches and nothing raises: the absorber simply updates a halo cell
instead of the owned cell it should. Two or more cells of overflow run past
the slab end, the clipped window is shorter than the psi arrays, and XLA
raises `mul got incompatible shapes for broadcasting: (20, 1, 1), (15, 53,
53)` — which names no feature either. Both ends of that band are now refused
with one message.

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
  cpml_layers, lane=...)` — class 5, position-dependent, `ValueError`. Called
  inside `run_distributed()` at the point where `nx_per` / `pad_x` / `ghost`
  are known and before the state is split into slabs. The message carries
  `nx`, `n_devices`, `nx_per`, `pad_x`, `cpml_layers`, which face overflows
  and by how much, and the largest device count that *does* fit at this `nx`
  (searched exactly rather than estimated as `nx // cpml_layers`, because
  `pad_x` is not monotone in `N`).

Placement inside `run_distributed()` is deliberate on both sides:

- **after** the TFSF and waveguide-port fallbacks — those run the whole model
  on one device, which is a right answer rather than a silently wrong one, and
  that single-device lane honours all five features;
- **before** the `n_devices == 1` delegation to the pmap runner — that runner
  carries the same four gaps.

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
  `test_the_tfsf_and_waveguide_fallbacks_are_deliberately_unchanged` so the
  admission gate cannot quietly swallow them. Whether they should become
  refusals is a separate decision with a separate cost (they work today).
- **The cut-plane census and automatic cut-plane moving** (§4 (2)(3) of the
  direction note, lane B2) and **the grade map** (§4 (4)). Nothing here
  inspects *where* the cut falls relative to a dielectric interface, a PEC
  cell, a dispersive pole or a source index. B0 is position-independent by
  construction, with class 5 the single exception, because the absorber's
  position is fixed by the frame.
- **The x-hi CPML window on a PEC x-hi face.** The runner applies both x-face
  windows whenever `sim._boundary == "cpml"` and `grid.cpml_layers > 0`,
  without consulting `grid.face_pads` — so the asymmetric fixture in §2.5 gets
  an x-hi absorber correction at a reflector face. That is a *different* gap
  (a face-pad-blind runner, visible as the 9.7e-05-of-peak floor in the
  fitting control), it is not in the note's §3 list, and closing it changes
  numbers rather than closing a silent path. Left for lane B, where the
  distributed CPML outer termination is the subject.

## 6. Tests

`tests/unit/runners/test_distributed_admission_refusals.py`, 31 tests. For each
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
