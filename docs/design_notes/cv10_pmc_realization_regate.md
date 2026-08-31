# cv10 re-gate — PMC realization + image-doubling control arm

Lane: `agent/regate-cv10`. Case: `validation/crossval/10_pmc_cpml_half_symmetric.py`.
Tracker: issue #812 (crossval gate audit), critical tier, first entry.
Status of this section: **PRE-DECLARATION**. Written and committed BEFORE the
measurement that judges it. Append-only — later sections correct, never rewrite.

This is instrument work. **No physics verdict of cv10 is challenged.** The
v1.7.5 PMC/CPML composition fix that cv10 locks is not in question; what is in
question is whether cv10's gate can fail when that fix is absent.

## 1. What the audit measured (prior measurement, not this lane's)

Issue #812 reports that monkeypatching `rfx.boundaries.pmc.apply_pmc_faces` to
a no-op — i.e. deleting the boundary condition the case is *named for* — makes
cv10 score **better**, not worse:

| leg | peak spread `(max−min)/max` | gate |
|---|---|---|
| uniform, correct code | 0.2487 % | < 2 % PASS |
| uniform, `apply_pmc_faces` → no-op | 0.0667 % | < 2 % PASS (better) |
| non-uniform, correct code | 0.00784 % | < 2 % PASS |
| non-uniform, `apply_pmc_faces` → no-op | 0.0201 % | < 2 % PASS |

with the probe peaks scaled by 0.5077 and the field bit-identical to a PEC wall.

## 2. The mechanism, stated so the repair is forced to address it

Every current cv10 gate is a **within-path relative spread**:

```
peak_range = (peak_max - peak_min) / peak_max      # over cpml_layers ∈ {2,4,6,8}
```

`peak_range` is invariant under `peak_i → c · peak_i` for any constant `c > 0`.
Deleting a boundary condition on a face that the direct source→probe path never
touches is *exactly* such a constant factor: the array truncation silently
supplies a PEC (`E_tan = 0`) wall in place of the absent PMC (`H_tan = 0`) wall,
which flips the image-source sign and rescales every leg of the sweep by the
same factor. The gate divides that factor out by construction.

So the repair cannot be a tighter relative spread. It must add (i) a **direct
realization check** — is the wall the thing it claims to be — and (ii) an
**absolute amplitude reference** that a constant factor cannot cancel.

## 3. Gate 3 (new) — PMC realization, bit-exact

**Statement.** At the end of every swept run, on the declared `y_lo` face:

```
max |Hx[:, 0, :]| == 0.0   and   max |Hz[:, 0, :]| == 0.0    (bit-exact)
```

with the non-degeneracy guard `max |Hx| over the whole array > 0` (so an
all-zero field cannot satisfy the gate vacuously).

**Threshold derivation — definitional, nothing fitted.** `0.0` is not a
tolerance. `rfx/boundaries/pmc.py::apply_pmc_faces` writes the literal `0.0`
into `hx[:, 0, :]` and `hz[:, 0, :]` on a `y_lo` face, and the scan body
(`rfx/simulation.py:1338-1340`, `rfx/nonuniform.py:1326-1327`) applies it after
the CPML-H stage, so nothing re-populates those cells before the state is
returned. Any code path in which the wall is not realized leaves the ordinary
H update in place and the face is generically non-zero. There is no window to
widen and no number to fit. The guard is likewise threshold-free (`> 0`).

**Both paths, all four `cpml_layers` values.** Free — no extra run; it reads
`result.state` of runs the case already performs. This is the leg that covers
`84b11aa` (the NU scan body previously never called `apply_pmc_faces` at all),
which is the historical defect cv10 was written for and which its relative
spread cannot see.

## 4. Gate 4 (new) — image-doubling control arm, absolute amplitude

**Geometry, derived from the PMC-plane convention this file already states.**

Half domain: `y' ∈ [0, 20] mm`, `dx = 1 mm`, 21 y-nodes `j' = 0…20` at
`y' = j'·dx`. `apply_pmc_faces` zeros `Hx`/`Hz` at array index 0, and those
components sit at `y' = 0.5 mm`. `H_tan = 0` there ⇒ `Ez` is **even** about
`y' = 0.5 mm` ⇒ node `j'` mirrors onto node `1 − j'`.

Images of the interior nodes `j' = 1…20` therefore land on
`y' = 0, −1, …, −19 mm`. The full symmetric domain is `y' ∈ [−19, 20] mm`
= **39 mm**, 40 nodes, 19.5 mm on each side of the plane. Map
`y_full = y' + 19 mm`; the mirror plane is at `y_full = 19.5 mm`.

- half-domain source `Ez` at `y' = 1 mm` ⇒ full-domain `Ez` **pair** at
  `y_full = 20 mm` and `y_full = 19 mm`, equal amplitude, **same sign** (a PMC
  images a tangential electric current in phase; a PEC would reverse it — this
  is the sign the defect flips);
- half-domain probe `y' = 5 mm` ⇒ full-domain probe `y_full = 24 mm`;
- full domain: CPML on **both** y faces, x/z extents and `cpml_layers`
  identical to the half domain.

The full-domain problem is then mirror-symmetric about `y_full = 19.5 mm` in
geometry, sources, and both y absorbers. Its `H_tan` is therefore identically
zero on that plane for all time — exactly the condition the half domain
imposes. The two runs are the same discrete problem, so

```
R = peak|Ez(probe)|_half-PMC / peak|Ez(probe)|_full-image   ==  1
```

identically, in exact arithmetic. The reference leg contains **no PMC face**,
so it is untouched by any defect in `apply_pmc_faces` — the control arm is an
independent measurement of the amplitude the relative spread threw away.

**Gate.** `|R − 1| < 0.02`, evaluated on each path at
`cpml_layers = max(CPML_VALUES) = 8`.

**Threshold derivation — three classes, none of them the data being judged.**

1. *Prior provenance.* `0.02` is the tolerance gate 1 has carried since the
   v1.7.5 lock (2026-04-20). It is re-used **unchanged and not widened**, now
   applied to an ABSOLUTE amplitude comparison where a constant multiplicative
   factor cannot cancel. Moving a number sideways onto a stricter comparator is
   not a new window.
2. *First-principles residual budget.* The discrete image identity above is
   exact, so the only residuals are (a) float32 accumulation over 800 steps,
   `≈ √800 · 2⁻²⁴ ≈ 2e−6` relative, and (b) the CPML reflection floor, budgeted
   at the conventional **−40 dB = 1 %** for an 8-layer polynomial-graded
   absorber. `0.02` is ≥ 2× that floor and ~10⁴× above (a).
3. *Discrimination margin.* The competing hypothesis — the wall is not
   realized, so array truncation supplies a PEC wall — **reverses the image
   sign**, turning a constructive pair into a destructive one. That is an O(1)
   change in absolute amplitude for any sub-wavelength plane offset (here
   `k·d = 2π·0.5 mm / 60 mm = 0.052 rad`), i.e. tens of percent at minimum. A
   2 % window sits more than an order of magnitude inside it.

`cpml_layers = 8` is chosen a priori as the thickest absorber in the existing
sweep — the leg for which the −40 dB budget in (2b) is best justified — not
because of anything observed.

**Cost.** Two extra runs (one per path), same scale as the eight the case
already performs.

## 5. Correction to the module docstring of `10_pmc_cpml_half_symmetric.py`

The docstring at `649b2cf` states, of the half-cell PMC-plane offset:

> "UNLIKE cv09, this offset biases NO comparator here: this crossval is a
> self-consistency / regression lock … so the half-cell shift changes no
> pass/fail gate … there is no gate-3 class comparison against a mirrored full
> geometry to protect."

**That was true only while every gate was a within-path relative spread, and
gate 4 makes it false.** Gate 4 is precisely a comparison against a mirrored
full geometry. The `y' = 0.5 mm` plane location is now load-bearing: it fixes
the full-domain length at **39 mm, not 40 mm**, and the image pair at
`y_full = 19/20 mm`. Getting the plane wrong by a half cell would move the
control-arm geometry and break gate 4. The docstring is updated accordingly;
the superseded sentence is recorded here rather than merely deleted.

## 6. Acceptance criteria for this lane (two-sided, both required)

- **(A)** the re-gated case still exits 0 on today's `main`;
- **(B)** with `rfx.boundaries.pmc.apply_pmc_faces` monkeypatched to
  `lambda state, faces: state`, the re-gated case exits 1, and it does so on
  gates 3 and 4 rather than on gate 1 (gate 1 is expected to keep passing —
  that is the audit's whole point, and it is not a defect of gate 1 but of
  relying on gate 1 alone).

(A) alone is cosmetic; (B) alone means the case is broken. A permanent pytest
falsifier pins (B) so it cannot silently regress.
