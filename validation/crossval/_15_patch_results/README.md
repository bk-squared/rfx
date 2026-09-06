# `_15_patch_results/` — record ledger

Producer: `validation/crossval/15_patch_antenna_rt5880.py`. Unlike case 05 this script
writes fixed filenames (`rfx.json`, `openems.json`), so a regeneration replaces the file
in place; the superseded content lives in git history.

| File | Leg | Status |
|---|---|---|
| `rfx.json` | rfx, `two_plane=True` ground, uniform `dx = h/4` | **CURRENT.** Regenerated 2026-09-06 under issue #912: VESSL run 369367258715 on remilab-c0, repo SHA `f5ee3b59c3e83832f7df1682f0ad30abb260a42e` (SHA-guarded), image `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8`, `jax[cpu]==0.6.2`, command `15_patch_antenna_rt5880.py rfx --num-periods 45 --n-freqs 181 --gain` (the committed leg's own command). |
| `openems.json` | openEMS, same geometry | Unchanged. PR #897 is rfx code and cannot move an external-solver record. |
| `rfx_one_plane_ground_b29f9de7.json` | rfx, PRE-#740-fix one-plane ground | Frozen historical negative control. The filename pins commit `b29f9de7`; the shipped CLI has no one-plane mode (`build_rfx_sim(two_plane=False)` is reachable only from the test suite), so it is not regenerated. Its cited quantity `f_harminv_hz` is a ring-down number that no S-parameter extractor change can move. |

Superseded `rfx.json` (the leg PR #768 committed at `1f005d0d`, 2026-08-29):

    git show 7d1e9733:validation/crossval/_15_patch_results/rfx.json

## What moved, and what actually caused it (issue #912, measured)

The 2026-09-06 job ran the same command twice under one image and one set of pip pins:
leg `main` at `f5ee3b59` and an A/B control leg `ctrl` at `aa888b7a`, PR #897's immediate
parent. Both SHA-guarded.

**Committed (`1f005d0d`) → regenerated:** `f_dip_hz` 2.310000 → 2.320000 GHz (+10 MHz,
exactly one frequency bin); `s11_dip_db` −4.4298391 → −0.3181958 (+4.1116 dB; |S11| at
the dip +60.5 %); `max_abs_s11` 0.786966 → 0.998132; `max |dS11| = 0.9767` over the band.

**That move is NOT PR #897, and #912's ~0.002–0.005 premise does not describe it.** The
control leg at PR #897's parent already reads `s11_dip_db = -0.3448075`,
`max_abs_s11 = 0.996699`, `f_dip_hz = 2.320000 GHz` — i.e. the whole 4.1 dB is present
*before* the half-step correction. It came from the wire-port one-port diagonal being
rewritten after this leg was recorded: PR #776 (`7c80714c`, 2026-08-30, whole-port driven
diagonal — "matched load reads ~0, PEC short reads −1") and PR #777 (`ad13b4c7`,
2026-08-31, uniform-lane POST flip + decomposer recalibration). The committed leg dates
from 2026-08-29 and has been stale against the committed solver since 2026-08-31. This staleness was
NOT first found here: `docs/design_notes/20260901_patch_mode_identification_predeclaration.md`
§6.6 (2026-09-01) already bisected 1f005d0 → ad13b4c^ → main, named PR #776/#777, wrote "it is
stale, not unprovenanced" and asked for a separate issue; the #812 round-1 lane then regenerated
the leg to −0.3448 dB and KNOWINGLY reverted it (test docstring, design-note §6.9), which
restored an artifact no SHA produces. #912 reverses that revert. The physics question the
live leg raises — rfx dips only −0.32 dB where openEMS dips −20.1 dB on the same geometry —
is tracked as **issue #920** (it predates #897; #897 moves it by 0.027 dB). See also the
appended 2026-09-06 section of `docs/design_notes/20260901_numeric_provenance_gate.md`.

**PR #897's own contribution**, isolated on `ctrl` → `main`: `f_dip_hz` unchanged;
`s11_dip_db` −0.3448075 → −0.3181958 (+0.026612 dB, |S11| +0.307 %); `max_abs_s11`
0.996699 → 0.998132; `max |dS11| = 0.007284` at the top bin (3.4 GHz). The per-bin
current rotation recovered from the pair is `pi*f*dt` with `dt = 1.513304e-12` s against
the run's own grid `dt = 1.513439e-12` s (1 part in 1e4) and a linearity residual of
6.4e-6 rad; re-rotating the `ctrl` S11 by `exp(+j*pi*f*dt)` on the current channel
reproduces `main` to `3.8e-6`. That is the advertised mechanism and nothing else.

**Unchanged in every leg** (the correction touches a DFT accumulator, not the field
update): `f_harminv_hz`, `f_primary_hz`, `q_harminv`, `settle_db` (−53.99 dB, SETTLED
against the −40 dB bar), `settled`, `n_steps`, `n_sub_cells`, `dx_um`, `f_analytic_hz`,
the full `preflight` text and `stack_check`. `gain_dbi` moves in the 7th significant
digit (7.2424231 → 7.2424226), a float32 reduction-order difference between hosts, not a
model change.

**Gate status after the regeneration** (no gate value was changed): cv15's `compare`
passivity bound is `max|S11| <= 1.05` and the new leg reads 0.998132 — inside it, but
with far less headroom than the superseded 0.786966. The 8 % `f0` envelope is judged on
`f_harminv_hz`, which did not move.

### Provenance-gate remedy: what was actually done (disclosure)

The #812 remedy text reads "re-anchor criterion (B) on ANOTHER measured defect ... do not delete the
arm". This ingest did NOT pick another defect: it re-anchored the SAME key `rfx.json::s11_dip_db` on
its live value (−0.3182 dB) and left `CV15_REGENERATED_VALUE = −0.3448069` as the mutation (now a
value `aa888b7a` actually produces). The arm still fires, no constant moved, `tests/` is untouched —
but the wording of the remedy was not followed literally, and that is stated here rather than implied.
`rfx.json` itself carries no provenance block (the script at f5ee3b59 writes none); run id and SHA live
in this README and the CHANGELOG.
