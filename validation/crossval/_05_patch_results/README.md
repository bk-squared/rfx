# `_05_patch_results/` — record ledger

Producer: `validation/crossval/05_patch_antenna.py` (full three-part run; the JSON is
written only when `RFX_CROSSVAL05_JSON` is set). Filenames carry the VESSL run id that
produced them.

| File | Run | Repo SHA | Status |
|---|---|---|---|
| `cv05_run_openems_369367258715.json` / `.log` | VESSL 369367258715, remilab-c0, 2026-09-06 | `f5ee3b59c3e83832f7df1682f0ad30abb260a42e` | **CURRENT.** Post-PR-#897 (wire-port half-step current-DFT phase) and post-PR-#847 (mode-resolved selector). |
| `cv05_run_openems_369367257743.json` / `.log` | VESSL 369367257743, remilab-c0, 2026-09-02 | the #812 working checkout, pre-#847 | **HISTORICAL, kept on purpose.** This is the artifact `validation/crossval/manifest.json` (case `05_patch_antenna`) and `docs/design_notes/20260901_patch_mode_identification_predeclaration.md` §6.11 cite by name for the #812 mode-identification measurement. Its `rfx_s11*` block predates PR #897; do not read S11 from it. |
| `cv05_ringdown_fixture_rebuild_369367257743.log` | VESSL 369367257743 | — | build log for `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`. |

## What differs between the two records (issue #912, measured)

The 2026-09-06 job ran the same script twice under one image and one set of pip pins:
leg `main` at `f5ee3b59` and an A/B control leg `ctrl` at `aa888b7a`, which is PR #897's
immediate parent. Both SHA-guarded in the job.

1. **`ctrl` reproduces the 2026-09-02 record's `rfx_s11` array bit-for-bit**
   (bit-identical: `max |dS11| = 0.000000` over all 101 bins; 2.5e-16 is the mechanism-fit residual at theta~0, not this comparison), and every openEMS-leg value to the digit
   (`max |dS11_openEMS| = 0.0`). The committed record was therefore a faithful pre-#897
   measurement, and the only physics difference between the two files is PR #897.
2. **PR #897's effect**, isolated on `ctrl` → `main`: `rfx_s11_dip_hz` unchanged at
   2.32 GHz; `rfx_s11_min_db` −1.614843 → −1.583059 (+0.031784 dB, i.e. |S11| at the dip
   +0.367 %); `rfx_s11_max_abs` 0.970578 → 0.972184; `max |dS11| = 0.006850` at the top
   bin (3.5 GHz). The per-bin current rotation recovered from the pair is `pi*f*dt` with
   `dt = 7.78397e-13` s and a linearity residual of 1.1e-5 rad; re-rotating the `ctrl`
   S11 by that factor reproduces `main` to `8.9e-6`. That is the PR #897 mechanism and
   nothing else.
3. **Everything not derived from the port S-parameters is identical** in all three
   files: `rfx_harminv_hz`, `rfx_vs_analytic_pct`, `rfx_vs_openems_harminv_pct`,
   `rfx_internal_pct`, `rfx_mode_id_ok`, `openems_mode_id_ok`, `openems_harminv_hz`,
   `openems_s11_*`, `analytic_resonance_hz`, `mode_identification_tol`, `rfx_modes_hz`.
   PR #897 changes a DFT accumulator, not the field update, so this is what it must do.
4. **Two differences that are NOT PR #897 and not physics.** The 2026-09-02 file carries
   `status: "failed"` and lacks `openems_mode_id_gated`, `openems_mode_id_reasons`,
   `openems_modes_hz`, `openems_mode_assignment`. Both legs of the new job read
   `status: "passed"` and carry those four keys. The cause is PR #847 (`7794a334`,
   2026-09-03), which took the openEMS identification leg out of `all_ok` and added the
   reported mode list — a change to the *script*, landed after the 2026-09-02 run. The
   new record's openEMS mode list is exactly what the manifest's case-05 `claim_scope`
   describes in prose (seven poles; TM010 claimed by 2, TM100 by 3, TM110 by 2), which
   the older file never contained.
