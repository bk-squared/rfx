# Brief for Codex — option B sweep driver (measurement only, no product change)

Read `/root/workspace/bk-workspace/.801-measure/B/DESIGN.md` in full: it is the leader's design and
you build exactly the measurement it describes. Reporting rules as in
`/root/workspace/bk-workspace/.801-measure/BRIEF_implement_conductor_continuation.md` ("Command safety
(hard rules)" and "Report" sections apply verbatim; you do not interpret results; `Conclusion: leader
fills after the sweep.` where one would go). Work in a NEW worktree:
`git -C /root/workspace/bk-workspace/rfx worktree add --detach /root/workspace/bk-workspace/rfx-wt-B-sweep origin/main`
(STOP if it exists). No commits are needed in the repository: everything you write goes under
`/root/workspace/bk-workspace/.801-measure/B/` (drivers, YAMLs, results, `REPORT.md`). Nothing over
~10 CPU minutes on the pod; every solve runs on VESSL (remilab-c0, gpu-rtx4090; job-file rules in
`/root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md`; working examples of a job file and
driver: `/root/workspace/bk-workspace/.801-measure/alpha/`, which scaled alpha the same way). Launch each
job once; record run ids in `run_id.txt` files; poll with short `vessl run read` calls.

## FACT (leader)
- `alpha = 0.05 * (1.0 - rho)` is a literal in `rfx/boundaries/cpml.py:_cpml_profile` (:158); the
  `.801-measure/alpha/` driver scaled it by monkeypatching that function and the scaling reached the
  solve (the four factors gave four different ring-downs). Reuse that mechanism; add factor 0.
- `add_msl_port(..., terminates=())` (since #1178) makes the port end no conductor, so the strip
  continues through the absorber while the port still excites and probes.
- The four rigs are named in DESIGN.md with their files. The WR-90 battery builder and the reflectivity
  oracle rig may need a small local variant (an absorber end instead of a second port; a frequency
  window down to 0.2 GHz); write the variant in your driver, never in the repository files, and print
  the realized grid, pads and port apertures of every arm ("assert realized, not declared").

## ASSUMPTION (check, report)
- B1: the reflectivity oracle's clean-reference method still resolves reflections at −60 dB with the
  band widened to 0.2–6 GHz (the reference domain must be resized so its wall echo lands after the
  window). Report the window and the reference size you used.
- B2: on rig 1 with `terminates=()` on both ports, the one-port |S11| from the MSL lane's extractor is
  a valid reading of the absorber termination (the far port removed or made passive with no strip
  under it). Report which of the two you did and why the extractor accepts it.

## Deliverables
1. `B/results/<rig>/<s>_<N>.json` per arm with the numbers DESIGN.md names, the witnesses (end-of-run
   energy vs post-source peak; settling dB; preflight text verbatim), realized grid and pads, commit,
   run id, dtype.
2. `B/TABLE_<rig>.md`: one table per rig, rows N, columns s, cell = the rig's headline number; a second
   table with the witness. Rig 4 additionally: reflection vs frequency per (s, N) as a CSV, and the
   frequency at which |R| crosses −40 dB and −60 dB for each arm.
3. The low-frequency arm of rig 1 (band down to 0.5 GHz) as its own table.
4. `B/REPORT.md` (under 120 lines): B1/B2 outcomes, every STOP with options, the tables, commands
   with last lines, run ids. No sentence that says what a number means.
When done, remove the worktree: `git -C /root/workspace/bk-workspace/rfx worktree remove /root/workspace/bk-workspace/rfx-wt-B-sweep`.

## Addendum 1 (leader, after REPORT.md's STOP-1)
`git -c safe.directory=<worktree>` cannot work in the image: a worktree's `.git` is a FILE pointing at
`/root/workspace/bk-workspace/rfx/.git/worktrees/<name>`, a second path Git also refuses. Do not run
git in the job at all. Export the source once on the pod, the way `.801-measure/src-main` was made:
`git -C /root/workspace/bk-workspace/rfx archive --format=tar origin/main | tar -x --no-same-owner -C /root/workspace/bk-workspace/.801-measure/B/src`
(after `mkdir` of that directory; STOP if it exists), write the commit hash to `B/src/PROVENANCE.txt`
with `git -C /root/workspace/bk-workspace/rfx rev-parse origin/main`, and point every job's `$src` at
`B/src`; the job copies `PROVENANCE.txt` into its output instead of calling `git rev-parse`, and the
clean-tree check becomes a sha256 of the driver files against `job_source_hashes.json`. The removed
worktree is not recreated. Relaunch the four jobs once each (new run ids), then complete the
deliverables. Report as `B/REPORT_2.md`; leave `REPORT.md` as the record of the failed launch.

## Addendum 2 (leader, after REPORT_2.md) — comparator first, on rig 4
Rig 4's numbers cannot yet be read: at s = 1, N = 8 the variant reads |R| = −20 dB at 2 GHz and gets
WORSE with frequency (−36 dB at 1 GHz, −13 dB at 6 GHz), while the committed oracle
`tests/oracle/test_pml_reflectivity.py` reads −68.3 dB at 2 GHz with the same eight layers. A graded
absorber does not reflect more at higher frequency; the variant (uniform 5×5 source, periodic
transverse boundaries, two-ended-box residual over incident) is the suspect, not the absorber.
1. Run the committed oracle EXACTLY as the test does (its own rig, point source, its window and
   reference) at s ∈ {0, 1, 3} × N ∈ {4, 8, 16} on GPU; record its number per arm. That is rig 4's
   table now; the variant's tables are moved under `failed_instrument/` and are not cited.
2. Then, with the oracle's method unchanged, sweep the source centre frequency over {0.5, 1, 2, 4} GHz
   (window and reference resized each time so that the reference's wall echo lands after the window,
   as the oracle's docstring prescribes; report the sizes) at s ∈ {0, 1, 3}, N = 8 — this is how f_alpha
   is read, one number per centre frequency, not from a wide-band spectrum.
3. State, in the report, the one difference between the variant and the oracle that accounts for
   the 48 dB, if you can find it by measurement (for example: drive the oracle rig with the variant's
   source; drive the variant rig with the point source); if not, say "not found".
On rigs 1 and 2: the energy witness reads −39.3 / −36.4 dB (end / last 5 %) against the post-source
peak while the field probes settle at −104 dB; report, for s = 0 N = 8 on rig 1, what carries that
residual energy (its spatial location: interior, under the port, in the absorber) with a field dump
at the end of the record. For rig 2 at N = 4 the +24 dB |S11| is non-physical: report the same
diagnostic and the extractor's own passivity warning text. No other arms are re-run.
Report as `B/REPORT_3.md`.
