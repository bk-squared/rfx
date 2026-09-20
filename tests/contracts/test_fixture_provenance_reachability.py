"""Every commit sha committed into a fixture still resolves for a reviewer.

Issue #1013. A fixture that records ``"rfx_commit": "6a369c27..."`` is making a
claim a reader is supposed to be able to check. After the producing PR is
squash-merged that sha is reachable from nothing::

    $ git merge-base --is-ancestor 6a369c27 origin/main; echo $?
    1

Nothing in this repository noticed. There was no fixture-provenance contract
test at all before this module -- ``git grep rfx_commit -- tests/contracts/``
returned zero hits -- and the three generators that write the key had three
independent copies of ``subprocess.run(["git","rev-parse","HEAD"]).stdout
.strip()``, none of which checked the return code. Outside a work tree that
call exits 128 and the ``.strip()`` writes ``""``.

WHAT THIS GATE IS. An **enumerate-and-classify** contract in the shape #928
established: every tracked JSON under ``tests/fixtures/`` is walked for
provenance-shaped sha keys, and every site found must land in exactly one
named bucket. A site in no bucket fails. There is no wildcard and no silent
skip, so a generator that stops emitting the witness turns its own site red
rather than disappearing from the census.

THE BUCKETS, in the order they are tried:

  A  ``reachable``      -- the sha is an ancestor of the reference ref.
  A' ``reachable-head`` -- not on the reference ref, but an ancestor of HEAD.
                           This is the in-flight arm; see REFERENCE REF below.
  B  ``code-tree``      -- the sha is gone, but the site carries a sibling
                           ``rfx_code_tree`` equal to the ``rfx/`` subtree of
                           some commit on the reference ref.
  B' ``code-tree-head`` -- same, witnessed on HEAD's history only.
  C  ``annotated``      -- listed in :data:`ANNOTATED` with a written reason.
  -- anything else FAILS.

REFERENCE REF, and why the answer differs on a PR and on main. The ref is
``$RFX_PROVENANCE_REF``, else ``origin/main``, else ``main``
(:func:`tests._fixture_provenance.resolve_reference_ref`). Buckets A and B are
ALSO evaluated against ``HEAD``, and that is deliberate: a pull request that
changes ``rfx/`` and regenerates a fixture stamps a sha and a code tree that
exist on the branch and on no main commit yet. Measured: of the 26 recent
merged PRs whose head object is still fetchable, 2 have an ``rfx/`` subtree on
no main commit. Without the HEAD arm those PRs could not go green until they
merged, and the required check is what blocks merging. With it, they are green
at PR time (on the ``pull_request`` event HEAD is the merge commit, whose
history contains both sides) and the gate collapses to the reference ref on the
``push: branches: [main]`` run, where HEAD *is* main.

That asymmetry is the whole mechanism, not a loophole: at PR time the sha IS
reachable, so a PR-only gate can never catch this class. **This gate is a
post-merge detector and its true positive lands on main.** When it does, the
remedy is a follow-up commit that either regenerates the fixture on main or
adds an :data:`ANNOTATED` entry with the measured reason. Widening a bucket to
make main green again is not a remedy.

WHAT A GREEN HERE IS NOT. This is an E0 contract in the sense of
``docs/agent-memory/physics_validation_evidence_rule.md``: it asserts schema
and reachability, moves no gate value and no physical number, and is never
evidence for a physics claim. ``rfx_code_tree`` says "the ``rfx/`` content that
produced this number is on main" -- not "this fixture replays byte-for-byte",
and not "this exact commit is on main". Its measured resolving power is coarse:
787 distinct ``rfx/`` subtrees over 1936 main commits (measured at
``origin/main`` = 541f703f), worst case one subtree
covering 95 commits, and two different fixture-producing commits (``5b588f3c``,
``f914a7ca``) collide on subtree ``7d93cbe0``.

WHO WRITES THE SITES THIS GATE GUARDS, and whether a regeneration keeps the
witness. An annotation whose producer drops it on the next run is the #928
defect ("producer-owned home") repeated, so the answer is written down rather
than assumed:

  KEEPS IT -- the producer emits ``rfx_code_tree`` through
  ``tests._fixture_provenance.capture()``:
    tests/fixtures/{rcs_cube_bem,rcs280_reference_subtraction}/generate.py,
    tests/fixtures/rcs_sphere_three_way/generate_bempp.py,
    scripts/diagnostics/waveguide_chain_battery_measure.py (three fixtures,
      16 sites; its three fixed provenance key tuples now carry the key too,
      which is what a hand edit alone would NOT have survived),
    scripts/diagnostics/waveguide_chain_battery_closure_measure.py,
    scripts/diagnostics/thru_singular_value_dx_ladder.py (3 rung files),
    scripts/diagnostics/build_waveguide_wr90_nu_flux_broad_e4_comparison.py.

  NOT ROUTED, deliberately -- scripts/diagnostics/waveguide_false_lane_column
  _power_suspects.py. An earlier revision routed it and was reverted: it read
  the witness out of another lane's fixture by index, so #873's live diagnostic
  would have started dying the day that fixture stopped carrying the key. Its
  three sites are declared in :data:`ANNOTATED` instead, and they clear when
  #873's lane routes its own producer.

  KEEPS IT CONDITIONALLY -- scripts/diagnostics/cv02_harminv_decimation_ladder
  .py copies the witness from ``validation/crossval/_02_ring_resonator_results/
  crossval.json``, which does not record one yet (it is outside this gate's
  tests/fixtures/ scope). In a clone that cannot resolve ``296cabad`` -- which
  is every fresh clone, origin refuses the object -- a regeneration writes a
  null plus a reason and this gate goes RED naming the file. That is the
  intended failure: visible, not silent. The durable repair is to record the
  witness in the crossval record, and it is the named #1013 follow-up.

  NO LIVE PRODUCER -- hand-maintained adjudication records, so nothing will
  drop the annotation: tests/fixtures/thru_singular_value_dx_ladder/
  verdict.json, tests/fixtures/harminv_decimation/stage1/receipt.json, and the
  ``setup`` / ``normalization_source_audit_2026_09_13`` blocks in
  tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests._fixture_provenance import (  # noqa: E402
    RECORDED_CODE_TREES,
    CODE_TREE_KEY,
    code_tree_of,
    code_trees_on,
    git_available,
    is_reachable,
    is_shallow,
    landing_commit,
    resolve_reference_ref,
)
from tests._git_tracked import tracked_set  # noqa: E402

FIXTURE_ROOT = "tests/fixtures/"

#: Set by the one CI job whose checkout is full-history. Where it is set, a
#: skip is a failure; everywhere else a skip states its reason and stops.
REQUIRED_ENV = "RFX_PROVENANCE_GATE_REQUIRED"

#: Keys whose value, when it looks like a git object name, is a claim about
#: which code produced the number. Measured over the whole repo: every hit
#: under the generic ``sha``/``commit`` names in tests/fixtures/ was a genuine
#: commit reference, not a content digest.
SHA_KEYS = frozenset({
    "rfx_commit", "commit", "git_sha", "git_commit",
    "source_commit", "producer_commit", "rfx_sha", "sha",
})
_HEX = re.compile(r"^[0-9a-f]{7,40}$")

#: Keys whose 7-40 hex value is NOT a git object reference, so a site under one
#: of them is not a provenance claim this gate can or should check. Measured,
#: not assumed: the run-id family is decimal VESSL job numbers
#: (``369367258638``) that merely match the hex pattern.
NOT_A_COMMIT_REFERENCE: dict[str, str] = {
    "run_id": "VESSL job number, decimal",
    "source_run_id": "VESSL job number, decimal",
    "vessl_run": "VESSL job number, decimal",
    "vessl_run_id": "VESSL job number, decimal",
    "vessl_run_good": "VESSL job number, decimal",
    "vessl_run_failed_port_off_grid": "VESSL job number, decimal",
    "pre_fix_run_id": "VESSL job number, decimal",
    "post_fix_run_id": "VESSL job number, decimal",
    "run": "VESSL job number, decimal",
    "coarse": "VESSL job number, decimal (a per-rung run-id map)",
    "mid": "VESSL job number, decimal (a per-rung run-id map)",
    "external_reference_sha256": "truncated content digest of an external file, not a git object",
    CODE_TREE_KEY: "this gate's own witness: a tree sha, not a commit",
}

#: Commit-shaped keys this gate does NOT enumerate yet, with the measured size
#: of the gap. They are the same class of claim as :data:`SHA_KEYS` and they
#: are not fixed here: 28 sites live under these names and **22 of them do not
#: resolve to a commit reachable from main today**, most because they are
#: 7-8 character short shas (``deead58``, ``e682814``, ``287b281``,
#: ``8da9e39``, ``8cdebb9``, ``b0322c1``) that no clone can expand.
#:
#: Widening :data:`SHA_KEYS` to cover them would turn a tight gate into a
#: ~25-entry exemption list overnight, which is how #928's bootstrap exemption
#: became permanent. They are named here instead, with a count, so the hole is
#: a number a reader can see -- and so that a producer cannot escape the gate
#: by renaming its key: :func:`test_every_commit_shaped_key_name_is_classified`
#: fails on any key name in no table at all.
#: Key names classified BEFORE any committed fixture carries them. Exactly one
#: thing belongs here and it is a trap review found: ``tests/_fixture_provenance``
#: `capture()` EMITS ``generator_blob``, twelve producers call `capture()`, and
#: the name was in no table -- so the first fixture regenerated after this PR
#: would have reddened a gate nobody had connected to it. Classifying it now
#: costs nothing; discovering it on somebody else's PR costs them an afternoon.
#:
#: The exact-table rule still applies in both directions: an anticipated name
#: that never lands is caught by nothing, so keep this list at the one or two
#: entries you can justify, and the assertion below moves a name OUT the moment
#: it appears for real.
ANTICIPATED: frozenset[str] = frozenset({"generator_blob"})

COMMIT_SHAPED_NOT_YET_GATED: dict[str, str] = {
    # ARRIVED WITH THE POST-GATE FIXTURES (see the note in ANNOTATED). Counted
    # here rather than gated, for the same reason as the rest of this table: a
    # name is promoted into SHA_KEYS when its producer is routed, not before.
    "sha_full": "8 sites, graded-mesh S-parameter control arms",
    "commit_full": "4 sites, graded-mesh S-parameter control arms",
    # capture() EMITS THIS, and it was in no table until review found it. A key
    # the repo's own shared helper writes on every routed fixture has to be
    # classified, or the first fixture to carry it reds a gate nobody expected.
    # It is a BLOB sha, not a commit, so it is here and not in SHA_KEYS: the
    # reachability question this gate asks is meaningless for a blob.
    "generator_blob": "the generator file's own blob sha, written by "
                      "tests._fixture_provenance.capture(); a blob, so neither "
                      "reachable-from-main nor code-tree applies to it",
    "commit_hash": "16 sites, ALL unreachable -- review re-derived this; the "
                   "earlier '12' was wrong (broad-e5 envelopes; mostly 7-char)",
    "repo_commit": "4 sites, 2 unreachable (patch mode-identification records)",
    "predeclaration_sha": "4 sites, short git object shas of the predeclaration doc",
    "source_sha": "2 sites, both unreachable (msl_s_matrix_golden)",
    "artifact_commit": "1 site, 7-char (broad-e5 rectangular comparison)",
    "baseline_commit": "1 site, reachable today (harminv stage1 receipt)",
    "captured_from_commit": "1 site, unreachable (slab_family_windows_baseline, #928)",
    "verified_identical_to_main_commit": "1 site, reachable today (slab_family baseline)",
    "pre_fix_commit": "1 site (broad-e5 rectangular comparison)",
    "post_fix_commit": "1 site, unreachable; same sha as the gated setup.commit beside it",
}

#: Why the 17 sites that landed on main after this gate was written are
#: declared rather than repaired. One string, because it is one finding, and
#: fifteen copies of a paragraph is fifteen places for it to go stale -- which
#: is what happened: every copy said a code-tree witness "can NOT be derived
#: from any clone", and that is FALSE. Measured 2026-09-20: `git fetch origin
#: refs/pull/<N>/head` brings every one of these objects (f5712d6b, 460bb9b7,
#: 427fc97a), their shared rfx/ subtree is f6dbf87d, and f6dbf87d IS one of the
#: 805 distinct rfx/ subtrees on origin/main. So each of these sites is
#: RESCUABLE, by recording the sha->tree pair in RECORDED_CODE_TREES and having
#: the fixture carry the witness -- exactly what this PR did for the legacy
#: backlog.
#:
#: It is still not done here, and the reason is ownership rather than
#: derivability. The fixture belongs to #873/#1083, both open investigations
#: somebody else is running, and their producers still call git directly and
#: write "unknown" on failure. Back-filling the key without routing the producer
#: is the #928 defect: the next regeneration drops it again. Whoever routes
#: those producers clears these sites in the same change.
_ARRIVED_AFTER_THE_GATE = (
    "arrived on main AFTER this gate was written, and is the #1013 defect itself "
    "in a new fixture: a branch-only sha, unreachable from main and absent from a "
    "fresh clone. It IS derivable -- `git fetch origin refs/pull/<N>/head` brings "
    "the object and its rfx/ subtree f6dbf87d is on main (measured 2026-09-20) -- "
    "so this site is rescuable rather than lost. It is not rescued here because "
    "the fixture and its producer belong to another lane's open investigation, "
    "and back-filling the key without routing the producer only lasts until the "
    "next regeneration"
)


#: Sites that cannot be rescued honestly, each with the measured reason.
#: An entry here is a debt with a name on it, not an exemption: the stale-entry
#: test below refuses any key that no longer resolves to a real site AND any
#: key whose site would now pass on its own, so the table cannot rot into
#: permanence the way #928's bootstrap exemption did.
ANNOTATED: dict[tuple[str, str], str] = {
    # #873's evidence, and #873 is somebody else's open investigation. An earlier
    # revision of this PR annotated this fixture AND edited its producer
    # (waveguide_false_lane_column_power_suspects.py) to emit the witness. Both
    # are reverted: the producer edit read the witness out of another lane's
    # fixture by index, so #873's script would have started dying the day that
    # fixture stopped carrying the key. Changing how somebody's live diagnostic
    # fails is not this PR's to do.
    #
    # So the site is declared instead. It clears when #873's lane routes its own
    # producer through tests._fixture_provenance.capture().
    ("tests/fixtures/waveguide_false_lane_column_power/suspects.json",
     "source_commit"):
        "#873's evidence; its producer is not routed and this PR does not touch "
        "it. sha ca168584 is unreachable from main, and the rfx/ subtree that "
        "would witness it (0cd2ab4a) is recorded in RECORDED_CODE_TREES but the "
        "fixture does not carry it",
    # ---------------------------------------------------------------
    # ARRIVED AFTER THIS GATE, and every one is the #1013 defect in a NEW
    # fixture rather than a legacy one.
    #
    # They are listed rather than repaired because their shas (f5712d6b,
    # 460bb9b7, 427fc97a) resolve in NO clone available anywhere -- not this
    # one, not origin, not the read-only checkout that holds the two objects
    # origin refuses. Checked all three before writing this, and again on
    # 2026-09-20 including `git fetch origin refs/pull/N/head` for the pull
    # requests they came from. Without the object there is no rfx/ subtree to
    # record, so no annotation can make these sites checkable and only a
    # regeneration would.
    #
    # NOT BEING FIXED IN THIS PR, and that is a decision rather than an
    # oversight. Their three producers still call git directly and write
    # "unknown" on failure. Routing them is the repair, and routing them means
    # editing scripts that belong to #873 and #1083, both open investigations
    # somebody else is running. This PR does not touch them.
    #
    # So this list can still grow, and the PR body says so. Whoever routes those
    # producers clears these sites at the same time.
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.A|pec_short.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.A|slab.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.A|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.B|pec_short.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.B|slab.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.B|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.C|pec_short.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.C|slab.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.C|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.D1|pec_short.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.D1|slab.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.D1|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.D2|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "arms.E|thru.provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json",
     "provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json",
     "provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    ("tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json",
     "provenance.commit"):
        _ARRIVED_AFTER_THE_GATE,
    (
        "tests/fixtures/rcs280_reference_subtraction/fixture.json",
        "provenance.rfx_commit",
    ): (
        "6a369c27 is the #1005 branch commit; the squash 8586f549 rebased onto a main "
        "that had moved four commits in rfx/sparams/, so neither the root tree "
        "(d71fce95, on 0 of 1905 main trees) nor the rfx/ subtree (2a076e71, on 0 of "
        "787 main subtrees) survives. This is the issue's own motivating fixture and "
        "rfx_code_tree does NOT rescue it. The recorded tree is kept so the derivation "
        "stays checkable from refs/pull/1005/head; the only anchor reachable from main "
        "is the landing commit 8586f549. The obvious repair -- regenerate on main -- was "
        "TRIED and rejected on measurement: the run reproduces every headline metric to "
        "~1e-6 relative but is not bit-identical, and committing it would move 149 of the "
        "208 floats in this fixture (worst metric delta "
        "full_curve_mean_abs_db_corrected 0.7048104545801226 -> 0.7048094138678332). "
        "Provenance repair is not worth moving 149 committed numbers, so the sha stays "
        "and is annotated instead."
    ),
    (
        "tests/fixtures/harminv_decimation/stage1/receipt.json",
        "source_commit",
    ): (
        "98d31987's rfx/ subtree is d4702ee9, on 0 of 787 main subtrees -- rfx/ moved "
        "in flight between the branch base and the squash, so no tree witness exists. "
        "The receipt is a hand-written capture record with no live producer, and the "
        "capture cannot be cheaply re-run. Anchor reachable from main: the landing "
        "commit of the file."
    ),
    (
        "tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json",
        "setup.commit",
    ): (
        "'6fd6ea0' is 7 characters, written by a `git rev-parse --short HEAD` call in "
        "scripts/diagnostics/build_waveguide_wr90_nu_flux_broad_e4_comparison.py that "
        "this PR changes to full length. It resolves in no clone available here -- not "
        "this worktree and not the primary checkout -- so no code tree can be derived "
        "from it and no annotation can make it checkable. Unrecoverable; the landing "
        "commit 4353a16d is the only surviving anchor. Recorded rather than deleted so "
        "the loss is legible."
    ),
}

#: Fixture JSONs that declare no commit sha anywhere. Asserted EXACTLY, both
#: directions, so the hole is a number a reader can see rather than a silence.
#: Measured at 53 of 80 when this gate landed, 55 of 82 after the rebase below,
#: then 54 once the value filter came out of the walker: rcs_mie_e4 does carry
#: a provenance key (meta.commit), it just holds a label rather than a sha, so
#: it is a NOT_A_SHA entry and not a file that declares nothing.
#: It must only shrink. A new fixture is expected to embed
#: ``tests._fixture_provenance.capture()`` rather than be added here.
#:
#: TWO ENTRIES WERE ADDED AFTER THE LIST WAS FROZEN, and the reason is recorded
#: because "the list must only shrink" is otherwise violated by them. Both
#: arrived on main in ``fc7f7202`` (#1051, "docs/research_notes is local-only:
#: untrack it and add the contract"), which RELOCATED them out of the
#: now-untracked ``docs/research_notes/`` into ``tests/fixtures/``. They are not
#: new fixtures somebody forgot to instrument; they are pre-existing evidence
#: that had to stay tracked and landed here on the way. The gate caught them on
#: the first run after this branch rebased onto that merge, which is the
#: behaviour it was built for.
NO_PROVENANCE_YET: frozenset[str] = frozenset({
    "tests/fixtures/coax_broad_e4/coaxial_line_meep_broad_comparison.json",
    # relocated by #1051 (fc7f7202) out of docs/research_notes/
    "tests/fixtures/harminv_record_support/capacity_impact_report.json",
    "tests/fixtures/msl_notch_uniform_environment/realized_metal_plan.json",
    "tests/fixtures/coax_broad_e5/coaxial_line_broad_e5_envelope.json",
    "tests/fixtures/cv02_ring_judge/tautology_trials_200k.json",
    "tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json",
    "tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json",
    "tests/fixtures/experiments/multilayer_fresnel_v2.json",
    "tests/fixtures/experiments/patch_antenna_cpu_v1.json",
    "tests/fixtures/experiments/patch_antenna_v2.json",
    "tests/fixtures/experiments/wr90_waveguide_v2.json",
    "tests/fixtures/golden_workflows/multilayer_fresnel.json",
    "tests/fixtures/golden_workflows/patch_antenna.json",
    "tests/fixtures/golden_workflows/wr90_waveguide.json",
    "tests/fixtures/gradient_dx_ladder/ad_fd_ladder.json",
    "tests/fixtures/gradient_dx_ladder/per_cell_snr.json",
    "tests/fixtures/mcp/rfx_tools_v1.json",
    "tests/fixtures/msl_notch_e4/comparison_summary.json",
    "tests/fixtures/msl_notch_e4/msl_stub_notch_openems_dx50.json",
    "tests/fixtures/msl_notch_e4/msl_stub_notch_palace_referee.json",
    "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json",
    "tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json",
    "tests/fixtures/msl_s_matrix_golden.json",
    "tests/fixtures/optimizer_bakeoff/bakeoff_results.json",
    "tests/fixtures/patch_canonical_farfield_e4/canonical_farfield_e4_measured_369367259302.json",
    "tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json",
    "tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json",
    "tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json",
    "tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json",
    "tests/fixtures/rcs_dielectric_sphere_mie/fixture.json",
    "tests/fixtures/rcs_mie_ka_sweep/fixture.json",
    "tests/fixtures/rcs_sphere_mie/fixture.json",
    "tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json",
    "tests/fixtures/slab_family_windows_baseline.json",
    "tests/fixtures/waveguide_broad_e5/noise_floor_measurement.json",
    "tests/fixtures/waveguide_broad_e5/phase_falsifier_and_domain_invariance.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr10_wband_broad_e5_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr10_wband_broad_e5_phase_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr15_vband_broad_e5_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr15_vband_broad_e5_phase_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr28_kaband_broad_e5_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr28_kaband_broad_e5_phase_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr340_sband_broad_e5_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr340_sband_broad_e5_phase_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr62_kuband_broad_e5_envelope.json",
    "tests/fixtures/waveguide_broad_e5/waveguide_wr62_kuband_broad_e5_phase_envelope.json",
    "tests/fixtures/waveguide_chain_battery/enforcement_931_realized_pec_forward2_run369367259427.json",
    "tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json",
    "tests/fixtures/waveguide_nu_beta_cell_size_envelope.json",
    "tests/fixtures/waveguide_nu_broad_e5/waveguide_wr90_nu_flux_broad_e5_envelope.json",
    "tests/fixtures/waveguide_tjunction_e4/waveguide_tjunction_broad_e5_envelope.json",
    "tests/fixtures/waveguide_tjunction_e4/waveguide_tjunction_meep_external_comparison.json",
    "tests/fixtures/wr90_iris_filter/fixture.json",
    "tests/fixtures/wr90_iris_modematch/fixture.json",
})


def _walk(node, prefix, out):
    """Yield (keypath, sha, container) for every provenance-shaped sha site."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in SHA_KEYS and isinstance(value, str):
                # The _HEX test used to live HERE, and that was the hole. A key
                # named `rfx_commit` holding "" or "unknown" is exactly the
                # fail-open value the writer half of #1013 was hardened against,
                # and filtering on the value made it not a site at all: it
                # vanished from the census instead of failing. Record every
                # string under a provenance key; _classify decides what it is.
                out.append((path, value, node))
            else:
                _walk(value, path, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{prefix}[{index}]", out)


def _fixture_jsons() -> list[str]:
    return sorted(p for p in tracked_set(REPO)
                  if p.startswith(FIXTURE_ROOT) and p.endswith(".json"))


@functools.lru_cache(maxsize=1)
def _sites() -> tuple[tuple[str, str, str, str | None], ...]:
    """(relpath, keypath, sha, code_tree_witness) for every site, sorted.

    Memoised: the three chain-battery fixtures are ~15 MB each, and the tests
    below ask for the census several times.
    """
    found: list[tuple[str, str, str, str | None]] = []
    for rel in _fixture_jsons():
        try:
            data = json.loads((REPO / rel).read_text())
        except (OSError, json.JSONDecodeError) as exc:  # a fixture that will not parse
            pytest.fail(f"{rel} does not parse as JSON: {exc}")
        out: list = []
        _walk(data, "", out)
        for keypath, sha, container in out:
            witness = container.get(CODE_TREE_KEY)
            found.append((rel, keypath, sha, witness if isinstance(witness, str) else None))
    return tuple(sorted(found))


def _require_answerable() -> str:
    """The reference ref, or skip loudly. Never a silent pass."""
    if not git_available(REPO):
        pytest.skip("NOT A PASS: git cannot answer here (source tarball or sandbox); "
                    "provenance reachability is unanswerable, not satisfied")
    if is_shallow(REPO):
        pytest.skip("NOT A PASS: shallow clone -- 'not on main' and 'never fetched' are "
                    "indistinguishable. CI must use actions/checkout fetch-depth: 0; "
                    "locally run `git fetch --unshallow`")
    ref = resolve_reference_ref(REPO)
    if ref is None:
        pytest.skip("NOT A PASS: no reference ref resolves (tried $RFX_PROVENANCE_REF, "
                    "origin/main, main); reachability is unanswerable")
    return ref


#: Provenance-shaped keys whose value is deliberately a human LABEL and not a
#: sha. Asserted exactly, both directions, for the same reason ANNOTATED is: the
#: alternative is filtering non-sha values out of the census, which is precisely
#: how a fail-open ``""`` or ``"unknown"`` escaped this gate. A label is fine; an
#: empty string is the defect. Listing them is what separates the two.
NOT_A_SHA: dict[tuple[str, str], str] = {
    ("tests/fixtures/msl_z0_length_invariance/platform_datums.json", "datums[1].commit"):
        "a dated main reference ('main@2026-08-10'), written by hand as a datum label",
    ("tests/fixtures/msl_z0_length_invariance/platform_datums.json", "datums[2].commit"):
        "a dated main reference ('main@2026-08-17'), written by hand as a datum label",
    ("tests/fixtures/msl_z0_length_invariance/platform_datums.json", "datums[3].commit"):
        "a dated main reference ('main@2026-08-24'), written by hand as a datum label",
    ("tests/fixtures/msl_z0_length_invariance/platform_datums.json", "datums[9].commit"):
        "the literal string 'not recorded', written by hand. The row's own `source` "
        "field says why: the run copied a work tree without its git directory, so "
        "`git rev-parse HEAD` printed 'no git' and there was nothing to record. That "
        "is the writer-side half of this issue, and it is what "
        "tests._fixture_provenance.capture() now raises on instead of stamping. The "
        "row itself cannot be repaired after the fact -- the tree it measured was a "
        "branch work tree that no longer exists -- so the label stays and is declared. "
        "Arrived on main in 906ce0eb (#1099) while this branch was open",
    ("tests/fixtures/rcs_mie_e4/rcs_pec_sphere_mie.json", "meta.commit"):
        "a descriptive label ('post-276-97ca6a5') that embeds a short sha rather "
        "than being one",
}

#: Verdicts that are not a failure. Everything else reds, and that list is the
#: gate. An earlier revision failed only on 'unresolved', so adding a bucket was
#: a way to make a defect stop failing -- the inverse of what a bucket is for.
ACCEPTED = frozenset({
    "reachable", "reachable-head", "code-tree", "code-tree-head",
    "annotated", "labelled",
})

#: NOT accepted, and the reason is the whole point of the binding.
#:
#: ``code-tree-unbound`` is what _classify returns when a site records a witness
#: but the commit it claims to witness cannot be resolved in this clone, so the
#: two cannot be tied together. It used to be in ACCEPTED, and review showed what
#: that bought: a fixture carrying a commit that exists NOWHERE
#: (``deadbeef``x5) plus `git rev-parse origin/main:rfx` as its witness passed
#: the gate green. Any 40-hex value that happens to be some rfx/ tree on main was
#: a valid witness for any sha.
#:
#: That is the forgery the module docstring says this gate closed, and it was
#: closed only for the ten shas in RECORDED_CODE_TREES -- i.e. only for sites
#: that already existed. Every FUTURE occurrence of #1013 passed.
#:
#: So an unbindable witness is now a failure. A site in this state has two
#: honest exits: record the sha->tree pair in RECORDED_CODE_TREES with the clone
#: it was read from (which is what makes it checkable in CI, where the object is
#: absent), or declare it in ANNOTATED with the measured reason.
_UNBINDABLE = "code-tree-unbound"


def _classify(sha: str, witness: str | None, ref: str) -> str:
    if not _HEX.match(sha):
        # Not a sha at all. Either a fail-open write (``""``, ``"unknown"``) or a
        # human label under a provenance key; NOT_A_SHA says which are the latter.
        return "not-a-sha"
    if is_reachable(sha, ref, REPO):
        return "reachable"
    if is_reachable(sha, "HEAD", REPO):
        return "reachable-head"
    if witness:
        # The witness has to witness THIS commit. Set membership over every rfx/
        # subtree on main accepts any 40-hex value that happens to be one of
        # them -- main's own tip tree pasted into an unrelated fixture passed.
        # When the recorded commit resolves we can bind them, and then the
        # witness is a claim about the fixture rather than about the repo.
        # The binding, and it has to work in a clone that never had the object.
        # RECORDED_CODE_TREES is consulted FIRST for exactly that reason: CI
        # checks out main at full depth and still does not carry a branch-only
        # commit, so `code_tree_of` returns None for precisely the shas this
        # gate exists for. An earlier revision bound only on the git-derived
        # value and was therefore inert in CI while passing locally -- the check
        # ran nowhere it mattered. test_recorded_code_trees_match_git keeps the
        # table honest against git wherever git can answer.
        recorded = RECORDED_CODE_TREES.get(sha)
        derived = recorded[0] if recorded else code_tree_of(sha, REPO)
        if derived is not None:
            if witness != derived:
                return "witness-mismatch"
            if witness in code_trees_on(ref, REPO):
                return "code-tree"
            if witness in code_trees_on("HEAD", REPO):
                return "code-tree-head"
            return "unresolved"
        # The commit object is absent from this clone, so the binding cannot be
        # checked here and only membership is available. A weaker bucket, named
        # so nobody reads it as the strong one.
        if witness in code_trees_on(ref, REPO):
            return "code-tree-unbound"
        if witness in code_trees_on("HEAD", REPO):
            return "code-tree-head"
    return "unresolved"


def test_the_enumeration_finds_the_fixtures() -> None:
    """Anti-vacuity: a broken glob must fail loudly, not report zero problems."""
    files = _fixture_jsons()
    assert len(files) >= 70, f"only {len(files)} tracked fixture JSONs -- glob is broken"
    sites = _sites()
    assert len(sites) >= 30, f"only {len(sites)} provenance sites found -- walker is broken"


def test_every_provenance_site_is_reachable_or_witnessed_or_annotated() -> None:
    """The gate. Every site lands in a named bucket or the build is red."""
    ref = _require_answerable()
    failures = []
    trace = []
    for rel, keypath, sha, witness in _sites():
        key = (rel, keypath)
        verdict = _classify(sha, witness, ref)
        if verdict == "unresolved" and key in ANNOTATED:
            verdict = "annotated"
        if verdict == "not-a-sha" and key in NOT_A_SHA:
            verdict = "labelled"
        trace.append(f"  {verdict:<15} {sha[:8]:<9} {rel}::{keypath}"
                     + (f"  [{CODE_TREE_KEY}={witness[:8]}]" if witness else ""))
        if verdict in ACCEPTED:
            continue
        landing = landing_commit(rel, ref, REPO) or "<none>"
        if verdict == "not-a-sha":
            why = (f"the value {sha!r} is not a sha. If that is a deliberate human "
                   f"label, declare it in NOT_A_SHA with the reason; if it is a "
                   f"generator that failed open and wrote a placeholder, the fix is "
                   f"tests._fixture_provenance.capture(), which raises instead")
        elif verdict == _UNBINDABLE:
            why = (f"{CODE_TREE_KEY}={witness} is on some commit of {ref}, but the "
                   f"commit it claims to witness ({sha}) resolves in no clone here, "
                   f"so the two cannot be tied together. Membership alone is not a "
                   f"witness: any tree that exists on main would pass. Record the "
                   f"pair in RECORDED_CODE_TREES with its donor clone, or declare "
                   f"the site in ANNOTATED")
        elif verdict == "witness-mismatch":
            why = (f"{CODE_TREE_KEY}={witness} is NOT the rfx/ subtree of the "
                   f"recorded commit ({code_tree_of(sha, REPO)}). A witness has to "
                   f"witness THIS commit; any tree that merely exists on main would "
                   f"otherwise pass")
        elif witness is None:
            why = (f"no {CODE_TREE_KEY} sibling recorded -- the producer of this file does "
                   f"not emit it (or dropped it); add tests._fixture_provenance.capture()")
        else:
            why = f"{CODE_TREE_KEY}={witness} is on no commit of {ref} or HEAD"
        failures.append(
            f"{rel}::{keypath}\n"
            f"    verdict    {verdict}\n"
            f"    sha        {sha} -- not reachable from {ref} and not from HEAD\n"
            f"    witness    {why}\n"
            f"    landing    {landing} (the file's last commit on {ref}; an anchor only "
            f"in combination with a code tree, never on its own)")
    assert not failures, (
        f"{len(failures)} provenance site(s) in no bucket against ref={ref}.\n"
        "Each names a committed number whose producing code a reviewer cannot get "
        "back to.\n\n" + "\n".join(failures)
        + "\n\nFull per-site trace (R5 -- read it before believing any verdict):\n"
        + "\n".join(trace))


def test_the_annotation_table_has_no_stale_entries() -> None:
    """An exemption must name a real, still-failing site and give a reason."""
    ref = _require_answerable()
    live = {(rel, keypath): (sha, witness) for rel, keypath, sha, witness in _sites()}
    problems = []
    for key, reason in ANNOTATED.items():
        if not reason or not reason.strip():
            problems.append(f"{key}: empty reason -- an exemption without a written "
                            "reason is a silence with a keyname")
            continue
        if key not in live:
            problems.append(f"{key}: names no site that exists today -- delete it")
            continue
        sha, witness = live[key]
        if _classify(sha, witness, ref) != "unresolved":
            problems.append(f"{key}: this site passes on its own now -- delete the "
                            "exemption rather than let it outlive its reason")
    assert not problems, "stale ANNOTATED entries:\n  " + "\n  ".join(problems)


def test_the_not_a_sha_table_has_no_stale_entries() -> None:
    """The same two directions ANNOTATED gets, because it is the same kind of table.

    NOT_A_SHA's own comment claimed it was "asserted exactly, both directions"
    and nothing asserted it: an entry whose site was deleted, or whose value was
    later replaced by a real sha, would have sat here forever declaring a label
    that no longer exists. The forward direction -- a label nobody declared --
    already reds in the gate above; these are the two reverse ones.
    """
    live = {(rel, keypath): sha for rel, keypath, sha, _ in _sites()}
    problems = []
    for key, reason in NOT_A_SHA.items():
        if not reason or not reason.strip():
            problems.append(f"{key}: empty reason -- a declared label with no reason "
                            "is indistinguishable from a generator that failed open")
            continue
        if key not in live:
            problems.append(f"{key}: names no site that exists today -- delete it")
            continue
        if _HEX.match(live[key]):
            problems.append(f"{key}: the value is a sha now ({live[key][:8]}) -- "
                            "delete the entry and let the reachability arms judge it")
    assert not problems, "stale NOT_A_SHA entries:\n  " + "\n  ".join(problems)


def test_no_provenance_yet_is_exact() -> None:
    """The files declaring no sha at all are an asserted number, not a silence."""
    with_sites = {s[0] for s in _sites()}
    measured = frozenset(rel for rel in _fixture_jsons() if rel not in with_sites)
    missing = NO_PROVENANCE_YET - measured
    extra = measured - NO_PROVENANCE_YET
    assert not missing and not extra, (
        f"NO_PROVENANCE_YET is stale: {len(NO_PROVENANCE_YET)} declared, "
        f"{len(measured)} measured.\n"
        f"  now declare provenance (delete from the list): {sorted(missing)}\n"
        f"  declare none and are not listed: {sorted(extra)}\n"
        "A new fixture should embed tests._fixture_provenance.capture() rather than "
        "join this list; the list must only shrink.")


def test_the_gate_is_answerable_where_it_is_declared_required() -> None:
    """Fail-closed where it matters: a shallow checkout there is a misconfiguration.

    Locally, and in any CI job that has not opted in, a shallow clone only
    skips -- with a stated reason. In the job that declares
    ``RFX_PROVENANCE_GATE_REQUIRED`` a skip is indistinguishable from a pass in
    the summary, which is the exact failure shape #1013 is about, so that job
    gets an assertion instead.

    The flag is an explicit opt-in rather than a bare ``$CI`` check because CI
    is not uniform: only pr-tests.yml's fast-suite job has ``fetch-depth: 0``.
    The weekly validation lane runs the same ``tests/`` tree from a depth-1
    checkout, where this question is genuinely unanswerable -- asserting there
    would red-light a lane for a condition it cannot satisfy.
    """
    if not os.environ.get(REQUIRED_ENV):
        pytest.skip(f"{REQUIRED_ENV} is not set; a shallow clone here skips the gate "
                    "with a stated reason rather than asserting")
    assert git_available(REPO), "no git; the provenance gate cannot run"
    assert not is_shallow(REPO), (
        "checkout is shallow -- the provenance gate would skip and protect nothing. "
        "actions/checkout needs `with: fetch-depth: 0` on the job that runs "
        "tests/contracts/.")
    assert resolve_reference_ref(REPO) is not None, (
        "no reference ref resolves (origin/main / main); fetch-depth: 0 fetches "
        "history but the job must also have the remote ref.")


def test_the_pr_workflow_still_makes_this_gate_answerable() -> None:
    """The two yaml settings the gate depends on, pinned so an edit cannot drop them.

    Parsed, not grepped, so reformatting the workflow cannot fool it. Without
    ``fetch-depth: 0`` the gate skips on every CI run and protects nothing;
    without :data:`REQUIRED_ENV` that skip is silent.
    """
    import yaml

    workflow = yaml.safe_load((REPO / ".github/workflows/pr-tests.yml").read_text())
    job = workflow["jobs"]["fast-suite"]
    checkouts = [s for s in job["steps"] if "checkout" in str(s.get("uses", ""))]
    assert checkouts, "fast-suite has no checkout step"
    assert any((s.get("with") or {}).get("fetch-depth") == 0 for s in checkouts), (
        "fast-suite's checkout lost `fetch-depth: 0`; tests/contracts/ runs in this "
        "job and the provenance gate cannot tell 'not on main' from 'never fetched' "
        "in a depth-1 clone")
    assert any(str((s.get("env") or {}).get(REQUIRED_ENV, "")) == "1" for s in job["steps"]), (
        f"fast-suite no longer sets {REQUIRED_ENV}; without it a shallow checkout "
        "would make this gate skip silently instead of failing the job")


def test_the_1005_regression_is_red_on_main_and_green_on_its_own_pr() -> None:
    """Non-vacuity, the hard way: the bug this gate exists for, both directions.

    ``6a369c27`` with no code-tree witness is exactly the state
    ``tests/fixtures/rcs280_reference_subtraction/fixture.json`` shipped in.
    Against ``origin/main`` it must classify ``unresolved`` -- if it does not,
    the gate cannot see the defect it was written for. Against
    ``refs/pull/1005/head`` (2e3601e2), where the sha genuinely is an ancestor,
    it must classify ``reachable`` -- if it does not, the gate is red for a
    reason other than reachability and its greens mean nothing either.
    """
    ref = _require_answerable()
    sha = "6a369c2730c4904f71187f3cd4bddff943e91f22"
    assert _classify(sha, None, ref) == "unresolved", (
        f"{sha[:8]} classifies as passable against {ref}; the #1005 regression would "
        "not be caught and this gate is vacuous")

    pr_ref = None
    for candidate in ("refs/pull/1005/head", "refs/rfx1013/pr1005"):
        if is_reachable(sha, candidate, REPO):
            pr_ref = candidate
            break
    if pr_ref is None:
        pytest.skip("the green half needs the #1005 head: "
                    "`git fetch origin refs/pull/1005/head:refs/rfx1013/pr1005`. "
                    "The red half above ran and is the load-bearing arm.")
    assert _classify(sha, None, pr_ref) == "reachable", (
        f"{sha[:8]} does not classify reachable against {pr_ref} either; the gate is "
        "failing for some reason other than reachability")


def test_capture_refuses_to_stamp_a_record_outside_a_checkout(tmp_path) -> None:
    """The writer half fails closed, which is the other half of #1013.

    The three generators this replaced each ran
    ``subprocess.run(["git","rev-parse","HEAD"], capture_output=True,
    text=True).stdout.strip()`` and checked nothing. Measured in a directory
    that is not a work tree: ``returncode=128``, ``stdout.strip() == ""``. The
    fixture would have been stamped ``"rfx_commit": ""`` and committed, and no
    gate could tell that from a sha that has merely become unreachable.

    That path was reachable: ``scripts/vessl_gpu_suite.yaml`` runs from a
    ``git archive`` export under /tmp that has no ``.git`` at all.
    """
    from tests._fixture_provenance import ProvenanceUnavailable, capture

    with pytest.raises(ProvenanceUnavailable) as excinfo:
        capture(tmp_path)
    assert "128" in str(excinfo.value) or "not a git repository" in str(excinfo.value)


def test_capture_refuses_a_short_commit_override() -> None:
    """A 7-char sha is how the one unrecoverable site in the backlog was made."""
    from tests._fixture_provenance import ProvenanceUnavailable, capture

    with pytest.raises(ProvenanceUnavailable, match="not a 40-hex sha"):
        capture(REPO, commit_override="6fd6ea0")


def test_every_commit_shaped_key_name_is_classified() -> None:
    """A producer cannot escape this gate by renaming its key.

    Enumerate-and-classify one level up: every key name under
    ``tests/fixtures/`` that carries a 7-40 hex value must appear in exactly
    one of :data:`SHA_KEYS` (gated), :data:`NOT_A_COMMIT_REFERENCE` (run ids and
    content digests that merely match the pattern) or
    :data:`COMMIT_SHAPED_NOT_YET_GATED` (the named, counted gap). A new name in
    none of them fails here, so the choice to gate it or not is always made
    deliberately.
    """
    measured: dict[str, int] = {}
    for rel in _fixture_jsons():
        data = json.loads((REPO / rel).read_text())

        def collect(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, str) and _HEX.match(value):
                        measured[key] = measured.get(key, 0) + 1
                    else:
                        collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(data)

    classified = set(SHA_KEYS) | set(NOT_A_COMMIT_REFERENCE) | set(COMMIT_SHAPED_NOT_YET_GATED)
    unclassified = {k: n for k, n in measured.items() if k not in classified}
    assert not unclassified, (
        f"key name(s) carrying a commit-shaped value and in no table: {unclassified}.\n"
        "Add each to SHA_KEYS (it is a commit claim this gate should check),\n"
        "NOT_A_COMMIT_REFERENCE (it is a run id or a content digest), or\n"
        "COMMIT_SHAPED_NOT_YET_GATED (it is a commit claim, with the measured gap).")
    stale = sorted(k for k in set(NOT_A_COMMIT_REFERENCE) | set(COMMIT_SHAPED_NOT_YET_GATED)
                   if k not in measured and k not in ANTICIPATED)
    assert not stale, (
        f"classified key name(s) that appear nowhere any more: {stale}. Delete them so "
        "the tables describe the tree as it is.")
    landed = sorted(k for k in ANTICIPATED if k in measured)
    assert not landed, (
        f"{landed} now appears in a committed fixture, so it is no longer "
        "anticipated. Move it out of ANTICIPATED and give its table entry the "
        "measured counts, the way every other entry carries them.")


def test_recorded_code_trees_match_git() -> None:
    """RECORDED_CODE_TREES is the binding's input in CI, so it must not drift.

    The table exists because a clone of main does not carry a branch-only commit
    and ``code_tree_of`` cannot answer there. That makes it the one place a wrong
    value would be invisible: the gate would happily bind a witness to a tree
    nobody can check. So wherever git CAN answer -- which is here, and in any
    clone that has fetched the objects -- the recorded value is re-derived and
    compared.
    """
    if not git_available(REPO):
        pytest.skip("NOT A PASS: git is unavailable")
    checked = 0
    for sha, (tree, where) in sorted(RECORDED_CODE_TREES.items()):
        derived = code_tree_of(sha, REPO)
        if derived is None:
            continue  # object absent from this clone; `where` names the donor
        assert derived == tree, (
            f"RECORDED_CODE_TREES[{sha}] says {tree} (from {where}) but "
            f"`git rev-parse {sha}:rfx` in this clone gives {derived}. One of "
            "them is wrong, and the table is what CI trusts."
        )
        checked += 1

    # NO FLOOR ON `checked`, and that absence is the point.
    #
    # An earlier revision asserted `checked >= 4` and reddened CI. The reason is
    # that how many of these shas a clone can resolve is a property of which
    # ABANDONED BRANCHES origin still happens to serve, not a property of the
    # repository's content. Measured over origin's 88 refs: 2 of the 10 resolve,
    # 6a369c27 only via refs/remotes/origin/fix/888-aux-absorber-r2 and 98d31987
    # only via refs/remotes/origin/fix/872-harminv-fir-boundaries. Prune either
    # stale branch and the count drops. The floor was set from a development
    # clone that had fetched more, so it passed locally and failed in CI --
    # exactly the shape of defect this gate exists to catch, one level up.
    #
    # What this test can honestly assert is the IMPLICATION: every entry git can
    # check here is right. That holds in any clone, including one that can check
    # none, and it is the property that keeps the table from drifting. A table
    # that is never exercised anywhere is caught by the separate staleness test
    # below, which needs no objects at all.
    print(f"RECORDED_CODE_TREES: {checked}/{len(RECORDED_CODE_TREES)} re-derivable "
          f"in this clone (clone state, not repo state)")


def test_recorded_code_trees_are_all_used() -> None:
    """The table must not accumulate entries nothing reads.

    This is the half of the previous test that does NOT depend on which objects a
    clone holds: every recorded sha has to be one some committed fixture actually
    records, or the table is carrying a witness for a site that no longer exists.
    Checkable in any clone, including one that can resolve none of the objects.
    """
    recorded = set(RECORDED_CODE_TREES)
    in_fixtures = {sha for _, _, sha, _ in _sites()}
    unused = sorted(recorded - in_fixtures)
    assert not unused, (
        f"RECORDED_CODE_TREES carries {len(unused)} sha(s) no committed fixture "
        f"records any more: {unused}. Delete them; a witness for a site that does "
        "not exist is not evidence of anything."
    )
