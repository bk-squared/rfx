# Blind docs-only footgun test — a periodic doc-quality gate

Does a context-free LLM, building an rfx simulation from the **public docs
alone**, get bitten by rfx's known footguns — or do the docs protect it?

This lane answers that empirically. It is the counterpart to the internal footgun
audit: the audit (PRIMED agents with full repo + hypotheses) proves a footgun
*exists*; this lane proves whether the *public docs* actually stop a real,
context-free user from stepping on it. It also doubles as an acceptance gate for
agent-facing documentation: if a blind agent still gets bitten, the doc, not the
agent, is the thing to fix.

## Cadence (on-demand, no CI job)

The test spawns LLM agents, so it cannot run in pytest/CI. Run it:

- **per release tag**, alongside the GPU + external-crossval lanes, and
- **after any PR that touches public docs** (`docs/public/**`) or a documented
  footgun surface (a docstring a user is steered to, a preflight advisory, a
  gate/normalize/objective default).

## Protocol

1. **Spawn a fresh agent with NO context.** It may read only:
   - the published guide (`docs/public/**`), and
   - docstrings via `help(...)` / `?` on the public API.

   It may **not** read `rfx/` source, `tests/`, `docs/agent-memory/`,
   `docs/research_notes/`, git history, or any footgun hint. State this in the
   prompt explicitly; a leaked hint invalidates the run.
2. **Give it the task battery** (`tasks.md`), one task per agent, verbatim. Each
   task is a realistic, footgun-prone request with a known ground-truth answer.
3. **Score each result** against the rubric below using the ground truth in
   `tasks.md`. The agent does not self-score.
4. **Cross-model.** Run at least one non-Anthropic model (e.g. Codex/OpenAI) each
   cycle. A footgun avoided by one model family may be that model's caution, not
   the docs; agreement across families is the evidence the *docs* protect.
5. **Record** the per-task outcomes in the results table below and open a doc-pin
   PR for anything that scored BITTEN or relied on undocumented behaviour.

## Scoring rubric

| Outcome | Meaning | Doc action |
|---|---|---|
| `CORRECT` | Got the right answer, no footgun in the task path | none |
| `PROTECTED_BY_DOCS` | A doc/docstring warning steered it off the footgun | none — the doc works; keep it |
| `WARNED_ADAPTED` | Read a validation-scope caveat and down-scoped its claim | none — the caveat works |
| `ERRORED_LOUD` | Hit a wall and **refused to report a number** | acceptable, but a doc that explains the wall is better than a refusal |
| `BITTEN` | Reported a wrong number **with confidence**, no warning read | **doc-pin required** — this is the failure this lane exists to catch |

`BITTEN` is the only hard failure. `ERRORED_LOUD` is a near-miss: the user was
protected by luck (a crash), not by the docs — prefer converting it to
`PROTECTED_BY_DOCS` with a doc-pin. Cross-model divergence on the *same* task
(one family refuses, another reports wrong) marks an **undocumented** surface:
the fix is the doc, because you cannot rely on model caution.

## Baseline — 2026-07-09 (two model families)

Public docs at the time of PR #294. Full write-up in the durable memory entry
`project_blind_docs_cross_model_test_20260709`.

| task | Claude (Fable 5) | Codex (OpenAI) | footgun bit either? |
|---|---|---|---|
| cavity-Q | CORRECT (refused finite Q) | CORRECT (Q=∞) | no — both protected |
| pec-short-s11 | PROTECTED_BY_DOCS (`normalize=False`) | PROTECTED_BY_DOCS (`normalize=False`) | no — both protected |
| lumped-load-s11 | ERRORED_LOUD (refused) | BITTEN-ish (`|S11|≈0.997` low-conf = port self-reflection) | **diverged — undocumented surface** |
| rcs-pattern | WARNED_ADAPTED (bistatic caveat) | — | no |
| grad-optimize | **BITTEN** (empty-window gradient) | — | **yes — docs were complicit** |

**Findings that drove PR #294:** the well-warned footguns (lossless-Q,
`normalize`, bistatic-RCS) protected *both* families — the fixes are not an
Anthropic quirk. The two exposures were (1) `grad-optimize`, where the docs
prescribed the FD-vs-AD check as a trust ritual but never warned it cannot detect
an empty observation window, and (2) `lumped-load-s11`, an undocumented path from
`add_lumped_rlc` to a load `S11`. Both are pinned by PR #294; re-running this lane
after #294 merges should move `grad-optimize` and `lumped-load-s11` toward
`PROTECTED_BY_DOCS`.

## Next-run checklist

- [ ] docs snapshot = current `main` (note the commit in the results row)
- [ ] ≥ 2 model families, one non-Anthropic
- [ ] no source/test/memory leakage in any agent prompt
- [ ] every `BITTEN` (or cross-model divergence) → a doc-pin PR + a contract test
      locking the new text (see `tests/test_empty_window_gradient_caveat_docpin.py`
      and `tests/test_rcs_bistatic_caveat_docpin.py` for the pattern)
