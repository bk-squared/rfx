# changelog.d — one changelog entry per PR

`CHANGELOG.md` used to be the repo's main merge-conflict source: every PR
appended to the same `## [Unreleased]` block of one 5500-line file, so any two
concurrent PRs conflicted on text adjacency alone (#934). Entries are written
here instead — one file per PR, so two PRs never touch the same file — and
assembled into `CHANGELOG.md` at release time.

## Naming

```
changelog.d/<number>.<type>.md
```

`<number>` is the PR number, or the issue number when the entry is better
identified by the issue. `<type>` is one of `added`, `breaking`, `changed`,
`deprecated`, `fixed`, `removed`.

## Content

Exactly the text you would have pasted under `## [Unreleased]`: a heading line
and its bullets, nothing else. No version heading, no date. `changelog.d/934.changed.md`:

```markdown
### Changed — changelog entries are written as `changelog.d/` fragments, not appended to `CHANGELOG.md` (#934)

- Each PR adds its own `changelog.d/<number>.<type>.md`, so two concurrent PRs
  never touch the same file.
- `scripts/changelog/assemble.py` moves them into `CHANGELOG.md` at release.
```

The heading is `### <Type> — <one sentence> (#NNN)` with an em dash (U+2014),
matching the entries already in `CHANGELOG.md`; `breaking` renders as
`### BREAKING — …`. The number in the filename must appear among the `#NNN`
references on that line, so a copy-pasted fragment cannot file itself under
someone else's number; extra references (`(#1041, #1055)`) are fine.

## Assembling

```
python scripts/changelog/assemble.py --dry-run     # print the diff, change nothing
python scripts/changelog/assemble.py               # move fragments into Unreleased
python scripts/changelog/assemble.py --check       # validate names and headings
python scripts/changelog/assemble.py --release 2.0.0 [--date YYYY-MM-DD]
```

The default run inserts every fragment directly under the `## [Unreleased …]`
heading, newest number first, and deletes the fragment files. `--release`
does that first, then renames the heading to `## [X.Y.Z] - <date>` and opens a
fresh `## [Unreleased]` above it — so an entry can never land in an already
released section. Stdlib only; no install.

## What CI enforces

`.github/workflows/changelog-fragment.yml` runs
`scripts/ci/check_changelog_fragment.py` on every PR:

- **A PR that changes anything under `rfx/` must add a fragment.** Docs, tests,
  scripts and records may add one; they are not required to.
- **`CHANGELOG.md` may only be edited by a PR carrying the `release` label.**
  That is the assembling PR. Every other edit is rejected with the fragment
  filename to use instead.

Malformed names and heading lines fail the same job, using the assembler's own
validator so the gate and the release step cannot drift apart.
