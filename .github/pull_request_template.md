<!--
The `pr-body` check (scripts/ci/check_pr_body.py) requires a Physics first line
on PRs created at or after 2026-10-27 00:00 KST, plus Lane and Review lines.
All must start flush left, outside code fences. Replace every placeholder.
The first non-blank body line describes what changes for a user:
  Physics: <user number that changes>
  Physics: none — <reason>
The text must be non-empty, without angle brackets. `none` (case-insensitive)
requires a reason after an em dash (—), en dash (–), or hyphen (-).
  Review: <who read it> (separate instance) - ACCEPT   |   ... - ACCEPT WITH CHANGES
  Review: skipped - (a) <easily reverted pure docs/comment change>
  Review: skipped - (b) <the PI instructed the skip>
An unedited `<...>` placeholder does not pass.

`lane:*` labels appear on this PR automatically, from the paths it touches. They
do not replace the `Lane:` line — most PRs earn two of them or none, and which
lane OWNS the change is not derivable from paths. When labels are present the
line must name one of them.

To close an issue, START a line with the keyword (`Closes #N.` then the summary);
for several issues repeat the keyword. To NOT close one, do not write a closing
keyword in front of its number at all: GitHub closes on it even when the sentence
negates it, and the `pr-body` check fails it inside a sentence.

Keep the repo habit of quoting measured evidence below.
-->

Physics: <user number that changes, or none — reason>
Lane: lane:
Review:
