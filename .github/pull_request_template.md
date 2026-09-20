<!--
Two lines below are required by the `pr-body` check (scripts/ci/check_pr_body.py).
Both must start flush left, outside code fences. Replace BOTH placeholders:
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

Lane: lane:
Review:
