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

A closing keyword must START its line (`Closes #N.` then the summary). Inside a
sentence GitHub still closes the issue on merge - even negated, even in inline
code - and the `pr-body` check fails it. Several issues: repeat the keyword for each.

Keep the repo habit of quoting measured evidence below.
-->

Lane: lane:
Review:
