<!--
The `pr-body` check (scripts/ci/check_pr_body.py) requires a review line and a
lane. Both must start flush left, outside code fences. Replace BOTH placeholders:
  Review: <who read it> (separate instance) - ACCEPT   |   ... - ACCEPT WITH CHANGES
  Review: skipped - (a) <easily reverted pure docs/comment change>
  Review: skipped - (b) <the PI instructed the skip>
An unedited `<...>` placeholder does not pass.

The `Lane:` line may be DELETED once .github/workflows/labeler.yml has put
exactly one `lane:*` label on this PR — the paths are the better witness. Keep
it, naming the primary lane, when the PR earns two lane labels or when the paths
it touches belong to no lane. A line and a label that disagree fail.

Keep the repo habit of quoting measured evidence below.
-->

Lane: lane:
Review:
