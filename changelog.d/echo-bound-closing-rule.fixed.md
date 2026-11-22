Records whose identification window holds fewer than 2.05 longest round trips now turn a ring-down settling pass into "cannot judge". (#PR)
Ring-down completion uses the same limit and reports its reason when it cannot judge.
`until_decay` runs longer where it would have stopped earlier: 550 → 606 steps (+10 %) on the decay-drive-length test.
When coverage and this limit both refuse a record, the coverage warning starts with "no ring-down settling witness" and includes both reasons.
