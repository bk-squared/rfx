### Fixed — ring-down echo delay closing rule
A settling pass becomes undetermined with NaN dB when its identification window contains fewer than 2.05 longest round trips of the declared materials in the realized domain.
Ring-down completion uses the same limit and reports its reason when it cannot judge.
`until_decay` runs longer where it would have stopped earlier: the decay-drive-length stop counts are measured with this change.
When coverage and this limit both refuse a record, the coverage warning starts with "no ring-down settling witness" and includes both reasons.
