### Fixed — ring-down echo delay closing rule (#PR)
A settling pass becomes undetermined with NaN dB when its identification window contains fewer than 2.05 longest round trips of the declared materials in the realized domain.
Ring-down completion uses the same limit and reports its reason when it cannot judge.
`until_decay` waits for the same window: the dielectric-box decay-drive stops increase from 550 to 650 steps (uniform) and from 700 to 800 (graded), both on the threshold.
When coverage and this limit both refuse a record, the coverage warning starts with "no ring-down settling witness" and includes both reasons.
