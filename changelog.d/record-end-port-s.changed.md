(#PR)

Port S record-end verdicts now read the estimated relative change to S.
Deep nulls that passed the channel-level check can now fail.
An S entry with no field recorded at its receiving port over the whole record (a PEC short, an isolated port, or a record too short for the wave to arrive) now reads `undetermined`, with `settling_db` NaN, instead of `pass`.
The energy-envelope warning stays only on runs that read no frequency: no
NTFF box, no DFT plane and no same-run S.
A failing record-end witness means lengthen the record.
