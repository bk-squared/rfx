(#PR)

Port S record-end verdicts now read the estimated relative change to S.
Deep nulls that passed the channel-level check can now fail.
The #332 energy-envelope truncation warning has been removed.
Fixed-period, pulsed probe-only runs now warn from their ring-down settling
witness when it fails, or report its reason when it cannot judge.
A failing record-end witness means lengthen the record.
