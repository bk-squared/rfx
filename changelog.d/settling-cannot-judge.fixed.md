### Fixed — settling warnings identify memory refusals (#PR)

- When every reported row is undetermined only because pole identification was
  refused by `rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES`, the warning reads
  "cannot judge (identification refused by the memory bound)" instead of
  "NOT PASSED" and no longer says truncation is suspect. The memory remedies stay.
- Every other warning keeps its text. No number or verdict changes.
