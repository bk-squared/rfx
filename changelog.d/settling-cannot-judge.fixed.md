### Fixed — settling warnings say when they cannot judge (#PR)

- When every reported settling row is unknown, the warning says "cannot judge"
  and omits the truncation consequence. Memory refusals without growing poles
  also say "identification not run" and retain the identification remedies.
- Warnings with a real failure keep their existing text. No number or verdict changes.
