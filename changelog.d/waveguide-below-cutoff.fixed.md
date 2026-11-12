### Fixed — waveguide S parameters below cutoff (#PR)

Waveguide modes at or below cutoff now report NaN for every affected S entry,
with one warning identifying the ports, modes and frequency bins that have no
power wave. Previously these bins could report |S| up to 7.5e25 or 3.82 in an
empty matched guide. Entries above both ports' cutoffs are unchanged.
