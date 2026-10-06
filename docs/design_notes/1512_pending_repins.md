# #1512 geometry conversion — measurements pending

The edge-fed patch traces now start at their MSL port planes: 5 mm of metal
behind the port is removed. The device-side trace, patch, substrate, grid and
record lengths are unchanged. The previous figures describe a different feed.
No numerical pin or historical evidence record is changed in this commit.

| Lock | Old pin / reading | New value |
| --- | --- | --- |
| Uniform S11 and NU S11 (shared constants) | RES_BAND_GHZ = (7.4, 8.2); min S11 > 0.70; max Re(Zin) > 500 ohm; crossing in band; dip > 8.2 GHz | Pending measurement |
| Uniform S11 historical readings (#931) | max S11 0.9837; crossing 7.7620 GHz; peak Re(Zin) 4157 ohm at 7.7 GHz; band min S11 0.9096; dip 8.800 GHz, S11 0.6418 | Pending measurement |
| Harminv Leg B feed pull | centre −6.8795%, half-width 0.772 percentage points; window [−7.6515, −6.1075]% | Pending measurement |
| Harminv Leg A control (unfed geometry unchanged) | centre −3.8198%, half-width 0.182 percentage points | Re-measure as the same-job reference |

Passivity limit 1.05 and settling bar −40 dB remain requirements, not new
measurement-derived pins. Harminv's configuration and extractor envelopes need
their own evidence before selecting new widths: one nominal measurement cannot
reconstruct those envelopes. The measurement script emits both raw traces/census
and the quantities read by the locks; it does not silently widen any gate.

Shared builder importers inherit the geometry change. Direct patch diagnostic
copies are converted too. The broad-E5 builder trims both ends only where the
odd-resonance rule selects the case: RO4003C high sub4/sub6, RO4003C low sub4,
and Teflon high sub4 (each for thru/open_stub). All other broad-E5 endpoint
coordinates remain unchanged. Broad-E5 has no numerical pin changed here.

Reason for every eventual old→new entry: #1512 removes the open stub behind the
line port; remeasure the new geometry on the current solver. Until that job is
run, these numerical locks are pending, not claimed green.

Conclusions: 리더가 채움.
