# cv15 result legs — `validation/crossval/15_patch_antenna_rt5880.py`

What each committed JSON in this directory is, and which are CURRENT versus
HISTORICAL. A historical leg is kept because a committed document or test still
reasons about it; it is never the leg `compare()` gates.

| file | status | leg | headline numbers | why |
|---|---|---|---|---|
| `rfx.json` | **CURRENT** | rfx, galvanic probe feed | dip **-21.92 dB** @ 2.360 GHz; ring-down f0 2.3646 GHz, Q 10.30; max\|S11\| 0.989; D 7.24 dBi; settling -52.6 dB (SETTLED); 111.8 s CPU | the leg `compare()` gates. Feed runs from the ground body's realized node plane to the patch body's, both end cells dead/shorted inside their sheets (`feed_check.galvanic`), matching openEMS's galvanic `AddLumpedPort`. |
| `openems.json` | **CURRENT** | openEMS, same geometry | dip -20.10 dB @ 2.330 GHz; max\|S11\| 0.992; D 7.34 dBi; 47 s | the external reference. Unchanged by #920. |
| `rfx_floating_post_1f005d0d.json` | HISTORICAL | rfx, **floating post** (issue #920) | dip -4.43 dB @ 2.310 GHz; ring-down f0 2.3139 GHz, Q 18.90; max\|S11\| 0.787; D 7.24 dBi | the leg cv15 shipped until 2026-09-06, byte-identical to the file committed at `1f005d0d` (#768). Its port spanned only the two INTERIOR substrate cells and touched neither conductor, so a series gap capacitance (~-347j ohm) sat between feed and patch. Kept because `tests/fixtures/patch_mode_identification/*` were recorded through that builder and `test_cv15_reproduction_ringdown_matches_the_floating_post_leg` pins the correspondence. |
| `rfx_one_plane_ground_b29f9de7.json` | HISTORICAL | rfx, **one-plane ground** (issue #740) | ring-down f0 2.4719 GHz (+6.09 % vs openEMS), Q 17.04; dip -3.23 dB @ 2.470 GHz | the pre-#768 realization whose ground wall sat one cell below the substrate floor, leaving a vacuum cell in the cavity. Kept as the #812 blindness exhibit: it passes the 8 % f0 gate and only `assert_realized_stack` catches it. Also carries the floating post — #740 and #920 are independent defects, and this leg has both. |

## Reading the two historical legs together

They are not a ladder. `rfx_one_plane_ground_b29f9de7.json` differs from
`rfx_floating_post_1f005d0d.json` in the GROUND WALL PLANE (#740); both share
the floating feed. `rfx.json` differs from `rfx_floating_post_1f005d0d.json` in
the FEED (#920); both have the correct wall plane. Any comparison that crosses
both axes at once is comparing two changes.

Extractor caveat on the floating-post leg: it was written under the pre-#776
wire-port extractor. Re-running that same fixture on today's extractor gives
-0.3448 dB, not -4.43 dB — #776/#777 changed how fully the frame reports the
gap reactance, not the fixture. See the #920 CHANGELOG entry.
