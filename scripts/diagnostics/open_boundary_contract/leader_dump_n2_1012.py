"""Leader's one-off: dump the still-growing 4-layer patch arm on the tree with PR #1012.

Question: the realized patch rig is mirror-symmetric in its conductor (PEC edges span exactly the
declared domain on both sides), yet on main the growing wave sits on the +x / +y absorber faces only.
#1012 corrects a lo/hi asymmetry of the absorber itself (magnetic profiles sampled half a cell off).
If the hi-side localization comes from that asymmetry, this arm's growing wave on main + #1012 should
no longer sit on the hi faces only.

Uses Codex's dump_fields.py unchanged: adds one label at import time and calls its main().
"""
import sys

import dump_fields as d

d.LABELS["n2_pad10_main1012"] = dict(tree="src-main-plus-1012", n=2, pad_h=10, cpml=4, ceil=False)
sys.argv = [sys.argv[0], "n2_pad10_main1012"]
d.main()
