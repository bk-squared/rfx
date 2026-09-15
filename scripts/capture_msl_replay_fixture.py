#!/usr/bin/env python3
"""Retired one-sided MSL replay capture entry point (issue #726).

The historical accumulators and golden remain unchanged as evidence. Their
capture did not record both H planes needed by the current extractor, and
its hand-copied NumPy reference implemented the old current-plane rule.
Current structural tests manufacture complete bracketing records and use
an independent planted S matrix. Real FDTD qualification uses the aligned
coupon and records both side currents; see docs/research_notes/issue726/collocation/.
The former capture implementation remains available in git history.
"""


def main():
    raise SystemExit(
        "Retired MSL replay capture (#726): historical one-sided H records "
        "must not be overwritten or used as a collocated-current oracle. "
        "See docs/research_notes/issue726/collocation/ for current evidence "
        "and qualification drivers. No simulation or fixture write was performed."
    )


if __name__ == "__main__":
    main()
