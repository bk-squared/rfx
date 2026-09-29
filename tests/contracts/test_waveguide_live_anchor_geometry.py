import numpy as np
import pytest

from tests._realized_pec import (
    assert_no_wall_at, assert_walls_at, realize, wall_positions,
)
from tests.oracle.test_waveguide_live_anchors import (
    BAND_HZ, LIVE_DX, N_FREQS, PEC_SHORT_THICKNESS, PEC_SHORT_X, _live_build_sim,
)


def test_live_pec_short_realizes_the_block_it_declares():
    """BUILD-TIME (no solve) witness for the live anchor's reflector (#931).

    The short is a metal BLOCK, so a volume: the contract realizes tangential
    walls on BOTH drawn faces and shorts the normal edge between them. Read
    back through the one realized-edge reader — a check that only asked for
    the near face would not notice the far one going missing, which is what
    the pre-#931 rule did at every thickness.
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs, pec_short_x=PEC_SHORT_X)
    realized = realize(sim)
    # footprint=None: the WHOLE cross-section. A footprint taken from the
    # plug's own cells passed on the auto mesh while the plug was one row
    # short of the top wall (the slot the builder's comment describes).
    assert_walls_at(realized, 0, [PEC_SHORT_X, PEC_SHORT_X + PEC_SHORT_THICKNESS],
                    what="live-anchor PEC short (full cross-section)")
    assert wall_positions(realized, 0) == pytest.approx(
        [PEC_SHORT_X, PEC_SHORT_X + PEC_SHORT_THICKNESS], abs=1e-9), (
        "the reflector realizes wall planes it did not declare: "
        f"{wall_positions(realized, 0)}")
    # Falsifier arm: no wall one cell in front of the reflecting face, so a
    # body that grew a cell would be caught rather than absorbed by |S11| ~ 1.
    assert_no_wall_at(realized, 0, [PEC_SHORT_X - LIVE_DX],
                      what="live-anchor PEC short")
