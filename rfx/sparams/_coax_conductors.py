"""Coax conductor-edge realization shared by the line calculators."""

import numpy as np


def _coax_pec_edge_masks(pec_cells, periodic=(False, False, False), merge_with=None):
    """The conductor cells of a coax line, as PEC E-edge masks.

    ``rfx.boundaries.pec.realized_pec_edge_masks`` is the repo's one rule for
    turning conductor geometry into shorted edges: walls on BOTH faces, every
    normal edge between them shorted. The coax lanes used to leave their
    conductors as ``sigma = PEC_SIGMA``, which ``rfx/core/yee.py`` applies per
    NODE to the three E components co-indexed with that node — so only the
    plus-side edges of each conductor cell were damped and the edges entering it
    from the minus side stayed live. See ``stamp_coaxial_line`` for what that
    cost and what it was measured at.

    ``merge_with`` is another lane's already-realized masks (the coax-to-MSL
    transition has its own board); the two are unioned rather than replaced.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks

    edges = realized_pec_edge_masks(np.asarray(pec_cells), sheets=(), wires=(),
                                    periodic=periodic)
    if merge_with is None:
        return tuple(edges)
    return tuple(np.asarray(e) | np.asarray(m) for e, m in zip(edges, merge_with))
