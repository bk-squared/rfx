# Handoff from group T (tests) — `Simulation.conductor_mask` docstring (api group)

`rfx/api/_compile.py::conductor_mask` already unions PEC sheet footprints
(`[sp.footprint for sp in pec_sheets]`), so its BEHAVIOUR is right under the
contract and `tests/unit/materials/test_conductor_mask_accessor.py` passes.

Two wording debts in its docstring, which T cannot edit:

1. "a surface-impedance thin conductor is a **node-thin** per-step operator" —
   the phrase carries the retired reading (a conductor occupying one cell
   layer). Under the contract it is "a per-step operator on ONE node plane".
   T made the same replacement in the three test docstrings that quoted it.
2. The formula in the docstring is still
   `pec_mask | (sigma > sigma_threshold) | union(f0 sheet masks)`; the code
   also unions PEC sheet footprints. Say so, and say what the accessor MEANS
   under the contract: the CELL footprint of everything conducting — volumes'
   cells plus sheets' node footprints — which is what a connectivity or
   occupancy check wants, while the realized EDGE set (the thing the solver
   applies) comes from `realized_pec_edge_masks`. Those are two different
   objects and every consumer should be reading the one it needs.
