# Issue 752 material-origin tie correction re-review

The final source SHA-256 is `58a9f0e1a3e98a8a2df19c1aa12d32e6814fec5a024510f3a5c751333cbd13a1`. Removing exactly the newly inserted five-line normal-index override reconstructs the previously reviewed preflight file byte-for-byte (SHA-256 `0bac79f7528efe60a7035c1ab703c4727b98e8c7fc6ace5af714fb5d86d2d152`). No other preflight change is hidden in this follow-up.

Verdict: SOUND; no new material blocker. The material walk now starts from `_msl_normal_bounds(grid, msl_port_from_entry(pe))[0]`, the same lower-tie normal index used by the actual source cross-section in `rfx/sources/msl_port.py::_msl_yz_cells`. Generic conversion of propagation/width positions is retained. The subsequent volume-ground skip, epsilon selection, same-permittivity tolerance, contiguous-slot loop and extent sum are unchanged. Only the previously inconsistent normal origin changes at tie cases; no source/operator/AD array is mutated.

The added layered epsilon3/epsilon5 controls distinguish the first dielectric above the actual source ground from the next layer. They cover odd/even lower-node parity, +x/-y propagation and uniform/NU grids. Author logs document two failing odd-parity uniform cases before correction and all43 build cases passing afterward. These are inspected author results, not reviewer reruns.

The final AD audit has completed: its log records18 passed in105.11seconds, and its source receipt matches SHA58a9f0e1 exactly. This includes the existing stated AD/FD checks; the reviewer did not rerun them or expand their physical qualification. The earlier absolute-face helper checks remain applicable because its bytes were not changed by this five-line correction.

No field simulation, source edit or whole-suite rerun was performed by this reviewer. Final exact-commit inclusion and applicable required CI remain outstanding before merge.
