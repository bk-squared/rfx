### Fixed — estimate_ad_memory with several outputs pulled back at once counted one carry per extra output; it now counts the measured 7.5 (#2089)

Each extra simultaneous pullback now adds one boundary carry plus 6.5 carries of
backward working arrays in the estimate, explanation, and checkpoint planner.
Single-output memory estimates are unchanged.
On the measured graded scene (737,009 nodes, RTX A6000) the 16-output estimate goes
from 0.44 to 0.99 of the measured device peak (computed from the new count, not re-measured).
The coefficient is calibrated on one graded whole-grid scene, GPU, JAX 0.10.2.
Design boxes use the same whole-grid workspace estimate and remain unmeasured.
