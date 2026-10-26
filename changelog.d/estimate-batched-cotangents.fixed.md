### Fixed — estimate_ad_memory with several outputs pulled back at once counted one carry per extra output; it now counts the measured 7.5 (#2089)

Each extra simultaneous pullback now adds one boundary carry plus 6.5 carries of
backward working arrays in the estimate, explanation, and checkpoint planner.
Single-output memory estimates are unchanged.
On the frozen scene S anchor, the 16-output estimate/measured-peak ratio changes
from 0.44 to a computed 0.989296 (4510.69413 MB / 4559.5 MB), not a new measurement.
The coefficient is calibrated on one graded whole-grid scene, GPU, JAX 0.10.2.
Design boxes use the same whole-grid workspace estimate and remain unmeasured.
