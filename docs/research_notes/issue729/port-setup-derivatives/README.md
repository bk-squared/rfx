# Setup-only derivative diagnostic

The production compute_msl_s_matrix -> forward -> core-run boundary was
intercepted before any FDTD scan. On the legacy fixture, JVP and central FD
agree for epsilon and source waveforms; sigma derivatives are both zero.
Source indices and field precision are identical in all three setups.
This finds no source-parameter derivative omission, but is not an RF or
full-solver acceptance test. The run was from the working tree before the
height/source fixture repair; no original clean-source receipt was captured.
The driver is retained for reproduction. Its initial zero/zero relative
sigma summary was NaN; comparison-corrected.json records that undefined
relative metric as null and the exact zero absolute difference, recomputed
from the retained arrays without another run.
