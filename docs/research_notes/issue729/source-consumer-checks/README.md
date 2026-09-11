# Source consumer checks

The candidate passes six real clearing-call captures (uniform run/forward,
x/y axes, and the NU run path) plus the direct coax-MSL setup capture. No
FDTD solve occurs: each test stops at the actual production setup boundary.
A two-cell-thick trace supplies a real PEC normal edge that the old inclusive
source support incorrectly released. The new tests check that edge and both
tangential masks. The direct transition check reads actual wall bounds and
one-volt modal normalization, without claiming transition calibration.

On current main e32e386d, six of these seven tests fail: five trace-clearing
cases and the direct coax source count. The -y forward clearing-only case
passes because the independently diagnosed old forward frame bug moves the
feed away from the intended location. It is caught by the separate existing
run/forward physical-frame test. The two checks are complementary.
The archived baseline diagnostic initially asserted seven failures and thus
exited 1 after observing six; retain that honest log. No physical test or
production behavior was changed to obtain a different baseline count.
