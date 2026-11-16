"""Messages shared with the coax calculator's public refusal contract."""

# Appended to the shared passivity guard's message when THIS lane refuses, so
# the exception a caller actually reads names the way back to the matrix. The
# shared message (rfx/sparams/_common.py) is used by five extractors and is
# left untouched; only this lane refuses by default.
COAX_MSL_TRANSITION_REFUSAL_HINT = (
    "compute_coax_msl_transition is an EXPERIMENTAL cross-family method and "
    "refuses a non-passive S by default (strict_passivity=True); pass "
    "strict_passivity=False to get the diagnostic matrix back with a "
    "UserWarning instead of this error."
)
