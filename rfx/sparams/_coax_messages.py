"""Messages shared with the coax calculator's public refusal contract."""

COAX_MSL_TRANSITION_REFUSAL_HINT = (
    "compute_coax_msl_transition is an EXPERIMENTAL cross-family method and "
    "refuses a non-passive S by default (strict_passivity=True); pass "
    "strict_passivity=False to get the diagnostic matrix back with a "
    "UserWarning instead of this error."
)
