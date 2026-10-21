"""The gradient witness's bar and its two comparison forms, read by rfx.ringdown.gradient_witness."""

#: The bar of the gradient witness (PI, 2026-09-25): the largest difference
#: of the two gradients, relative to the largest magnitude of the gradient.
GRADIENT_WITNESS_BAR = 1.0e-2

_GRADIENT_WITNESSES = {
    "early_start": (
        "WE_gradient",
        "max over the parameters of |g - g_E| / max |g|: g through "
        "ringdown.s_params (window [window_start T, T]), g_E through "
        "ringdown.s_params_long (window [window_start/2 T, T]), same record"),
    "longer_record": (
        "record_length_gradient",
        "max over the parameters of |g - g_L| / max |g|: g through "
        "ringdown.s_params, g_L through ringdown.s_params of a record 1.5-2x "
        "longer"),
}
