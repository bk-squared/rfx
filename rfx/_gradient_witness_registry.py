"""The gradient witness's bar and its two comparison forms, read by rfx.ringdown.gradient_witness.

In probe mode the same forms compare gradients of probe_spectra and
probe_spectra_long; the port-mode rule strings are kept for compatibility.

``GRADIENT_WITNESS_BAR`` bounds each leaf's largest difference of the two gradients relative to that leaf's
largest magnitude, per frequency bin when ``bin_axis`` is given."""

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
