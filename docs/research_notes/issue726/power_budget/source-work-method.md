# Scope and metric definitions for the667 source-work readout

The source-work matrix is the response of two fixed distributed source-model
power ports. It does not replace, calibrate, or prove the accuracy of the
first-probe-plane MSL S. It does not prove global CPML passivity or identify
a unique cause of lateral power inflow. Results cover the stored161 frequency
samples, not every intervening frequency or arbitrary source profiles.

The four reported source-projection settling values use

    10*log10(mean(phi_mid[-floor(N/10):]**2) / max(phi_mid**2))

This is last10percent RMS divided by whole-record peak. It is neither the
maximum tail amplitude nor a rigorous remaining-frequency-error bound.
The independent reader reproduces all four values and also records a
separate last5percent peak measure. Original producer and readout files
are retained; this note makes the metric explicit.

The production source increments are float32 fl(Cb*e*u). The recorder
stores the actual base waveform u and fixed profile e, not every rounded
cell increment. A pre-run direct reconstruction matched increments to
2.22e-7 relative peak error. The independent midpoint DTFT uses float64
phase evaluation; streaming E DFTs retain the production float32 clock.
Their measured differences are recorded and are not treated as zero.

The effective source references are derived from actual added sigma and
fixed profile norm (about49.999999877 ohm); they are not fitted to the
response. An independent active-response plant remains nonpassive through
the same reader, so the calculation does not enforce the reported result.
