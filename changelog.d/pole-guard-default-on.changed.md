The float32 reverse-mode pole-strength underflow warning is now on by default.
Measured cost per traced call: the upper bound of the median on/off time ratio is
at most 1.026 on CPU and 1.016 on GPU. Set `RFX_POLE_UNDERFLOW_CHECK=0` before
tracing to turn it off; a cached compile keeps its setting. Under `jax.vmap`,
the check makes one host call per call and reports the total flagged count.
