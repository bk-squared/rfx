`compute_mixed_s_matrix` now refuses traced array inputs and simulation
declarations at entry with an `RfxRefusal` naming the input, instead of a raw
JAX or NumPy error. The message identifies the differentiable microstrip and
lumped-port routes.
