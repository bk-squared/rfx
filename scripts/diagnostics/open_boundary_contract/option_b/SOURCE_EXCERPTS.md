rfx/sparams/msl.py:260 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
260:     if not self._msl_ports:
261:         raise ValueError("No MSL ports registered. Call add_msl_port() first.")
262:     if self._ports or self._waveguide_ports or self._floquet_ports:
263:         raise NotImplementedError(
264:             "compute_msl_s_matrix() is defined only for add_msl_port(...) "
265:             "families in the current simulation. Use separate "
266:             "simulations for add_port(...), add_waveguide_port(...), "
267:             "or add_floquet_port(...) S-parameter workflows."
268:         )
```

rfx/sparams/msl.py:310 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
310:     entries = list(self._msl_ports)
311:     n_ports = len(entries)
312: 
313:     # Probe placement, material assembly and each run share the resolved
314:     # mesh. Missing z profiles are synthesized locally by the grid builder;
315:     # writing a derived profile into the declaration would freeze auto-mesh
316:     # state and change the resolved domain during the driver.
317:     grid = self._build_realized_grid()
318: 
319:     if freqs is None:
320:         freqs_arr = np.asarray(jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs))
321:     else:
322:         freqs_arr = np.asarray(freqs)
323:     n_freqs_used = int(freqs_arr.shape[0])
324: 
325:     # Issue #469: solve the probe-offset interval for AUTO ports (the
326:     # downstream reflector term is only computable here, with the full
327:     # geometry registered — see _resolve_msl_auto_offsets).
328:     entries = self._resolve_msl_probe_entries(grid)
329: 
330:     # Build MSLPort descriptors and probe coords once (geometry shared).
331:     # Issue #661: msl_port_from_entry projects ``position`` onto the
332:     # port frame for whichever in-plane axis ``direction`` names.
333:     msl_ports = [msl_port_from_entry(pe) for pe in entries]
```

rfx/sparams/msl.py:496 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
496:         meta = port_idx_meta[p_idx]
497:         # Walk UP the substrate-normal axis (always z) from the
498:         # substrate top, at the feed cell on the propagation axis and
499:         # the trace centre on the width axis (issue #661).
500:         _ij = tuple(
501:             meta["i_feed"] if c == meta["prop_idx"] else meta["j_centre"]
502:             for c in range(3) if c != meta["normal_idx"]
503:         )
504:         _k_lo_tr, _k_hi_tr = _trace_planes(
505:             _msl_pec_edge_masks, meta["normal_idx"], _ij, meta["k_top"],
506:             periodic=self._periodic_flags())
507:         if _k_lo_tr is None:
508:             raise RuntimeError(
509:                 "compute_msl_s_matrix: no realized PEC trace conductor "
510:                 "found above the substrate top for MSL port "
511:                 f"{entries[p_idx].name!r}; the closed Ampere-loop "
512:                 "current (issue #80 stage S1) needs the trace. Declare "
513:                 "the microstrip trace as a Box(material='pec') (a "
514:                 "volume) or as a zero-thickness Box / add_thin_conductor "
515:                 "(a sheet, #931). A surface_impedance_f0 thin conductor "
516:                 "is NOT a trace conductor here — it realizes no PEC "
517:                 "edge, and the Ampere-loop current and V span anchor on "
518:                 "realized PEC wall planes. Keep the trace PEC and use f0 "
519:                 "sheets for auxiliary lossy metal only."
520:             )
521:         trace_k_per_port.append((_k_lo_tr, _k_hi_tr))
```

tests/oracle/test_waveguide_port_validation_battery.py:49 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
49: 
50: DOMAIN = (0.12, 0.04, 0.02)
51: PORT_LEFT_X = 0.01
52: PORT_RIGHT_X = 0.09
53: F_CUTOFF_HZ = 3.75e9
54: TARGET_CPML_M = 0.030  # 30 mm physical CPML absorber target
55: #: whole cells of the PEC short (#931: a Box is a volume; a sub-cell
56: #: extent is refused, and this module does not pin dx).
57: SHORT_CELLS = 2
58: 
59: 
60: def _build_sim(
61:     freqs_hz,
```

tests/oracle/test_pml_reflectivity.py:34 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
34:         f0 = 2e9
35:         n_steps = 400
36: 
37:         pulse = GaussianPulse(f0=f0, bandwidth=0.5)
38: 
39:         # --- Reference: large PEC domain (no reflections reach probe) ---
40:         grid_ref = Grid(freq_max=freq_max, domain=(0.20, 0.20, 0.20),
41:                         cpml_layers=0)
42:         state_ref = init_state(grid_ref.shape)
43:         materials_ref = init_materials(grid_ref.shape)
44: 
45:         cx_r = grid_ref.nx // 2
46:         cy_r = grid_ref.ny // 2
47:         cz_r = grid_ref.nz // 2
48:         probe_ref = (cx_r + 3, cy_r, cz_r)
49:         dt_r, dx_r = grid_ref.dt, grid_ref.dx
50: 
51:         ts_ref = np.zeros(n_steps)
52:         for n in range(n_steps):
53:             t = n * dt_r
54:             state_ref = update_h(state_ref, materials_ref, dt_r, dx_r)
55:             state_ref = update_e(state_ref, materials_ref, dt_r, dx_r)
56:             state_ref = apply_pec(state_ref)
57:             ez = state_ref.ez.at[cx_r, cy_r, cz_r].add(pulse(t))
58:             state_ref = state_ref._replace(ez=ez)
59:             ts_ref[n] = float(state_ref.ez[probe_ref])
60: 
```

rfx/boundaries/cpml.py:153 at e7f7e02704fd46ea7e21f127b19fc81cb66d6148

```text
153:     # The hi face uses jnp.flip() to reverse this.
154:     rho = 1.0 - xp.arange(n_layers, dtype=work_dtype) / max(n_layers - 1, 1)
155:     sigma = sigma_max * rho**order
156:     # κ graded from kappa_max (outer) to 1.0 (inner): κ(ρ) = 1 + (κ_max - 1) * ρ^m
157:     kappa = 1.0 + (kappa_max - 1.0) * rho**order
158:     alpha = 0.05 * (1.0 - rho)  # α: small, decreasing toward outer boundary
159: 
160:     # Update coefficients
161:     denom = sigma * kappa + kappa**2 * alpha
162:     b = xp.exp(-(sigma / kappa + alpha) * dt / EPS_0)
163:     c = xp.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
164: 
165:     return CPMLParams(
166:         sigma=jnp.asarray(sigma, dtype=jnp.float32),
```
