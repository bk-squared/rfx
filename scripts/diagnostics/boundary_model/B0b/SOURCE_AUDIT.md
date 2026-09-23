# Source record

Commit: `798ec64e5cda057318664bc431e528819f9e371a`.

| Symbol | Calls under rfx/ |
|---|---:|
| `apply_bloch_periodic_x` | 0 |
| `extract_floquet_modes` | 3 |
| `update_floquet_dft` | 0 |
| `inject_floquet_source` | 0 |
| `compute_floquet_s_params` | 0 |

## Call paths read

`Simulation.run -> run_uniform -> rfx.simulation.run -> _build_step_setup -> make_core_step -> Yee updates`.
`Simulation.forward -> _forward_from_materials -> rfx.simulation.run -> _build_step_setup -> make_core_step -> Yee updates`.
`RISUnitCell.sweep_angle -> _build_sim -> add_floquet_port; sweep_angle -> sim.run -> _extract_reflection`.
`compute_floquet_s_params -> extract_floquet_modes` (three syntactic calls).

## Exact searches

`rg -n apply_bloch_periodic_x rfx`
Exit: 0
```text
rfx/floquet.py:122:def apply_bloch_periodic_x(

```

`rg -n floquet_port_configs|update_floquet_dft|inject_floquet_source|compute_floquet_s_params|extract_floquet_modes rfx`
Exit: 0
```text
rfx/api/__init__.py:2699:        # Only the specular (0,0) TE mode is implemented in extract_floquet_modes; n_modes>1 and
rfx/__init__.py:203:    update_floquet_dft,
rfx/__init__.py:204:    inject_floquet_source,
rfx/__init__.py:205:    extract_floquet_modes,
rfx/__init__.py:206:    compute_floquet_s_params,
rfx/__init__.py:242:#     (coaxial_tem_*, *_plane_vi*, extract_floquet_modes, extract_multimode_*,
rfx/__init__.py:286:    "extract_s_matrix_wire", "compute_floquet_s_params",
rfx/runners/uniform.py:650:    floquet_port_configs = []
rfx/runners/uniform.py:665:        floquet_port_configs.append({
rfx/floquet.py:223:def update_floquet_dft(
rfx/floquet.py:297:def extract_floquet_modes(
rfx/floquet.py:339:            f"extract_floquet_modes only implements the specular (0,0) mode "
rfx/floquet.py:434:def inject_floquet_source(
rfx/floquet.py:525:def compute_floquet_s_params(
rfx/floquet.py:590:    inc_modes = extract_floquet_modes(
rfx/floquet.py:592:    ref_modes = extract_floquet_modes(
rfx/floquet.py:601:        trans_modes = extract_floquet_modes(

```

`rg -n scan_theta|bloch rfx/runners rfx/api/_execute.py rfx/api/_preflight.py rfx/preflight`
Exit: 0
```text
rfx/api/_execute.py:1480:                method=getattr(self._tfsf, "method", "bloch"),
rfx/runners/distributed.py:1315:        sim, lane="distributed (v1) pmap runner", bloch=kwargs.get("bloch"))
rfx/runners/uniform.py:616:            method=getattr(sim._tfsf, 'method', 'bloch'),
rfx/runners/uniform.py:669:            "scan_theta": fpe.scan_theta,
rfx/runners/distributed_v2.py:455:def refuse_unsupported_distributed_features(sim, *, lane, bloch=None):
rfx/runners/distributed_v2.py:459:    ``bloch`` also accepts an explicit phase from a direct caller.
rfx/runners/distributed_v2.py:467:    if bloch is None:
rfx/runners/distributed_v2.py:468:        bloch = getattr(sim, "_bloch", None)
rfx/runners/distributed_v2.py:469:    if periodic_axes or bloch is not None:
rfx/runners/distributed_v2.py:473:        if bloch is not None:
rfx/runners/distributed_v2.py:577:        sim, lane="distributed (v2) runner", bloch=kwargs.get("bloch"))

```

`git status --porcelain`
Exit: 0
```text


```

## Source excerpts

### rfx/api/__init__.py

```text
2641:     def add_floquet_port(
2642:         self,
2643:         position: float,
2644:         *,
2645:         axis: str = "z",
2646:         scan_theta: float = 0.0,
2647:         scan_phi: float = 0.0,
2648:         polarization: str = "te",
2649:         n_modes: int = 1,
2650:         freqs: jnp.ndarray | None = None,
2651:         n_freqs: int = 50,
2652:         f0: float | None = None,
2653:         bandwidth: float = 0.5,
2654:         amplitude: float = 1.0,
2655:         name: str | None = None,
2656:     ) -> "Simulation":
2657:         """Add a Floquet port for periodic structure / phased array analysis.
2658: 
2659:         The Floquet port injects a plane wave at the given scan angle
2660:         and extracts Floquet mode amplitudes (S-parameters) from the
2661:         unit cell response.  Requires periodic BC on the two axes
2662:         perpendicular to the port normal.
2663: 
2664:         Parameters
2665:         ----------
2666:         position : float
2667:             Physical coordinate along the port normal axis (metres).
2668:         axis : str
2669:             Port normal axis: ``"x"``, ``"y"``, or ``"z"``.
2670:         scan_theta : float
2671:             Scan angle theta from broadside (degrees). Default 0.
2672:         scan_phi : float
2673:             Scan angle phi in the transverse plane (degrees). Default 0.
2674:         polarization : str
2675:             ``"te"`` or ``"tm"``. Default ``"te"``.
2676:         n_modes : int
2677:             Number of Floquet modes to extract (default 1 = specular).
2678:         freqs : array or None
2679:             Analysis frequencies. Auto-generated if None.
2680:         n_freqs : int
2681:             Number of frequency points when ``freqs`` is None.
2682:         f0 : float or None
2683:             Source center frequency. Default: ``freq_max / 2``.
2684:         bandwidth : float
2685:             Source fractional bandwidth. Default 0.5.
2686:         amplitude : float
2687:             Source amplitude. Default 1.0.
2688:         name : str or None
2689:             Optional name for the port. Auto-generated if None.
2690:         """
2691:         if axis not in ("x", "y", "z"):
2692:             raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
2693:         if polarization not in ("te", "tm"):
2694:             raise ValueError(f"polarization must be 'te' or 'tm', got {polarization!r}")
2695:         if scan_theta < 0 or scan_theta >= 90:
2696:             raise ValueError(f"scan_theta must be in [0, 90), got {scan_theta}")
2697:         if n_modes < 1:
2698:             raise ValueError(f"n_modes must be >= 1, got {n_modes}")
2699:         # Only the specular (0,0) TE mode is implemented in extract_floquet_modes; n_modes>1 and
2700:         # TM were silently accepted and returned wrong results (higher-order lobes dropped; TM read
2701:         # the TE field pair + impedance). Fail loud until implemented (RF-audit 2026-07-23).
2702:         if n_modes > 1:
2703:             raise NotImplementedError(
2704:                 f"add_floquet_port(n_modes={n_modes}): only the specular (0,0) Floquet mode is "
2705:                 f"extracted; higher-order grating lobes are not implemented. Use n_modes=1."
2706:             )
2707:         if polarization == "tm":
2708:             raise NotImplementedError(
2709:                 "add_floquet_port(polarization='tm'): TM Floquet S-parameter extraction is not "
2710:                 "implemented — the extractor is hardwired to the TE (Ex,Hy) pair and TE wave "
2711:                 "impedance, so a TM drive yields wrong S-parameters. Use polarization='te'. "
2712:                 "(For a TM plane-wave field study without S-parameters, use add_tfsf_source, "
2713:                 "which supports oblique TM incidence.)"
2714:             )
2715:         if self._tfsf is not None:
2716:             raise ValueError(
2717:                 "Floquet ports are not supported together with TFSF sources"
2718:             )
2719: 
2720:         # Auto-set periodic axes for the two transverse directions
2721:         transverse = "".join(a for a in "xyz" if a != axis)
2722:         if not self._periodic_axes:
2723:             self._periodic_axes = transverse
2724:         else:
2725:             for a in transverse:
2726:                 if a not in self._periodic_axes:
2727:                     raise ValueError(
2728:                         f"Floquet port on axis={axis!r} requires periodic BC on {transverse!r}, "
2729:                         f"but periodic_axes={self._periodic_axes!r}"
2730:                     )
2731: 
2732:         # Reject an explicit incompatible declaration at registration. Auto
2733:         # mesh depends on the completed model and is checked by preflight.
2734:         if self._declared_mesh["_dz_profile"] is not None:
2735:             raise ValueError(
2736:                 "Floquet ports do not support non-uniform z mesh (dz_profile). "
2737:                 "Set dx explicitly to prevent auto-mesh from creating NU grid."
2738:             )
2739: 
2740:         if name is None:
2741:             name = f"floquet_{len(self._floquet_ports)}"
2742: 
2743:         self._floquet_ports.append(_FloquetPortEntry(
2744:             name=name,
2745:             position=position,
2746:             axis=axis,
2747:             scan_theta=scan_theta,
2748:             scan_phi=scan_phi,
2749:             polarization=polarization,
2750:             n_modes=n_modes,
2751:             freqs=freqs,
2752:             n_freqs=n_freqs,
2753:             f0=f0,
2754:             bandwidth=bandwidth,
2755:             amplitude=amplitude,
2756:         ))
2757:         return self
2758: 
2759:     # ---- probes ----
2760: 
2761:     def add_probe(
2762:         self,
```

### rfx/runners/uniform.py

```text
649:     # Floquet port sources — inject plane wave via standard source mechanism
650:     floquet_port_configs = []
651:     axis_map_str = {"x": 0, "y": 1, "z": 2}
652:     for fpe in sim._floquet_ports:
653:         axis_idx = axis_map_str[fpe.axis]
654:         pos_vec = [0.0, 0.0, 0.0]
655:         pos_vec[axis_idx] = fpe.position
656:         port_grid_index = grid.position_to_index(tuple(pos_vec))[axis_idx]
657: 
658:         fp_f0 = fpe.f0 if fpe.f0 is not None else sim._freq_max / 2
659:         fp_freqs = (
660:             fpe.freqs
661:             if fpe.freqs is not None
662:             else jnp.linspace(sim._freq_max / 10, sim._freq_max, fpe.n_freqs)
663:         )
664: 
665:         floquet_port_configs.append({
666:             "name": fpe.name,
667:             "axis": axis_idx,
668:             "port_index": port_grid_index,
669:             "scan_theta": fpe.scan_theta,
670:             "scan_phi": fpe.scan_phi,
671:             "polarization": fpe.polarization,
672:             "n_modes": fpe.n_modes,
673:             "freqs": fp_freqs,
674:             "f0": fp_f0,
675:             "bandwidth": fpe.bandwidth,
676:             "amplitude": fpe.amplitude,
677:         })
678: 
679:         # Add a soft source at the port plane (uniform across the plane)
680:         from rfx.sources.sources import GaussianPulse as GP
681:         wf = GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
682:         # Place source at center of transverse plane
683:         center = [sim._domain[i] / 2.0 for i in range(3)]
684:         center[axis_idx] = fpe.position
685:         if fpe.polarization == "te":
686:             # TE: inject first tangential E component
687:             if fpe.axis == "z":
688:                 comp = "ex"
689:             elif fpe.axis == "x":
690:                 comp = "ey"
691:             else:
692:                 comp = "ex"
693:         else:
694:             # TM: inject second tangential E component
695:             if fpe.axis == "z":
696:                 comp = "ey"
697:             elif fpe.axis == "x":
698:                 comp = "ez"
699:             else:
700:                 comp = "ez"
701:         from rfx.simulation import make_source as _make_src
702:         sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))
703: 

777:             sheet_impedance=sheet_ctx,
778:             **({} if report_every is None else
779:                {"report_every": report_every, "report_label": report_label}),
780:         )
781:     else:
782:         sim_result = _simulation.run(
783:             grid, materials, n_steps,
784:             boundary=sim._boundary,
785:             cpml_axes=cpml_axes,
786:             pec_axes=pec_axes,
787:             periodic=periodic,
788:             debye=debye,
789:             lorentz=lorentz,
790:             tfsf=tfsf,
791:             sources=sources,
792:             probes=probes,
793:             dft_planes=dft_planes,
794:             flux_monitors=flux_monitors,
795:             waveguide_ports=waveguide_ports,
796:             ntff=ntff_box,
797:             snapshot=snapshot,
798:             checkpoint=checkpoint,
799:             aniso_eps=aniso_eps,
800:             aniso_inv_eps=aniso_inv_eps,
801:             pec_mask=pec_mask,
802:             pec_edge_masks=pec_edge_masks,
803:             pec_sheets=pec_sheets,
804:             pec_wires=pec_wires,
805:             conformal_weights=conformal_weights,
806:             wire_port_sparams=wire_sparam_specs or None,
807:             lumped_rlc=rlc_metas,
808:             kerr_chi3=kerr_chi3,
809:             field_dtype=field_dtype,
810:             mag_sources=mag_sources or None,
811:             stencil_order=sim._stencil_order,
812:             sheet_impedance=sheet_ctx,
813:             **({} if report_every is None else
814:                {"report_every": report_every, "report_label": report_label}),
815:         )

836: 
837:     s_params = None
838:     freqs_out = None
839: 
840:     _single_wire_fastpath = (
841:         compute_s_params
842:         and wire_ports
843:         and not lumped_ports
844:         and len(wire_ports) == 1
845:         and bool(sim_result.wire_port_sparams)
846:     )
847: 
```

### rfx/api/_execute.py

```text
1981:         # Floquet ports — inject soft source, same as run_uniform.py:274-327
1982:         if self._floquet_ports:
1983:             axis_map_str = {"x": 0, "y": 1, "z": 2}
1984:             for fpe in self._floquet_ports:
1985:                 axis_idx = axis_map_str[fpe.axis]
1986:                 fp_f0 = fpe.f0 if fpe.f0 is not None else self._freq_max / 2
1987:                 from rfx.sources.sources import GaussianPulse as _GP
1988:                 wf = _GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
1989:                 center = [self._domain[i] / 2.0 for i in range(3)]
1990:                 center[axis_idx] = fpe.position
1991:                 if fpe.polarization == "te":
1992:                     comp = {"z": "ex", "x": "ey", "y": "ex"}[fpe.axis]
1993:                 else:
1994:                     comp = {"z": "ey", "x": "ez", "y": "ez"}[fpe.axis]
1995:                 from rfx.simulation import make_source as _make_src
1996:                 sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))
1997:         # ── Port guard on the occupancy (issue #82; runs on both lanes) ──
1998:         # The tensor lane (``RFX_PEC_OCC_KOTTKE=1``) used to dilate the
1999:         # occupancy by one cell on every face, so a probe-fed patch one

2119:                 ))
2120: 
2121:         result = _run(
2122:             grid,
2123:             materials,
2124:             n_steps,
2125:             boundary=self._boundary,
2126:             cpml_axes=cpml_axes_run,
2127:             pec_axes=pec_axes_run,
2128:             periodic=periodic_bool,
2129:             tfsf=tfsf_run,
2130:             debye=debye,
2131:             lorentz=lorentz,
2132:             sources=sources,
2133:             probes=probes,
2134:             waveguide_ports=waveguide_ports if waveguide_ports else None,
2135:             ntff=ntff_box,
2136:             checkpoint=checkpoint,
2137:             checkpoint_segments=checkpoint_segments,
2138:             pec_mask=pec_mask_local,
2139:             pec_edge_masks=pec_edge_masks_local,
2140:             pec_sheets=pec_sheets,
2141:             pec_wires=pec_wires,
2142:             pec_occupancy=pec_occupancy_for_run,
2143:             aniso_inv_eps=aniso_inv_eps_run,
2144:             aniso_inv_eps_smooth=(aniso_inv_eps_run is not None),
2145:             lumped_port_sparams=lumped_port_sparam_specs or None,
2146:             wire_port_sparams=wire_port_sparam_specs or None,
2147:             wire_refplane_sparams=wire_refplane_specs or None,
2148:             lumped_rlc=rlc_metas,
2149:             kerr_chi3=kerr_chi3,
2150:             dft_planes=dft_planes if dft_planes else None,
2151:             flux_monitors=flux_monitor_cfgs if flux_monitor_cfgs else None,
2152:             return_state=False,
2153:             stencil_order=self._stencil_order,
2154:             sheet_impedance=sheet_impedance,
2155:             design_box=design_box,
2156:             design_occupancy=design_occupancy,
2157:             field_dtype=self._resolve_field_dtype(),
2158:         )
2159: 

2259:             warn_if_nonpassive_lumped_s11(
2260:                 s_params_out, freqs_out,
2261:                 extractor="forward(port_s11_freqs=...)",
2262:             )
2263: 
2264:         # Convert tuple → name-keyed dict, mirroring runners/uniform.py:704
2265:         # so consumers can index by the same name they registered with.
2266:         dft_planes_out = None
2267:         sim_dft_planes = getattr(result, "dft_planes", None)
2268:         if self._dft_planes and sim_dft_planes:
2269:             dft_planes_out = {
2270:                 entry.name: probe
2271:                 for entry, probe in zip(self._dft_planes, sim_dft_planes)
2272:             }
2273: 
2274:         return ForwardResult(
2275:             time_series=result.time_series,
2276:             ntff_data=result.ntff_data,
2277:             ntff_box=result.ntff_box,
2278:             grid=result.grid,
2279:             s_params=s_params_out,
2280:             freqs=freqs_out,
2281:             lumped_port_sparams=result.lumped_port_sparams,
2282:             wire_port_sparams=result.wire_port_sparams,
2283:             dft_planes=dft_planes_out,
2284:         )
2285: 
2286:     @staticmethod
2287:     def _pack_nu_forward_result(
2288:         *,
2289:         time_series,
2290:         grid,
```

### rfx/floquet.py

```text
122: def apply_bloch_periodic_x(
123:     state: FDTDState,
124:     phase_x: jnp.ndarray,
125: ) -> FDTDState:
126:     """Apply Bloch-periodic BC on x-axis boundaries.
127: 
128:     For real-valued FDTD with complex Bloch phase, we split into
129:     real and imaginary parts. However, since standard FDTD operates
130:     on real fields, we use the split-field technique: the actual
131:     fields carry the spatially-varying phase factor.
132: 
133:     For broadside (phase_x = 1), this reduces to standard periodic BC
134:     (copy field from one side to the other).
135: 
136:     Parameters
137:     ----------
138:     state : FDTDState
139:         Current field state.
140:     phase_x : complex scalar
141:         Bloch phase factor exp(j * kx * Lx).
142: 
143:     Returns
144:     -------
145:     Updated FDTDState with Bloch-periodic x boundaries.
146:     """
147:     # For the Yee grid, periodic wrapping copies the field from the
148:     # last interior cell to the ghost cell at the opposite end, with
149:     # the appropriate phase factor applied.
150:     #
151:     # For H-field forward differences: F[N] = F[0] * phase
152:     # For E-field backward differences: F[-1] = F[N-1] * conj(phase)
153:     #
154:     # In the standard periodic case (phase=1), jnp.roll handles this
155:     # automatically. For Bloch-periodic, we need explicit wrapping.
156:     phase_re = jnp.real(phase_x)
157: 
158:     # Apply to each field component at x boundaries
159:     # x=0 boundary: field = field[nx-1] * conj(phase_x)
160:     # x=nx-1 boundary: field = field[0] * phase_x
161:     def _wrap(f):
162:         # For real FDTD, we only apply the real part of the phase shift.
163:         # This is exact for broadside and a good approximation for small angles.
164:         # For full complex Bloch, a split-field formulation would be needed.
165:         f = f.at[0, :, :].set(f[-1, :, :] * phase_re)
166:         f = f.at[-1, :, :].set(f[0, :, :] * phase_re)
167:         return f
168: 
169:     return state._replace(
170:         ex=_wrap(state.ex),
171:         ey=_wrap(state.ey),
172:         ez=_wrap(state.ez),
173:         hx=_wrap(state.hx),
174:         hy=_wrap(state.hy),
175:         hz=_wrap(state.hz),
176:     )
177: 
178: 
179: # ---------------------------------------------------------------------------
180: # Floquet mode DFT accumulation
181: # ---------------------------------------------------------------------------
182: 

296: 
297: def extract_floquet_modes(
298:     acc: FloquetDFTAccumulator,
299:     dx: float,
300:     Lx: float,
301:     Ly: float,
302:     freqs: jnp.ndarray,
303:     theta_deg: float = 0.0,
304:     phi_deg: float = 0.0,
305:     n_modes: int = 1,
306: ) -> dict:
307:     """Extract Floquet mode amplitudes from accumulated DFT data.
308: 
309:     Only the specular ``(0,0)`` mode with **TE** polarization is implemented (the
310:     extractor reads the (Ex, Hy) tangential pair and the TE wave impedance
311:     ``eta0/cos(theta)``). ``n_modes > 1`` (higher-order grating lobes) and TM
312:     polarization are NOT implemented and are rejected fail-loud rather than
313:     silently returning wrong results (RF-audit 2026-07-23).
314: 
315:     Parameters
316:     ----------
317:     acc : FloquetDFTAccumulator
318:         Accumulated field DFTs on the port plane.
319:     dx : float
320:         Grid cell size (metres).
321:     Lx, Ly : float
322:         Unit cell periods (metres).
323:     freqs : (n_freqs,) array
324:         Frequencies.
325:     theta_deg, phi_deg : float
326:         Scan angles (degrees).
327:     n_modes : int
328:         Number of Floquet modes to extract.
329: 
330:     Returns
331:     -------
332:     dict with keys:
333:         'S' : (n_modes, n_freqs) complex array of Floquet S-parameters
334:         'modes' : list of (m, n) mode index tuples
335:         'freqs' : frequency array
336:     """
337:     if int(n_modes) != 1:
338:         raise NotImplementedError(
339:             f"extract_floquet_modes only implements the specular (0,0) mode "
340:             f"(n_modes=1); got n_modes={n_modes}. Higher-order Floquet grating "
341:             f"lobes are not yet extracted — this fails loud instead of silently "
342:             f"returning only the (0,0) mode."
343:         )
344: 
345:     # Spatial averaging for the (0,0) mode = mean over the plane
346:     # This is the 2D spatial DFT at (kx=0, ky=0) normalized by area
347:     e1_avg = jnp.mean(acc.e_tang1_dft, axis=(1, 2))  # (n_freqs,)
348:     h2_avg = jnp.mean(acc.h_tang2_dft, axis=(1, 2))
349: 
350:     # Wave impedance for the specular mode
351:     theta = math.radians(theta_deg)
352:     eta0 = jnp.sqrt(MU_0 / EPS_0)  # ~377 ohms
353: 
354:     # TE mode impedance: eta_TE = eta0 / cos(theta)
355:     cos_theta = max(math.cos(theta), 1e-10)
356:     eta_te = eta0 / cos_theta
357: 
358:     # For the specular mode, decompose into forward/backward waves
359:     # using the E/H ratio. For a +z traveling wave: Hy = Ex / eta
360:     # Reflected: Hy = -Ex / eta
361:     # Forward amplitude: a = (E + eta*H) / 2
362:     # Backward amplitude: b = (E - eta*H) / 2
363:     a_te = (e1_avg + eta_te * h2_avg) / 2.0  # forward TE
364:     b_te = (e1_avg - eta_te * h2_avg) / 2.0  # backward TE
365: 
366:     modes_list = [(0, 0)]
367:     S_00 = b_te / jnp.where(jnp.abs(a_te) > 1e-30, a_te, 1e-30)
368: 
369:     result = {
370:         'S': S_00[None, :],  # (1, n_freqs)
371:         'modes': modes_list,
372:         'freqs': freqs,
373:         'forward_amplitude': a_te,
374:         'backward_amplitude': b_te,
375:     }
376: 
377:     return result
378: 
379: 
380: # ---------------------------------------------------------------------------
381: # FloquetPort configuration
382: # ---------------------------------------------------------------------------
383: 
384: @dataclass(frozen=True)
385: class FloquetPort:
386:     """Floquet port for periodic structure excitation and mode extraction.
387: 
388:     For a unit cell with periodic BC, the Floquet port injects a
389:     plane wave at a specified scan angle and extracts reflected
390:     and transmitted Floquet modes.
391: 
392:     Parameters
393:     ----------
394:     position : float

434: def inject_floquet_source(
435:     state: FDTDState,
436:     port_index: int,
437:     axis: int,
438:     dt: float,
439:     dx: float,
440:     step: int,
441:     f0: float,
442:     bandwidth: float,
443:     amplitude: float,
444:     polarization: str = "te",
445:     theta_deg: float = 0.0,
446:     phi_deg: float = 0.0,
447:     Lx: float = 0.0,
448:     Ly: float = 0.0,
449: ) -> FDTDState:
450:     """Inject a Floquet plane-wave source at the port plane.
451: 
452:     Uses a soft source (additive) that launches a Gaussian-modulated
453:     plane wave. For non-zero scan angles, the spatial phase gradient
454:     across the port plane is applied.
455: 
456:     Parameters
457:     ----------
458:     state : FDTDState
459:     port_index : int
460:         Grid index along the normal axis.
461:     axis : int
462:         Normal axis (0=x, 1=y, 2=z).
463:     dt, dx : float
464:         Timestep and cell size.
465:     step : int
466:         Current timestep index.
467:     f0 : float
468:         Center frequency (Hz).
469:     bandwidth : float
470:         Fractional bandwidth.
471:     amplitude : float
472:         Source amplitude (V/m).
473:     polarization : str
474:         "te" or "tm".
475:     theta_deg, phi_deg : float
476:         Scan angles (degrees).
477:     Lx, Ly : float
478:         Unit cell periods for phase computation.
479:     """
480:     t = step * dt
481:     tau = 1.0 / (f0 * bandwidth * jnp.pi)
482:     t0 = 3.0 * tau
483: 
484:     # Gaussian pulse envelope
485:     arg = (t - t0) / tau
486:     pulse = amplitude * (-2.0 * arg) * jnp.exp(-(arg ** 2))
487: 
488:     if axis == 2:  # z-normal
489:         if polarization == "te":
490:             # TE: inject Ex
491:             field = state.ex
492:             field = field.at[:, :, port_index].add(pulse)
493:             state = state._replace(ex=field)
494:         else:
495:             # TM: inject Ey
496:             field = state.ey
497:             field = field.at[:, :, port_index].add(pulse)
498:             state = state._replace(ey=field)
499:     elif axis == 0:  # x-normal
500:         if polarization == "te":
501:             field = state.ey
502:             field = field.at[port_index, :, :].add(pulse)
503:             state = state._replace(ey=field)
504:         else:
505:             field = state.ez
506:             field = field.at[port_index, :, :].add(pulse)
507:             state = state._replace(ez=field)
508:     else:  # y-normal (axis == 1)
509:         if polarization == "te":
510:             field = state.ex
511:             field = field.at[:, port_index, :].add(pulse)
512:             state = state._replace(ex=field)
513:         else:
514:             field = state.ez
515:             field = field.at[:, port_index, :].add(pulse)
516:             state = state._replace(ez=field)
517: 
518:     return state
519: 
520: 
```

### rfx/ris.py

```text
297:         f_center = (self._freq_range[0] + self._freq_range[1]) / 2.0
298:         sim.add_floquet_port(
299:             port_z,
300:             axis="z",
301:             scan_theta=theta,
302:             scan_phi=phi,
303:             polarization=self._polarization,
304:             n_freqs=self._n_freqs,
305:             f0=f_center,
306:         )
307: 
308:         # Probe above the patch
309:         probe_z = h_sub + (z_total - h_sub) * 0.3
310:         component = "ex" if self._polarization == "te" else "ey"
311:         sim.add_probe((Lx / 2, Ly / 2, probe_z), component=component)
312: 
313:         return sim
314: 
315:     # ---- sweep methods ----
316: 
317:     def sweep_capacitance(
318:         self,
319:         values: Sequence[float],
320:         freq: np.ndarray | None = None,

368:         self,
369:         theta_values: Sequence[float],
370:         freq: np.ndarray | None = None,
371:         phi: float = 0.0,
372:     ) -> RISSweepResult:
373:         """Sweep scan angle and extract reflection vs angle.
374: 
375:         Parameters
376:         ----------
377:         theta_values : sequence of float
378:             Scan angles theta in degrees to sweep over.
379:         freq : array or None
380:             Frequency points. If None, auto-generated from freq_range.
381:         phi : float
382:             Azimuth angle in degrees. Default 0.
383: 
384:         Returns
385:         -------
386:         RISSweepResult
387:         """
388:         theta_values = list(theta_values)
389:         if not theta_values:
390:             raise ValueError("theta_values must not be empty")
391: 
392:         all_phases = []
393:         all_amps = []
394:         result_freqs = None
395:         cap_val = self._varactors[0].capacitance_range[0] if self._varactors else None
396: 
397:         for theta in theta_values:
398:             sim = self._build_sim(
399:                 capacitance_override=cap_val,
400:                 theta=theta,
401:                 phi=phi,
402:             )
403:             result = sim.run(n_steps=self._n_steps)
404: 
405:             s11, freqs = _extract_reflection(result, self._freq_range, self._n_freqs)
406:             phase_deg = np.angle(s11, deg=True)
407:             amplitude = np.abs(s11)
408: 
409:             all_phases.append(phase_deg)
410:             all_amps.append(amplitude)
411:             if result_freqs is None:
412:                 result_freqs = freqs
413: 
414:         return RISSweepResult(
415:             phases=np.array(all_phases),
416:             amplitudes=np.array(all_amps),

534:     return _substrate_material_kwargs(material)["eps_r"]
535: 
536: 
537: def _extract_reflection(
538:     result,
539:     freq_range: tuple[float, float],
540:     n_freqs: int,
541: ) -> tuple[np.ndarray, np.ndarray]:
542:     """Extract S11 reflection from a simulation result.
543: 
544:     Tries Floquet S-params first — the only path here that returns a
545:     physically valid S11.
546: 
547:     WARNING: the time-domain FFT fallback below does NOT produce a valid
548:     S11. It FFTs the raw probe time series (total field, with no
549:     incident-field reference subtraction) and peak-normalizes the
550:     magnitude. This violates the repo rule "never FFT-of-probe for R(f)"
551:     and yields a spectrum shaped
552:     by the source, not a reflection coefficient. The fallback is kept
553:     only so RIS demos run end-to-end; treat its output as a qualitative
554:     placeholder, not a measured S-parameter.
555: 
556:     Returns
557:     -------
558:     s11 : (n_freqs,) complex array
559:     freqs : (n_freqs,) float array
560:     """
561:     # If the result has S-parameters from ports (rare for Floquet-only),
562:     # use them.
563:     if result.s_params is not None and result.freqs is not None:
564:         s11 = np.asarray(result.s_params)[0, 0, :]
565:         freqs = np.asarray(result.freqs)
566:         # Filter to freq_range
567:         mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
568:         if np.sum(mask) >= 2:
569:             return s11[mask], freqs[mask]
570: 
571:     # Fallback: FFT of probe time series
572:     ts = np.asarray(result.time_series).ravel()
573:     if result.dt is not None:
574:         dt = result.dt
575:     else:
576:         # Estimate dt from the simulation parameters
577:         dt = 1.0 / (freq_range[1] * 40)  # rough estimate
578: 
579:     n = len(ts)
580:     freqs_fft = np.fft.rfftfreq(n, d=dt)
581:     spectrum = np.fft.rfft(ts)
582: 
583:     # Interpolate to desired frequency points
584:     target_freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
585:     s11 = np.interp(target_freqs, freqs_fft, spectrum, left=0, right=0)
586: 
587:     # Normalize by peak
588:     peak = np.max(np.abs(s11))
589:     if peak > 0:
590:         s11 = s11 / peak
591: 
592:     return s11, target_freqs
```

### rfx/api/_preflight.py

```text
380:                 "result.waveguide_sparams but not Result.s_params"
381:             )
382:         if self._floquet_ports:
383:             messages.append(
384:                 "add_floquet_port(...) is experimental and has no "
385:                 "claims-bearing run(compute_s_params=True) S-matrix path"
386:             )
387:         if self._tfsf is not None:
388:             messages.append(
```

### tests/oracle/test_oblique_fresnel_magnitude.py

```text
1: """Differentiable OBLIQUE Fresnel |Γ|(θ) magnitude via forward() — ground-truth-gated.
2: 
3: Extends the normal-incidence complex Γ helper (#419) to oblique incidence (the RIS/absorber
4: condition), MAGNITUDE ONLY. forward() returns the raw complex Bloch envelope for oblique, so
5: ``oblique_reflection_magnitude`` uses the +j (conjugate) DFT kernel and a de-embed-free
6: magnitude ratio (a de-embedded complex mean cancels at oblique — the reflected phase spans
7: many π across the plateau).
8: 
9: Validated vs analytic oblique Fresnel ``fresnel_r_te(θ,εr)``: |Γ| err 7.2% / 3.9% @ θ=30°/45°
10: (εr=4); d|Γ|/dε AD==FD. 60° is CPML-contaminated (memory) so it is NOT gated. PHASE (complex
11: Γ) is a documented open follow-up (needs the exact discrete k_x de-embed).
12: Harness: docs/research_notes/experiments/i404_oblique_20260720/oblique_gamma_validate.py
13: """
14: import numpy as np
15: import jax
16: import jax.numpy as jnp
17: import pytest
18: 
19: from rfx.api import Simulation
20: from rfx.grid import Grid
21: from rfx.probes import oblique_reflection_magnitude, fresnel_r_te
22: 
23: F0, BW = 5e9, 0.15
24: DOMAIN = (0.60, 0.12, 0.006)
25: DX = 0.002
26: X_IFACE, X_SLAB_END = 0.15, 0.50
27: PLATEAU = (0.05, 0.13)
28: C0 = 299792458.0
29: EPS_R = 4.0
30: 
31: 
32: def _build(theta):
33:     grid = Grid(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=10)
34:     xi = grid.position_to_index((X_IFACE, 0.06, 0.003))[0]
35:     xe = grid.position_to_index((X_SLAB_END, 0.06, 0.003))[0]
36:     xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
37:     n_slab = np.sqrt(EPS_R)
38:     t_back = (2 * (X_IFACE - PLATEAU[1]) / C0) + (2 * (X_SLAB_END - X_IFACE) / (C0 / n_slab))
39:     ns = min(int(0.9 * t_back / grid.dt), 1700)
40:     sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
41:     sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
42:                         angle_deg=theta, waveform="modulated_gaussian")
43:     for xp in xprobes:
44:         sim.add_probe((float(xp), 0.06, 0.003), component="ez")
45:     return sim, grid.shape, xi, xe, grid.dt, ns
46: 
47: 
48: def _eps_slab(shape, xi, xe, eps):
49:     a = jnp.ones(shape, jnp.float32)
50:     return a if eps == 1.0 else a.at[xi:xe, :, :].set(eps)
51: 
52: 
53: def _build_lean(theta):
54:     """Small oblique cell for the GRADIENT test only. AD=FD is config-independent, and the
55:     reverse tape of the full-size config OOMs the GPU (~5.6GB alloc on rtx4090); this lean
56:     domain/step-count keeps the AD peak well under GPU memory for the pre-release gate."""
57:     dom = (0.40, 0.08, 0.006)
58:     xif, xend, plat = 0.10, 0.30, (0.04, 0.08)
59:     yc, zc = dom[1] / 2, dom[2] / 2
60:     grid = Grid(freq_max=10e9, domain=dom, dx=DX, cpml_layers=10)
61:     xi = grid.position_to_index((xif, yc, zc))[0]
62:     xe = grid.position_to_index((xend, yc, zc))[0]
63:     xprobes = np.arange(plat[0], plat[1], DX)
64:     n = np.sqrt(EPS_R)
65:     t_back = (2 * (xif - plat[1]) / C0) + (2 * (xend - xif) / (C0 / n))
66:     ns = min(int(0.9 * t_back / grid.dt), 1000)
67:     sim = Simulation(freq_max=10e9, domain=dom, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
68:     sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
69:                         angle_deg=theta, waveform="modulated_gaussian")
70:     for xp in xprobes:
71:         sim.add_probe((float(xp), yc, zc), component="ez")
72:     probe_idx = np.array([grid.position_to_index((float(xp), yc, zc))[0] for xp in xprobes], float)
73:     return sim, grid.shape, xi, xe, probe_idx, grid.dt, ns
74: 
75: 
76: def _series(sim, eps_arr, ns, ckpt=False):
77:     return sim.forward(eps_override=eps_arr, n_steps=ns, checkpoint=ckpt,
78:                        skip_preflight=True).time_series  # complex for oblique
79: 
80: 
81: @pytest.mark.slow
82: @pytest.mark.parametrize("theta,tol", [(30.0, 0.10), (45.0, 0.08)])
83: def test_oblique_fresnel_magnitude_vs_analytic(theta, tol):
84:     """|Γ|(θ) matches analytic oblique Fresnel R_TE and is passive; +j kernel is mandatory."""
85:     sim, shape, xi, xe, dt, ns = _build(theta)
86:     inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns)
87:     tot = _series(sim, _eps_slab(shape, xi, xe, EPS_R), ns)
88:     g = float(oblique_reflection_magnitude(tot, inc, f0=F0, dt=dt, n_gate=ns))
89:     ga = abs(fresnel_r_te(theta, EPS_R))
90:     assert g < 1.0, f"nonphysical |Γ|={g:.3f}≥1 (θ={theta})"
91:     assert abs(g - ga) / ga < tol, f"|Γ|={g:.3f} vs R_TE={ga:.3f} (θ={theta}, {abs(g-ga)/ga*100:.1f}%)"
92: 
93: 
94: # highmem (issue #545): this test's own footprint is UNMEASURED in CI -- it
95: # was the test killed in shard 2 (weekly run 31103127491), so its own
96: # post-test [rss] hook line never fired (the process died first). The
97: # process was already at 12,187 MB *entering* this test -- that number
98: # belongs to the PRECEDING msl-referee highmem test, not this one.
99: @pytest.mark.slow
100: @pytest.mark.highmem
101: def test_oblique_reflection_magnitude_differentiable():
102:     """d|Γ|/dε flows through the checkpointed oblique forward and matches finite difference."""
103:     sim, shape, xi, xe, _idx, dt, ns = _build_lean(30.0)  # lean config: AD tape fits GPU memory
104:     inc = _series(sim, _eps_slab(shape, xi, xe, 1.0), ns, ckpt=True)
105: 
106:     def absg(eps):
```

### rfx/probes/fresnel.py

```text
321: def oblique_reflection_magnitude(
322:     total_series: jnp.ndarray,
323:     incident_series: jnp.ndarray,
324:     *,
325:     f0: float,
326:     dt: float,
327:     n_gate: int | None = None,
328: ) -> jnp.ndarray:
329:     """Differentiable |Γ|(θ) MAGNITUDE for OBLIQUE incidence from the complex Bloch envelope.
330: 
331:     For oblique TFSF, ``Simulation.forward()`` returns the RAW complex Bloch envelope P
332:     (``time_series`` is complex64), not a real physical field. Two differences from the
333:     normal-incidence :func:`fresnel_reflection_coefficient`:
334: 
335:     * **+j (conjugate) DFT kernel** — the envelope ``P(t) ∝ exp(-j2πf0t)`` has its f0 content
336:       at ``-f0``, so it is extracted with ``exp(+j2πf0t)``. A ``-j`` kernel averages the
337:       incident to noise ⇒ nonphysical |Γ|≫1 (the #404 trap).
338:     * **magnitude ratio, de-embed-free** — ``mean|T−I| / mean|I|``. At oblique the reflected
339:       phase varies by many π across the plateau, so a de-embedded COMPLEX mean cancels
340:       (a wrong/nominal k_x gives |Γ|→0). The magnitude ratio is robust; the reflected phase
341:       is NOT recovered here (see Notes).
342: 
343:     Same physics protocol as :func:`fresnel_reflection_coefficient` but OBLIQUE-specific:
344:     narrowband source (``bandwidth≲0.15`` — the Bloch phase ``k_y=k0 sinθ`` is single-f0),
345:     finite THICK slab before the CPML, time-gate to the front-face, plateau probes.
346: 
347:     Parameters
348:     ----------
349:     total_series, incident_series : (n_steps, n_probes) COMPLEX arrays
350:         Bloch-envelope probe series from the scatterer and vacuum runs (oblique forward()).
351:     f0, dt : float
352:     n_gate : int, optional
353:         DFT window (time-gate). Defaults to the full series.
354: 
355:     Returns
356:     -------
357:     jnp.ndarray
358:         Real scalar |Γ|(f0) at the injected oblique angle, differentiable in ``total_series``.
359: 
360:     Notes
361:     -----
362:     MAGNITUDE ONLY. The reflected PHASE (complex Γ, the RIS phase-steering knob) is not
363:     provided: de-embedding to the interface needs the exact DISCRETE k_x (the injected angle
364:     is dispersion-shifted from the nominal, e.g. 29.5° for a 30° request), which is an open
365:     follow-up. Validated vs analytic ``fresnel_r_te`` at θ=30°/45° to ~4–7%.
366: 
367:     See Also
368:     --------
369:     fresnel_reflection_coefficient : normal-incidence COMPLEX Γ (amplitude + phase).
370:     fresnel_r_te : analytic |R_TE|(θ) ground truth.
371:     """
372:     total = jnp.asarray(total_series)
373:     inc = jnp.asarray(incident_series)
374:     if total.ndim != 2 or inc.shape != total.shape:
375:         raise ValueError(
376:             "total_series and incident_series must both be (n_steps, n_probes) and "
377:             f"equal-shaped; got {total.shape} and {inc.shape}"
378:         )
379:     ns = total.shape[0] if n_gate is None else int(n_gate)
380:     t = jnp.arange(ns) * dt
381:     kern = jnp.exp(+1j * 2.0 * jnp.pi * f0 * t) * dt  # +j conjugate kernel (oblique envelope)
382:     tp = jnp.sum(total[:ns].astype(jnp.complex64) * kern[:, None], axis=0)
383:     ip = jnp.sum(inc[:ns].astype(jnp.complex64) * kern[:, None], axis=0)
384:     return jnp.mean(jnp.abs(tp - ip)) / jnp.mean(jnp.abs(ip))
385: 
386: 
387: def oblique_reflection_coefficient(
388:     total_series: jnp.ndarray,
389:     incident_series: jnp.ndarray,
390:     *,
391:     f0: float,
392:     dt: float,
393:     probe_index: jnp.ndarray,
394:     interface_index: float,
395:     n_gate: int | None = None,
```

### rfx/core/yee.py

```text
172: def _diff_fwd_o(arr, axis, periodic, order, bloch=None):
173:     """Forward staggered first difference (f[i+1]-f[i] family), at i+1/2; the
174:     caller divides by dx. order=2 is byte-identical to ``_shift_fwd(arr)-arr``
175:     when ``bloch is None``.
176: 
177:     ``bloch`` (oblique-periodic Bloch field-transformation, #404): a length-3
178:     tuple of per-axis complex phases ``exp(-j·k_axis·dx)``.  On a PERIODIC axis
179:     the forward-rolled neighbour is multiplied by ``bloch[axis]`` so the plain
180:     ``jnp.roll`` wrap represents the exact discrete Yee derivative of a wave with
181:     transverse wavenumber ``k_axis``.  ``bloch=None`` (default) leaves the real
182:     path untouched; ``bloch`` is only threaded on order-2 (guarded upstream)."""
183:     if order == 2:
184:         if periodic[axis]:
185:             nxt = jnp.roll(arr, -1, axis)
186:             if bloch is not None:
187:                 nxt = nxt * bloch[axis]
188:             return nxt - arr
189:         return _shift_fwd(arr, axis) - arr
190:     # order == 4:  c1*(f[i+1]-f[i]) + c2*(f[i+2]-f[i-1])
191:     if periodic[axis]:
192:         near = jnp.roll(arr, -1, axis) - arr
193:         far = jnp.roll(arr, -2, axis) - jnp.roll(arr, 1, axis)
194:         return _C4_NEAR * near + _C4_FAR * far
195:     near = _shift_fwd(arr, axis) - arr
196:     far = _shift_fwd(_shift_fwd(arr, axis), axis) - _shift_bwd(arr, axis)
197:     return _ribbon_2nd(_C4_NEAR * near + _C4_FAR * far, near, axis)
198: 
199: 
200: def _diff_bwd_o(arr, axis, periodic, order, bloch=None):
201:     """Backward staggered first difference (f[i]-f[i-1] family), at i-1/2; the
202:     caller divides by dx. order=2 is byte-identical to ``arr-_shift_bwd(arr)``
203:     when ``bloch is None``.
204: 
205:     For the Bloch field-transformation (#404) the BACKWARD-rolled neighbour
206:     carries the CONJUGATE phase ``exp(+j·k_axis·dx)`` (``bloch[axis].conjugate()``),
207:     the exact discrete adjoint of the forward stagger.  See ``_diff_fwd_o``."""
208:     if order == 2:
209:         if periodic[axis]:
210:             prv = jnp.roll(arr, 1, axis)
211:             if bloch is not None:
212:                 prv = prv * bloch[axis].conjugate()
213:             return arr - prv
214:         return arr - _shift_bwd(arr, axis)
215:     # order == 4:  c1*(f[i]-f[i-1]) + c2*(f[i+1]-f[i-2])
216:     if periodic[axis]:
217:         near = arr - jnp.roll(arr, 1, axis)
218:         far = jnp.roll(arr, -1, axis) - jnp.roll(arr, 2, axis)
```

Conclusion: leader fills.
