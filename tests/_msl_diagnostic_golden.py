"""Main-export wording, retained as adjacent literals to keep source lines readable."""

TEXT_0 = (
    "MSL port 'checked' (trace W=488µm, h_sub=244µm): lateral clearance to −y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 488µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_1 = (
    "MSL port 'checked' (trace W=488µm, h_sub=244µm): lateral clearance to +y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 488µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_2 = (
    "MSL port 'checked': only 2 normal interval(s) between the validated ground and trace "
    'planes. Validated conductor-plane gap=244.1µm over 2 normal interval(s), from z=0.0µm'
    ' to 244.1µm (declared height=244.1µm). Separately, the declared-material column has 2'
    ' same-permittivity sample slot(s), extent 244.1µm. This material extent is not the '
    'conductor-plane gap. The existing resolution recommendation is at least 4 normal '
    'intervals and an aligned declared substrate interface. On a uniform mesh, refine to '
    'dx ≤ 61.0µm and align the declared ground z=0.0µm and trace z=244.1µm with nodes; on '
    'a profiled mesh, place sufficient nodes between those faces. Geometry screening does '
    'not quantify Z0 error. The historical sweep and its realized-board Hammerstad-Jensen '
    'anchor are pre-#802 records; a new matched-geometry measurement is needed before '
    'quoting a current accuracy bound.'
)

TEXT_3 = (
    "MSL port 'checked' at x=0.49mm, direction='+x': distance to nearest x-CPML = 244µm "
    '(domain edge + 244.1µm calibrated CPML buffer) < recommended 488µm (= 2·h_sub). '
    'Source-side CPML reflection may inflate |S11|. Move port further from boundary OR '
    'increase domain x-extent.'
)

TEXT_4 = (
    "MSL port 'checked' (direction='+x'): n_probe_offset=3 puts probe 0 366.2µm "
    "(1.50·h_sub) from this port's OWN feed plane, inside the source near-field standoff "
    "of 10 cells (1.221mm = 5·h_sub, the issue-#80 Fix B constant add_msl_port's auto "
    'offset already floors to). Within a few substrate thicknesses of the feed the '
    'launched field is not the guided mode yet: the evanescent content decays with the '
    "substrate's own transverse-resonance length 2·h_sub/π = 155.4µm for THIS board (on "
    'the issue-#823 fixture, h_sub=300µm, that length measured 0.1932mm against a '
    'predicted 0.19099mm — 1.1%). The decay LENGTH is a property of the substrate; the '
    'near-feed AMPLITUDE is not, so no error magnitude is predicted for your port here — '
    'read result diagnostics (the two-wave fit residual, and on the coax<->MSL lane the '
    'ladder-split witness) rather than trusting this offset. For reference, the #823 '
    "fixture's own measured amplitude (11.3 at the feed plane) put 5·h_sub at 4.4e-3 "
    'against the 0.02 two-wave residual bar this family holds itself to, and 1.50·h_sub at'
    ' 1.1e+00. Set n_probe_offset >= 10, or leave it None: the automatic offset counts '
    "5·h_sub = 1.221mm in this runway's 122.1µm cells, at least 10 cells. REPORT-ONLY: "
    'nothing is refused, and the rule is derived from ONE fixture at W/h = 2 — a much '
    'wider trace may need more (the first higher-order microstrip mode scales with W + '
    '2·h, which one fixture cannot separate from h).'
)

TEXT_5 = (
    "MSL port 'checked' (trace W=488µm, h_sub=275µm): lateral clearance to −y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 549µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_6 = (
    "MSL port 'checked' (trace W=488µm, h_sub=275µm): lateral clearance to +y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 549µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_7 = (
    "MSL port 'checked': only 2 normal interval(s) between the validated ground and trace "
    'planes. Validated conductor-plane gap=244.1µm over 2 normal interval(s), from z=0.0µm'
    ' to 244.1µm (declared height=274.7µm). Separately, the declared-material column has 3'
    ' same-permittivity sample slot(s), extent 366.2µm. This material extent is not the '
    'conductor-plane gap. The existing resolution recommendation is at least 4 normal '
    'intervals and an aligned declared substrate interface. On a uniform mesh, refine to '
    'dx ≤ 68.7µm and align the declared ground z=0.0µm and trace z=274.7µm with nodes; on '
    'a profiled mesh, place sufficient nodes between those faces. Geometry screening does '
    'not quantify Z0 error. The historical sweep and its realized-board Hammerstad-Jensen '
    'anchor are pre-#802 records; a new matched-geometry measurement is needed before '
    'quoting a current accuracy bound.'
)

TEXT_8 = (
    "MSL port 'checked': the declared trace plane at z=274.7µm sits 0.250 of a cell above "
    'its lower mesh node (that cell is 122.1µm) — this lands in the [0.10, 0.40] mixed-'
    'cell danger zone of the existing declared-face alignment heuristic. The fraction '
    'alone does not establish material/PEC overlap. Historical substrate-air/trace mixed-'
    'cell runs with AD-traceable ``pec_occupancy_override`` reported unphysical |S21|² > 1'
    ' (cited, not remeasured on this checkout: runs #563/#567, 2026-05-08, dx∈[75,82]µm '
    "h_sub=254µm). A hard ``Box(material='pec')`` avoids that specific bug ON THE DEFAULT "
    'RUN PATH (subpixel_smoothing=False and no conformal PEC face — the shipped defaults):'
    ' a PEC Box is a VOLUME that occupies whole primal cells with walls on both faces '
    '(lattice ownership contract #931), and a foil trace declared as a SHEET (zero-'
    'thickness Box / add_thin_conductor) is one node plane; neither enters '
    '``pec_occupancy_override``. That is NOT a blanket exemption — two opt-in lanes DO '
    "give a hard PEC box fractional cell occupancy: subpixel_smoothing='kottke_pec' (the "
    'inv-eps tensor is built over the PEC shapes) and the Stage-1 conformal PEC lane '
    '(which replaces the binary pec_mask with fractional weights); rfx/runners/uniform.py.'
    " Plain subpixel_smoothing=True does NOT: 'pec' carries eps_r=1.0 in the material "
    'library, so a PEC box enters the smoother as vacuum and stays whole-cell through '
    'pec_mask. The alignment advice below still applies on the two lanes that do. '
    'Validated conductor-plane gap=244.1µm over 2 normal interval(s), from z=0.0µm to '
    '244.1µm (declared height=274.7µm). Separately, the declared-material column has 3 '
    'same-permittivity sample slot(s), extent 366.2µm. This material extent is not the '
    'conductor-plane gap. The declared-face fraction and material extent describe '
    'geometry; neither predicts a Z0 error. The frozen sweep cannot establish a current '
    'extractor accuracy bound. To snap onto a mesh matching the DECLARED board instead, '
    'set dx = 91.6µm (= h_sub/3) or 137.3µm (= h_sub/2), aligning the declared ground '
    'z=0.0µm and trace z=274.7µm.'
)

TEXT_9 = (
    "MSL port 'checked' at x=0.49mm, direction='+x': distance to nearest x-CPML = 244µm "
    '(domain edge + 244.1µm calibrated CPML buffer) < recommended 549µm (= 2·h_sub). '
    'Source-side CPML reflection may inflate |S11|. Move port further from boundary OR '
    'increase domain x-extent.'
)

TEXT_10 = (
    "MSL port 'checked' (direction='+x'): n_probe_offset=3 puts probe 0 366.2µm "
    "(1.33·h_sub) from this port's OWN feed plane, inside the source near-field standoff "
    "of 11 cells (1.343mm = 5·h_sub, the issue-#80 Fix B constant add_msl_port's auto "
    'offset already floors to). Within a few substrate thicknesses of the feed the '
    'launched field is not the guided mode yet: the evanescent content decays with the '
    "substrate's own transverse-resonance length 2·h_sub/π = 174.9µm for THIS board (on "
    'the issue-#823 fixture, h_sub=300µm, that length measured 0.1932mm against a '
    'predicted 0.19099mm — 1.1%). The decay LENGTH is a property of the substrate; the '
    'near-feed AMPLITUDE is not, so no error magnitude is predicted for your port here — '
    'read result diagnostics (the two-wave fit residual, and on the coax<->MSL lane the '
    'ladder-split witness) rather than trusting this offset. For reference, the #823 '
    "fixture's own measured amplitude (11.3 at the feed plane) put 5·h_sub at 4.4e-3 "
    'against the 0.02 two-wave residual bar this family holds itself to, and 1.33·h_sub at'
    ' 1.4e+00. Set n_probe_offset >= 11, or leave it None: the automatic offset counts '
    "5·h_sub = 1.373mm in this runway's 122.1µm cells, at least 11 cells. REPORT-ONLY: "
    'nothing is refused, and the rule is derived from ONE fixture at W/h = 2 — a much '
    'wider trace may need more (the first higher-order microstrip mode scales with W + '
    '2·h, which one fixture cannot separate from h).'
)

TEXT_11 = (
    "MSL port 'checked' (trace W=488µm, h_sub=543µm): lateral clearance to −y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 1086µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~12%, mesh-'
    'conv may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_12 = (
    "MSL port 'checked' (trace W=488µm, h_sub=543µm): lateral clearance to +y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 1086µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~12%, mesh-'
    'conv may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_13 = (
    "MSL port 'checked': conductor-plane separation differs from the declared height by "
    '-10.1%, beyond the existing 8.3% geometry-advisory threshold. Validated conductor-'
    'plane gap=488.3µm over 4 normal interval(s), from z=0.0µm to 488.3µm (declared '
    'height=543.2µm). Separately, the declared-material column has 5 same-permittivity '
    'sample slot(s), extent 610.4µm. This material extent is not the conductor-plane gap. '
    'Place mesh nodes at the declared ground z=0.0µm and trace z=543.2µm or refine the '
    'normal mesh. This geometry difference does not predict a Z0 change or establish an '
    'extractor accuracy bound.'
)

TEXT_14 = (
    "MSL port 'checked' at x=0.49mm, direction='+x': distance to nearest x-CPML = 244µm "
    '(domain edge + 244.1µm calibrated CPML buffer) < recommended 1086µm (= 2·h_sub). '
    'Source-side CPML reflection may inflate |S11|. Move port further from boundary OR '
    'increase domain x-extent.'
)

TEXT_15 = (
    "MSL port 'checked' (direction='+x'): n_probe_offset=3 puts probe 0 366.2µm "
    "(0.67·h_sub) from this port's OWN feed plane, inside the source near-field standoff "
    "of 22 cells (2.686mm = 5·h_sub, the issue-#80 Fix B constant add_msl_port's auto "
    'offset already floors to). Within a few substrate thicknesses of the feed the '
    'launched field is not the guided mode yet: the evanescent content decays with the '
    "substrate's own transverse-resonance length 2·h_sub/π = 345.8µm for THIS board (on "
    'the issue-#823 fixture, h_sub=300µm, that length measured 0.1932mm against a '
    'predicted 0.19099mm — 1.1%). The decay LENGTH is a property of the substrate; the '
    'near-feed AMPLITUDE is not, so no error magnitude is predicted for your port here — '
    'read result diagnostics (the two-wave fit residual, and on the coax<->MSL lane the '
    'ladder-split witness) rather than trusting this offset. For reference, the #823 '
    "fixture's own measured amplitude (11.3 at the feed plane) put 5·h_sub at 4.4e-3 "
    'against the 0.02 two-wave residual bar this family holds itself to, and 0.67·h_sub at'
    ' 3.9e+00. Set n_probe_offset >= 22, or leave it None: the automatic offset counts '
    "5·h_sub = 2.716mm in this runway's 122.1µm cells, at least 22 cells. REPORT-ONLY: "
    'nothing is refused, and the rule is derived from ONE fixture at W/h = 2 — a much '
    'wider trace may need more (the first higher-order microstrip mode scales with W + '
    '2·h, which one fixture cannot separate from h).'
)

TEXT_16 = (
    "MSL port 'checked' (direction='+x'): the 3-probe ladder (n_probe_offset=40, "
    'n_probe_spacing=2 cells) runs past the grid and CLAMPS — only 1 of 3 probes land on '
    'distinct grid cells (2 duplicate probe position(s): (4.15, 4.15, 4.15)mm). The '
    'N-probe least-squares wave-decomposition fit is rank-deficient on duplicated '
    "positions; `compute_msl_s_matrix`'s Z0/S11 extraction is unreliable for this port. "
    'Shorten n_probe_offset/n_probe_spacing or extend the domain so the full ladder stays '
    'in-grid.'
)

TEXT_17 = (
    "MSL port 'checked' (direction='+x'): probe 2 (deepest, x=4.15mm) is past the domain "
    'edge (domain x-extent [0, 3.91]mm) — inside the CPML absorbing region. The N-probe '
    "extractor's clean-travelling-wave assumption is void there: signal is attenuated and "
    'the fitted Z0/S11 are corrupted. compliant n_probe_offset interval ≈ [10, 22] cells.'
)

TEXT_18 = (
    "MSL port 'checked': reflector clearance could not be evaluated: realized grid or "
    'positive design frequency is unavailable.'
)

TEXT_19 = (
    "MSL port 'checked' (direction='+x'): probe 2 (deepest, x=5.86mm) is past the domain "
    'edge (domain x-extent [0, 3.91]mm) — inside the CPML absorbing region. The N-probe '
    "extractor's clean-travelling-wave assumption is void there: signal is attenuated and "
    'the fitted Z0/S11 are corrupted. compliant n_probe_offset interval ≈ [10, 22] cells.'
)

TEXT_20 = (
    "MSL port 'checked' (direction='+x'): probe 2 (deepest, x=3.78mm) is within 2 cells "
    '(244.1µm) of the domain edge, just past which the CPML absorber is active. Fields '
    'there carry CPML fringe/reflection error, biasing the fitted Z0/S11. compliant '
    'n_probe_offset interval ≈ [10, 22] cells.'
)

TEXT_21 = (
    "MSL port 'checked' (direction='+x'): deepest probe at x=1.34mm sits -122µm from a "
    "strong reflector candidate (conductor 'pec' at x∈[1.22,1.34]mm y∈[0.49,0.98]mm; "
    'distance estimated from registered conductor bounds); recommended ≥ 1676µm (= λ_g/4 '
    'at f_max with ε_eff_proxy=5.0). Measured on ONE fixture (VESSL 369367260508, '
    "microstrip open-stub notch fixed-source, same source, load and DUT; only p1's "
    'observation offset varied; both arms settled below -118 dB): standing-wave content at'
    " the probes corrupts the FITTED Z0/beta - the near-reflector arm's beta scan railed "
    'on 51/51 bins over 3-5 GHz against 0/51 for the compliant arm - while raw S11 at the '
    '3.77125 GHz notch bin read +0.026895 dB on the near-reflector arm and +0.018065 dB on'
    ' the compliant one, against the analytic 0 dB that quarter-wave open-stub notch has. '
    'Both are ABOVE unity on a passive structure, by 0.31 % and 0.21 %; that is never '
    'reported here as physics - they are raw, unprojected values carrying the coherent '
    'power excess recorded in #838 (closed as not planned). Their 0.009 dB difference is '
    'one fixture, one bin, an arm-to-arm difference, NOT a bound on S, and that '
    "comparison's producer verdict was not_read. What it does show is that S11/S21 move "
    'far less than the fit does, because they normalize with the analytic Hammerstad-'
    "Jensen Z0 rather than with the fit. Neither 'S11/S21 are unaffected' nor the retired "
    "'-5 to -10 dB' figure is right. This layout warning does not certify accuracy when "
    'absent. Available layout: no compliant n_probe_offset exists on this feed length '
    '(interval empty). Choose an offset within a nonempty interval; if it is empty, extend'
    ' the uniform feed region to fit the source standoff, full probe ladder and reflector '
    'clearance. Gate on probe_clearance for the geometric condition and beta_railed for '
    'the fitted-value symptom; reliable is a per-bin fit-quality mask and gates neither. '
    'When no compliant offset exists on the available feed length, keep the analytic '
    'Hammerstad-Jensen Z0 for normalization (already the production path) and treat the '
    'fitted Z0/beta as UNREADABLE rather than merely uncertain. S11/S21 are the better-'
    'behaved of the two, but nothing here BOUNDS their error. The fix is to lengthen the '
    'uniform feed region or move the reference plane; raising n_probe_offset alone moves '
    'the probes toward the reflector. Check settling and observation-plane sensitivity '
    'before interpreting S.'
)

TEXT_22 = (
    "MSL port 'checked': the upstream and downstream probe clearances are mutually "
    'unsatisfiable on this feed (upstream needs n_probe_offset >= 10 cells = max(λ/4π, '
    '5·h_sub)/dx; downstream needs <= -11 cells to keep the deepest of 3 probes ≥ 1676µm '
    '(λ_g/4 at f_max) clear of the reflector 0.73mm from the feed). The feed line is too '
    'short for a clean N-probe measurement (issue #469) — keeping the upstream-priority '
    "offset 10, which puts the deep probes inside the reflector's near field. Measured on "
    'ONE fixture (VESSL 369367260508, microstrip open-stub notch fixed-source, same '
    "source, load and DUT; only p1's observation offset varied; both arms settled below "
    '-118 dB): standing-wave content at the probes corrupts the FITTED Z0/beta - the near-'
    "reflector arm's beta scan railed on 51/51 bins over 3-5 GHz against 0/51 for the "
    'compliant arm - while raw S11 at the 3.77125 GHz notch bin read +0.026895 dB on the '
    'near-reflector arm and +0.018065 dB on the compliant one, against the analytic 0 dB '
    'that quarter-wave open-stub notch has. Both are ABOVE unity on a passive structure, '
    'by 0.31 % and 0.21 %; that is never reported here as physics - they are raw, '
    'unprojected values carrying the coherent power excess recorded in #838 (closed as not'
    ' planned). Their 0.009 dB difference is one fixture, one bin, an arm-to-arm '
    "difference, NOT a bound on S, and that comparison's producer verdict was not_read. "
    'What it does show is that S11/S21 move far less than the fit does, because they '
    'normalize with the analytic Hammerstad-Jensen Z0 rather than with the fit. Neither '
    "'S11/S21 are unaffected' nor the retired '-5 to -10 dB' figure is right. Gate on "
    'probe_clearance for the geometric condition and beta_railed for the fitted-value '
    'symptom; reliable is a per-bin fit-quality mask and gates neither. When no compliant '
    'offset exists on the available feed length, keep the analytic Hammerstad-Jensen Z0 '
    'for normalization (already the production path) and treat the fitted Z0/beta as '
    'UNREADABLE rather than merely uncertain. S11/S21 are the better-behaved of the two, '
    'but nothing here BOUNDS their error. The fix is to lengthen the uniform feed region '
    'or move the reference plane; raising n_probe_offset alone moves the probes toward the'
    ' reflector.'
)

TEXT_23 = (
    "MSL port 'checked' (direction='+x'): deepest probe at x=2.20mm sits -977µm from a "
    "strong reflector candidate (conductor 'pec' at x∈[1.22,1.34]mm y∈[0.49,0.98]mm; "
    'distance estimated from registered conductor bounds); recommended ≥ 1676µm (= λ_g/4 '
    'at f_max with ε_eff_proxy=5.0). Measured on ONE fixture (VESSL 369367260508, '
    "microstrip open-stub notch fixed-source, same source, load and DUT; only p1's "
    'observation offset varied; both arms settled below -118 dB): standing-wave content at'
    " the probes corrupts the FITTED Z0/beta - the near-reflector arm's beta scan railed "
    'on 51/51 bins over 3-5 GHz against 0/51 for the compliant arm - while raw S11 at the '
    '3.77125 GHz notch bin read +0.026895 dB on the near-reflector arm and +0.018065 dB on'
    ' the compliant one, against the analytic 0 dB that quarter-wave open-stub notch has. '
    'Both are ABOVE unity on a passive structure, by 0.31 % and 0.21 %; that is never '
    'reported here as physics - they are raw, unprojected values carrying the coherent '
    'power excess recorded in #838 (closed as not planned). Their 0.009 dB difference is '
    'one fixture, one bin, an arm-to-arm difference, NOT a bound on S, and that '
    "comparison's producer verdict was not_read. What it does show is that S11/S21 move "
    'far less than the fit does, because they normalize with the analytic Hammerstad-'
    "Jensen Z0 rather than with the fit. Neither 'S11/S21 are unaffected' nor the retired "
    "'-5 to -10 dB' figure is right. This layout warning does not certify accuracy when "
    'absent. Available layout: no compliant n_probe_offset exists on this feed length '
    '(interval empty). Choose an offset within a nonempty interval; if it is empty, extend'
    ' the uniform feed region to fit the source standoff, full probe ladder and reflector '
    'clearance. Gate on probe_clearance for the geometric condition and beta_railed for '
    'the fitted-value symptom; reliable is a per-bin fit-quality mask and gates neither. '
    'When no compliant offset exists on the available feed length, keep the analytic '
    'Hammerstad-Jensen Z0 for normalization (already the production path) and treat the '
    'fitted Z0/beta as UNREADABLE rather than merely uncertain. S11/S21 are the better-'
    'behaved of the two, but nothing here BOUNDS their error. The fix is to lengthen the '
    'uniform feed region or move the reference plane; raising n_probe_offset alone moves '
    'the probes toward the reflector. Check settling and observation-plane sensitivity '
    'before interpreting S.'
)

TEXT_24 = (
    "MSL port 'checked' (direction='+x'): probe span x∈[0.49, 2.20]mm crosses the feed "
    "plane of MSL port 'opposite' at x=1.46mm. A feed is a source discontinuity the "
    'reflector scan above cannot see; probes sampling across it break the N-probe '
    "extractor's uniform-line assumption. If this crossing is intentional, verify the "
    'extracted Z0/S11 independently.'
)

TEXT_25 = (
    "MSL port 'opposite' (trace W=488µm, h_sub=244µm): lateral clearance to −y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 488µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_26 = (
    "MSL port 'opposite' (trace W=488µm, h_sub=244µm): lateral clearance to +y absorbing "
    'boundary = 244µm (domain edge + 244.1µm calibrated CPML buffer) < recommended 488µm '
    '(= 2·h_sub). Fringing field will be clipped → Z0 may be biased HIGH by ~8%, mesh-conv'
    ' may diverge. Increase domain y-extent OR move port further from sidewall.'
)

TEXT_27 = (
    "MSL port 'opposite': only 2 normal interval(s) between the validated ground and trace"
    ' planes. Validated conductor-plane gap=244.1µm over 2 normal interval(s), from '
    'z=0.0µm to 244.1µm (declared height=244.1µm). Separately, the declared-material '
    'column has 2 same-permittivity sample slot(s), extent 244.1µm. This material extent '
    'is not the conductor-plane gap. The existing resolution recommendation is at least 4 '
    'normal intervals and an aligned declared substrate interface. On a uniform mesh, '
    'refine to dx ≤ 61.0µm and align the declared ground z=0.0µm and trace z=244.1µm with '
    'nodes; on a profiled mesh, place sufficient nodes between those faces. Geometry '
    'screening does not quantify Z0 error. The historical sweep and its realized-board '
    'Hammerstad-Jensen anchor are pre-#802 records; a new matched-geometry measurement is '
    'needed before quoting a current accuracy bound.'
)

TEXT_28 = (
    "MSL port 'opposite' (direction='-x'): probe 2 (deepest, x=-0.24mm) is past the domain"
    ' edge (domain x-extent [0, 3.91]mm) — inside the CPML absorbing region. The N-probe '
    "extractor's clean-travelling-wave assumption is void there: signal is attenuated and "
    'the fitted Z0/S11 are corrupted. no compliant n_probe_offset exists on this feed '
    'length (interval empty).'
)

TEXT_29 = (
    "MSL port 'opposite' (direction='-x'): probe span x∈[-0.24, 1.46]mm crosses the feed "
    "plane of MSL port 'checked' at x=0.49mm. A feed is a source discontinuity the "
    'reflector scan above cannot see; probes sampling across it break the N-probe '
    "extractor's uniform-line assumption. If this crossing is intentional, verify the "
    'extracted Z0/S11 independently.'
)

TEXT_30 = (
    "MSL port 'checked': conductor attachment could not be validated because the run "
    'geometry could not be assembled; the conductor-plane gap is unavailable.'
)

TEXT_31 = (
    'attachment unavailable The conductor-plane gap is unavailable.'
)

TEXT_32 = (
    'MSL probe placement could not be resolved: placement unavailable'
)

TEXT_33 = (
    'placement detail'
)

TEXT_34 = (
    "MSL port 'checked': reflector clearance could not be evaluated: no probe metadata."
)

TEXT_35 = (
    "MSL port 'checked' (direction='+x'): the downstream-reflector clearance scan could "
    "NOT evaluate 1 registered conductor(s), so a 'clear' result here is not evidence that"
    ' the probes are clear — unsupported conductor. Give those shapes an axis-aligned '
    'bounding box (or place the probes explicitly with n_probe_offset) before trusting '
    "`compute_msl_s_matrix`'s Z₀ / |S11| here."
)

TEXT_36 = (
    "MSL port 'checked': probe clearance is UNAVAILABLE (no probe metadata); a clean read "
    'here is not evidence that the probes are clear of a downstream reflector. Gate on '
    'probe_clearance for the geometric condition and beta_railed for the fitted-value '
    'symptom; reliable is a per-bin fit-quality mask and gates neither. When no compliant '
    'offset exists on the available feed length, keep the analytic Hammerstad-Jensen Z0 '
    'for normalization (already the production path) and treat the fitted Z0/beta as '
    'UNREADABLE rather than merely uncertain. S11/S21 are the better-behaved of the two, '
    'but nothing here BOUNDS their error. The fix is to lengthen the uniform feed region '
    'or move the reference plane; raising n_probe_offset alone moves the probes toward the'
    ' reflector.'
)

TEXT_37 = (
    "MSL port 'checked': the probe-clearance scan could not run on this route (ValueError:"
    ' scan unavailable), so a clean read here is not evidence that the probes are clear.'
)

TEXT_38 = (
    "MSL port 'checked': probe clearance is INSUFFICIENT before compute_msl_s_matrix runs "
    "— the deepest probe sits -122um from conductor 'pec' at x∈[1.22,1.34]mm "
    'y∈[0.49,0.98]mm, against the 1676um layout recommendation. Measured on ONE fixture '
    '(VESSL 369367260508, microstrip open-stub notch fixed-source, same source, load and '
    "DUT; only p1's observation offset varied; both arms settled below -118 dB): standing-"
    "wave content at the probes corrupts the FITTED Z0/beta - the near-reflector arm's "
    'beta scan railed on 51/51 bins over 3-5 GHz against 0/51 for the compliant arm - '
    'while raw S11 at the 3.77125 GHz notch bin read +0.026895 dB on the near-reflector '
    'arm and +0.018065 dB on the compliant one, against the analytic 0 dB that quarter-'
    'wave open-stub notch has. Both are ABOVE unity on a passive structure, by 0.31 % and '
    '0.21 %; that is never reported here as physics - they are raw, unprojected values '
    'carrying the coherent power excess recorded in #838 (closed as not planned). Their '
    '0.009 dB difference is one fixture, one bin, an arm-to-arm difference, NOT a bound on'
    " S, and that comparison's producer verdict was not_read. What it does show is that "
    'S11/S21 move far less than the fit does, because they normalize with the analytic '
    "Hammerstad-Jensen Z0 rather than with the fit. Neither 'S11/S21 are unaffected' nor "
    "the retired '-5 to -10 dB' figure is right. Gate on probe_clearance for the geometric"
    ' condition and beta_railed for the fitted-value symptom; reliable is a per-bin fit-'
    'quality mask and gates neither. When no compliant offset exists on the available feed'
    ' length, keep the analytic Hammerstad-Jensen Z0 for normalization (already the '
    'production path) and treat the fitted Z0/beta as UNREADABLE rather than merely '
    'uncertain. S11/S21 are the better-behaved of the two, but nothing here BOUNDS their '
    'error. The fix is to lengthen the uniform feed region or move the reference plane; '
    'raising n_probe_offset alone moves the probes toward the reflector. preflight() '
    'reports the full layout interval for this port.'
)

TEXT_39 = (
    'conductor realization failed'
)

TEXT_40 = (
    'Line-stub check could not inspect 1 conductor shape(s) (unsupported conductor). They '
    'were skipped; the other conductors were checked. This is not evidence that the '
    'skipped shapes leave no tail behind a port.'
)

TEXT_41 = (
    'The strip continues 1.95312 mm behind the port and ends there (realized L; declared '
    '1.95312 mm); with the open-end extension 0.0989573 mm its effective length is 2.05208'
    ' mm. It is an open stub that shorts the port near 21.7003 GHz (quarter wave); stub '
    'frequencies 21.7003 GHz (order 1), inside/near the band you read. Read band 0..20 '
    "GHz; eps_eff=2.83269; port 'checked'. The port's realized grid node is x=1.953125 mm."
    " Fix: start the signal strip at that coordinate (the port's grid node), so it covers "
    'the port node and nothing behind it (#1512).'
)

TEXT_42 = (
    "MSL port 'checked': the declared trace plane at z=274.7µm sits 0.250 of a cell above "
    'its lower mesh node (that cell is 122.1µm) — this lands in the [0.10, 0.40] mixed-'
    'cell danger zone of the existing declared-face alignment heuristic. The fraction '
    'alone does not establish material/PEC overlap. Historical substrate-air/trace mixed-'
    'cell runs with AD-traceable ``pec_occupancy_override`` reported unphysical |S21|² > 1'
    ' (cited, not remeasured on this checkout: runs #563/#567, 2026-05-08, dx∈[75,82]µm '
    "h_sub=254µm). A hard ``Box(material='pec')`` avoids that specific bug ON THE DEFAULT "
    'RUN PATH (subpixel_smoothing=False and no conformal PEC face — the shipped defaults):'
    ' a PEC Box is a VOLUME that occupies whole primal cells with walls on both faces '
    '(lattice ownership contract #931), and a foil trace declared as a SHEET (zero-'
    'thickness Box / add_thin_conductor) is one node plane; neither enters '
    '``pec_occupancy_override``. That is NOT a blanket exemption — two opt-in lanes DO '
    "give a hard PEC box fractional cell occupancy: subpixel_smoothing='kottke_pec' (the "
    'inv-eps tensor is built over the PEC shapes) and the Stage-1 conformal PEC lane '
    '(which replaces the binary pec_mask with fractional weights); rfx/runners/uniform.py.'
    " Plain subpixel_smoothing=True does NOT: 'pec' carries eps_r=1.0 in the material "
    'library, so a PEC box enters the smoother as vacuum and stays whole-cell through '
    'pec_mask. The alignment advice below still applies on the two lanes that do. '
    'Validated conductor-plane gap=244.1µm over 2 normal interval(s), from z=0.0µm to '
    '244.1µm (declared height=274.7µm). Separately, the declared-material column has 3 '
    'same-permittivity sample slot(s), extent 366.2µm. This material extent is not the '
    'conductor-plane gap. The declared-face fraction and material extent describe '
    'geometry; neither predicts a Z0 error. The frozen sweep cannot establish a current '
    'extractor accuracy bound. On the non-uniform profile, place mesh nodes at the '
    'declared ground z=0.0µm and trace z=274.7µm.'
)

TEXT_43 = (
    "MSL port 'checked' (direction='+x'): the 3-probe ladder (n_probe_offset=40, "
    'n_probe_spacing=2 cells) runs past the grid and CLAMPS — only 1 of 3 probes land on '
    'distinct grid cells (2 duplicate probe position(s): (3.91, 3.91, 3.91)mm). The '
    'N-probe least-squares wave-decomposition fit is rank-deficient on duplicated '
    "positions; `compute_msl_s_matrix`'s Z0/S11 extraction is unreliable for this port. "
    'Shorten n_probe_offset/n_probe_spacing or extend the domain so the full ladder stays '
    'in-grid.'
)

TEXT_44 = (
    "MSL port 'checked' (direction='+x'): probe 2 (deepest, x=3.91mm) is within 2 cells "
    '(244.1µm) of the domain edge, just past which the CPML absorber is active. Fields '
    'there carry CPML fringe/reflection error, biasing the fitted Z0/S11. compliant '
    'n_probe_offset interval ≈ [10, 22] cells.'
)

TEXT_45 = (
    "MSL port 'opposite' (direction='-x'): the 3-probe ladder (n_probe_offset=10, "
    'n_probe_spacing=2 cells) runs past the grid and CLAMPS — only 2 of 3 probes land on '
    'distinct grid cells (1 duplicate probe position(s): (0.24, 0.0, 0.0)mm). The N-probe '
    'least-squares wave-decomposition fit is rank-deficient on duplicated positions; '
    "`compute_msl_s_matrix`'s Z0/S11 extraction is unreliable for this port. Shorten "
    'n_probe_offset/n_probe_spacing or extend the domain so the full ladder stays in-grid.'
)

TEXT_46 = (
    "MSL port 'opposite' (direction='-x'): probe 2 (deepest, x=0.00mm) is within 2 cells "
    '(244.1µm) of the domain edge, just past which the CPML absorber is active. Fields '
    'there carry CPML fringe/reflection error, biasing the fitted Z0/S11. no compliant '
    'n_probe_offset exists on this feed length (interval empty).'
)

TEXT_47 = (
    "MSL port 'opposite' (direction='-x'): probe span x∈[0.00, 1.46]mm crosses the feed "
    "plane of MSL port 'checked' at x=0.49mm. A feed is a source discontinuity the "
    'reflector scan above cannot see; probes sampling across it break the N-probe '
    "extractor's uniform-line assumption. If this crossing is intentional, verify the "
    'extracted Z0/S11 independently.'
)

GOLDEN = {
    'uniform/base': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'uniform/alignment': [
        TEXT_5,
        TEXT_6,
        TEXT_7,
        TEXT_8,
        TEXT_9,
        TEXT_10,
    ],
    'uniform/gap_difference': [
        TEXT_11,
        TEXT_12,
        TEXT_13,
        TEXT_14,
        TEXT_15,
    ],
    'uniform/clamped': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_16,
        TEXT_17,
    ],
    'uniform/unavailable_grid_extrapolation': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_18,
        TEXT_19,
    ],
    'uniform/near_absorber': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_20,
    ],
    'uniform/reflector': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_21,
        TEXT_4,
    ],
    'uniform/automatic_placement': [
        TEXT_22,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_23,
    ],
    'uniform/cross_feed': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_24,
        TEXT_25,
        TEXT_26,
        TEXT_27,
        TEXT_28,
        TEXT_29,
    ],
    'uniform/assembly_unavailable': [
        TEXT_30,
        TEXT_0,
        TEXT_1,
        TEXT_3,
        TEXT_4,
    ],
    'uniform/attachment_unavailable': [
        TEXT_31,
        TEXT_0,
        TEXT_1,
        TEXT_3,
        TEXT_4,
    ],
    'uniform/placement_unavailable': [
        TEXT_32,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'uniform/placement_note': [
        TEXT_33,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'uniform/incomplete_scan': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_34,
        TEXT_35,
        TEXT_4,
    ],
    'uniform/calculator_unavailable': [
        TEXT_36,
    ],
    'uniform/calculator_scan_failed': [
        TEXT_37,
    ],
    'uniform/calculator_clearance': [
        TEXT_38,
    ],
    'uniform/stub_realization': [
        TEXT_39,
    ],
    'uniform/stub_inspection': [
        TEXT_40,
    ],
    'uniform/stub_behind_port': [
        TEXT_41,
    ],
    'graded/base': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'graded/alignment': [
        TEXT_5,
        TEXT_6,
        TEXT_7,
        TEXT_42,
        TEXT_9,
        TEXT_10,
    ],
    'graded/gap_difference': [
        TEXT_11,
        TEXT_12,
        TEXT_13,
        TEXT_14,
        TEXT_15,
    ],
    'graded/clamped': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_43,
        TEXT_44,
    ],
    'graded/unavailable_grid_extrapolation': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_18,
        TEXT_19,
    ],
    'graded/near_absorber': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_20,
    ],
    'graded/reflector': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_21,
        TEXT_4,
    ],
    'graded/automatic_placement': [
        TEXT_22,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_23,
    ],
    'graded/cross_feed': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_24,
        TEXT_25,
        TEXT_26,
        TEXT_27,
        TEXT_45,
        TEXT_46,
        TEXT_47,
    ],
    'graded/assembly_unavailable': [
        TEXT_30,
        TEXT_0,
        TEXT_1,
        TEXT_3,
        TEXT_4,
    ],
    'graded/attachment_unavailable': [
        TEXT_31,
        TEXT_0,
        TEXT_1,
        TEXT_3,
        TEXT_4,
    ],
    'graded/placement_unavailable': [
        TEXT_32,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'graded/placement_note': [
        TEXT_33,
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_4,
    ],
    'graded/incomplete_scan': [
        TEXT_0,
        TEXT_1,
        TEXT_2,
        TEXT_3,
        TEXT_34,
        TEXT_35,
        TEXT_4,
    ],
    'graded/calculator_unavailable': [
        TEXT_36,
    ],
    'graded/calculator_scan_failed': [
        TEXT_37,
    ],
    'graded/calculator_clearance': [
        TEXT_38,
    ],
    'graded/stub_realization': [
        TEXT_39,
    ],
    'graded/stub_inspection': [
        TEXT_40,
    ],
    'graded/stub_behind_port': [
        TEXT_41,
    ],
}
