"""The MSL diagnostic catalog and its byte-preserving message variants.

Lengths enter in metres, frequencies in Hz, fractions as ratios. Display-unit
conversion belongs here. A Message carries observations through composed
fragments; it is not accepted from arbitrary caller-authored message text.

Value names in the table name the input fields. Numeric tuples/lists flatten to
<name>_0, <name>_1, ...; indexed port summaries prefix these with port_<index>_.
Severity in the table is the default: a warning record is advisory, while the
same finding becomes refusal only at a blocking error/raise site.
"""

from dataclasses import dataclass
from string import Formatter

from rfx.diagnostic_records import Diagnostic
from rfx.preflight._msl_text import Message, render, join_messages as msl_join
from rfx.preflight._msl_text import DisplayedValue as msl_value

#: Witness for :data:`MSL_PROBE_CLEARANCE_EFFECT`; kept separate so a test
#: can assert the run id survives every rewording of the sentence.
MSL_PROBE_CLEARANCE_WITNESS = (
    "VESSL 369367260508, microstrip open-stub notch fixed-source"
)

#: The ONE sentence allowed to quote either retired claim, because it is what
#: retracts them. Named so every test that asserts "this site does not state a
#: retired claim" can strip exactly this and nothing else -- four of them do,
#: including the docs scan.
MSL_PROBE_CLEARANCE_RETRACTION = (
    "Neither 'S11/S21 are unaffected' nor the retired '-5 to -10 dB' figure is right."
)

#: What probe-clearance corruption does, with its numbers and their limits.
#: Embedded by the preflight layout warning, ``compute_msl_s_matrix``'s Z0
#: guard (only when the port's clearance is NOT satisfied) and the
#: auto-offset resolver's empty-interval warning, so the three cannot drift
#: apart again (#726).
MSL_PROBE_CLEARANCE_EFFECT = (
    "Measured on ONE fixture (" + MSL_PROBE_CLEARANCE_WITNESS + ", same "
    "source, load and DUT; only p1's observation offset varied; both arms "
    "settled below -118 dB): standing-wave content at the probes corrupts "
    "the FITTED Z0/beta - the near-reflector arm's beta scan railed on 51/51 "
    "bins over 3-5 GHz against 0/51 for the compliant arm - while raw S11 at "
    "the 3.77125 GHz notch bin read +0.026895 dB on the near-reflector arm "
    "and +0.018065 dB on the compliant one, against the analytic 0 dB that "
    "quarter-wave open-stub notch has. Both are ABOVE unity on a passive "
    "structure, by 0.31 % and 0.21 %; that is never reported here as "
    "physics - they are raw, unprojected values carrying the coherent power "
    "excess recorded in #838 (closed as not planned). Their 0.009 dB difference is one fixture, one "
    "bin, an arm-to-arm difference, NOT a bound on S, and that comparison's "
    "producer verdict was not_read. What it does show is that S11/S21 move "
    "far less than the fit does, because they normalize with the analytic "
    "Hammerstad-Jensen Z0 rather than with the fit. " + MSL_PROBE_CLEARANCE_RETRACTION
)

#: What a caller should do, including when the feed admits no compliant
#: offset at all ("interval empty") - #726 item 3.
MSL_PROBE_CLEARANCE_GUIDANCE = (
    "Gate on probe_clearance for the geometric condition and beta_railed "
    "for the fitted-value symptom; reliable is a per-bin fit-quality mask "
    "and gates neither. When no compliant offset exists on the available "
    "feed length, keep the analytic Hammerstad-Jensen Z0 for normalization "
    "(already the production path) and treat the fitted Z0/beta as "
    "UNREADABLE rather than merely uncertain. S11/S21 are the "
    "better-behaved of the two, but nothing here BOUNDS their error. The fix "
    "is to lengthen the uniform feed region or move the reference plane; "
    "raising n_probe_offset alone moves the probes toward the reflector."
)


TEMPLATES = {
    "probe_clearance_scan_failed": "MSL port {port_name!r}: the probe-clearance scan could not "
    "run on this route ({type_exc___name}: {exc}), so a clean read "
    "here is not evidence that the probes are clear.",
    "probe_clearance_unavailable": "MSL port {port_name!r}: probe clearance is UNAVAILABLE "
    "({record_note}); a clean read here is not evidence that the "
    "probes are clear of a downstream reflector. "
    "{msl_probe_clearance_guidance}",
    "clearance_gap": "{reflector_gap_m:um|.0f}um",
    "reflector_clearance_calculator": "MSL port {port_name!r}: probe clearance is INSUFFICIENT "
    "before compute_msl_s_matrix runs — the deepest probe sits "
    "{gap_txt} from {record_reflector}, against the "
    "{recommended_reflector_gap_m:um|.0f}um layout "
    "recommendation. {msl_probe_clearance_effect} "
    "{msl_probe_clearance_guidance} preflight() reports the "
    "full layout interval for this port.",
    "probe_placement_failed": "MSL probe placement could not be resolved: {exc}",
    "conductor_assembly_unavailable": "MSL port {port_name!r}: conductor attachment could not be "
    "validated because the run geometry could not be assembled; "
    "the conductor-plane gap is unavailable.",
    "conductor_attachment": "{exc} The conductor-plane gap is unavailable.",
    "standoff_ramp": "the source-fringing standoff crosses cells of more than one size on the "
    "{propagation_axis} runway, so a probe-offset in CELLS does not name one "
    "distance here",
    "declared_faces": "declared ground {normal_axis}={declared_ground_m:um|.1f}µm and trace "
    "{normal_axis}={declared_trace_m:um|.1f}µm",
    "width_low_side": "−{width_axis}",
    "width_high_side": "+{width_axis}",
    "lateral_clearance": "MSL port '{port_name}' (trace W={trace_width_m:um|.0f}µm, "
    "h_sub={substrate_height_m:um|.0f}µm): lateral clearance to {side} "
    "absorbing boundary = {lateral_clearance_m:um|.0f}µm (domain edge + "
    "{cpml_buffer_m:len|} calibrated CPML buffer) < recommended "
    "{recommended_clearance_m:um|.0f}µm (= 2·h_sub). Fringing field will be "
    "clipped → Z0 may be biased HIGH by ~{bias_fraction:percent|.0f}%, "
    "mesh-conv may diverge. Increase domain {width_axis}-extent OR move port "
    "further from sidewall.",
    "material_column": " Separately, the declared-material column has {material_slots} "
    "same-permittivity sample slot(s), extent {material_extent_m:um|.1f}µm. "
    "This material extent is not the conductor-plane gap.",
    "conductor_gap": "Validated conductor-plane gap={conductor_gap_m:um|.1f}µm over "
    "{normal_intervals} normal interval(s), from "
    "{normal_axis}={ground_m:um|.1f}µm to {trace_m:um|.1f}µm (declared "
    "height={substrate_height_m:um|.1f}µm).",
    "normal_resolution": "MSL port '{port_name}': only {normal_intervals} normal interval(s) "
    "between the validated ground and trace planes. {gap_txt}{material_txt} "
    "The existing resolution recommendation is at least 4 normal intervals "
    "and an aligned declared substrate interface. On a uniform mesh, refine "
    "to dx ≤ {recommended_cell_m:um|.1f}µm and align the {absolute_faces} "
    "with nodes; on a profiled mesh, place sufficient nodes between those "
    "faces. Geometry screening does not quantify Z0 error. The historical "
    "sweep and its realized-board Hammerstad-Jensen anchor are pre-#802 "
    "records; a new matched-geometry measurement is needed before quoting a "
    "current accuracy bound.",
    "declared_interface": "the declared trace plane at {normal_axis}={declared_trace_m:um|.1f}µm "
    "sits {interface_fraction:.3f} of a cell above its lower mesh node "
    "(that cell is {interface_cell_m:um|.1f}µm)",
    "alignment_profile": "On the non-uniform profile, place mesh nodes at the {absolute_faces}.",
    "alignment_translated": "For this translated board, place mesh nodes at the {absolute_faces}. "
    "A height/n spacing alone does not ensure that both absolute faces "
    "are on nodes.",
    "alignment_uniform_two_spacings": "To snap onto a mesh matching the DECLARED board instead, "
    "set dx = {finer_cell_m:um|.1f}µm (= "
    "h_sub/{finer_intervals}) or {coarser_cell_m:um|.1f}µm (= "
    "h_sub/{coarser_intervals}), aligning the {absolute_faces}.",
    "alignment_uniform_one_spacing": "To snap onto a mesh matching the DECLARED board instead, "
    "set dx = {finer_cell_m:um|.1f}µm (= "
    "h_sub/{finer_intervals}), aligning the {absolute_faces}; "
    "there is no coarser positive-interval candidate. Check 2 "
    "still recommends at least four normal intervals.",
    "trace_face_alignment": "MSL port '{port_name}': {iface_txt} — this lands in the [0.10, 0.40] "
    "mixed-cell danger zone of the existing declared-face alignment "
    "heuristic. The fraction alone does not establish material/PEC "
    "overlap. Historical substrate-air/trace mixed-cell runs with "
    "AD-traceable ``pec_occupancy_override`` reported unphysical |S21|² > "
    "1 (cited, not remeasured on this checkout: runs #563/#567, "
    "2026-05-08, dx∈[75,82]µm h_sub=254µm). A hard "
    "``Box(material='pec')`` avoids that specific bug ON THE DEFAULT RUN "
    "PATH (subpixel_smoothing=False and no conformal PEC face — the "
    "shipped defaults): a PEC Box is a VOLUME that occupies whole primal "
    "cells with walls on both faces (lattice ownership contract #931), "
    "and a foil trace declared as a SHEET (zero-thickness Box / "
    "add_thin_conductor) is one node plane; neither enters "
    "``pec_occupancy_override``. That is NOT a blanket exemption — two "
    "opt-in lanes DO give a hard PEC box fractional cell occupancy: "
    "subpixel_smoothing='kottke_pec' (the inv-eps tensor is built over "
    "the PEC shapes) and the Stage-1 conformal PEC lane (which replaces "
    "the binary pec_mask with fractional weights); "
    "rfx/runners/uniform.py. Plain subpixel_smoothing=True does NOT: "
    "'pec' carries eps_r=1.0 in the material library, so a PEC box enters "
    "the smoother as vacuum and stays whole-cell through pec_mask. The "
    "alignment advice below still applies on the two lanes that do. "
    "{gap_txt}{material_txt} The declared-face fraction and material "
    "extent describe geometry; neither predicts a Z0 error. The frozen "
    "sweep cannot establish a current extractor accuracy bound. "
    "{snap_txt}",
    "conductor_gap_mismatch": "MSL port '{port_name}': conductor-plane separation differs from "
    "the declared height by {gap_difference_fraction:percent|+.1f}%, "
    "beyond the existing {gap_tolerance_fraction:percent|.1f}% "
    "geometry-advisory threshold. {gap_txt}{material_txt} Place mesh "
    "nodes at the {absolute_faces} or refine the normal mesh. This "
    "geometry difference does not predict a Z0 change or establish an "
    "extractor accuracy bound.",
    "source_absorber_clearance": "MSL port '{port_name}' at {propagation_axis}={feed_m:mm|.2f}mm, "
    "direction={direction!r}: distance to nearest "
    "{propagation_axis}-CPML = {source_clearance_m:um|.0f}µm (domain "
    "edge + {source_buffer_m:len|} calibrated CPML buffer) < "
    "recommended {recommended_clearance_m:um|.0f}µm (= 2·h_sub). "
    "Source-side CPML reflection may inflate |S11|. Move port "
    "further from boundary OR increase domain "
    "{propagation_axis}-extent.",
    "ladder_ramp": "the probe ladder crosses cells of more than one size on the "
    "{propagation_axis} runway, so a probe-offset in CELLS does not name one "
    "distance here",
    "probe_ladder_clamped": "MSL port '{port_name}' (direction={direction!r}): the "
    "{probe_count}-probe ladder (n_probe_offset={probe_offset_cells}, "
    "n_probe_spacing={probe_spacing_cells} cells) runs past the grid and "
    "CLAMPS — only {distinct_probe_count} of {probe_count} probes land on "
    "distinct grid cells ({duplicate_probe_count} duplicate probe "
    "position(s): {probe_m:mm_tuple|}mm). The N-probe least-squares "
    "wave-decomposition fit is rank-deficient on duplicated positions; "
    "`compute_msl_s_matrix`'s Z0/S11 extraction is unreliable for this "
    "port. Shorten n_probe_offset/n_probe_spacing or extend the domain so "
    "the full ladder stays in-grid.",
    "reflector_scan_unavailable": "MSL port {port_name!r}: reflector clearance could not be "
    "evaluated: {clearance_note}.",
    "reflector_scan_incomplete": "MSL port '{port_name}' (direction={direction!r}): the "
    "downstream-reflector clearance scan could NOT evaluate "
    "{unevaluated_conductor_count} registered conductor(s), so a "
    "'clear' result here is not evidence that the probes are clear "
    "— ",
    "reflector_interval": "compliant n_probe_offset interval ≈ [{reflector_offset_min_cells}, "
    "{reflector_offset_max_cells}] cells",
    "reflector_clearance_general": "MSL port '{port_name}' (direction={direction!r}): deepest "
    "probe at {propagation_axis}={deepest_probe_m:mm|.2f}mm sits "
    "{reflector_gap_m:um|.0f}µm from a strong reflector candidate "
    "({nearest_label}; distance estimated from registered "
    "conductor bounds); recommended ≥ "
    "{recommended_reflector_gap_m:um|.0f}µm (= λ_g/4 at f_max with "
    "ε_eff_proxy={eps_eff_proxy:.1f}). "
    "{msl_probe_clearance_effect} This layout warning does not "
    "certify accuracy when absent. Available layout: "
    "{interval_txt}. Choose an offset within a nonempty interval; "
    "if it is empty, extend the uniform feed region to fit the "
    "source standoff, full probe ladder and reflector clearance. "
    "{msl_probe_clearance_guidance} Check settling and "
    "observation-plane sensitivity before interpreting S.",
    "absorber_interval": "compliant n_probe_offset interval ≈ [{absorber_offset_min_cells}, "
    "{absorber_offset_max_cells}] cells",
    "probe_in_absorber": "MSL port '{port_name}' (direction={direction!r}): probe "
    "{deepest_probe_index} (deepest, "
    "{propagation_axis}={deepest_probe_m:mm|.2f}mm) is past the domain edge "
    "(domain {propagation_axis}-extent [0, {domain_extent_m:mm|.2f}]mm) — "
    "inside the CPML absorbing region. The N-probe extractor's "
    "clean-travelling-wave assumption is void there: signal is attenuated "
    "and the fitted Z0/S11 are corrupted. {abs_interval_txt}.",
    "probe_near_absorber": "MSL port '{port_name}' (direction={direction!r}): probe "
    "{deepest_probe_index} (deepest, "
    "{propagation_axis}={deepest_probe_m:mm|.2f}mm) is within "
    "{absorber_proximity_cells} cells ({absorber_margin_m:len|}) of the "
    "domain edge, just past which the CPML absorber is active. Fields "
    "there carry CPML fringe/reflection error, biasing the fitted Z0/S11. "
    "{abs_interval_txt}.",
    "other_msl_owner": "MSL port '{other_name}'",
    "other_lumped_owner": "the lumped/wire port (component={other_component!r})",
    "probe_crosses_feed": "MSL port '{port_name}' (direction={direction!r}): probe span "
    "{propagation_axis}∈[{span_lo_m:mm|.2f}, {span_hi_m:mm|.2f}]mm crosses "
    "the feed plane of {other_owner_txt} at "
    "{propagation_axis}={other_feed_m:mm|.2f}mm. A feed is a source "
    "discontinuity the reflector scan above cannot see; probes sampling "
    "across it break the N-probe extractor's uniform-line assumption. If "
    "this crossing is intentional, verify the extracted Z0/S11 "
    "independently.",
    "source_snap": " The grid stamps the source on the node at "
    "{propagation_axis}={source_node_m:mm|.4f}mm, {source_snap_m:len|} from the "
    "declared feed.",
    "near_field_ramp": "MSL port '{port_name}' (direction={direction!r}): the source fringing "
    "decays over about five substrate thicknesses, {five_heights_m:len|} on "
    "this board, and that is a LENGTH. This port's feed sits where the mesh "
    "changes cell size: {ramp_txt}. Neither the distance an offset buys nor "
    "the offset that would clear the transient is one number here.",
    "near_field_distance": " The grid puts probe 0 {first_probe_distance_m:len|} "
    "({first_probe_height_ratio:.2f}·h_sub) from the feed plane.",
    "automatic_ramp": "counts {nf_none_term} in the scalar dx cell ({scalar_cell_m:len|}), "
    "{automatic_offset_cells} cells, because counted in this runway's own cells "
    "its probe ladder would cross a grading ramp",
    "automatic_uniform": "counts {nf_none_term} in this runway's {automatic_cell_m:len|} cells, ",
    "automatic_count": "{automatic_offset_cells} cells",
    "near_field_automatic_opening": "the automatic n_probe_offset={near_field_offset_cells} puts "
    "probe 0 ",
    "near_field_explicit_opening": "n_probe_offset={near_field_offset_cells} puts probe 0 ",
    "automatic_choice": " add_msl_port chose {near_field_offset_cells} by counting ",
    "automatic_shortfall": " in the scalar dx cell ({scalar_cell_m:len|}); this port's runway "
    "cells are {runway_cell_m:len|}, so on this runway the automatic floor "
    "falls short.",
    "remedy_automatic": "Set n_probe_offset >= {standoff_cells} explicitly on this port; leaving "
    "it None chooses {near_field_offset_cells} again.",
    "remedy_none_clears": "Set n_probe_offset >= {standoff_cells}, or leave it None: the "
    "automatic offset {nf_none_txt}.",
    "remedy_none_short": "Set n_probe_offset >= {standoff_cells}; leaving it None {nf_none_txt}, "
    "and falls short on this runway.",
    "remedy_explicit": "Set n_probe_offset >= {standoff_cells}.",
    "near_field_owner": "MSL port '{port_name}' (direction={direction!r}): ",
    "near_field_uniform": "{first_probe_distance_m:len|} ({first_probe_height_ratio:.2f}·h_sub) "
    "from this port's OWN feed plane, inside the source near-field standoff "
    "of {standoff_cells} cells ({standoff_m:len|} = 5·h_sub, the issue-#80 "
    "Fix B constant",
    "near_field_witness": " Within a few substrate thicknesses of the feed the launched field is "
    "not the guided mode yet: the evanescent content decays with the "
    "substrate's own transverse-resonance length 2·h_sub/π = "
    "{decay_length_m:len|} for THIS board (on the issue-#823 fixture, "
    "h_sub=300µm, that length measured 0.1932mm against a predicted "
    "0.19099mm — 1.1%). The decay LENGTH is a property of the substrate; "
    "the near-feed AMPLITUDE is not, so no error magnitude is predicted for "
    "your port here — read result diagnostics (the two-wave fit residual, "
    "and on the coax<->MSL lane the ladder-split witness) rather than "
    "trusting this offset. For reference, the #823 fixture's own measured "
    "amplitude (11.3 at the feed plane) put "
    "{reference_height_ratio:.0f}·h_sub at 4.4e-3 against the 0.02 two-wave "
    "residual bar this family holds itself to, and "
    "{first_probe_height_ratio:.2f}·h_sub at {reference_amplitude:.1e}. ",
    "probe_placement_note": "{detail}",
}

TEMPLATES.update(
    {
        "material_unavailable": " The declared-material column extent is unavailable.",
        "gap_unavailable": "The conductor-plane gap is unavailable because attachment was not validated.",
        "offset_interval_empty": "no compliant n_probe_offset exists on this feed length (interval empty)",
        "near_field_ramp_remedy": (
            ' Put the port and its probes inside one uniform zone of the profile, or extend that '
            'zone to hold the standoff. REPORT-ONLY: nothing is refused.'
        ),
        "near_field_scope": (
            ' REPORT-ONLY: nothing is refused, and the rule is derived from ONE fixture at W/h = 2'
            ' — a much wider trace may need more (the first higher-order microstrip mode scales '
            'with W + 2·h, which one fixture cannot separate from h).'
        ),
        "reflector_bounds_remedy": (
            '. Give those shapes an axis-aligned bounding box (or place the probes explicitly with'
            " n_probe_offset) before trusting `compute_msl_s_matrix`'s Z₀ / |S11| here."
        ),
        "automatic_kept_count": (
            " The driver keeps that count because, counted in this runway's own cells, the probe "
            'ladder would cross a grading ramp.'
        ),
        "automatic_floor": " add_msl_port's auto offset already floors to",
    }
)

CODE_LEVELS = {
    "msl.probe_clearance_scan_failed": ("msl_port_geometry", "advisory"),
    "msl.probe_clearance_unavailable": ("msl_port_geometry", "advisory"),
    "msl.reflector_clearance": ("msl_port_geometry", "advisory"),
    "msl.probe_placement_note": ("msl_port_geometry", "advisory"),
    "msl.probe_placement_failed": ("msl_port_geometry", "refusal"),
    "msl.conductor_assembly_unavailable": ("msl_port_conductor_planes", "refusal"),
    "msl.conductor_attachment": ("msl_port_conductor_planes", "refusal"),
    "msl.lateral_clearance": ("msl_port_geometry", "advisory"),
    "msl.normal_resolution": ("msl_port_geometry", "advisory"),
    "msl.trace_face_alignment": ("msl_port_geometry", "advisory"),
    "msl.conductor_gap_mismatch": ("msl_port_geometry", "advisory"),
    "msl.source_absorber_clearance": ("msl_port_geometry", "advisory"),
    "msl.probe_ladder_clamped": ("msl_port_geometry", "advisory"),
    "msl.reflector_scan_unavailable": ("msl_port_geometry", "advisory"),
    "msl.reflector_scan_incomplete": ("msl_port_geometry", "advisory"),
    "msl.probe_in_absorber": ("msl_port_geometry", "advisory"),
    "msl.probe_near_absorber": ("msl_port_geometry", "advisory"),
    "msl.probe_crosses_feed": ("msl_port_geometry", "advisory"),
    "msl.source_near_field": ("msl_port_geometry", "advisory"),
}


TEMPLATES.update(
    {
        "stub_odd_frequencies": "{frequency_hz:GHz|.6g}, {third_frequency_hz:GHz|.6g}, "
        "{fifth_frequency_hz:GHz|.6g}, ... GHz (odd multiples)",
        "stub_single_order": "{first_frequency_hz:GHz|.6g} GHz (order {first_order})",
        "stub_order_interval": "{first_frequency_hz:GHz|.6g}..{last_frequency_hz:GHz|.6g} GHz (odd "
        "orders {first_order}..{last_order})",
        "stub_band": "Read band {band_lo_hz:GHz|.6g}..{band_hi_hz:GHz|.6g} GHz; ",
        "stub_permittivity_mismatch": "Realized substrate eps_r={substrate_eps_r:.6g}; declared port "
        "eps_r_sub={declared:.6g}. ",
        "stub_body": "The strip continues {overhang_m:mm|.6g} mm behind the port and ends there "
        "(realized L; declared {declared_overhang_m:mm|.6g} mm); with the open-end "
        "extension {end_extension_m:mm|.6g} mm its effective length is "
        "{effective_length_m:mm|.6g} mm. It is an open stub that shorts the port near "
        "{frequency_hz:GHz|.6g} GHz (quarter wave); stub frequencies {frequencies}, "
        "{relation}. {band_text}eps_eff={eps_eff:.6g}; port {port_name!r}. {mismatch}The "
        "port's realized grid node is {axis}={port_node_m:mm|.9g} mm. Fix: start the "
        "signal strip at that coordinate (the port's grid node), so it covers the port "
        "node and nothing behind it (#1512).",
    }
)
TEMPLATES.update(
    {
        "line_stub_realization": "{detail}",
        "line_stub_inspection_unavailable": (
            "Line-stub check could not inspect {shape_count} conductor shape(s) "
            "({shapes}). They were skipped; the other conductors were "
            "checked. This is not evidence that the skipped shapes leave no tail behind a port."
        ),
    }
)
CODE_LEVELS.update(
    {
        "msl.line_stub_realization": ("line_stub_realization", "refusal"),
        "msl.line_stub_inspection_unavailable": (
            "line_stub_inspection_unavailable",
            "advisory",
        ),
        "msl.line_stub_behind_port": ("line_stub_behind_port", "advisory"),
    }
)


TEMPLATES.update(
    {
        "automatic_wavelength_term": "λ_eff/(4π) at f_max = {automatic_wavelength_m:len|}",
        "automatic_height_term": "5·h_sub = {automatic_height_m:len|}",
        "automatic_minimum_term": "the {automatic_minimum_cells}-cell minimum",
        "reflector_bounds": (
            "{owner}{kind} at {propagation_axis}∈[{reflector_lo_m:mm|.2f},"
            "{reflector_hi_m:mm|.2f}]mm "
            "{width_axis}∈[{reflector_width_lo_m:mm|.2f},{reflector_width_hi_m:mm|.2f}]mm"
        ),
    }
)


TEMPLATES.update(
    {
        "attachment_axes": "{owner}: propagation and width axes must both be resolved",
        "attachment_width": "{owner}: the declared width contains no grid node",
        "attachment_wall": "{owner}: the source touches domain PEC face {face}, which zeros its "
        "substrate-normal E component. Move the port off that wall.",
        "attachment_missing": "{owner}: declared {role} at z={declared_plane_m:.9g} m maps to node "
        "{plane_index} (z={realized_plane_m:.9g} m), but no longitudinal "
        "conductor edge meets it at width node {width_node}. Observed conductor "
        "planes on this column are {observed_plane_m} m. Make the port "
        "ground/trace declarations agree with the realized conductor surfaces.",
        "attachment_occupied": "{owner}: substrate interval [{lower_node}, {upper_node}) intersects "
        "PEC normal edges {occupied_node} at width node {width_node}; use the "
        "substrate-facing surfaces, not a plane inside the ground/trace "
        "volume.",
        "attachment_loaded": "{owner}: surface-impedance sheet edges {loaded_node} load the "
        "substrate-normal source at width node {width_node}. Move the port off "
        "the intersecting sheet.",
        "attachment_intervening": "{owner}: additional conductor planes {intervening_node} lie inside "
        "the substrate interval [{lower_node}, {upper_node}); the port "
        "cannot span them.",
    }
)


TEMPLATES.update(
    {
        "placement_traced": "{port_name!r} (direction={direction!r}): the {axis}-axis cell sizes are "
        "a traced mesh-as-design-variable profile and cannot be inspected "
        "host-side; the stored offset and spacing are kept",
        "placement_missing_lengths": "{port_name!r} (direction={direction!r}): the propagation axis "
        "{propagation_axis} is GRADED and the lengths its automatic "
        "probe ladder was counted from were not recorded; the stored "
        "offset and spacing are kept",
        "placement_ladder": "offset {runway_offset_cells} + {probe_intervals} x spacing "
        "{runway_spacing_cells} cells of {runway_cell_m:len|}",
        "placement_crosses_ramp": "{port_name!r} (direction={direction!r}): the propagation axis "
        "{propagation_axis} is GRADED and this port's automatic probe "
        "ladder counted in its runway cell ({ladder}) would cross a grading "
        "ramp, so no cell count names one length along it; the offset "
        "{stored_offset_cells} and spacing {stored_spacing_cells} counted "
        "in the boundary cell at registration are kept",
        "placement_uniform_zone": "{port_name!r} (direction={direction!r}): the propagation axis "
        "{propagation_axis} is GRADED; the automatic probe ladder is "
        "counted in this port's own runway cell ({ladder}, probe 0 "
        "{first_probe_distance_m:len|} from the source)",
        "placement_uninspected_remedy": (
            "; the upstream-only offset and the registration spacing, counted in this axis's cell, are kept"
        ),
        "placement_uninspected": "{port_name!r} (direction={direction!r}): the downstream reflector "
        "scan could not evaluate {unevaluated_conductor_count} conductor(s) "
        "— ",
        "placement_empty_interval": "MSL port {port_name!r}: the upstream and downstream probe "
        "clearances are mutually unsatisfiable on this feed (upstream "
        "needs n_probe_offset >= {offset_min_cells} cells = max(λ/4π, "
        "5·h_sub)/dx; downstream needs <= {offset_max_cells} cells to "
        "keep the deepest of {probe_count} probes ≥ "
        "{recommended_reflector_gap_m:um|.0f}µm (λ_g/4 at f_max) clear of "
        "the reflector {feed_reflector_gap_m:mm|.2f}mm from the feed). "
        "The feed line is too short for a clean N-probe measurement "
        "(issue #469) — keeping the upstream-priority offset "
        "{offset_min_cells}, which puts the deep probes inside the "
        "reflector's near field. ",
        "placement_graded_skipped": "MSL auto probe-offset interval solve (issue #469) SKIPPED for "
        "{port_count} port(s), so the downstream reflector clearance is "
        "NOT enforced for them and no auto probe spacing is widened "
        "(#681; on a graded propagation axis both would count reflector "
        "and absorber distances in cells of more than one size). Per "
        "port: {details}. This used to be silent for every non-uniform "
        "mesh (issue #686). Set n_probe_offset explicitly on these ports, "
        "or make the propagation axis uniform, if the deepest probe's "
        "clearance matters.",
    }
)


# These are wording variants of the SAME reflector-distance check: both call
# msl_probe_clearance_for_port and test the same threshold on the same ladder.
# The calculator route gives shorter advice; general preflight gives intervals.
VARIANTS = {
    "msl.probe_placement_note": (
        "placement_traced",
        "placement_missing_lengths",
        "placement_ladder",
        "placement_crosses_ramp",
        "placement_uniform_zone",
        "placement_uninspected",
        "placement_uninspected_remedy",
        "placement_empty_interval",
        "placement_graded_skipped",
    ),
    "msl.conductor_attachment": (
        "attachment_axes",
        "attachment_width",
        "attachment_wall",
        "attachment_missing",
        "attachment_occupied",
        "attachment_loaded",
        "attachment_intervening",
    ),
    "msl.line_stub_behind_port": (
        "stub_body",
        "stub_odd_frequencies",
        "stub_single_order",
        "stub_order_interval",
        "stub_band",
        "stub_permittivity_mismatch",
    ),
    "msl.reflector_clearance": (
        "reflector_clearance_general",
        "reflector_clearance_calculator",
    ),
    "msl.trace_face_alignment": (
        "alignment_profile",
        "alignment_translated",
        "alignment_uniform_two_spacings",
        "alignment_uniform_one_spacing",
    ),
    "msl.source_near_field": (
        "near_field_ramp",
        "near_field_uniform",
        "near_field_automatic_opening",
        "near_field_explicit_opening",
        "automatic_ramp",
        "automatic_uniform",
        "remedy_automatic",
        "remedy_none_clears",
        "remedy_none_short",
        "remedy_explicit",
    ),
}


@dataclass(frozen=True)
class CodeDefinition:
    legacy_slug: str
    severity: str
    templates: tuple[str, ...]
    value_names: tuple[str, ...]


# Shared fragments used by more than one check remain in this same table.
_FRAGMENTS = {
    "reflector_scan_incomplete": ("reflector_bounds_remedy",),
    "normal_resolution": ("conductor_gap", "material_column", "declared_faces"),
    "trace_face_alignment": (
        "declared_interface",
        "conductor_gap",
        "material_column",
        "declared_faces",
    ),
    "conductor_gap_mismatch": ("conductor_gap", "material_column", "declared_faces"),
    "reflector_clearance": (
        "clearance_gap",
        "reflector_interval",
        "standoff_ramp",
        "ladder_ramp",
        "reflector_bounds",
    ),
    "probe_in_absorber": ("absorber_interval", "standoff_ramp", "ladder_ramp"),
    "probe_near_absorber": ("absorber_interval", "standoff_ramp", "ladder_ramp"),
    "source_near_field": (
        "source_snap",
        "near_field_owner",
        "near_field_distance",
        "near_field_witness",
        "automatic_choice",
        "automatic_shortfall",
        "automatic_count",
        "standoff_ramp",
        "automatic_wavelength_term",
        "automatic_height_term",
        "automatic_minimum_term",
        "near_field_scope",
    ),
    "probe_crosses_feed": ("other_msl_owner", "other_lumped_owner"),
}


def _definition(code, slug, severity):
    name = code.split(".", 1)[1]
    keys = tuple(
        dict.fromkeys(
            (
                *((name,) if name in TEMPLATES else ()),
                *VARIANTS.get(code, ()),
                *_FRAGMENTS.get(name, ()),
            )
        )
    )
    names = tuple(
        sorted(
            {
                field
                for key in keys
                for _, field, _, _ in Formatter().parse(TEMPLATES[key])
                if field is not None
            }
        )
    )
    names = tuple(
        sorted(set(names).union(*(STATIC_VALUES.get(key, {}) for key in keys)))
    )
    return CodeDefinition(slug, severity, keys, names)


# Quantities in the retained historical prose are observations too. Issue/run
# identifiers and channel names (S11/S21) are identities, not SI quantities.
STATIC_VALUES = {
    "reflector_clearance_general": {"reflector_wavelength_divisor": 4},
    "automatic_wavelength_term": {"automatic_wavelength_divisor": 4},
    "placement_empty_interval": {
        "upstream_wavelength_divisor": 4, "near_field_height_ratio": 5,
        "reflector_wavelength_divisor": 4,
    },
    "lateral_clearance": {"recommended_height_ratio": 2},
    "source_absorber_clearance": {"recommended_height_ratio": 2},
    "normal_resolution": {"recommended_normal_intervals": 4},
    "near_field_scope": {
        "reference_width_height_ratio": 2,
        "higher_mode_height_multiplier": 2,
    },
    "near_field_uniform": {"standoff_height_ratio": 5},
    "probe_in_absorber": {"domain_start_m": 0},
    "trace_face_alignment": {
        "danger_fraction_lo": 0.10,
        "danger_fraction_hi": 0.40,
        "historical_dx_lo_m": 75e-6,
        "historical_dx_hi_m": 82e-6,
        "historical_substrate_height_m": 254e-6,
        "historical_power_limit": 1,
        "pec_library_eps_r": 1.0,
    },
    "near_field_witness": {
        "reference_substrate_height_m": 300e-6,
        "reference_measured_decay_m": 0.1932e-3,
        "reference_predicted_decay_m": 0.19099e-3,
        "reference_decay_difference_fraction": 0.011,
        "reference_feed_amplitude": 11.3,
        "reference_five_heights_amplitude": 4.4e-3,
        "reference_residual_bar": 0.02,
        "decay_height_multiplier": 2,
    },
}
_CLEARANCE_HISTORY = {
    "witness_settling_ratio": 10 ** (-118 / 20),
    "witness_near_railed_bins": 51,
    "witness_bin_count": 51,
    "witness_clear_railed_bins": 0,
    "witness_band_lo_hz": 3e9,
    "witness_band_hi_hz": 5e9,
    "witness_notch_hz": 3.77125e9,
    "witness_near_s11_ratio": 10 ** (0.026895 / 20),
    "witness_clear_s11_ratio": 10 ** (0.018065 / 20),
    "witness_analytic_s11_ratio": 10 ** (0 / 20),
    "witness_near_excess_fraction": 0.0031,
    "witness_clear_excess_fraction": 0.0021,
    "witness_difference_ratio": 10 ** (0.009 / 20),
    "retired_error_lo_ratio": 10 ** (-5 / 20),
    "retired_error_hi_ratio": 10 ** (-10 / 20),
}
for _key in (
    "reflector_clearance_general",
    "reflector_clearance_calculator",
    "placement_empty_interval",
):
    STATIC_VALUES[_key] = {**STATIC_VALUES.get(_key, {}), **_CLEARANCE_HISTORY}


CODES = {code: _definition(code, *entry) for code, entry in CODE_LEVELS.items()}


def msl_text(key, **values):
    message = render(key, TEMPLATES[key], values)
    message.values.update(STATIC_VALUES.get(key, {}))
    return message


def msl_diagnostic(code, message, *, subject=None, source="_check_msl_port_geometry", severity=None):
    if not isinstance(message, Message):
        raise TypeError("MSL messages must be rendered through the catalog")
    forwarded = getattr(message, "diagnostic", None)
    if (
        forwarded is not None
        and forwarded.code == code
        and forwarded.message == str(message)
    ):
        return forwarded
    entry = CODES[code]
    return Diagnostic(
        code,
        severity or entry.severity,
        subject,
        str(message),
        message.values,
        source=source,
    )


def stub_diagnostic(finding, band=None, *, severity="advisory"):
    fq = finding.frequency_hz / 1e9
    from rfx.preflight.line_stub import resonant_odd_orders

    orders = None if band is None else resonant_odd_orders(finding, band)
    if orders is None:
        relation = (
            "at odd quarter-wave resonances"
            if band is None
            else "outside the refusal interval for the band you read"
        )
        frequencies = msl_text(
            "stub_odd_frequencies",
            frequency_hz=msl_value(finding.frequency_hz, fq),
            third_frequency_hz=msl_value(3 * finding.frequency_hz, 3 * fq),
            fifth_frequency_hz=msl_value(5 * finding.frequency_hz, 5 * fq),
        )
    else:
        first, last = orders
        relation = "inside/near the band you read"
        frequencies = (
            msl_text(
                "stub_single_order",
                first_frequency_hz=msl_value(first * finding.frequency_hz, first * fq),
                first_order=first,
            )
            if first == last
            else msl_text(
                "stub_order_interval",
                first_frequency_hz=msl_value(first * finding.frequency_hz, first * fq),
                last_frequency_hz=msl_value(last * finding.frequency_hz, last * fq),
                first_order=first,
                last_order=last,
            )
        )
    band_text = (
        ""
        if band is None
        else msl_text(
            "stub_band",
            band_lo_hz=msl_value(band[0], band[0] / 1e9),
            band_hi_hz=msl_value(band[1], band[1] / 1e9),
        )
    )
    mismatch = ""
    declared = finding.declared_eps_r_sub
    if (
        declared is not None
        and abs(declared - finding.substrate_eps_r) > 0.01 * finding.substrate_eps_r
    ):
        mismatch = msl_text(
            "stub_permittivity_mismatch",
            substrate_eps_r=finding.substrate_eps_r,
            declared=declared,
        )
    message = msl_text(
        "stub_body",
        overhang_m=finding.overhang_m,
        declared_overhang_m=finding.declared_overhang_m,
        end_extension_m=finding.end_extension_m,
        effective_length_m=finding.effective_length_m,
        frequency_hz=msl_value(finding.frequency_hz, fq),
        frequencies=frequencies,
        relation=relation,
        band_text=band_text,
        eps_eff=finding.eps_eff,
        port_name=finding.port_name,
        mismatch=mismatch,
        axis=finding.axis,
        port_node_m=finding.port_node_m,
    )

    return msl_diagnostic(
        "msl.line_stub_behind_port",
        message,
        subject=finding.port_name,
        severity=severity,
        source="line_stub_findings",
    )


def msl_geometry_error(message, *, subject=None):
    diagnostic = msl_diagnostic(
        "msl.conductor_attachment",
        message,
        subject=subject,
        source="validate_msl_port_geometry",
    )
    return msl_error(diagnostic)


def msl_error(diagnostic, *, error=None):
    """Attach the record while preserving the legacy exception type and text."""
    if error is None:
        error = ValueError(diagnostic.message)
    error.diagnostic = diagnostic
    error.diagnostics = (diagnostic,)
    return error


def placement_warning(message):
    """Keep the original UserWarning category and caller's stack level."""
    diagnostic = msl_diagnostic(
        "msl.probe_placement_note",
        message,
        subject=message.values.get("port_name"),
        source="_resolve_msl_auto_offsets",
    )
    warning = UserWarning(diagnostic.message)
    warning.diagnostic = diagnostic
    return warning
