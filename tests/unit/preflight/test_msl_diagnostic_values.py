"""Read displayed observations independently, including display precision and SI units."""
import math
import re

import pytest

from tests._msl_diagnostic_cases import captured_cases, cases


N = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
L = N + r'\s*(µm|um|mm|nm|m)'
SCALE = {'m': 1., 'mm': 1e-3, 'µm': 1e-6, 'um': 1e-6, 'nm': 1e-9}
# Patterns are transcribed from the user-visible sentences, not catalog keys,
# rendering helpers, raw geometry inputs, or the calculations that produced them.
LENGTHS = {
    'trace_width_m': 'trace W=' + L,
    'substrate_height_m': r'(?:h_sub=|declared height=)' + L,
    'lateral_clearance_m': 'absorbing boundary = ' + L,
    'cpml_buffer_m': r'domain edge \+ ' + L + ' calibrated',
    'recommended_clearance_m': 'recommended ' + L,
    'conductor_gap_m': 'conductor-plane gap=' + L,
    'ground_m': r'from [xyz]=' + L,
    'trace_m': r' to ' + L + r' \(declared height=',
    'material_extent_m': 'sample slot\\(s\\), extent ' + L,
    'recommended_cell_m': 'refine to dx ≤ ' + L,
    'declared_ground_m': r'declared ground [xyz]=' + L,
    'declared_trace_m': r'(?:and trace [xyz]=|declared trace plane at [xyz]=)' + L,
    'feed_m': r"' at [xyz]=" + L,
    'source_clearance_m': r'CPML = ' + L,
    'source_buffer_m': r'domain edge \+ ' + L + ' calibrated',
    'first_probe_distance_m': r'(?:puts probe 0 |grid puts probe 0 |, probe 0 )' + L,
    'standoff_m': r'standoff of \d+ cells \(' + L,
    'decay_length_m': '2·h_sub/π = ' + L,
    'automatic_cell_m': r"in this runway's " + L + ' cells',
    'interface_cell_m': r'that cell is ' + L,
    'finer_cell_m': r'set dx = ' + L,
    'coarser_cell_m': r'\(= h_sub/\d+\) or ' + L,
    'deepest_probe_m': r'(?:deepest, [xyz]=|deepest probe at [xyz]=)' + L,
    'absorber_margin_m': r'within \d+ cells \(' + L,
    'reflector_gap_m': r'(?:mm sits |deepest probe sits )' + L,
    'recommended_reflector_gap_m': r'(?:recommended ≥ |against the |probes ≥ )' + L,
    'other_feed_m': r' at [xyz]=' + L,
    'overhang_m': r'strip continues ' + L,
    'declared_overhang_m': r'realized L; declared ' + L,
    'end_extension_m': r'open-end extension ' + L,
    'effective_length_m': r'effective length is ' + L,
    'port_node_m': r'grid node is [xyz]=' + L,
    'automatic_height_m': r'automatic offset counts 5·h_sub = ' + L,
}
SCALARS = {
    'bias_fraction': (r'biased HIGH by ~' + N + '%', .01),
    'normal_intervals': (r'(?:only |over )' + N + ' normal interval', 1),
    'material_slots': (r'column has ' + N + ' same-permittivity', 1),
    'near_field_offset_cells': (r'n_probe_offset=' + N + ' puts', 1),
    'first_probe_height_ratio': (r'\(' + N + '·h_sub\\) from', 1),
    'standoff_cells': (r'standoff of ' + N + ' cells', 1),
    'reference_height_ratio': (r'feed plane\) put ' + N + '·h_sub', 1),
    'reference_amplitude': (r'and [\d.]+·h_sub at ' + N, 1),
    'automatic_offset_cells': (r'automatic offset counts .*? cells, (?:at least )?' + N + ' cells', 1),
    'interface_fraction': (r'sits ' + N + ' of a cell', 1),
    'finer_intervals': (r'\(= h_sub/' + N + r'\) or ', 1),
    'coarser_intervals': (r' or [\d.]+µm \(= h_sub/' + N, 1),
    'gap_difference_fraction': (r'height by ' + N + '%', .01),
    'gap_tolerance_fraction': (r'existing ' + N + '% geometry-advisory', .01),
    'probe_count': (r'the ' + N + '-probe ladder', 1),
    'probe_offset_cells': (r'ladder \(n_probe_offset=' + N, 1),
    'probe_spacing_cells': (r'n_probe_spacing=' + N + ' cells', 1),
    'distinct_probe_count': (r'CLAMPS — only ' + N + ' of', 1),
    'duplicate_probe_count': (r'\(' + N + ' duplicate probe', 1),
    'deepest_probe_index': (r'probe ' + N + ' \\(deepest,', 1),
    'absorber_proximity_cells': (r'within ' + N + ' cells', 1),
    'eps_eff_proxy': (r'ε_eff_proxy=' + N, 1),
    'unevaluated_conductor_count': (r'NOT evaluate ' + N + ' registered', 1),
    'shape_count': (r'could not inspect ' + N + ' conductor', 1),
    'eps_eff': (r'eps_eff=' + N, 1),
    'frequency_hz': (r'shorts the port near ' + N + ' GHz', 1e9),
    'third_frequency_hz': (r'stub frequencies [\d.eE+-]+, ' + N + ',', 1e9),
    'fifth_frequency_hz': (r'stub frequencies [\d.eE+-]+, [\d.eE+-]+, ' + N + ',', 1e9),
    'first_frequency_hz': (r'stub frequencies ' + N, 1e9),
    'first_order': (r'order ' + N, 1),
    'band_lo_hz': (r'Read band ' + N, 1e9),
    'band_hi_hz': (r'Read band [\d.e+-]+?\.\.' + N + ' GHz', 1e9),
}


LENGTHS.update({
    'reference_substrate_height_m': r'fixture, h_sub=' + L,
    'reference_measured_decay_m': r'length measured ' + L,
    'reference_predicted_decay_m': r'against a predicted ' + L,
    'historical_substrate_height_m': r'\]µm h_sub=' + L,
})
SCALARS.update({
    'recommended_height_ratio': (r'\(= ' + N + '·h_sub', 1),
    'recommended_normal_intervals': (r'at least ' + N + ' normal intervals', 1),
    'danger_fraction_lo': (r'in the \[' + N + ', ', 1),
    'danger_fraction_hi': (r'in the \[[\d.]+, ' + N + r'\]', 1),
    'historical_dx_lo_m': (r'dx∈\[' + N + ',', 1e-6),
    'historical_dx_hi_m': (r'dx∈\[[\d.]+,' + N + r'\]µm', 1e-6),
    'reference_decay_difference_fraction': (r'predicted [\d.]+mm — ' + N + '%', .01),
    'reference_feed_amplitude': (r'measured amplitude \(' + N + ' at the feed', 1),
    'reference_five_heights_amplitude': (r'put [\d.]+·h_sub at ' + N, 1),
    'reference_residual_bar': (r'against the ' + N + ' two-wave residual', 1),
    'witness_settling_ratio': (r'settled below ' + N + ' dB', 1),
    'witness_near_railed_bins': (r'scan railed on ' + N + '/', 1),
    'witness_bin_count': (r'scan railed on \d+/' + N, 1),
    'witness_clear_railed_bins': (r'against ' + N + '/51', 1),
    'witness_band_lo_hz': (r'bins over ' + N + '-', 1e9),
    'witness_band_hi_hz': (r'bins over 3-' + N + ' GHz', 1e9),
    'witness_notch_hz': (r'S11 at the ' + N + ' GHz notch', 1e9),
    'witness_near_s11_ratio': (r'notch bin read ' + N + ' dB', 1),
    'witness_clear_s11_ratio': (r'arm and ' + N + ' dB on the compliant', 1),
    'witness_analytic_s11_ratio': (r'analytic ' + N + ' dB', 1),
    'witness_near_excess_fraction': (r'structure, by ' + N + ' %', .01),
    'witness_clear_excess_fraction': (r'% and ' + N + ' %', .01),
    'witness_difference_ratio': (r'Their ' + N + ' dB difference', 1),
    'retired_error_lo_ratio': (r"retired '" + N + ' to', 1),
    'retired_error_hi_ratio': (r"retired '-5 to " + N + ' dB', 1),
})

SCALARS.update({
    'decay_height_multiplier': (r'transverse-resonance length ' + N + '·h_sub/π', 1),
    'reference_width_height_ratio': (r'at W/h = ' + N, 1),
    'higher_mode_height_multiplier': (r'W \+ ' + N + '·h', 1),
    'standoff_height_ratio': (r'mm = ' + N + '·h_sub', 1),
    'domain_start_m': (r'domain [xy]-extent \[' + N + ',', 1),
    'historical_power_limit': (r'\|S21\|² > ' + N, 1),
    'pec_library_eps_r': (r"'pec' carries eps_r=" + N, 1),
})


SCALARS.update({
    'offset_min_cells': (r'needs n_probe_offset >= ' + N + ' cells', 1),
    'offset_max_cells': (r'downstream needs <= ' + N + ' cells', 1),
    'upstream_wavelength_divisor': (r'max\(λ/' + N + 'π', 1),
    'near_field_height_ratio': (r'π, ' + N + '·h_sub', 1),
    'reflector_wavelength_divisor': (r'λ_g/' + N + ' at f_max', 1),
    'automatic_wavelength_divisor': (r'λ_eff/\(' + N + 'π', 1),
    'declared_plane_m': (r'declared (?:ground|trace) at z=' + N + ' m', 1),
    'plane_index': (r'maps to node ' + N, 1),
    'realized_plane_m': (r'node \d+ \(z=' + N + ' m', 1),
    'width_node': (r'width node ' + N, 1),
    'lower_node': (r'substrate interval \[' + N + ',', 1),
    'upper_node': (r'substrate interval \[\d+, ' + N, 1),
})
LENGTHS['feed_reflector_gap_m'] = r'the reflector ' + L + ' from the feed'


# Additional observations found by the reviewer's independent setup matrix.
# No unknown numeric key is ignored: printed() raises for any new key.
LENGTHS.update({
    'five_heights_m': r'five substrate thicknesses, ' + L,
    'automatic_height_m': r'5·h_sub = ' + L,
    'automatic_wavelength_m': r'λ_eff/\(4π\) at f_max = ' + L,
    'source_snap_m': r'mm, ' + L + ' from the declared feed',
    'scalar_cell_m': r'scalar dx cell \(' + L,
    'runway_cell_m': r'cells of ' + L,
})
SCALARS.update({
    'port_count': (r'SKIPPED for ' + N + ' port', 1),
    'runway_offset_cells': (r'(?:runway cell|own runway cell) \(offset ' + N, 1),
    'probe_intervals': (r'offset [\d.]+ \+ ' + N + ' x spacing', 1),
    'runway_spacing_cells': (r'x spacing ' + N + ' cells', 1),
    'stored_offset_cells': (r'; the offset ' + N + ' and spacing', 1),
    'stored_spacing_cells': (r'; the offset [\d.]+ and spacing ' + N + ' counted', 1),
    'source_node_m': (r'source on the node at [xy]=' + N + 'mm', 1e-3),
    'substrate_eps_r': (r'Realized substrate eps_r=' + N, 1),
    'declared': (r'declared port eps_r_sub=' + N, 1),
    'last_frequency_hz': (r'stub frequencies [\d.eE+-]+?\.\.' + N + ' GHz', 1e9),
    'first_order': (r'orders? ' + N, 1),
    'last_order': (r'odd orders [\d]+\.\.' + N, 1),
    'standoff_height_ratio': (r'(?:µm|mm|m) = ' + N + '·h_sub', 1),
    'automatic_offset_cells': (r'(?:cells, |cell \([^)]+\), )(?:at least )?' + N + ' cells', 1),
})


def precision(token, scale):
    mantissa, _, exponent = token.lower().partition('e')
    decimals = len(mantissa.split('.')[1]) if '.' in mantissa else 0
    return .5 * 10 ** (int(exponent or 0) - decimals) * scale


def printed(key, text):
    if key.startswith('port_') and key.split('_', 2)[1].isdigit():
        _, index, field = key.split('_', 2)
        # Each indexed port owns one semicolon-separated per-port description.
        details = text.split('Per port: ', 1)[1]
        sections = re.split(r"; (?=['\"])", details)
        return printed(field, sections[int(index)])
    if key.startswith(('observed_plane_m_', 'occupied_node_', 'loaded_node_', 'intervening_node_')):
        prefix, index = key.rsplit('_', 1)
        wording = {
            'observed_plane_m': r'Observed conductor planes on this column are \[([^]]*)\] m',
            'occupied_node': r'intersects PEC normal edges \[([^]]*)\]',
            'loaded_node': r'surface-impedance sheet edges \[([^]]*)\]',
            'intervening_node': r'conductor planes \[([^]]*)\]',
        }
        match = re.search(wording[prefix], text)
        assert match, (key, text)
        token = re.findall(N, match.group(1))[int(index)]
        return float(token), precision(token, 1)
    if key == 'probe_count' and 'deepest of ' in text:
        token = re.search(r'deepest of ' + N + ' probes', text).group(1)
        return float(token), precision(token, 1)
    if key == 'substrate_height_m' and 'declared height=' in text:
        pattern = 'declared height=' + L
    else:
        pattern = LENGTHS.get(key)
    if pattern is not None:
        match = re.search(pattern, text)
        assert match, (key, text)
        token, unit = match.groups()
        scale = SCALE[unit]
    elif key in SCALARS:
        pattern, scale = SCALARS[key]
        match = re.search(pattern, text)
        assert match, (key, text)
        token = match.group(1)
    elif key.startswith('probe_m_'):
        items = re.search(r'duplicate probe position\(s\): \(([^)]+)\)mm', text).group(1).split(',')
        token, scale = items[int(key.rsplit('_', 1)[1])].strip(), 1e-3
    else:
        pair_specs = {
            'span_lo_m': (r'probe span [xy]∈\[' + N + ', ' + N + r'\]mm', 0, 1e-3),
            'span_hi_m': (r'probe span [xy]∈\[' + N + ', ' + N + r'\]mm', 1, 1e-3),
            'absorber_offset_min_cells': (r'interval ≈ \[' + N + ', ' + N + r'\] cells', 0, 1),
            'absorber_offset_max_cells': (r'interval ≈ \[' + N + ', ' + N + r'\] cells', 1, 1),
            'domain_extent_m': (r'extent \[0, ' + N + r'\]mm', 0, 1e-3),
            'reflector_lo_m': (r'(?:conductor|box).*? at [xy]∈\[' + N + ',' + N + r'\]mm', 0, 1e-3),
            'reflector_hi_m': (r'(?:conductor|box).*? at [xy]∈\[' + N + ',' + N + r'\]mm', 1, 1e-3),
            'reflector_width_lo_m': (r'\]mm [xy]∈\[' + N + ',' + N + r'\]mm', 0, 1e-3),
            'reflector_width_hi_m': (r'\]mm [xy]∈\[' + N + ',' + N + r'\]mm', 1, 1e-3),
        }
        pattern, index, scale = pair_specs[key]
        match = re.search(pattern, text)
        assert match, (key, text)
        token = match.groups()[index]
    if key.endswith('_ratio') and key.startswith(('witness_', 'retired_')):
        center, delta = float(token), precision(token, 1)
        ratio = 10 ** (center / 20)
        return ratio, max(10 ** ((center + delta) / 20) - ratio,
                          ratio - 10 ** ((center - delta) / 20))
    return float(token) * scale, precision(token, scale)


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
def test_every_family_value_is_recovered_from_printed_text(graded):
    codes = set()
    checked = set()
    for messages in captured_cases(graded).values():
        for warning in messages:
            d = warning.diagnostic
            codes.add(d.code)
            for key, value in d.values.items():
                if not isinstance(value, (int, float)):
                    continue
                parsed, tolerance = printed(key, d.message)
                assert math.isfinite(value)
                assert abs(parsed - value) <= tolerance + math.ulp(parsed) + math.ulp(value), (
                    d.code,
                    key,
                    parsed,
                    value,
                )
                checked.add(key)
    assert len(codes) == 22
    assert len(checked) >= 70


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
def test_family_values_survive_actual_run_results_and_refusals(graded):
    from dataclasses import replace
    from unittest.mock import patch
    from rfx import Box
    import rfx.preflight.msl as msl
    import rfx.preflight.line_stub as stub
    from tests._msl_diagnostic_cases import structure, U

    returned, refused, checked = 0, 0, set()

    def inspect(name, result):
        for diagnostic in result.diagnostics:
            if not diagnostic.code.startswith('msl.'):
                continue
            checked.add(diagnostic.code)
            for key, value in diagnostic.values.items():
                if isinstance(value, (int, float)):
                    parsed, tolerance = printed(key, diagnostic.message)
                    assert abs(parsed - value) <= tolerance + math.ulp(parsed) + math.ulp(value), (
                        name, diagnostic.code, key, parsed, value)

    def solve(name, sim):
        nonlocal returned, refused
        try:
            result = sim.run(
                n_steps=2,
                compute_s_params=False,
            )
            returned += 1
        except (ValueError, NotImplementedError) as error:
            result = error
            refused += 1
        inspect(name, result)

    for name, sim in cases(graded):
        solve(name, sim)

    # Keep both open tails outside the read band, so unconditional stub
    # admission allows the crossed-probe geometry to reach its run warning.
    sim = structure(
        graded,
        offset=10,
        freq_max=1e9,
    )
    sim._msl_ports.append(
        replace(
            sim._msl_ports[0],
            name="opposite",
            position=(12 * U, 6 * U, 0),
            direction="-x",
        )
    )
    solve('cross_feed_outside_stub_band', sim)
    # The graded ladder clamps to realized end nodes. A declared end a
    # quarter-cell before the last node makes that node lie in the absorber.
    solve(
        "realized_end_outside_declared_domain",
        structure(
            graded,
            offset=40,
            length=31.75 * U,
        ),
    )

    sim = structure(graded)
    grid = sim._build_realized_grid()
    clearance = msl.msl_probe_clearance_for_port(sim, sim._msl_ports[0], grid)
    unavailable = replace(
        clearance,
        status="unavailable",
        note="no probe metadata",
        unevaluated_conductors=("unsupported conductor",),
    )
    with patch.object(
        msl,
        "msl_probe_clearance_for_port",
        return_value=unavailable,
    ):
        solve("incomplete_scan", sim)
        # These two calculator-report checks are not called by run(). Exercise
        # their public entry on each grid rather than claim solver reach.
        inspect(
            "calculator_unavailable",
            sim.preflight_sparameters(
                calculator="msl",
            ),
        )
    with patch.object(
        msl,
        "msl_probe_clearance_for_port",
        side_effect=ValueError("scan unavailable"),
    ):
        inspect(
            "calculator_scan_failed",
            sim.preflight_sparameters(
                calculator="msl",
            ),
        )
    for name, error in [('stub_realization', ValueError('conductor realization failed')),
                        ('stub_inspection', NotImplementedError('unsupported conductor'))]:
        with patch.object(
            stub,
            "line_stub_findings",
            side_effect=error,
        ):
            solve(name, structure(graded))
    sim = structure(
        graded,
        feed=16 * U,
    )
    sim._geometry[-1] = replace(
        sim._geometry[-1],
        shape=Box((0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)),
    )
    solve('stub_behind_port', sim)
    assert returned > 0 and refused > 0
    # Literal identities are independently declared in the path contract.
    from tests.unit.preflight.test_msl_diagnostic_identity import EXPECTED_LEGACY
    assert checked == set(EXPECTED_LEGACY)


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["base_z", "graded_z"],
)
def test_review_matrix_numeric_keys_are_all_parsed(graded):
    import warnings
    from tests._msl_review_cases import additional_cases
    observed = set()
    for sim in additional_cases(graded):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reports = [
                sim.preflight(),
                sim.preflight_sparameters(
                    calculator="msl",
                ),
            ]
        for report in reports:
            for diagnostic in report.diagnostics:
                if not diagnostic.code.startswith('msl.'):
                    continue
                for key, value in diagnostic.values.items():
                    if not isinstance(value, (int, float)):
                        continue
                    parsed, tolerance = printed(key, diagnostic.message)
                    assert abs(parsed - value) <= tolerance + math.ulp(parsed) + math.ulp(value), (
                        diagnostic.code, key, parsed, value)
                    observed.add((diagnostic.code, key))
    expected = {
        ('msl.source_near_field', key) for key in (
            'five_heights_m', 'standoff_height_ratio', 'automatic_wavelength_m',
            'automatic_height_m', 'source_node_m', 'source_snap_m',
            'scalar_cell_m', 'automatic_offset_cells')
    } | {
        ('msl.probe_placement_note', key) for key in (
            'port_count', 'port_0_runway_offset_cells', 'port_0_probe_intervals',
            'port_0_runway_spacing_cells', 'port_0_runway_cell_m',
            'port_0_stored_offset_cells', 'port_0_stored_spacing_cells',
            'port_0_first_probe_distance_m')
    } | {
        ('msl.line_stub_behind_port', key) for key in (
            'substrate_eps_r', 'declared', 'last_frequency_hz', 'first_order', 'last_order')
    }
    assert expected <= observed, expected - observed
    with pytest.raises(KeyError):
        printed('new_unparsed_quantity', 'some number 1')


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
def test_nonzero_read_band_values_on_a_real_refusal(graded):
    from tests._msl_review_cases import additional_cases
    from rfx.preflight.line_stub import require_no_resonant_line_stub
    sim = list(additional_cases(graded))[-1]
    with pytest.raises(ValueError) as caught:
        require_no_resonant_line_stub(sim, [3e9, 100e9])
    record = caught.value.diagnostics[-1]
    assert record.severity == 'refusal'
    assert record.values['band_lo_hz'] == 3e9
    for key, value in record.values.items():
        if isinstance(value, (int, float)):
            parsed, tolerance = printed(key, record.message)
            assert abs(parsed - value) <= tolerance + math.ulp(parsed) + math.ulp(value), (key, parsed, value)
