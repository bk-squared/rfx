"""Fast contracts for the waveguide source-path isolation spike."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port
from scripts import _waveguide_source_path_spike as spike


def _tiny_cfg():
    port = WaveguidePort(
        x_index=2,
        y_slice=(0, 4),
        z_slice=(0, 3),
        a=4.0e-3,
        b=3.0e-3,
        mode=(1, 0),
        mode_type="TE",
        direction="+x",
    )
    return init_waveguide_port(
        port,
        dx=1.0e-3,
        freqs=jnp.asarray([6.0e9]),
        dft_total_steps=64,
        waveform="modulated_gaussian",
    )


def test_strict_physical_verdict_uses_both_pec_and_asym_gates():
    assert spike.strict_physical_verdict(
        {"pec_min_abs_s11": 0.99, "asym_mean_reciprocity_rel_diff": 0.019}
    ) == "GO_STRICT"
    assert spike.strict_physical_verdict(
        {"pec_min_abs_s11": 0.989, "asym_mean_reciprocity_rel_diff": 0.019}
    ) == "NO_GO_PHYSICAL"
    assert spike.strict_physical_verdict(
        {"pec_min_abs_s11": 0.99, "asym_mean_reciprocity_rel_diff": 0.02}
    ) == "NO_GO_PHYSICAL"


def test_h_inc_table_transform_scale_and_fractional_shift_are_reproducible():
    cfg = _tiny_cfg()
    table = jnp.asarray(np.sin(np.linspace(0.0, 4.0 * np.pi, 64)), dtype=jnp.float32)
    cfg = cfg._replace(h_inc_table=table, dt=1.0e-12)
    scaled = spike.transform_h_inc_table(cfg, h_amplitude_scale=0.9)
    np.testing.assert_allclose(
        np.asarray(scaled.h_inc_table),
        0.9 * np.asarray(cfg.h_inc_table),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    shifted = spike.transform_h_inc_table(cfg, h_time_shift_dt=0.5)
    assert shifted.h_inc_table.shape == cfg.h_inc_table.shape
    assert np.all(np.isfinite(np.asarray(shifted.h_inc_table)))
    assert not np.allclose(np.asarray(shifted.h_inc_table), np.asarray(cfg.h_inc_table))


def test_control_rows_are_separate_and_never_claim_strict_closure():
    rows = spike.run_control_rows()
    assert {row.row_id for row in rows} == {
        "control_known_modal_coeffs_scalar",
        "control_known_modal_coeffs_biortho",
        "control_probe_only_field_snapshot",
    }
    assert all(row.section == "control_rows" for row in rows)
    assert all(row.verdict == "CONTROL_ONLY" for row in rows)
    assert all(row.verdict != "GO_STRICT" for row in rows)


def test_format_sections_emits_machine_readable_sectioned_json_lines():
    rows = [
        spike.MatrixRow(
            "physical_rows",
            "phys_example",
            "biortho",
            "current_source",
            {"pec_min_abs_s11": 0.8, "asym_mean_reciprocity_rel_diff": 0.1},
            "NO_GO_PHYSICAL",
        ),
        spike.MatrixRow(
            "control_rows",
            "control_example",
            "scalar",
            "synthetic",
            {"coefficient_error_max": 0.0},
            "CONTROL_ONLY",
        ),
    ]
    text = spike.format_sections(rows)
    assert "physical_rows:" in text
    assert "control_rows:" in text
    assert "geometry_rows:" in text

    json_lines = [line for line in text.splitlines() if line.startswith("{")]
    decoded = [json.loads(line) for line in json_lines]
    assert decoded[0]["row_id"] == "phys_example"
    assert decoded[1]["verdict"] == "CONTROL_ONLY"


def test_physical_spec_names_include_required_toggle_rows():
    names = {row_id for _section, row_id, _extractor, _cpml, _options in spike._physical_specs(True)}
    assert {
        "phys_scalar_baseline",
        "phys_biortho_current_source",
        "phys_h_phase_minus",
        "phys_h_phase_plus",
        "phys_h_impedance_scale_0p9",
        "phys_h_impedance_scale_1p0",
        "phys_h_impedance_scale_1p1",
        "phys_reference_no_reflection_subtract",
        "phys_reference_no_through_norm",
        "phys_cpml_geometry_sanity_8",
        "phys_cpml_geometry_sanity_10",
        "phys_cpml_geometry_sanity_12",
    } <= names
