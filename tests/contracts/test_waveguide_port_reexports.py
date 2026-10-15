"""The historical waveguide port surface preserves object identity."""

from importlib import import_module


def test_waveguide_port_reexports():
    old = import_module("rfx.sources.waveguide_port")
    expected = {
        "WaveguidePort": "rfx.model.waveguide_modes",
        "WaveguidePortConfig": "rfx.model.waveguide_modes",
        "cutoff_frequency": "rfx.model.waveguide_modes",
        "init_waveguide_port": "rfx.model.waveguide_modes",
        "solve_rectangular_modes": "rfx.model.waveguide_modes",
        "init_multimode_waveguide_port": "rfx.model.waveguide_modes",
        "C0_LOCAL": "rfx.model.waveguide_modes",
        "inject_waveguide_port": "rfx.stepping.waveguide_source",
        "apply_waveguide_port_h": "rfx.stepping.waveguide_source",
        "apply_waveguide_port_e": "rfx.stepping.waveguide_source",
        "modal_voltage": "rfx.measurement.waveguide_modal",
        "modal_current": "rfx.measurement.waveguide_modal",
        "mode_self_overlap": "rfx.measurement.waveguide_modal",
        "overlap_modal_amplitude": "rfx.measurement.waveguide_modal",
        "update_waveguide_port_probe": "rfx.measurement.waveguide_modal",
        "OverlapDFTAccumulators": "rfx.measurement.waveguide_modal",
        "init_overlap_dft": "rfx.measurement.waveguide_modal",
        "update_overlap_dft": "rfx.measurement.waveguide_modal",
        "extract_waveguide_sparams_overlap": "rfx.measurement.waveguide_modal",
        "settling_db_from_named_records": "rfx.measurement.settling",
        "settling_db_from_port_records": "rfx.measurement.settling",
        "waveguide_plane_positions": "rfx.measurement.waveguide_s",
        "extract_waveguide_sparams": "rfx.measurement.waveguide_s",
        "extract_waveguide_s21": "rfx.measurement.waveguide_s",
        "extract_waveguide_s11": "rfx.measurement.waveguide_s",
        "extract_waveguide_port_waves": "rfx.measurement.waveguide_s",
        "extract_waveguide_s_matrix": "rfx.measurement.waveguide_s",
        "extract_waveguide_s_matrix_flux": "rfx.measurement.waveguide_s",
        "extract_waveguide_s_params_normalized": "rfx.measurement.waveguide_s",
        "extract_multimode_s_matrix": "rfx.measurement.waveguide_s",
        "extract_multimode_s_matrix_flux": "rfx.measurement.waveguide_s",
    }
    assert set(old.__all__) == set(expected)
    for name, home in expected.items():
        assert getattr(old, name) is getattr(import_module(home), name)
