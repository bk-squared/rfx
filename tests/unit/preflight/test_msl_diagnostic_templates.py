"""Previously uncovered wording, copied by hand from main's family source.

These 16 cases render fixed fields directly; they do not claim solver reach.
"""
import pytest

from rfx.preflight.msl_codes import msl_text


CASES = [
    (
        "alignment_uniform_one_spacing",
        dict(
            finer_cell_m=0.0001,
            finer_intervals=3,
            absolute_faces="faces",
        ),
        "To snap onto a mesh matching the DECLARED board instead, set dx = 100.0µm (= h_sub/3), "
        "aligning the faces; there is no coarser positive-interval candidate. Check 2 "
        "still recommends at least four normal intervals.",
    ),
    (
        "attachment_axes",
        dict(
            owner="MSL port 'p'",
        ),
        "MSL port 'p': propagation and width axes must both be resolved",
    ),
    (
        "attachment_width",
        dict(
            owner="MSL port 'p'",
        ),
        "MSL port 'p': the declared width contains no grid node",
    ),
    (
        "attachment_loaded",
        dict(
            owner="MSL port 'p'",
            loaded_node=[2, 3],
            width_node=4,
        ),
        "MSL port 'p': surface-impedance sheet edges [2, 3] load the substrate-normal source at width node 4. "
        "Move the port off the intersecting sheet.",
    ),
    (
        "automatic_choice",
        dict(
            near_field_offset_cells=3,
        ),
        " add_msl_port chose 3 by counting ",
    ),
    (
        "automatic_kept_count",
        {},
        " The driver keeps that count because, counted in this runway's own cells, "
        "the probe ladder would cross a grading ramp.",
    ),
    (
        "automatic_minimum_term",
        dict(
            automatic_minimum_cells=3,
        ),
        "the 3-cell minimum",
    ),
    (
        "automatic_shortfall",
        dict(
            scalar_cell_m=0.0001,
            runway_cell_m=0.0002,
        ),
        " in the scalar dx cell (100µm); this port's runway cells are 200µm, "
        "so on this runway the automatic floor falls short.",
    ),
    (
        "near_field_automatic_opening",
        dict(
            near_field_offset_cells=3,
        ),
        "the automatic n_probe_offset=3 puts probe 0 ",
    ),
    (
        "placement_missing_lengths",
        dict(
            port_name="p",
            direction="+x",
            propagation_axis="x",
        ),
        "'p' (direction='+x'): the propagation axis x is GRADED and the lengths its automatic "
        "probe ladder was counted from were not recorded; the stored offset and spacing are kept",
    ),
    (
        "placement_traced",
        dict(
            port_name="p",
            direction="+x",
            axis="x",
        ),
        "'p' (direction='+x'): the x-axis cell sizes are a traced mesh-as-design-variable profile "
        "and cannot be inspected host-side; the stored offset and spacing are kept",
    ),
    (
        "placement_uninspected",
        dict(
            port_name="p",
            direction="+x",
            unevaluated_conductor_count=2,
        ),
        "'p' (direction='+x'): the downstream reflector scan could not evaluate 2 conductor(s) — ",
    ),
    (
        "reflector_interval",
        dict(
            reflector_offset_min_cells=3,
            reflector_offset_max_cells=8,
        ),
        "compliant n_probe_offset interval ≈ [3, 8] cells",
    ),
    (
        "remedy_automatic",
        dict(
            standoff_cells=10,
            near_field_offset_cells=3,
        ),
        "Set n_probe_offset >= 10 explicitly on this port; leaving it None chooses 3 again.",
    ),
    (
        "remedy_none_short",
        dict(
            standoff_cells=10,
            nf_none_txt="counts 3 cells",
        ),
        "Set n_probe_offset >= 10; leaving it None counts 3 cells, and falls short on this runway.",
    ),
    (
        "source_snap",
        dict(
            propagation_axis="x",
            source_node_m=0.001,
            source_snap_m=0.0001,
        ),
        " The grid stamps the source on the node at x=1.0000mm, 100µm from the declared feed.",
    ),
]


@pytest.mark.parametrize(
    "key,fields,expected",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_fixed_main_wording(key, fields, expected):
    assert (
        msl_text(
            key,
            **fields,
        )
        == expected
    )
