"""TF/SF admission rows agree with the matrix models' real entry points."""
from types import SimpleNamespace

import jax
import pytest

from rfx.runners import _admission as admission
from tests.contracts.path_equivalence.builders import build
from tests.contracts.path_equivalence.generation import PAIRS


ROWS = (("_tfsf", "plane_wave"), ("_tfsf", "closed_box"))
LANES = tuple(dict.fromkeys(lane for a, b, _ in PAIRS for lane in (a, b)))


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row[1])
@pytest.mark.parametrize("lane", LANES)
def test_tfsf_row_matches_real_entry(row, lane):
    refusal = None
    try:
        # A closed box on a graded mesh is already refused at declaration.
        sim = build(row, lane)
        assert admission.DETECTORS[row](sim)
        assert sum(admission.DETECTORS[other](sim) for other in ROWS) == 1
        kwargs = dict(n_steps=1, skip_preflight=True)
        if "distributed" in lane:
            devices = jax.devices("cpu")
            assert len(devices) == 2, "requires two host devices"
            kwargs["devices"] = devices
        if lane.startswith("fwd_"):
            if "distributed" in lane:
                kwargs["distributed"] = True
            result = sim.forward(checkpoint=False, **kwargs)
        else:
            result = sim.run(compute_s_params=False, **kwargs)
        jax.block_until_ready(result.time_series)
        assert result.time_series.shape == (1, 2)
    except (NotImplementedError, ValueError) as exc:
        # Only admission refusals count, never an unrelated builder/solver error.
        assert any(text in str(exc) for text in (
            "CustomWaveform and closed_box TFSF require a uniform",
            "open-slab normal-incidence TFSF (periodic y/z is not on slabs)",
            "TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane",
        )), str(exc)
        refusal = str(exc)
    assert (refusal is None) == (row in admission.ADMITS[lane]), (row, lane, refusal)


@pytest.mark.parametrize("declaration,kind", (
    ({"face_record": object(), "angle_deg": 20., "method": "methodB"}, "TFSF face_record replay"),
    ({"angle_deg": 20., "method": "methodB"}, "TFSF Method B oblique (is_tfsf_methodB)"),
    ({"angle_deg": 20., "closed_box": True}, "TFSF 2-D Bloch line (is_tfsf_2d)"),
    ({"closed_box": True, "theta_deg": 20.}, "oblique closed-box TFSF (theta_deg or phi_deg != 0)"),
    ({"closed_box": True, "phi_deg": 20.}, "oblique closed-box TFSF (theta_deg or phi_deg != 0)"),
    ({"closed_box": False}, "open-slab normal-incidence TFSF (periodic y/z is not on slabs)"),
    ({"closed_box": True}, None),
))
def test_slab_refusal_text_and_precedence(declaration, kind):
    entry = SimpleNamespace(**declaration)
    assert admission.slab_tfsf_refusal(entry) == kind
    if kind is None:
        assert admission.refuse_slab_tfsf(entry) is None
    else:
        with pytest.raises(NotImplementedError) as caught:
            admission.refuse_slab_tfsf(entry)
        assert str(caught.value) == f"{kind} is not supported on uniform multi-device; use one device"
