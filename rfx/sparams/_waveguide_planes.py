"""One reported-plane convention for waveguide API results."""

from typing import NamedTuple

from rfx.nonuniform import NonUniformGrid
from rfx.sources.waveguide_port import waveguide_plane_positions
from rfx.sparams._common import _assert_nu_shift_span_in_one_grading_zone


class ReportedPlanes(NamedTuple):
    reference: float
    probe: float
    label: str


def resolve_reported_planes(entry, cfg, *, grid=None) -> ReportedPlanes:
    """Resolve reporting targets, leaving the config's sampling planes intact."""
    planes = waveguide_plane_positions(cfg)
    if entry.reference_plane is not None or entry.probe_plane is not None:
        reference = entry.reference_plane if entry.reference_plane is not None else planes["source"]
        probe = entry.probe_plane if entry.probe_plane is not None else planes["probe"]
        label = "explicit"
    elif entry.calibration_preset == "measured":
        reference, probe, label = planes["reference"], planes["probe"], "measured"
    else:
        reference, probe, label = planes["source"], planes["probe"], "source_to_probe"
    if isinstance(grid, NonUniformGrid) and reference != planes["reference"]:
        try:
            _assert_nu_shift_span_in_one_grading_zone(grid, cfg, reference, entry.name)
        except ValueError as exc:
            remedy = (
                "set reference_plane inside one uniform zone with the port and its ref_offset cells "
                "(move the port if needed)"
                if entry.reference_plane is not None or entry.probe_plane is not None
                else "place the port and its ref_offset cells inside one uniform zone, "
                     "or pass calibration_preset='measured'"
            )
            raise ValueError(
                f"{exc} Remedy: {remedy}."
            ) from exc
    return ReportedPlanes(reference, probe, label)
