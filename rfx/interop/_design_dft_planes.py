"""DFT-plane document fields, serialization and legacy normalization."""

from __future__ import annotations

from typing import Any

from rfx.api._spec import _DFTPlaneEntry
from rfx.interop._design_codec import (
    _ARRAY, _F, _INT, _NUM, _STR, _dump_entry, _integer_vector,
    _load_entry, _opt, _refuse,
)

_DFT_PLANE_FIELDS: dict[str, _F] = {
    "name": _STR,
    "axis": _STR,
    "coordinate": _NUM,
    "component": _STR,
    "freqs": _opt(_ARRAY),
    "n_freqs": _INT,
}


def _dft_plane_region(value: Any, *, what: str) -> tuple[int, ...]:
    region = _integer_vector(value, 4, what=what)
    if any(index < 0 for index in region):
        raise _refuse(f"{what} must contain non-negative node indices, got {region!r}")
    return region


_DFT_PLANE_DOCUMENT_FIELDS: dict[str, _F] = {
    **_DFT_PLANE_FIELDS,
    "region": _opt(_F(
        dump=lambda v, w: list(_dft_plane_region(v, what=w)),
        load=lambda v, w: _dft_plane_region(v, what=w),
    )),
}


def _dump_dft_planes(sim: Any) -> list[dict[str, Any]]:
    return [
        {
            **_dump_entry(entry, _DFT_PLANE_FIELDS, _DFTPlaneEntry,
                          what=f"_dft_planes[{index}]"),
            "region": _DFT_PLANE_DOCUMENT_FIELDS["region"].dump(
                getattr(sim, "_dft_plane_regions", {}).get(entry.name),
                f"_dft_planes[{index}].region"),
        }
        for index, entry in enumerate(sim._dft_planes)
    ]


def _load_dft_plane(payload: Any, *, what: str) -> dict[str, Any]:
    return _load_entry(
        payload, _DFT_PLANE_DOCUMENT_FIELDS, what=what,
        defaults={"region": None},
    )


def _normalize_dft_planes(document: dict) -> None:
    """Fill the optional crop rectangle omitted by older v2 documents."""
    document["observables"]["dft_planes"] = [
        {"region": None, **entry} for entry in document["observables"]["dft_planes"]
    ]
