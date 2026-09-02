"""Issue #685 — the MSL downstream-reflector scan must see ALL the metal.

``msl_nearest_downstream_reflector`` used to admit a conductor only when
``isinstance(shape, Box) and str(material_name).lower() == "pec"``. Two
classes were invisible:

* PEC-PROMOTED materials — a conductor registered with sigma >= 1e6 under
  any other name (the common case for imported CAD, where every conductor
  may be called "metal");
* non-``Box`` shapes — ``Sheet``, ``MeshShape`` (#358), CSG results — and
  thin conductors, which are not in ``geometry`` at all. Since #677 an
  ``surface_impedance_f0`` sheet is in neither ``pec_mask`` nor
  ``materials.sigma``, so nothing else would ever see it.

And the scan could not distinguish "nothing is nearby" from "I could not
look": a shape with no usable bounding box was skipped in silence and the
line read clean.

Measured on the Sheen-parameter geometry (dx=200um, h_sub=0.794mm,
er=2.2, f_max=20GHz -> msl_min_probe_clearance = 1676um), probe at
x=8.00mm:

  downstream element                 old scan     new scan
  Box, material "metal" (5.8e7 S/m)  inf (clean)  4466um
  thin_conductor f0 sheet            inf (clean)  4466um
  shape with no bounding box         inf (clean)  inf + 1 unevaluated
"""

import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._preflight import (
    msl_min_probe_clearance,
    msl_nearest_downstream_reflector,
)
from rfx.geometry.csg import Cylinder

DX = 2e-4
DOMAIN = (0.020, 0.02632, 0.0038)
Y_C = 0.01316
W_TRACE = 0.002413
H_SUB = 0.000794
X_PROBE = 0.008
PATCH = ((0.012466, 0.003, H_SUB), (0.015006, 0.02332, H_SUB + DX))


def _base():
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    # The port's own feed trace (contains the feed plane -> excluded).
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (0.012466, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    return sim


def _scan(sim, *, legacy=False):
    kw = {} if legacy else dict(
        resolve_material=sim._resolve_material,
        thin_conductors=getattr(sim, "_thin_conductors", ()),
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
    )
    return msl_nearest_downstream_reflector(
        sim._geometry, x_probe=X_PROBE, x_feed=0.0025, y_feed=Y_C,
        w_trace=W_TRACE, dx=DX, domain_y=DOMAIN[1], direction="+x", **kw)


def test_clearance_rule_is_the_one_this_module_quotes():
    assert round(msl_min_probe_clearance(20e9) * 1e6) == 1676


def test_pec_promoted_material_under_another_name_is_seen():
    """The imported-CAD case: every conductor is called 'metal'."""
    sim = _base()
    sim.add_material("metal", eps_r=1.0, sigma=5.8e7)
    sim.add(Box(*PATCH), material="metal")

    d, label, unevaluated = _scan(sim)
    assert np.isfinite(d), "a 5.8e7 S/m conductor must count as a reflector"
    assert round(d * 1e6) == 4466
    assert "metal" in label
    assert unevaluated == []

    # The old name test is blind to it — pinned so the widening is what
    # is being measured, not the fixture.
    d_legacy, label_legacy, _ = _scan(sim, legacy=True)
    assert not np.isfinite(d_legacy)
    assert label_legacy is None


def test_a_lossy_dielectric_is_still_not_a_reflector():
    """Widening the rule must not turn every lossy material into metal."""
    sim = _base()
    sim.add_material("lossy_sub", eps_r=4.0, sigma=1e-2)
    sim.add(Box(*PATCH), material="lossy_sub")
    d, label, unevaluated = _scan(sim)
    assert not np.isfinite(d), f"sigma=1e-2 must not read as a conductor ({label})"
    assert unevaluated == []


def test_thin_conductor_sheets_are_scanned():
    """An f0 sheet is in NEITHER pec_mask NOR sigma (#677) — the scan is
    the only thing that can see it, and it did not look."""
    sim = _base()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box(PATCH[0], (PATCH[1][0], PATCH[1][1], PATCH[0][2])),
            sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=20e9)

    d, label, unevaluated = _scan(sim)
    assert np.isfinite(d)
    assert round(d * 1e6) == 4466
    assert "thin_conductor[0]" in label
    assert unevaluated == []

    d_legacy, _, _ = _scan(sim, legacy=True)
    assert not np.isfinite(d_legacy), (
        "the legacy call passes no thin_conductors, so it must still miss it")


def test_non_box_shape_is_placed_by_its_bounding_box():
    sim = _base()
    sim.add(Cylinder(center=(0.0090, Y_C, H_SUB + DX / 2), radius=0.0004,
                     height=DX, axis="z"), material="pec")
    d, label, unevaluated = _scan(sim)
    assert np.isfinite(d)
    assert round(d * 1e6) == 600           # 9.0mm - 0.4mm - 8.0mm
    assert "bounding box" in label, label
    assert unevaluated == []


class _NoBBox:
    """A conductor shape the scan cannot place."""

    def mask(self, grid):                     # pragma: no cover - unused
        raise NotImplementedError


class _RaisingBBox(_NoBBox):
    def bounding_box(self):
        raise NotImplementedError("no bbox for this shape")


@pytest.mark.parametrize("shape_cls", [_NoBBox, _RaisingBBox])
def test_unplaceable_conductor_is_reported_not_silently_clean(shape_cls):
    sim = _base()
    entry_cls = type(sim._geometry[0])
    sim._geometry.append(entry_cls(shape=shape_cls(), material_name="pec"))

    d, label, unevaluated = _scan(sim)
    assert not np.isfinite(d)
    assert label is None
    assert len(unevaluated) == 1, unevaluated
    assert shape_cls.__name__ in unevaluated[0]
    assert "cannot be placed" in unevaluated[0]


def test_preflight_advisory_says_it_could_not_look():
    """The advisory, not just the return value, must carry the doubt."""
    sim = _base()
    entry_cls = type(sim._geometry[0])
    sim._geometry.append(entry_cls(shape=_NoBBox(), material_name="pec"))
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="p1")

    issues = sim.preflight(strict=False, check_ntff=False)
    msgs = [str(i) for i in issues]
    hits = [m for m in msgs if "could NOT evaluate" in m]
    assert hits, msgs
    assert "not evidence that the probes are clear" in hits[0]
    assert "_NoBBox" in hits[0]


def test_the_measured_board_shape_warns_now():
    """A conductor named 'metal' inside the clearance rule must warn.

    This is the #685 field report in miniature: the T-junction sits well
    inside msl_min_probe_clearance of the deepest probe and the old scan
    reported the line clean.
    """
    sim = _base()
    sim.add_material("metal", eps_r=1.0, sigma=5.8e7)
    # Junction placed deliberately close to the probe ladder.
    sim.add(Box((0.0088, 0.003, H_SUB), (0.0110, 0.02332, H_SUB + DX)),
            material="metal")
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="p1")

    issues = sim.preflight(strict=False, check_ntff=False)
    msgs = [str(i) for i in issues]
    hits = [m for m in msgs
            if "from a strong reflector" in m and "metal" in m]
    assert hits, msgs
