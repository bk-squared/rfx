"""Fixture BUILDER for the WR-90 chain-closure battery (v1.8 WP2).

This module constructs the ``Simulation`` for one ``(dut, dx)`` cell of the
battery and never runs it. It exists so that the geometry, the ladder, the
port/probe/reference planes, the drive and the absorber are fixed in one
importable place BEFORE the first S-parameter is measured — the
pre-declaration lives in
``docs/design_notes/waveguide_chain_battery_predeclaration.md`` and every
number here is restated there with its derivation.

Consumers:

* ``tests/test_waveguide_chain_battery_geometry.py`` (fast lane) — guard
  (iii) of plan decision 6: at every rung the realized guide node counts
  and the rasterized DUT cell counts scale exactly with ``1/dx``.
* ``tests/test_waveguide_chain_battery.py`` (the measurement, a later PR)
  — imports ``build_simulation`` and the constants; must not restate them.

Design rules baked in (plan WP2, "Fixture predeclaration"):

* Every rung is ``dx = a / N`` at integer ``N`` (9, 18, 36), so the three
  rungs realize ONE guide (a = 22.86 mm, b = 10.16 mm = a·4/9 → b is
  4 / 8 / 16 cells). Every x-coordinate below is an integer multiple of the
  coarse cell ``DX_COARSE = 2.54 mm``, so every rasterized face lands on a
  node at every rung and the half-open ``[lo, hi)`` Box rule
  (``rfx/geometry/csg.py``, class docstring) realizes the same physical
  extent at every rung.
* Reference planes are post-processing (``exp(∓jβΔ)`` in
  ``rfx/sources/waveguide_port.py::_shift_modal_waves``), so they need no
  node alignment; they are still multiples of ``DX_COARSE`` for legibility.
* The absorber is derived, not chosen: the rule at
  ``tests/test_waveguide_twoport_contract_v1.py:35-48``,
  ``CPML_LAYERS = ceil(0.75 · λ_g(f_low) / dx)`` with λ_g taken at the
  port's NUMERICAL TE10 cutoff — the cutoff preflight's
  ``_check_waveguide_port_evanescent`` derives from the realized guide
  (``_port_transverse_spans`` → ``guide``), not the analytic ``c/2a``.
  Only the rule transfers from that file; the cutoff and the cell count are
  this guide's own (see ``numerical_te10_cutoff_hz`` and ``cpml_layers``).
* Lanes: ``normalize=False`` and ``normalize="flux"`` only.
  ``normalize=True`` never enters a reflection gate
  (``docs/agent-memory/rfx-known-issues.md``, "normalize=True S11 formula
  wrong for strong reflectors").
* Nothing from ``rfx/probes/refplane.py`` is imported here or by the
  battery: that module carries a numpy round-trip the waveguide chain must
  not acquire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


C0 = 299_792_458.0

# --- guide ------------------------------------------------------------------
A_M = 0.02286   # WR-90 broad wall (y), metres
B_M = 0.01016   # WR-90 narrow wall (z), metres

# --- ladder (plan decision 6) -----------------------------------------------
# dx = a/N with N in {9, 18, 36}. Stated as the literal cell sizes the plan
# names — {2.54, 1.27, 0.635} mm — and checked against a/N at import.
DX_COARSE = 0.00254
DX_LADDER: tuple[float, ...] = (0.00254, 0.00127, 0.000635)
N_LADDER: tuple[int, ...] = (9, 18, 36)
for _dx, _n in zip(DX_LADDER, N_LADDER):
    assert abs(_dx * _n - A_M) < 1e-15, (_dx, _n)
    assert abs(_dx * _n * 4 / 9 - B_M) < 1e-15, (_dx, _n)

# --- x layout, in coarse cells (integers) -----------------------------------
# Every position is ``k * DX_COARSE``; the integers are the source of truth
# and the metre values below are derived from them so a reader can check
# each one by multiplication.
_NX_DOMAIN = 48          # 121.92 mm of guide between the two CPML pads
_K_PORT_LEFT = 5         # left port plane, +x launch
_K_PORT_RIGHT = 43       # right port plane, -x launch (48 - 5: symmetric)
_K_REF = 3               # default reference plane, cells inward of each port
_K_PROBE = 10            # probe plane, cells inward of each port
_K_PEC_LO, _K_PEC_HI = 23, 25     # PEC-short: 2 coarse cells = 5.08 mm thick
_K_SLAB_LO, _K_SLAB_HI = 22, 26   # eps_r=4 slab: 4 coarse cells = 10.16 mm
_K_SHIFT_LEFT = 12       # shifted reference plane, left port (WP2(b))
_K_SHIFT_RIGHT = 35      # shifted reference plane, right port (WP2(b))

DOMAIN_X_M = _NX_DOMAIN * DX_COARSE            # 0.12192
PORT_LEFT_X_M = _K_PORT_LEFT * DX_COARSE       # 0.01270
PORT_RIGHT_X_M = _K_PORT_RIGHT * DX_COARSE     # 0.10922
D_REF_M = _K_REF * DX_COARSE                   # 0.00762 (3 / 6 / 12 cells)
D_PROBE_M = _K_PROBE * DX_COARSE               # 0.02540 (10 / 20 / 40 cells)
PEC_SHORT_X_M = (_K_PEC_LO * DX_COARSE, _K_PEC_HI * DX_COARSE)   # 0.05842..0.06350
SLAB_X_M = (_K_SLAB_LO * DX_COARSE, _K_SLAB_HI * DX_COARSE)      # 0.05588..0.06604
SLAB_EPS_R = 4.0
# Default (measured) reference planes = port plane + D_REF inward.
REF_LEFT_DEFAULT_M = PORT_LEFT_X_M + D_REF_M     # 0.02032
REF_RIGHT_DEFAULT_M = PORT_RIGHT_X_M - D_REF_M   # 0.10160
# Probe (sampling) planes = port plane + D_PROBE inward.
PROBE_LEFT_M = PORT_LEFT_X_M + D_PROBE_M         # 0.03810
PROBE_RIGHT_M = PORT_RIGHT_X_M - D_PROBE_M       # 0.08382
# WP2(b) shifted pair — asymmetric on purpose (4 vs 5 coarse cells inward),
# the same reason the source pair at
# tests/test_waveguide_twoport_contract_v1.py:257 is asymmetric: a sign
# error on one port must not be cancelled by the other.
REF_LEFT_SHIFTED_M = _K_SHIFT_LEFT * DX_COARSE   # 0.03048 (+10.16 mm)
REF_RIGHT_SHIFTED_M = _K_SHIFT_RIGHT * DX_COARSE  # 0.08890 (-12.70 mm)

# --- band and drive ---------------------------------------------------------
# 17 bins, 0.2 GHz apart, centre bin (index 8) exactly 10.0 GHz. The top
# edge stays under 0.90 x fc_TE20 (= 0.90 x 13.115 GHz = 11.80 GHz), the
# preflight ``port_evanescent`` heuristic, so the fixture runs with that
# advisory silent; the bottom edge is 1.28 x fc_TE10.
FREQS = np.linspace(8.4e9, 11.6e9, 17)
F0_HZ = 10.0e9
BAND_CENTRE_BIN = 8
BANDWIDTH = 0.5
NUM_PERIODS = 40.0
FREQ_MAX_HZ = float(FREQS[-1])

DUTS: tuple[str, ...] = ("thru", "pec_short", "slab")
LANES: tuple = (False, "flux")
PORT_NAMES = ("left", "right")

# --- AD design variable θ (WP2(a)) -------------------------------------------
# θ enters through ``eps_override`` / ``sigma_override`` on the cells of ONE
# x-window of the guide, evaluated at θ0 (so the AD leg sees exactly the
# fixture the gates and referees see). Slab: the slab's own cells,
# eps_r = 4 + θ. PEC-short: a vacuum window of the same thickness ending on
# the short's front face, eps_r = 1 + θ (or sigma = θ for the loss leg).
_K_WINDOW_LO, _K_WINDOW_HI = 19, 23   # 0.04826..0.05842, 10.16 mm, ends on the PEC face
PEC_SHORT_WINDOW_X_M = (_K_WINDOW_LO * DX_COARSE, _K_WINDOW_HI * DX_COARSE)
THETA0_EPS = 0.0          # eps legs evaluate at the unperturbed fixture
THETA0_SIGMA_S_PER_M = 0.05   # loss leg: PEC-short window conductivity, S/m
FD_STEP_EPS = 0.05        # the step at tests/test_waveguide_flux_ad.py:80
FD_STEP_SIGMA_S_PER_M = 0.005


def design_region_x_m(dut: str) -> tuple[float, float]:
    """Absolute x-extent [lo, hi) of the θ window for ``dut``."""
    if dut == "slab":
        return SLAB_X_M
    if dut == "pec_short":
        return PEC_SHORT_WINDOW_X_M
    raise ValueError(f"no design region is declared for dut={dut!r}")


def guide_wavelength_m(freq_hz: float, fc_hz: float) -> float:
    """Continuous-medium TE guide wavelength λ_g = λ_0 / sqrt(1 - (fc/f)²)."""
    lam0 = C0 / float(freq_hz)
    return lam0 / math.sqrt(1.0 - (float(fc_hz) / float(freq_hz)) ** 2)


def cpml_layers_for(dx: float, fc_numerical_hz: float,
                    f_low_hz: float = float(FREQS[0])) -> int:
    """The absorber rule of ``tests/test_waveguide_twoport_contract_v1.py:35-48``:
    ``ceil(0.75 · λ_g(f_low) / dx)`` with λ_g at the NUMERICAL TE10 cutoff.
    """
    lam_g = guide_wavelength_m(f_low_hz, fc_numerical_hz)
    return int(math.ceil(0.75 * lam_g / dx))


def _boundary() -> BoundarySpec:
    # The canonical WR-90 construction in tests/test_waveguide_flux_ad.py:
    # CPML on the port-normal axis, PEC walls on both transverse axes.
    return BoundarySpec(
        x="cpml",
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    )


def _add_dut(sim: Simulation, dut: str) -> None:
    if dut == "thru":
        return
    if dut == "pec_short":
        # The construction at tests/test_waveguide_twoport_contract_v1.py:59-60
        # (pec_like: eps_r=1.0, sigma=1e10) restated in THIS guide's
        # coordinates: full cross-section, 5.08 mm thick.
        sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
        sim.add(Box((PEC_SHORT_X_M[0], 0.0, 0.0),
                    (PEC_SHORT_X_M[1], A_M, B_M)), material="pec_like")
        return
    if dut == "slab":
        sim.add_material("diel", eps_r=SLAB_EPS_R, sigma=0.0)
        sim.add(Box((SLAB_X_M[0], 0.0, 0.0),
                    (SLAB_X_M[1], A_M, B_M)), material="diel")
        return
    raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")


def build_simulation(
    dut: str,
    dx: float,
    *,
    cpml_layers: int | None = None,
    reference_planes: tuple[float | None, float | None] = (None, None),
    precision: str = "float32",
) -> Simulation:
    """Construct (do not run) the battery ``Simulation`` for ``(dut, dx)``.

    ``cpml_layers=None`` derives the absorber by :func:`cpml_layers_for`
    from this guide's numerical TE10 cutoff (a two-pass build: the cutoff
    is read off the realized grid of a first build, then the absorber is
    set on the returned one). Pass an explicit count only to reproduce a
    recorded fixture.

    ``reference_planes`` are the explicit ``reference_plane`` overrides for
    (left, right). ``None`` on either side resolves to the DECLARED default
    plane of that port (``REF_LEFT_DEFAULT_M`` / ``REF_RIGHT_DEFAULT_M`` =
    ``D_REF_M`` inward of the port, i.e. the raw record plane, ref_shift 0).
    It is passed explicitly because rfx's own default reports S at the
    PORT plane (``rfx/api/_sparams.py``, ``desired_ref = ... planes["source"]``,
    RF-audit 2026-07-23), not at ``source + ref_offset·dx``; the first
    coarse-rung plumbing run of the battery measured
    ``reference_planes = [0.0127, 0.10922]`` under ``(None, None)``, which is
    not the plane the pre-declaration (§2.3) references every oracle to.
    """
    if dut not in DUTS:
        raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")
    if not any(abs(dx - d) < 1e-15 for d in DX_LADDER):
        raise ValueError(f"dx={dx} is not a ladder rung {DX_LADDER}")
    if cpml_layers is None:
        probe = _build(dut, dx, cpml_layers=8, reference_planes=(None, None),
                       precision=precision)
        cpml_layers = cpml_layers_for(dx, numerical_te10_cutoff_hz(probe))
    left = REF_LEFT_DEFAULT_M if reference_planes[0] is None else float(reference_planes[0])
    right = REF_RIGHT_DEFAULT_M if reference_planes[1] is None else float(reference_planes[1])
    return _build(dut, dx, cpml_layers=cpml_layers,
                  reference_planes=(left, right), precision=precision)


def _build(dut, dx, *, cpml_layers, reference_planes, precision):
    n_cells = int(round(D_REF_M / dx)), int(round(D_PROBE_M / dx))
    ref_offset, probe_offset = n_cells
    assert abs(ref_offset * dx - D_REF_M) < 1e-15
    assert abs(probe_offset * dx - D_PROBE_M) < 1e-15
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=(DOMAIN_X_M, A_M, B_M),
        dx=dx,
        boundary=_boundary(),
        cpml_layers=int(cpml_layers),
        precision=precision,
    )
    _add_dut(sim, dut)
    freqs = jnp.asarray(FREQS)
    sim.add_waveguide_port(
        PORT_LEFT_X_M, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=F0_HZ, bandwidth=BANDWIDTH,
        ref_offset=ref_offset, probe_offset=probe_offset,
        name=PORT_NAMES[0], reference_plane=reference_planes[0],
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X_M, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=F0_HZ, bandwidth=BANDWIDTH,
        ref_offset=ref_offset, probe_offset=probe_offset,
        name=PORT_NAMES[1], reference_plane=reference_planes[1],
    )
    return sim


# --- realized-geometry readers (no run) ------------------------------------

@dataclass(frozen=True)
class TransverseSpans:
    """What preflight's ``_port_transverse_spans`` reports for one port."""
    a_aperture_m: float
    b_aperture_m: float
    a_guide_m: float
    b_guide_m: float
    guide_source: tuple[str, str]


def transverse_spans(sim: Simulation, port_index: int = 0) -> TransverseSpans:
    """Realized transverse spans of a port, read the way preflight reads
    them (``rfx/api/_preflight.py::_port_transverse_spans``)."""
    grid = sim._build_grid()
    pec_np = sim._port_pec_mask(grid)
    entry = sim._waveguide_ports[port_index]
    spans = sim._port_transverse_spans(entry, grid, pec_np)
    y, z = spans["y"], spans["z"]
    ap_y = y["aperture"] if y["aperture"] is not None else y["declared"]
    ap_z = z["aperture"] if z["aperture"] is not None else z["declared"]
    gd_y = y["guide"] if y["guide"] is not None else y["declared"]
    gd_z = z["guide"] if z["guide"] is not None else z["declared"]
    return TransverseSpans(
        a_aperture_m=float(max(ap_y, ap_z)), b_aperture_m=float(min(ap_y, ap_z)),
        a_guide_m=float(max(gd_y, gd_z)), b_guide_m=float(min(gd_y, gd_z)),
        guide_source=(str(y["guide_source"]), str(z["guide_source"])),
    )


def numerical_te10_cutoff_hz(sim: Simulation, port_index: int = 0) -> float:
    """TE10 cutoff of the REALIZED guide, the number preflight's
    ``port_evanescent`` advisory prints as ``fc_TE10`` — the formula of
    ``_emit_waveguide_port_cutoff_findings`` (``(c/2)·sqrt((m/a)²+(n/b)²)``)
    on the ``guide`` span, i.e. the wall-to-wall extent measured on the
    assembled PEC mask. This is the fixture's "numerical cutoff" in the sense
    of ``tests/test_waveguide_twoport_contract_v1.py:36-39``.
    """
    sp = transverse_spans(sim, port_index)
    return (C0 / 2.0) * math.sqrt((1.0 / sp.a_guide_m) ** 2 + (0.0 / sp.b_guide_m) ** 2)


def port_cutoff_hz(sim: Simulation, port_index: int = 0) -> float:
    """The TE10 cutoff the PORT CONFIG carries (``WaveguidePortConfig.f_cutoff``)
    — the number ``_shift_modal_waves`` / ``_compute_mode_impedance`` use for
    β and Z_TE. With ``mode_profile="discrete"`` it is the discrete 2D
    eigenvalue on the port aperture and is NOT the same quantity as
    :func:`numerical_te10_cutoff_hz` (preflight's wall-to-wall reader); the
    battery records both so a difference is visible in the artifact."""
    grid = sim._build_grid()
    entry = sim._waveguide_ports[port_index]
    cfg = sim._build_waveguide_port_config(
        entry, grid, jnp.asarray(FREQS), int(grid.num_timesteps(NUM_PERIODS)))
    return float(cfg.f_cutoff)


def dut_masks(sim: Simulation) -> dict[str, np.ndarray]:
    """Production rasterization of every geometry entry, keyed by material
    name, on the padded grid the solve uses (``Box.mask(grid)``)."""
    grid = sim._build_grid()
    out = {}
    for entry in sim._geometry:
        out[entry.material_name] = np.asarray(entry.shape.mask(grid), dtype=bool)
    return out


def axis_run_lengths(mask: np.ndarray) -> tuple[int, int, int]:
    """Occupied node count along each axis of an axis-aligned box mask —
    (nx, ny, nz) such that ``mask.sum() == nx*ny*nz``."""
    return tuple(int(mask.any(axis=tuple(k for k in range(3) if k != ax)).sum())
                 for ax in range(3))


def realized_guide_nodes(sim: Simulation) -> tuple[int, int]:
    """Node counts across the guide interior on y and z, from the grid's
    own pads: ``n_axis - pad_lo - pad_hi`` (PEC faces carry no pad). A guide
    of N cells has N + 1 nodes; the wall-to-wall extent is N·dx."""
    grid = sim._build_grid()
    nx, ny, nz = grid.shape
    fp = grid.face_pads
    return (int(ny - fp[2] - fp[3]), int(nz - fp[4] - fp[5]))


@dataclass(frozen=True)
class PortPlanes:
    """The three planes of one port in absolute metres, computed the way
    ``rfx/api/_compile.py::_build_waveguide_port_config`` computes them
    (index minus pad, times dx)."""
    source_m: float
    reference_m: float
    probe_m: float


def snapped_planes(sim: Simulation) -> dict[str, PortPlanes]:
    grid = sim._build_grid()
    out = {}
    for entry in sim._waveguide_ports:
        i = grid.position_to_index((entry.x_position, 0.0, 0.0))[0]
        src = (i - grid.axis_pads[0]) * grid.dx
        s = 1 if entry.direction.startswith("+") else -1
        out[entry.name] = PortPlanes(
            source_m=float(src),
            reference_m=float(src + s * entry.ref_offset * grid.dx),
            probe_m=float(src + s * entry.probe_offset * grid.dx),
        )
    return out


def design_region_index_range(sim: Simulation, dut: str) -> tuple[int, int]:
    """Padded-grid x-index range ``[i_lo, i_hi)`` of the θ window — the
    ``grid.position_to_index`` pattern of
    ``tests/test_waveguide_flux_ad.py::_eps_override_for``."""
    grid = sim._build_grid()
    lo, hi = design_region_x_m(dut)
    i_lo = grid.position_to_index((lo, 0.0, 0.0))[0]
    i_hi = grid.position_to_index((hi, 0.0, 0.0))[0]
    return int(i_lo), int(i_hi)


def design_override(sim: Simulation, dut: str, theta, *, kind: str = "eps"):
    """The full ``eps_override`` (or ``sigma_override``) array for
    ``compute_waveguide_s_matrix``: the fixture's OWN assembled material
    array with ``theta`` added on the θ window. The override replaces the
    assembled array wholesale (``rfx/api/_sparams.py``, "materials._replace"),
    so it must carry the slab's eps_r = 4 itself — a ``jnp.ones`` base would
    silently delete the DUT — and, for ``kind="sigma"``, the lane's own PEC
    fold (sigma = 1e10 on ``pec_mask``), or the override deletes the PEC
    short. ``theta`` may be a tracer.
    """
    grid = sim._build_grid()
    assembled = sim._assemble_materials(grid)
    mats, pec_mask = assembled[0], assembled[3]
    if kind == "eps":
        base = jnp.asarray(mats.eps_r)
    else:
        # ``_assemble_materials`` moves every conductor with sigma >= the PEC
        # threshold OUT of ``materials.sigma`` into ``pec_mask`` (the assembled
        # sigma is 0 inside ``pec_like``), and ``compute_waveguide_s_matrix``
        # folds that mask back into sigma = 1e10 before applying the
        # overrides ("Fold PEC mask back into high sigma", rfx/api/_sparams.py).
        # ``sigma_override`` replaces the FOLDED array wholesale, so a base
        # taken from the pre-fold assembly silently deletes the PEC short
        # (measured on the coarse rung: sigma_override(theta=0) gives the
        # empty guide's |S11| = 0.076 instead of 1.0). The base therefore
        # carries the fold itself, exactly as the lane would have built it.
        base = jnp.asarray(mats.sigma)
        if pec_mask is not None:
            base = jnp.where(jnp.asarray(pec_mask), jnp.asarray(1e10, dtype=base.dtype), base)
    i_lo, i_hi = design_region_index_range(sim, dut)
    return base.at[i_lo:i_hi, :, :].add(jnp.asarray(theta, dtype=base.dtype))
