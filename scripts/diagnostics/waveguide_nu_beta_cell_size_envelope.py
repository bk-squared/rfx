"""Boundary-cell vs local-cell beta on the nonuniform waveguide lane: a derived table.

The nonuniform (NU) rectangular-waveguide S-matrix lane evaluates the guided
propagation constant ``beta`` and the modal impedance ``Z_TE`` for the
reference-plane shift at ONE cell size -- ``cfg.dx``, which
``init_waveguide_port`` reads as ``float(grid_obj.dx)``: the grid's BOUNDARY
cell (``NonUniformGrid.dx``), not the cell the plane sits in. This script
does no FDTD run. It evaluates the production functions ``_compute_beta`` and
``_compute_mode_impedance`` (``rfx/sources/waveguide_port.py``) at the
boundary cell and at the fine cell of the committed NU AD fixture
(``tests/test_waveguide_nu_flux_ad.py``, coarse 1.5 mm / fine 0.75 mm,
``smooth_grading(max_ratio=1.3)``) over 8--12 GHz with the fixture's own
discrete cutoff and time step, and writes the table -- beta at both sizes,
their difference, and the phase error that difference makes over the
fixture's applied shift, over its port-to-reference offset, and over a
nominal 20 mm plane offset -- to
``tests/fixtures/waveguide_nu_beta_cell_size_envelope.json``.

Why arithmetic is enough here (v1.8 plan, WP4): the Yee correction to beta
is second order, ``beta(dx) = s_x * (1 + (s_x*dx)^2/24 + 3*(s_x*dx)^4/640 + ...)``
with ``s_x^2 = (sin(w*dt/2)/(c*dt/2))^2 - kc^2``, so the boundary-vs-local
difference is ``s_x^3 * (dx_b^2 - dx_l^2) / 24`` to leading order. The table
below carries that closed form beside the production value, and
``tests/test_waveguide_nu_beta_cell_size_envelope.py`` replays the JSON
against it. ``Z_TE`` needs no table: the discrete form
``mu0*dx*sin(w*dt/2) / (dt*sin(beta*dx/2))`` has ``sin(beta*dx/2) = s_x*dx/2``
by construction, so it does not depend on ``dx`` at all -- the JSON records
the relative difference so the replay can pin that.

The script also records, through the lane's own grading-zone check
(``rfx.api._sparams._assert_nu_shift_span_in_one_grading_zone``), which cells
each port's port-to-reference span crosses. For this fixture both spans lie
in uniform 1.5 mm cells (the first non-coarse cell begins at x = 0.030 m),
so the boundary cell IS the local cell there and the table is the envelope
for a plane that would sit in the fine block -- the fixture itself cannot
exercise the difference. A fixture with a reference plane inside the graded
region, and beta integration over the span, are deferred (issue #854 item 1).

No settling witness applies: nothing here is a time-domain measurement.

Arithmetic dtype: float64 (``jax_enable_x64`` is switched on at the top of
this standalone process -- never at a test module's import, that flip is
process-global). The production lane runs float32 (``cfg.freqs`` is float32),
so a float32 column is evaluated alongside and the replay test checks that
the effect is resolved at that precision too.

Run from the repository root::

    python scripts/diagnostics/waveguide_nu_beta_cell_size_envelope.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import jax

# Standalone process: float64 arithmetic for the reference column. This is
# the one place a global flip is allowed (see the module docstring).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from test_waveguide_nu_flux_ad import (  # noqa: E402  (the fixture itself, not a copy)
    NUM_PERIODS,
    _FREQS,
    _wr90_nu_sim,
)

import rfx  # noqa: E402
from rfx.api._sparams import _assert_nu_shift_span_in_one_grading_zone  # noqa: E402
from rfx.nonuniform import interior_cells  # noqa: E402
from rfx.runners.nonuniform import _build_waveguide_port_config_nu  # noqa: E402
from rfx.sources.waveguide_port import (  # noqa: E402
    C0_LOCAL,
    _compute_beta,
    _compute_mode_impedance,
    waveguide_plane_positions,
)

OUT_PATH = REPO / "tests" / "fixtures" / "waveguide_nu_beta_cell_size_envelope.json"
FIXTURE_MODULE = "tests/test_waveguide_nu_flux_ad.py"
PRODUCER = "scripts/diagnostics/waveguide_nu_beta_cell_size_envelope.py"

DX_BOUNDARY_M = 1.5e-3   # the fixture's coarse cell = NonUniformGrid.dx
DX_LOCAL_M = 0.75e-3     # the fixture's fine-block cell
NOMINAL_OFFSET_M = 0.020  # the plan's reference plane offset for the envelope
BAND_HZ = [8e9, 9e9, 10e9, 11e9, 12e9]


def closed_form_terms(f_hz: float, f_cutoff_hz: float, dt_s: float,
                      dx_b: float, dx_l: float) -> tuple[float, float, float]:
    """``(s_x, leading_delta_beta, next_order_delta_beta)`` in float64 numpy.

    ``s_x`` is the continuous-in-x wavenumber given the Yee temporal term;
    the leading boundary-minus-local difference is ``s_x^3 (dx_b^2 - dx_l^2)/24``
    and the next term of the arcsin series is ``3 s_x^5 (dx_b^4 - dx_l^4)/640``.
    """
    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * f_cutoff_hz / C0_LOCAL
    s_t_over_c = np.sin(omega * 0.5 * dt_s) / (C0_LOCAL * 0.5 * dt_s)
    s_x = np.sqrt(s_t_over_c ** 2 - kc ** 2)
    lead = s_x ** 3 * (dx_b ** 2 - dx_l ** 2) / 24.0
    nxt = 3.0 * s_x ** 5 * (dx_b ** 4 - dx_l ** 4) / 640.0
    return float(s_x), float(lead), float(nxt)


def _beta_real(f_hz: float, f_cutoff: float, dt: float, dx: float, dtype) -> float:
    b = _compute_beta(jnp.asarray([f_hz], dtype=dtype), f_cutoff, dt=dt, dx=dx)
    return float(np.real(np.asarray(b))[0])


def _z_te(f_hz: float, f_cutoff: float, dt: float, dx: float, dtype) -> float:
    z = _compute_mode_impedance(jnp.asarray([f_hz], dtype=dtype), f_cutoff, "TE",
                                dt=dt, dx=dx)
    return float(np.real(np.asarray(z))[0])


def main() -> dict:
    sim, domain_x = _wr90_nu_sim()
    grid = sim._build_nonuniform_grid()
    dt = float(grid.dt)
    freq_max = float(sim._freq_max)
    n_steps = int(np.ceil(NUM_PERIODS / freq_max / dt))

    cells = np.asarray(interior_cells(np.asarray(grid.dx_arr_f64, dtype=np.float64),
                                      int(grid.pad_x_lo), int(grid.pad_x_hi)))
    edges = np.insert(np.cumsum(cells), 0, 0.0)
    non_coarse = np.flatnonzero(cells < DX_BOUNDARY_M * (1.0 - 1e-9))
    first_non_coarse_x = float(edges[non_coarse[0]])
    last_non_coarse_end_x = float(edges[non_coarse[-1] + 1])

    ports = []
    f_cutoffs = []
    for entry in sim._waveguide_ports:
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, jnp.asarray(_FREQS), n_steps)
        planes = waveguide_plane_positions(cfg)
        desired = float(entry.reference_plane if entry.reference_plane is not None
                        else planes["source"])
        axis, lo, hi, sizes = _assert_nu_shift_span_in_one_grading_zone(
            grid, cfg, desired, entry.name)
        f_cutoffs.append(float(cfg.f_cutoff))
        ports.append({
            "name": entry.name,
            "direction": entry.direction,
            "requested_port_plane_m": float(entry.x_position),
            "requested_reference_plane_m": desired,
            "port_plane_m": float(planes["source"]),
            "modal_record_plane_m": float(planes["reference"]),
            "probe_plane_m": float(planes["probe"]),
            "applied_shift_m": desired - float(planes["reference"]),
            "port_to_reference_offset_m": desired - float(planes["source"]),
            "span_axis": axis,
            "span_lo_m": lo,
            "span_hi_m": hi,
            "cells_crossed_m": [float(v) for v in sizes],
            "distinct_cell_sizes_crossed_m": sorted({round(float(v), 12) for v in sizes}),
            "cfg_dx_m": float(cfg.dx),
            "cfg_dt_s": float(cfg.dt),
            "cfg_f_cutoff_hz": float(cfg.f_cutoff),
            "aperture_a_m": float(cfg.a),
            "aperture_b_m": float(cfg.b),
        })
    f_cutoff = f_cutoffs[0]
    assert all(abs(fc - f_cutoff) <= 1e-6 * f_cutoff for fc in f_cutoffs), f_cutoffs
    assert all(abs(p["cfg_dx_m"] - DX_BOUNDARY_M) < 1e-12 for p in ports), ports
    assert abs(float(grid.dx) - DX_BOUNDARY_M) < 1e-12, float(grid.dx)
    applied_shift_m = max(abs(p["applied_shift_m"]) for p in ports)
    port_to_ref_m = max(abs(p["port_to_reference_offset_m"]) for p in ports)

    freqs = sorted(set(BAND_HZ) | {float(f) for f in np.asarray(_FREQS, dtype=np.float64)})
    rows = []
    for f in freqs:
        s_x, lead, nxt = closed_form_terms(f, f_cutoff, dt, DX_BOUNDARY_M, DX_LOCAL_M)
        bb64 = _beta_real(f, f_cutoff, dt, DX_BOUNDARY_M, jnp.float64)
        bl64 = _beta_real(f, f_cutoff, dt, DX_LOCAL_M, jnp.float64)
        bb32 = _beta_real(f, f_cutoff, dt, DX_BOUNDARY_M, jnp.float32)
        bl32 = _beta_real(f, f_cutoff, dt, DX_LOCAL_M, jnp.float32)
        bc64 = _beta_real(f, f_cutoff, 0.0, 0.0, jnp.float64)
        zb = _z_te(f, f_cutoff, dt, DX_BOUNDARY_M, jnp.float64)
        zl = _z_te(f, f_cutoff, dt, DX_LOCAL_M, jnp.float64)
        d64 = bb64 - bl64
        rows.append({
            "f_hz": f,
            "in_fixture_freq_grid": bool(any(abs(f - float(g)) < 1.0 for g in np.asarray(_FREQS))),
            "s_x_rad_per_m": s_x,
            "beta_continuous_rad_per_m": bc64,
            "beta_boundary_cell_rad_per_m": bb64,
            "beta_local_cell_rad_per_m": bl64,
            "delta_beta_rad_per_m": d64,
            "delta_beta_closed_form_rad_per_m": lead,
            "delta_beta_next_order_rad_per_m": nxt,
            "beta_boundary_cell_float32_rad_per_m": bb32,
            "beta_local_cell_float32_rad_per_m": bl32,
            "delta_beta_float32_rad_per_m": bb32 - bl32,
            "z_te_boundary_cell_ohm": zb,
            "z_te_local_cell_ohm": zl,
            "z_te_rel_diff": abs(zb - zl) / abs(zb),
            "delta_phi_over_applied_shift_deg": float(np.degrees(d64 * applied_shift_m)),
            "delta_phi_over_port_to_reference_offset_deg": float(np.degrees(d64 * port_to_ref_m)),
            "delta_phi_over_nominal_20mm_deg": float(np.degrees(d64 * NOMINAL_OFFSET_M)),
        })

    out = {
        "producer": PRODUCER,
        "fixture": FIXTURE_MODULE + "::_wr90_nu_sim",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rfx_version": rfx.__version__,
        "jax_version": jax.__version__,
        "no_fdtd_run": True,
        "settling_db": None,
        "arithmetic": {
            "reference_columns_dtype": "float64 (jax_enable_x64 in this standalone process)",
            "production_lane_dtype": "float32 (cfg.freqs is float32 on the NU lane)",
            "closed_form": "delta_beta = s_x^3 (dx_b^2 - dx_l^2)/24; next order 3 s_x^5 (dx_b^4 - dx_l^4)/640; "
                           "s_x^2 = (sin(w dt/2)/(c dt/2))^2 - kc^2",
            "z_te_note": "discrete Z_TE = mu0 dx sin(w dt/2)/(dt sin(beta dx/2)) with sin(beta dx/2) = s_x dx/2 exactly, so it is cell-size independent",
        },
        "grid": {
            "domain_x_m": float(domain_x),
            "boundary_dx_m": float(grid.dx),
            "dt_s": dt,
            "n_steps_at_fixture_num_periods": n_steps,
            "num_periods": float(NUM_PERIODS),
            "freq_max_hz": freq_max,
            "n_interior_cells_x": int(cells.size),
            "distinct_cell_sizes_m": sorted({round(float(v), 12) for v in cells}),
            "first_non_coarse_cell_starts_x_m": first_non_coarse_x,
            "last_non_coarse_cell_ends_x_m": last_non_coarse_end_x,
        },
        "beta_inputs": {
            "f_cutoff_hz": f_cutoff,
            "f_cutoff_source": "cfg.f_cutoff from _build_waveguide_port_config_nu (discrete 2D eigenvalue on the snapped aperture)",
            "dx_boundary_m": DX_BOUNDARY_M,
            "dx_local_m": DX_LOCAL_M,
            "dt_s": dt,
            "applied_shift_m": applied_shift_m,
            "port_to_reference_offset_m": port_to_ref_m,
            "nominal_offset_m": NOMINAL_OFFSET_M,
        },
        "ports": ports,
        "rows": rows,
        "headline": {
            "max_delta_phi_over_nominal_20mm_deg": max(r["delta_phi_over_nominal_20mm_deg"] for r in rows),
            "max_delta_phi_over_applied_shift_deg": max(r["delta_phi_over_applied_shift_deg"] for r in rows),
            "max_z_te_rel_diff": max(r["z_te_rel_diff"] for r in rows),
            "both_spans_in_one_cell_size": all(len(p["distinct_cell_sizes_crossed_m"]) == 1 for p in ports),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    print(f"fixture {out['fixture']}: boundary dx {grid.dx * 1e3:.3f} mm, fine dx "
          f"{DX_LOCAL_M * 1e3:.3f} mm, dt {dt:.4e} s, f_cutoff {f_cutoff / 1e9:.4f} GHz")
    print(f"first non-coarse cell starts at x = {first_non_coarse_x * 1e3:.3f} mm; "
          f"last ends at {last_non_coarse_end_x * 1e3:.3f} mm of {domain_x * 1e3:.3f} mm")
    for p in ports:
        print(f"port {p['name']:>5} {p['direction']}: port plane {p['port_plane_m'] * 1e3:.4f} mm, "
              f"record plane {p['modal_record_plane_m'] * 1e3:.4f} mm, reference "
              f"{p['requested_reference_plane_m'] * 1e3:.4f} mm, applied shift "
              f"{p['applied_shift_m'] * 1e3:+.4f} mm, cells crossed "
              f"{len(p['cells_crossed_m'])} of sizes {[s * 1e3 for s in p['distinct_cell_sizes_crossed_m']]} mm")
    print(f"{'f GHz':>7} {'beta_b':>9} {'beta_l':>9} {'dbeta':>8} {'closed':>8} "
          f"{'dphi@shift':>10} {'dphi@5mm':>9} {'dphi@20mm':>9} {'Z_TE reldiff':>12}")
    for r in rows:
        print(f"{r['f_hz'] / 1e9:7.2f} {r['beta_boundary_cell_rad_per_m']:9.3f} "
              f"{r['beta_local_cell_rad_per_m']:9.3f} {r['delta_beta_rad_per_m']:8.4f} "
              f"{r['delta_beta_closed_form_rad_per_m']:8.4f} "
              f"{r['delta_phi_over_applied_shift_deg']:10.4f} "
              f"{r['delta_phi_over_port_to_reference_offset_deg']:9.4f} "
              f"{r['delta_phi_over_nominal_20mm_deg']:9.4f} {r['z_te_rel_diff']:12.2e}")
    print(f"wrote {OUT_PATH.relative_to(REPO)}")
    return out


if __name__ == "__main__":
    main()
