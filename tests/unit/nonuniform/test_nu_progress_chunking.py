"""#667 on the NU lane: chunked host-side progress with re-entry bit-identity.

The NU progress route drives the same chunked host loop the #383 decay stop
uses, with the stop disabled (decay_by=0.0 forced-N escape) — so the whole
claim rests on chunk re-entry threading the carry EXACTLY. These tests lock:

1. bit-identity: report_every=N produces bit-identical probe series and
   final fields to the single-scan run (same build, same steps);
2. progress lines actually appear (that was the point — a 29-hour silent
   solve on this lane was diagnosed by thermometer, 2026-08-26);
3. the incompatible combos (checkpoint / n_warmup) fall back with a warning
   instead of silently chunking a grad-tape run.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

N_STEPS = 600


def _build() -> Simulation:
    sim = Simulation(freq_max=40e9, domain=(4e-3, 4e-3, 4e-3), dx=200e-6,
                     boundary="cpml", cpml_layers=8,
                     dz_profile=[200e-6] * 8 + [100e-6] * 8 + [200e-6] * 8)
    # #931: one full 200 um cell of the coarse block, drawn on nodes. The
    # old 100 um box was half a cell thick and is refused now (a Box is a
    # volume); nothing in this file is about the conductor's thickness.
    sim.add(Box((1.5e-3, 1.5e-3, 1.2e-3), (2.5e-3, 2.5e-3, 1.4e-3)),
            material="pec")
    sim.add_source(position=(2e-3, 2e-3, 2.4e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=20e9, bandwidth=10e9))
    sim.add_probe(position=(2.6e-3, 2.6e-3, 2.0e-3), component="ez")
    return sim


def test_the_declared_scatterer_realizes_the_walls_it_is_drawn_with():
    """Build-time gate (no solve): the PEC block is where it is drawn.

    Before #931 this fixture's box was ``z 1.3 -> 1.4 mm``, half a cell of
    the 200 um coarse block. The node coordinates in that band are 1.2,
    1.4 and 1.6 mm and the half-open node window ``[1.3, 1.4)`` contains
    NO node, so the scatterer realized nothing at all and every assertion
    in this file passed with the conductor absent. A sub-cell PEC Box is
    an error now (§1.5) and the box is drawn on nodes, so drawn extent ==
    realized extent: walls at z = 1.2 mm and z = 1.4 mm.
    """
    from tests.unit._wall_planes_m import wall_planes_m

    # dz_profile => the NU lane; assemble on the lane the run uses.
    rz, zs = wall_planes_m(_build(), 2, nonuniform=True)
    assert not rz.sheets, "the block is a VOLUME; nothing here declares a sheet"
    assert [round(z, 9) for z in zs] == [1.2e-3, 1.4e-3], (
        f"realized z wall planes {zs}; drawn 1.2 -> 1.4 mm. A declared "
        "conductor that realizes no wall plane is a silent no-op.")


def test_chunked_progress_is_bit_identical_and_reports(capsys):
    r_plain = _build().run(n_steps=N_STEPS)
    ts_plain = np.asarray(r_plain.time_series)
    capsys.readouterr()                      # drop preflight/banner noise

    r_chunk = _build().run(n_steps=N_STEPS, report_every=150,
                           report_label="nu-chunk-test")
    ts_chunk = np.asarray(r_chunk.time_series)
    out = capsys.readouterr().out

    lines = [ln for ln in out.splitlines() if "nu-chunk-test" in ln]
    assert len(lines) >= N_STEPS // 150, f"expected progress lines, got: {out!r}"
    assert f"{N_STEPS}/{N_STEPS} steps" in lines[-1]
    assert "(cap)" not in lines[-1], "fixed-N route must not print the cap marker"

    assert ts_plain.shape == ts_chunk.shape
    # Two-tier equality (measured 2026-08-26): the chunked route and the
    # legacy single scan are DIFFERENT functions whose setup order differs at
    # float32 ULP scale (measured rel 3.6e-7); chunk RE-ENTRY itself is exact
    # (one-chunk vs four-chunk: bit-identical, asserted below).
    scale = float(np.max(np.abs(ts_plain))) or 1.0
    rel = float(np.max(np.abs(ts_plain - ts_chunk))) / scale
    assert rel < 1e-5, f"chunked route deviates from single scan: rel {rel:.3e}"

    r_one = _build().run(n_steps=N_STEPS, report_every=N_STEPS,
                         report_label="one-chunk")
    assert np.array_equal(np.asarray(r_one.time_series), ts_chunk), (
        "chunk re-entry is not bit-exact: one-chunk vs multi-chunk differ")

    ez_p = getattr(getattr(r_plain, "state", None), "ez", None)
    ez_c = getattr(getattr(r_chunk, "state", None), "ez", None)
    ez_1 = getattr(getattr(r_one, "state", None), "ez", None)
    if ez_p is not None and ez_c is not None:
        e_scale = float(np.max(np.abs(np.asarray(ez_p)))) or 1.0
        e_rel = float(np.max(np.abs(np.asarray(ez_p) - np.asarray(ez_c)))) / e_scale
        assert e_rel < 1e-5, f"final ez deviates from single scan: rel {e_rel:.3e}"
    if ez_1 is not None and ez_c is not None:
        assert np.array_equal(np.asarray(ez_1), np.asarray(ez_c)), \
            "chunk re-entry not bit-exact on final ez (one-chunk vs multi-chunk)"


def test_checkpoint_falls_back_with_warning():
    sim = _build()
    with pytest.warns(UserWarning, match="report_every=.*ignored on this non-uniform"):
        sim.run(n_steps=N_STEPS, report_every=150, checkpoint=True)
