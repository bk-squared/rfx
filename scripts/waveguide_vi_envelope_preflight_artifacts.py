"""The two zero-FDTD artifacts of the VI-envelope pre-declaration §7.

Both are milliseconds and both run BEFORE any case of the sweep.

    python scripts/waveguide_vi_envelope_preflight_artifacts.py [--out FILE]
        [--rungs 9 18 36 72]

1. **Mirror-covariance index audit.** Under the domain mirror the Yee grid maps
   E-node ``i -> nx-1-i`` and H-index ``k -> nx-2-k``. The audit asserts

       x_index_L + x_index_R = nx-1
       ref_x_L   + ref_x_R   = nx-1
       probe_x_L + probe_x_R = nx-1
       TFSF H-plane sum      = nx-2
       TFSF E-plane sum      = nx-1

   The last one **is expected to fail on shipped code at every rung** — it is
   ``nx``, because a ``+`` port corrects E at ``cfg.x_index`` and a ``-`` port at
   ``cfg.x_index + 1``. That failure is the §0 defect. It is RECORDED, not
   fixed: the sweep measures the shipped code and names its SHA, and merging a
   fix invalidates the antisymmetric clause of the envelope. This audit would
   have found the port-asymmetry arc with no FDTD at all.

   Exit status: nonzero if any of the four covariant sums breaks, and also
   nonzero if the E-plane sum is NOT ``nx`` — that would mean the library
   changed under the pre-declaration and the antisymmetric clause no longer
   describes the code being measured.

2. **Rasterization guard.** Asserts the realized grid at ``cpml_layers=8`` is
   (65,10,5) / (113,19,9) / (209,37,17) / (401,73,33) for N = 9/18/36/72, i.e.
   that ``dx`` came from the literal ladder and not from ``A_M/N`` (which lands
   one ULP low and moves the N=9 grid to (66,10,6) — a whole extra z node).
   **If this fails, stop.** Exit status nonzero, and the caller runs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.environ.get("RFX_WT") or str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp                                     # noqa: E402
import numpy as np                                          # noqa: E402

import tests._waveguide_chain_battery_fixture as F          # noqa: E402
import waveguide_vi_envelope_sweep as H                     # noqa: E402

# §7: the realized grid at cpml_layers=8, on the literal ladder.
EXPECTED_GRID_AT_8_LAYERS = {
    9: (65, 10, 5),
    18: (113, 19, 9),
    36: (209, 37, 17),
    72: (401, 73, 33),
}


def _r5_case(N: int) -> dict:
    return dict(case_id=f"audit_N{N}", N=N, r_lo=float(F.FREQS[0]) / H.FC_CONTINUOUS_HZ,
                r_hi=float(F.FREQS[-1]) / H.FC_CONTINUOUS_HZ, lock="none",
                freqs_hz=[float(v) for v in F.FREQS], f0_hz=float(F.F0_HZ),
                bandwidth=float(F.BANDWIDTH), dut="thru", precision="float32")


_INDEX_CACHE: dict[int, dict] = {}


def _port_indices(N: int) -> dict:
    """Build the sweep's own R5 geometry at cpml_layers=8 and read the two
    ports' COMPILED indices. No fields are stepped.

    The indices are read off ``_build_waveguide_port_config`` rather than
    re-derived from the grid: the point of the audit is what the solver will
    actually use, and a re-derivation would only re-state this file's own
    arithmetic. That build carries the discrete aperture mode solve, so it is
    seconds at N=9/18 and about 75 s / 145 s at N=36 / 72 on CPU — the
    pre-declaration's "milliseconds" holds for the arithmetic, not for the
    compile. Cached so the two artifacts pay for it once.
    """
    if N in _INDEX_CACHE:
        return _INDEX_CACHE[N]
    case = _r5_case(N)
    lay = H.layout(case["r_lo"])
    freqs = H.band_freqs(case, N)
    sim, meta = H._build(case, freqs, lay, cpml_layers=8)
    grid = sim._build_grid()
    fj = jnp.asarray(freqs)
    cfgs = [sim._build_waveguide_port_config(e, grid, fj, 16)
            for e in sim._waveguide_ports]
    nx = int(grid.shape[0])
    left, right = cfgs
    assert left.direction.startswith("+") and right.direction.startswith("-"), \
        (left.direction, right.direction)

    def h_plane(cfg):
        return cfg.x_index - 1 if cfg.direction.startswith("+") else cfg.x_index

    def e_plane(cfg):
        return cfg.x_index if cfg.direction.startswith("+") else cfg.x_index + 1

    out = dict(
        N=N, nx=nx, grid_shape=[int(v) for v in grid.shape],
        domain_x_m=meta["domain_x_m"], dx_m=meta["dx_m"],
        x_index=[int(left.x_index), int(right.x_index)],
        ref_x=[int(left.ref_x), int(right.ref_x)],
        probe_x=[int(left.probe_x), int(right.probe_x)],
        tfsf_h_plane=[int(h_plane(left)), int(h_plane(right))],
        tfsf_e_plane=[int(e_plane(left)), int(e_plane(right))],
    )
    _INDEX_CACHE[N] = out
    return out


def mirror_covariance_audit(rungs) -> dict:
    """Artifact 1. Returns the record; never edits the library."""
    rows, covariant_breaks, defect_absent = [], [], []
    for N in rungs:
        idx = _port_indices(N)
        nx = idx["nx"]
        checks = {
            "x_index": (sum(idx["x_index"]), nx - 1),
            "ref_x": (sum(idx["ref_x"]), nx - 1),
            "probe_x": (sum(idx["probe_x"]), nx - 1),
            "tfsf_h_plane": (sum(idx["tfsf_h_plane"]), nx - 2),
            "tfsf_e_plane": (sum(idx["tfsf_e_plane"]), nx - 1),
        }
        row = dict(idx)
        row["sums"] = {k: dict(got=g, mirror_covariant=e, ok=(g == e))
                       for k, (g, e) in checks.items()}
        for k, (g, e) in checks.items():
            if k == "tfsf_e_plane":
                # Expected to FAIL, and expected to fail in one specific way.
                row["e_plane_defect_matches_check2"] = (g == nx)
                if g != nx:
                    defect_absent.append((N, g, nx))
            elif g != e:
                covariant_breaks.append((N, k, g, e))
        rows.append(row)
    e_sums = {r["N"]: r["sums"]["tfsf_e_plane"]["got"] for r in rows}
    return dict(
        artifact="mirror_covariance_index_audit",
        predeclaration_section="7.1",
        rungs=list(rungs), rows=rows,
        covariant_breaks=covariant_breaks,
        e_plane_defect_absent_at=defect_absent,
        e_plane_sums=e_sums,
        expected_e_plane_failure=True,
        verdict=("RECORDED: E-plane sum = nx at every rung (the §0 defect), "
                 "every other index pair mirror-covariant"
                 if not covariant_breaks and not defect_absent else "BROKEN"),
        passed=(not covariant_breaks and not defect_absent),
    )


def rasterization_guard(rungs) -> dict:
    """Artifact 2. If this fails, STOP — the ladder is wrong before anything runs."""
    rows, breaks = [], []
    for N in rungs:
        dx = H.DX_BY_N[N]
        got_sweep = tuple(_port_indices(N)["grid_shape"])
        # Independent witness: the committed builder's own grid at the same rung
        # and the same absorber, built through a different code path.
        sim_fx = F._build("thru", dx, cpml_layers=8,
                          reference_planes=(F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M),
                          precision="float32")
        got_fixture = tuple(int(v) for v in sim_fx._build_grid().shape)
        want = EXPECTED_GRID_AT_8_LAYERS.get(N)
        b_cells = F.B_M / dx
        row = dict(N=N, dx_m=dx, dx_times_N_minus_a=dx * N - F.A_M,
                   b_over_dx=b_cells, b_over_dx_integral=abs(b_cells - round(b_cells)) < 1e-9,
                   d_ref_cells=F.D_REF_M / dx, d_probe_cells=F.D_PROBE_M / dx,
                   grid_shape_sweep=list(got_sweep), grid_shape_fixture=list(got_fixture),
                   expected=None if want is None else list(want))
        row["ok"] = (got_sweep == got_fixture
                     and (want is None or got_sweep == want)
                     and row["b_over_dx_integral"]
                     and abs(row["d_ref_cells"] - round(row["d_ref_cells"])) < 1e-9
                     and abs(row["d_probe_cells"] - round(row["d_probe_cells"])) < 1e-9)
        if not row["ok"]:
            breaks.append(row)
        rows.append(row)
    return dict(artifact="rasterization_guard", predeclaration_section="7.2",
                rungs=list(rungs), rows=rows, breaks=breaks,
                verdict="PASS" if not breaks else "STOP — the ladder is wrong",
                passed=not breaks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the record to this JSON")
    ap.add_argument("--rungs", nargs="*", type=int, default=[9, 18, 36, 72])
    a = ap.parse_args(argv)

    prov = H.assert_worktree_rfx()
    print("rfx provenance: " + json.dumps(prov), flush=True)

    guard = rasterization_guard(a.rungs)
    print("\n=== ARTIFACT 2 — rasterization guard (§7.2) ===")
    for r in guard["rows"]:
        print(f"  N={r['N']:3d} dx={r['dx_m']!r:12s} grid sweep={r['grid_shape_sweep']} "
              f"fixture={r['grid_shape_fixture']} expected={r['expected']} "
              f"b/dx={r['b_over_dx']:.1f} ok={r['ok']}")
    print(f"  verdict: {guard['verdict']}")

    audit = mirror_covariance_audit(a.rungs)
    print("\n=== ARTIFACT 1 — mirror-covariance index audit (§7.1) ===")
    for r in audit["rows"]:
        print(f"  N={r['N']:3d} nx={r['nx']:4d}")
        for k, s in r["sums"].items():
            flag = "ok" if s["ok"] else ("EXPECTED-FAIL" if k == "tfsf_e_plane" else "BREAK")
            print(f"      {k:14s} sum={s['got']:5d} mirror-covariant={s['mirror_covariant']:5d}  {flag}")
    print(f"  E-plane sums by rung: {audit['e_plane_sums']}")
    print(f"  verdict: {audit['verdict']}")

    record = dict(what="VI-envelope pre-declaration §7 zero-FDTD artifacts",
                  provenance=prov, rasterization_guard=guard,
                  mirror_covariance_audit=audit,
                  passed=bool(guard["passed"] and audit["passed"]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=1))
        print(f"\nrecord: {a.out}")

    if not guard["passed"]:
        print("\nFATAL: rasterization guard failed — STOP. Run nothing.")
        return 3
    if not audit["passed"]:
        print("\nFATAL: the index audit did not find the code the "
              "pre-declaration describes. Re-read §0 before measuring.")
        return 4
    print("\nBoth artifacts as pre-declared. The sweep may run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
