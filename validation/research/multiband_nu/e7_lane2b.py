"""Lane 2b — closing the recorded limits of E6 (in-plane design-variable AD) and AD-Q (z-stack L2).

Frozen rules: docs/design_notes/20260915_nu_lane2b_predeclaration.md.
Arms (one attempt each, run in this order, each committed before the next):
  y        E6's fixture rotated onto the y axis: band WIDTH w and POSITION y_c, L1 and L2s.
  pos      E6's x fixture with the probe on the SOURCE side of the strip, so the position
           control x_c is not a lead/tail cancellation; order test on x_c.
  zsmooth  the E6 smooth spectral observable (physical-time Hann + comb integral) applied to
           the AD-Q z stack thickness controls — the arm AD-Q recorded as NOT verified.
Judges: adq_designvar.fit_order / fd_budget / selfcheck, imported verbatim; lane-2b order
rule = R0 in [0.9, 1.1] and R1 >= 1.8 (lower side only; the upper edge has no
gradient-correctness meaning — AD-Q note, reviewer note).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp
import numpy as np
import rfx
from rfx.core.yee import MaterialArrays
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform

from . import adq_designvar as adq
from . import e4_diff_stackup as e4
from . import e6_inplane_designvar as e6

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'validation/research/multiband_nu/results'
assert str(Path(rfx.__file__).resolve()).startswith(str(ROOT) + '/'), rfx.__file__

HS = adq.HS
R0_BAND = (0.9, 1.1)
R1_MIN = 1.8
CANCEL_MIN = 0.5                     # pos arm eligibility: |g_lead + g_tail| / (|g_lead| + |g_tail|)
FLOOR_HS = tuple(np.linspace(1e-5, 1e-4, 64))
Z_COMB_DF = 0.25e9
Z_COMB_K = 10
Z_TW_FRACTION = 0.8


# ---------------------------------------------------------------- y-rotated E6 fixture
MAP_Y = e6.MeshMap.from_builder(e6.X_EDGES, e6.X_SIZES, e6.X_PROTECTED, e6.CAP, e6.PIN)
J_L, J_R = int(MAP_Y.edge_nodes[1]), int(MAP_Y.edge_nodes[2])
NX_Y = e6.NY                                              # 12 uniform x cells (the old y count)
DOMAIN_XY_Y = (NX_Y * e6.DYZ, float(np.sum(MAP_Y.cells0)))
SRC_Y = (6, 8, 6)
PRB_Y = (6, J_R + 3, e6.K_S + 2)
GRID_SHAPE_Y = (NX_Y + 1, MAP_Y.cells0.size + 1, e6.NZ + 1)


def strip_masks_y(shape, j_l=J_L, j_r=J_R, k_s=e6.K_S):
    mx = np.zeros(shape, bool); my = np.zeros(shape, bool); mz = np.zeros(shape, bool)
    my[:, j_l:j_r, k_s] = True          # tangential ey edges between the band-edge nodes
    mx[:, j_l:j_r + 1, k_s] = True      # ex edges on the nodes
    return (jnp.asarray(mx), jnp.asarray(my), jnp.asarray(mz))


MASKS_Y = strip_masks_y(GRID_SHAPE_Y)


def _grid_y(cells):
    return make_nonuniform_grid(DOMAIN_XY_Y, np.full(e6.NZ, e6.DYZ), e6.DYZ, cpml_layers=0, dy_profile=cells)


def _run_y(cells, n_steps, checkpoint_every=None, stop_dt=False):
    grid = _grid_y(cells)
    if stop_dt:
        grid = grid._replace(dt=jax.lax.stop_gradient(grid.dt))
    dt = grid.dt
    wf = e6._waveform(n_steps, dt)
    out = run_nonuniform(grid, e6._vacuum(grid), n_steps, pec_edge_masks=MASKS_Y,
                         sources=[(*SRC_Y, 'ez', wf)], probes=[(*PRB_Y, 'ez')], checkpoint_every=checkpoint_every)
    return out['time_series'][:, 0], dt


def _l2s_from(ts, dt, n2, t_w, comb):
    t = jnp.arange(n2, dtype=jnp.float32) * dt
    tw = jnp.float32(t_w)
    hann = jnp.where(t < tw, 0.5 * (1 - jnp.cos(2 * jnp.pi * t / tw)), 0.0).astype(jnp.float32)
    phase = 2 * jnp.pi * jnp.asarray(comb, jnp.float32)[:, None] * t[None, :]
    wts = hann * ts
    re = dt * jnp.sum(wts[None, :] * jnp.cos(phase), axis=1)
    im = dt * jnp.sum(wts[None, :] * jnp.sin(phase), axis=1)
    return jnp.sum(re ** 2 + im ** 2)


DT0_Y = e6.dt_f64(MAP_Y.cells0)
T_W_Y = e6.T_W_FRACTION * e6.N2 * DT0_Y


def l1_y(cells, stop_dt=False):
    ts, _ = _run_y(cells, e6.N1, stop_dt=stop_dt)
    return jnp.sum(ts ** 2)


def l2s_y(cells, stop_dt=False):
    ts, dt = _run_y(cells, e6.N2, e6.CHECKPOINT_EVERY, stop_dt)
    return _l2s_from(ts, dt, e6.N2, T_W_Y, e6.COMB)


# ---------------------------------------------------------------- pos: x fixture, probe on the source side
PRB_POS = (e6.I_L - 3, 6, e6.K_S + 2)


def _run_pos(cells, n_steps, checkpoint_every=None, stop_dt=False):
    grid = e6._grid(cells)
    if stop_dt:
        grid = grid._replace(dt=jax.lax.stop_gradient(grid.dt))
    dt = grid.dt
    wf = e6._waveform(n_steps, dt)
    out = run_nonuniform(grid, e6._vacuum(grid), n_steps, pec_edge_masks=e6.MASKS,
                         sources=[(*e6.SRC_IJK, 'ez', wf)], probes=[(*PRB_POS, 'ez')], checkpoint_every=checkpoint_every)
    return out['time_series'][:, 0], dt


def l1_pos(cells, stop_dt=False):
    ts, _ = _run_pos(cells, e6.N1, stop_dt=stop_dt)
    return jnp.sum(ts ** 2)


def l2s_pos(cells, stop_dt=False):
    ts, dt = _run_pos(cells, e6.N2, e6.CHECKPOINT_EVERY, stop_dt)
    return _l2s_from(ts, dt, e6.N2, e6.T_W, e6.COMB)


# ---------------------------------------------------------------- zsmooth: AD-Q stack, smooth spectral loss
F_NOM_Z = e4.f_res(e4.PARAMS0)
COMB_Z = np.asarray([F_NOM_Z + k * Z_COMB_DF for k in range(-Z_COMB_K, Z_COMB_K + 1)])
DT0_Z = e4.dt_f64(e4.PARAMS0[0])
T_W_Z = Z_TW_FRACTION * e4.N2_STEPS * DT0_Z


def l2s_z_cells(c, stop_dt=False):
    dz, ec = c[:64], c[64:]
    en = e4.dual_eps_from_cells(dz, ec)
    if stop_dt:
        from rfx.nonuniform import make_nonuniform_grid as _mk
        grid = _mk(domain_xy=e4.DOMAIN_XY, dz_profile=dz, dx=e4.DXY, cpml_layers=0)
        grid = grid._replace(dt=jax.lax.stop_gradient(grid.dt))
        i1, i2, j = e4._layout(e4.DOMAIN_XY, e4.DXY)
        wf = e4._drive(F_NOM_Z)(e4.N2_STEPS, grid.dt)
        out = run_nonuniform(grid, e4._materials(grid, en), e4.N2_STEPS,
                             sources=[(i1, j, e4.K_SRC, 'ey', wf), (i2, j, e4.K_SRC, 'ey', wf)],
                             probes=[(i1, j, e4.K_PRB_L2, 'ey')], checkpoint_every=e4.CHECKPOINT_EVERY)
        ts, dt = out['time_series'][:, 0], grid.dt
    else:
        ts, dt = e4._run(dz, en, e4.N2_STEPS, e4._drive(F_NOM_Z), e4.K_PRB_L2, e4.DOMAIN_XY, e4.DXY, e4.CHECKPOINT_EVERY)
    return _l2s_from(ts, dt, e4.N2_STEPS, T_W_Z, COMB_Z)


# ---------------------------------------------------------------- shared judging
def provenance():
    return {'rfx_file': rfx.__file__, 'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--short', '--', 'rfx', 'validation', 'tests'], cwd=ROOT, text=True).strip()),
            'jax': jax.__version__, 'platform': jax.devices()[0].platform}


def lane_order_verdict(order):
    if 'R1' not in order:
        return 'INCONCLUSIVE'
    r0, r1 = order['R0']['slope'], order['R1']['slope']
    return 'HELD' if (R0_BAND[0] <= r0 <= R0_BAND[1] and r1 >= R1_MIN) else 'FIRED'


def measure_floor(loss, x, v):
    ys = np.array([float(loss(jnp.asarray(x + h * v, jnp.float32))) for h in FLOOR_HS])
    hs = np.asarray(FLOOR_HS)
    res2 = ys - np.polyval(np.polyfit(hs, ys, 2), hs)
    res3 = ys - np.polyval(np.polyfit(hs, ys, 3), hs)
    sigma = float(np.sqrt(np.sum(res2 ** 2) / (len(hs) - 3)))
    return {'sigma': sigma,
            'reliable': bool(np.sqrt(np.mean(res3 ** 2)) >= 0.9 * np.sqrt(np.mean(res2 ** 2))),
            'hs': hs.tolist(), 'losses': ys.tolist(),
            'quadratic_residuals': res2.tolist(), 'cubic_residuals': res3.tolist()}


def judge_control(loss, x, v, ad_rel, scale, loss0, use_floor):
    pts = adq.losses_along(loss, x, v, HS)
    sigma = None
    floor = None
    if use_floor:
        floor = measure_floor(loss, x, v)
        sigma = floor['sigma']
        floor['sigma_in_ulp'] = sigma / adq.ulp(loss0)
    order = adq.fit_order(pts, loss0, ad_rel, sigma=sigma)
    fd = adq.fd_budget(pts, ad_rel, scale, sigma=sigma)
    return {'order': order, 'lane_order_verdict': lane_order_verdict(order), 'fd': fd, 'floor': floor,
            'points': pts}


def controls_arm(name, loss_fns, cells_of, jac, x0, names, scales, tied_cells, use_floor, revert_control=None):
    """Generic: loss_fns = {label: (loss_cells(cells, stop_dt), n_steps)}; x0 = nominal delta/param vector;
    jac = d cells / d x (n_cells x n_x); names/scales per control; tied_cells = indices of the dt-setting set."""
    out = {'arm': name, 'provenance': provenance(), 'selfcheck': adq.selfcheck(), 'losses': {},
           'nominal': np.asarray(x0, float).tolist(), 'jacobian': np.asarray(jac, float).tolist(),
           'control_names': list(names), 'control_scales': list(scales),
           'tied_cells': np.asarray(tied_cells).tolist(), 'hs': list(HS)}
    assert out['selfcheck']['all_pass'], out['selfcheck']
    for i in range(len(names)):
        assert float(np.ptp(jac[tied_cells, i])) == 0.0, (names[i], 'induced direction not constant on the tied set')
    for label, (loss_cells, _n) in loss_fns.items():
        loss = jax.jit(lambda q, lc=loss_cells: lc(cells_of(q)))
        loss0 = float(loss(jnp.asarray(x0, jnp.float32)))
        g_cell = np.asarray(jax.jit(jax.grad(loss_cells))(jnp.asarray(cells_of(jnp.asarray(x0, jnp.float32)))), float)
        g_x = jac.T @ g_cell
        rec = {'loss0': loss0, 'loss_ulp': adq.ulp(loss0), 'g_cell': g_cell.tolist(),
               'g_x_chain': g_x.tolist(), 'n_steps': _n, 'controls': {}}
        for i, nm in enumerate(names):
            # x0 may be longer than names (zsmooth: full P0 with the four thickness controls in front);
            # the step direction lives in x0's space, the control index is shared.
            v = np.zeros(len(x0)); v[i] = scales[i]
            ad_rel = float(g_x[i] * scales[i])
            r = judge_control(loss, np.asarray(x0, float), v, ad_rel, scales[i], loss0, use_floor)
            r['ad_relative'] = ad_rel
            rec['controls'][nm] = r
            print(name, label, nm, r['lane_order_verdict'], r['order'].get('R1', {}).get('slope'), r['fd']['verdict'], flush=True)
        if revert_control is not None:
            i = names.index(revert_control)
            loss_nodt = jax.jit(lambda q, lc=loss_cells: lc(cells_of(q), stop_dt=True))
            g_nodt = np.asarray(jax.jit(jax.grad(lambda c, lc=loss_cells: lc(c, stop_dt=True)))(jnp.asarray(cells_of(jnp.asarray(x0, jnp.float32)))), float)
            gx_nodt = jac.T @ g_nodt
            v = np.zeros(len(x0)); v[i] = scales[i]
            ad_nodt = float(gx_nodt[i] * scales[i])
            r = judge_control(loss, np.asarray(x0, float), v, ad_nodt, scales[i], loss0, use_floor)
            forward_nodt = float(loss_nodt(jnp.asarray(x0, jnp.float32)))
            rec['revert'] = {**r, 'control': revert_control,
                             'forward_identical': forward_nodt == loss0, 'forward_loss': forward_nodt,
                             'dt_share': float((g_x[i] - gx_nodt[i]) / g_x[i]) if g_x[i] else None,
                             'g_cell': g_nodt.tolist(), 'g_x_chain': gx_nodt.tolist(),
                             'ad_relative': ad_nodt, 'fd_verdict': r['fd']['verdict']}
            print(name, label, 'REVERT', revert_control, r['lane_order_verdict'], r['fd']['verdict'], 'dt_share', rec['revert']['dt_share'], flush=True)
        rec['coverage'] = adq.coverage(g_cell, jac)
        held = [i for i, nm in enumerate(names) if rec['controls'][nm]['lane_order_verdict'] == 'HELD' and rec['controls'][nm]['fd']['verdict'] == 'HELD']
        rec['coverage_verified'] = adq.coverage(g_cell, jac[:, held]) if held else None
        out['losses'][label] = rec
    return out


def arm_y():
    jac = MAP_Y.jacobian_seg() @ e6.A
    tied = np.flatnonzero(MAP_Y.cells0 == MAP_Y.cells0.min())
    cells_of = lambda d: MAP_Y.cells(jnp.asarray(e6.SEG_LEN0, jnp.float32) + jnp.asarray(e6.A, jnp.float32) @ d)
    out = controls_arm('y', {'l1': (l1_y, e6.N1), 'l2s': (l2s_y, e6.N2)}, cells_of, jac, np.zeros(2), ('w', 'y_c'),
                       (e6.W0, e6.W0), tied, use_floor=True, revert_control='w')
    out['fixture'] = {'domain_xy': DOMAIN_XY_Y, 'src': SRC_Y, 'prb': PRB_Y, 'grid_shape': GRID_SHAPE_Y, 'band_nodes': (J_L, J_R), 'tied_cells': tied.tolist()}
    return out


def arm_pos():
    jac = e6.J_PARAM
    tied = np.flatnonzero(e6.MAP.cells0 == e6.MAP.cells0.min())
    lead = e6.MAP.seg_id == 0; tail = e6.MAP.seg_id == 2
    # eligibility pre-check (declared): the position gradient must not be a lead/tail cancellation
    pre = {}
    for label, lc in (('l1', l1_pos), ('l2s', l2s_pos)):
        g_cell = np.asarray(jax.jit(jax.grad(lc))(e6.CELLS0_F32), float)
        gl = float(jac[lead, 1] @ g_cell[lead]); gt = float(jac[tail, 1] @ g_cell[tail])
        pre[label] = {'g_lead': gl, 'g_tail': gt, 'cancellation_ratio': abs(gl + gt) / (abs(gl) + abs(gt)) if (abs(gl) + abs(gt)) else None}
        print('pos precheck', label, pre[label], flush=True)
    out = controls_arm('pos', {'l1': (l1_pos, e6.N1), 'l2s': (l2s_pos, e6.N2)}, e6.cells_of, jac, np.zeros(2), ('w', 'x_c'),
                       (e6.W0, e6.W0), tied, use_floor=True, revert_control=None)
    out['precheck'] = pre
    out['fixture'] = {'prb': PRB_POS, 'src': e6.SRC_IJK, 'cancel_min': CANCEL_MIN}
    for label in ('l1', 'l2s'):
        if pre[label]['cancellation_ratio'] is None or pre[label]['cancellation_ratio'] < CANCEL_MIN:
            out['losses'][label]['controls']['x_c']['lane_order_verdict'] = 'INCONCLUSIVE (precheck)'
    return out


def arm_zsmooth():
    p0 = jnp.asarray(adq.P0, jnp.float32)
    jac = np.asarray(jax.jacfwd(adq.stack_cells)(p0), float)
    c0 = np.asarray(adq.stack_cells(p0), float)
    tied = np.flatnonzero(c0[:64] == c0[:64].min())
    names = adq.NAMES[:4]; scales = tuple(float(s) for s in adq.P0[:4])
    out = controls_arm('zsmooth', {'l2s': (l2s_z_cells, e4.N2_STEPS)}, adq.stack_cells, jac[:, :4], adq.P0, names, scales, tied,
                       use_floor=True, revert_control='h_thin')
    out['fixture'] = {'f_nom': F_NOM_Z, 'comb_hz': COMB_Z.tolist(), 't_w': T_W_Z, 'n2': e4.N2_STEPS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('y', 'pos', 'zsmooth'), required=True)
    ap.add_argument('--out', type=Path, help='Separate output path for a new, explicitly declared witness')
    a = ap.parse_args()
    out_path = a.out if a.out is not None else RESULTS / f'e7_{a.arm}.json'
    if out_path.exists():
        raise FileExistsError(out_path)
    claim = out_path.with_suffix('.started')
    with claim.open('x') as f:
        f.write(f'{time.time()}\n')
    t = time.monotonic()
    res = {'y': arm_y, 'pos': arm_pos, 'zsmooth': arm_zsmooth}[a.arm]()
    res['elapsed_s'] = time.monotonic() - t
    out_path.write_text(json.dumps(res, indent=1, allow_nan=False, default=float) + '\n')
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
