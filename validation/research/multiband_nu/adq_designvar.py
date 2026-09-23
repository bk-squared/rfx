"""NU design-variable order/error-budget witness, not a 15% accuracy claim.

Frozen rules: docs/design_notes/20260913_nu_ad_designvar_predeclaration.md.
Fixed-length layer edits and per-layer eps are tested by Taylor order;
coverage and every fired/inconclusive result are reported, never widened.
At a cellwise CFL tie AD is the equal split, not the derivative of an
arbitrary per-cell optimizer update. Production behavior is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import rfx

from . import e4_diff_stackup as e4
from . import w7_accuracy_ad as w7

ROOT = Path(__file__).resolve().parents[3]
HS = tuple(2.0 ** -k for k in range(17, 2, -1))
RES_HS = (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04)
N_QUANTA = 32
RHO_BAND = (2.0, 8.0)          # Richardson h^2 ratio 4, factor-2 model-validity band
COUNTS = (20, 4, 20, 20)
P0 = np.array([0.014, 0.002, 0.014, 0.014, 4.3, 3., 4.3, 1.])
NAMES = tuple(f'{kind}_{layer}' for kind in ('h', 'eps')
              for layer in ('core_left', 'thin', 'core_right', 'air'))
A = np.array([[1., -.5, -1., -.5], [0., 1., 0., 0.],
              [-1., -.5, 1., -.5], [0., 0., 0., 1.]])


def stack_cells(p):
    p = jnp.asarray(p)
    heights = jnp.asarray(P0[:4], p.dtype) + jnp.asarray(A, p.dtype) @ (p[:4] - jnp.asarray(P0[:4], p.dtype))
    dz = jnp.concatenate([jnp.full((n,), heights[i] / n) for i, n in enumerate(COUNTS)])
    ec = jnp.concatenate([jnp.full((n,), p[i + 4]) for i, n in enumerate(COUNTS)])
    return jnp.concatenate([dz, ec])


def node_map(p):
    c = stack_cells(p)
    return e4.dual_eps_from_cells(c[:64], c[64:])


def ulp(x):
    return float(abs(np.spacing(np.float32(abs(x)))))


def fit_order(points, loss0, ad, sigma=None):
    """Choose one longest eligible monotone run, without slope selection."""
    eligible = []
    for p in points:
        h, lp, lm = p['h'], p['loss_plus'], p['loss_minus']
        r0 = abs(lp - loss0)
        r1 = abs(lp - loss0 - h * ad)
        q = sigma if sigma is not None else max(ulp(lp), ulp(lm), ulp(loss0))
        eligible.append((r0, r1, r1 > N_QUANTA * q and
                         abs(lp + lm - 2 * loss0) <= .25 * abs(lp - lm)))
    runs, run = [], []
    for i, (_, r1, ok) in enumerate(eligible):
        if not ok or (run and r1 <= eligible[run[-1]][1]):
            if run:
                runs.append(run)
            run = []
        if ok:
            run.append(i)
    if run:
        runs.append(run)
    chosen = max(runs, key=lambda r: (len(r), -r[0]), default=[])
    out = {'r0': [e[0] for e in eligible], 'r1': [e[1] for e in eligible],
           'eligible': [bool(e[2]) for e in eligible], 'indices': chosen,
           'points': len(chosen), 'window': [points[i]['h'] for i in (chosen[0], chosen[-1])] if chosen else [],
           'verdict': 'INCONCLUSIVE'}
    if len(chosen) < 4:
        return out
    x = np.log([points[i]['h'] for i in chosen])
    for j, label in enumerate(('R0', 'R1')):
        y = np.log([eligible[i][j] for i in chosen])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        out[label] = {'slope': float(slope), 'intercept': float(intercept),
                      'residuals': residual.tolist(), 'rms': float(np.sqrt(np.mean(residual ** 2))),
                      'slope_stderr': float(np.sqrt(np.sum(residual ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))}
    out['verdict'] = 'HELD' if .9 <= out['R0']['slope'] <= 1.1 and 1.8 <= out['R1']['slope'] <= 2.2 else 'FIRED'
    return out


def fd_budget(points, ad, scale=1., sigma=None):
    """Central FD against its OWN estimated error, per step (note: FD error budget).

    HS is ascending with ratio exactly two. T(h) = 4|D(h)-D(h/2)|/3 (Richardson,
    D = g + c h^2); Q(h) = (ulp(L+) + ulp(L-))/(2h). A step is informative only
    if (i) the Richardson model is in its asymptotic regime there, checked by the
    centered ratio rho = (D(2h)-D(h))/(D(h)-D(h/2)) lying in RHO_BAND around its
    h^2 value 4, and (ii) the acceptance band 3B/|g| beats the legacy 15 %.
    Outside either, the step is INCONCLUSIVE: the FD reference cannot judge AD there.
    """
    ds = [(p['loss_plus'] - p['loss_minus']) / (2 * p['h']) for p in points]
    n = len(points)
    rows = []
    for i, p in enumerate(points):
        neighbor = i - 1 if i else 1
        fine, coarse = (ds[neighbor], ds[i]) if i else (ds[i], ds[neighbor])
        rich = (4 * fine - coarse) / 3
        trunc = abs(ds[i] - rich)
        rounding = (sigma / p['h'] if sigma is not None else
                    (ulp(p['loss_plus']) + ulp(p['loss_minus'])) / (2 * p['h']))
        bar = trunc + rounding
        error = abs(ad - ds[i])
        rel_bar = bar / abs(ad) if ad else None
        if 0 < i < n - 1 and ds[i] != ds[i - 1]:
            rho = (ds[i + 1] - ds[i]) / (ds[i] - ds[i - 1])
        else:
            rho = None
        asymptotic = rho is not None and RHO_BAND[0] <= rho <= RHO_BAND[1]
        if not ad or not asymptotic or 3 * rel_bar > .15:
            verdict = 'INCONCLUSIVE'
        else:
            verdict = 'HELD' if error <= 3 * bar else 'FIRED'
        rel = error / abs(ds[i]) if ds[i] else None
        rows.append({**p, 'fd': ds[i], 'richardson': rich, 'roundoff': rounding,
                     'truncation': trunc, 'bar': bar, 'bar_relative': rel_bar,
                     'richardson_rho': rho, 'asymptotic': bool(asymptotic),
                     'error': error, 'error_over_bar': error / bar if bar else None,
                     'error_over_3bar': error / (3 * bar) if bar else None,
                     'fd_physical': ds[i] / scale, 'bar_physical': bar / scale,
                     'roundoff_physical': rounding / scale, 'truncation_physical': trunc / scale,
                     'legacy_relative_error': rel,
                     'legacy_smoke_held': bool(rel is not None and rel <= .15 and np.sign(ad) == np.sign(ds[i])),
                     'verdict': verdict})
    informative = [i for i, r in enumerate(rows) if r['verdict'] != 'INCONCLUSIVE']
    verdict = ('FIRED' if any(r['verdict'] == 'FIRED' for r in rows) else
               'HELD' if informative else 'INCONCLUSIVE')
    narrowest = min(informative, key=lambda i: rows[i]['bar_relative']) if informative else None
    return {'steps': rows, 'verdict': verdict, 'narrowest_index': narrowest,
            'n_informative': len(informative)}


def coverage(g, matrix):
    g, matrix = np.asarray(g, float), np.asarray(matrix, float)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=0)
        matrix = matrix[:, norms > 0] / norms[norms > 0]
    if matrix.size:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.sum(s > 1e-12 * s[0]))
        proj = u[:, :rank] @ (u[:, :rank].T @ g)
    else:
        rank, proj = 0, np.zeros_like(g)
    norm = float(np.linalg.norm(g))
    return {'rank': rank, 'norm': norm, 'projected_norm': float(np.linalg.norm(proj)),
            'residual_norm': float(np.linalg.norm(g - proj)),
            'fraction': float(np.linalg.norm(proj) / norm) if norm else None}


def selfcheck():
    """Synthetic judges check, run before every arm (note: replay and tree checks).

    Losses are rounded to float32 exactly as the solver's are. A quadratic loss
    with the TRUE slope must give R0 ~ 1, R1 ~ 2 and an FD HELD; the same loss
    with a slope 10 % wrong must give R1 slope ~ 1 (FIRED). If either fails the
    judges cannot tell a right gradient from a wrong one, and no arm may run.
    """
    a, b, l0 = 0.37, 2.9, 0.1306

    def pts(slope_err):
        out = []
        for h in HS:
            lp = np.float32(l0 + a * h + b * h * h + 0.5 * h ** 3)
            lm = np.float32(l0 - a * h + b * h * h - 0.5 * h ** 3)
            out.append({'h': h, 'loss_plus': float(lp), 'loss_minus': float(lm)})
        return out, float(np.float32(l0)), a * (1 + slope_err)
    res = {}
    for label, err, want in (('true_quadratic', 0.0, 'HELD'), ('wrong_by_10pct', 0.1, 'FIRED')):
        p, loss0, ad = pts(err)
        order = fit_order(p, loss0, ad)
        res[label] = {'order_verdict': order['verdict'],
                      'r1_slope': order.get('R1', {}).get('slope'),
                      'r0_slope': order.get('R0', {}).get('slope'),
                      'fd_verdict': fd_budget(p, ad)['verdict'], 'expected_order': want}
    ok = (res['true_quadratic']['order_verdict'] == 'HELD'
          and res['true_quadratic']['fd_verdict'] == 'HELD'
          and res['wrong_by_10pct']['order_verdict'] == 'FIRED')
    res['all_pass'] = bool(ok)
    return res


def losses_along(loss, x, v, hs):
    return [{'h': h, 'loss_plus': float(loss(jnp.asarray(x + h * v, jnp.float32))),
             'loss_minus': float(loss(jnp.asarray(x - h * v, jnp.float32)))} for h in hs]


def measure_stack(arm):
    f_nom = e4.f_res(e4.PARAMS0)

    def cell_loss(c):
        dz, ec = c[:64], c[64:]
        en = e4.dual_eps_from_cells(dz, ec)
        if arm == 'stack_l1':
            return e4.l1_from_dz_eps(dz, en)
        ts, dt = e4._run(dz, en, e4.N2_STEPS, e4._drive(f_nom), e4.K_PRB_L2,
                         e4.DOMAIN_XY, e4.DXY, e4.CHECKPOINT_EVERY)
        phase = 2 * jnp.pi * jnp.float32(f_nom) * jnp.arange(e4.N2_STEPS, dtype=jnp.float32) * dt
        return (dt * jnp.sum(ts * jnp.cos(phase))) ** 2 + (dt * jnp.sum(ts * jnp.sin(phase))) ** 2

    loss = jax.jit(lambda p: cell_loss(stack_cells(p)))
    p0 = jnp.asarray(P0, jnp.float32)
    c0 = np.asarray(stack_cells(p0), float)
    g = np.asarray(jax.jit(jax.grad(cell_loss))(jnp.asarray(c0, jnp.float32)), float)
    gp = np.asarray(jax.jit(jax.grad(lambda p: cell_loss(stack_cells(p))))(p0), float)
    jac = np.asarray(jax.jacfwd(stack_cells)(p0), float)
    njac = np.asarray(jax.jacfwd(node_map)(p0), float)
    loss0 = float(loss(p0))
    tied = np.flatnonzero(c0[:64] == c0[:64].min())
    out = {'loss0': loss0, 'loss_ulp': ulp(loss0), 'f_nom': f_nom,
           'n_steps': e4.N1_STEPS if arm == 'stack_l1' else e4.N2_STEPS,
           'params0': P0.tolist(), 'cell0': c0.tolist(), 'node_eps0': np.asarray(node_map(p0)).tolist(),
           'jacobian': jac.tolist(), 'node_eps_jacobian': njac.tolist(),
           'g_cell': g.tolist(), 'g_param': gp.tolist(), 'chain_rule': (jac.T @ g).tolist(),
           'tied_cells_z': tied.tolist(), 'n_tied_xyz': [60, 6, len(tied)], 'directions': {}}
    held = []
    for i, name in enumerate(NAMES):
        v = np.eye(8)[i] * P0[i]
        points = losses_along(loss, P0, v, HS)
        ad = gp[i] * P0[i]
        order, fd = fit_order(points, loss0, ad), fd_budget(points, ad, P0[i])
        spread = float(np.ptp(jac[tied, i]))
        assert spread == 0.
        out['directions'][name] = {'induced_direction': jac[:, i].tolist(), 'tie_spread_xyz': [0., 0., spread],
                                   'ad_relative': float(ad), 'ad_physical': float(gp[i]),
                                   'order': order, 'fd': fd}
        if order['verdict'] == fd['verdict'] == 'HELD':
            held.append(i)
        print(arm, name, order.get('R0', {}).get('slope'), order.get('R1', {}).get('slope'),
              order['verdict'], fd['verdict'], flush=True)
    out['coverage_raw_m_eps'] = coverage(g, jac)
    out['coverage_verified_raw_m_eps'] = coverage(g, jac[:, held])
    out['verified_controls'] = [NAMES[i] for i in held]
    out['coverage_dz'] = coverage(g[:64], jac[:64])
    out['coverage_eps_cell'] = coverage(g[64:], jac[64:])
    out['coverage_dimensionless'] = coverage(g * c0, jac / c0[:, None])
    return out


def _l1_stop_dt(dz, eps_node):
    """E4's L1 with the CFL time step cut out of the gradient (revert-proof only).

    Identical forward value to e4.l1_from_dz_eps; only d(dt)/d(dz) is zeroed, so
    the gradient loses exactly the dt path -- the path the tied-minimum cells
    carry and the one the per-cell gate never saw.
    """
    from rfx.nonuniform import make_nonuniform_grid, run_nonuniform
    grid = make_nonuniform_grid(domain_xy=e4.DOMAIN_XY, dz_profile=dz, dx=e4.DXY, cpml_layers=0)
    grid = grid._replace(dt=jax.lax.stop_gradient(grid.dt))
    i1, i2, j = e4._layout(e4.DOMAIN_XY, e4.DXY)
    wf = e4._w5_waveform(e4.N1_STEPS)
    out = run_nonuniform(grid, e4._materials(grid, eps_node), e4.N1_STEPS,
                         sources=[(i1, j, e4.K_SRC, "ey", wf), (i2, j, e4.K_SRC, "ey", wf)],
                         probes=[(i1, j, e4.K_PRB_L1, "ey")])
    return jnp.sum(out["time_series"][:, 0] ** 2)


def measure_revert():
    """Revert-proof: true L1 losses along each thickness control, judged against
    the gradient WITHOUT its dt path. dt depends only on the minimum z cell (the
    thin layer), so h_thin must FIRE and the core/air controls must be unchanged
    from the true-gradient arm. If h_thin HOLDS here, the Taylor gate has no power
    against the dt path on this fixture, and the main h_thin verdict cannot be
    read as verifying it."""
    def true_cell_loss(c):
        return e4.l1_from_dz_eps(c[:64], e4.dual_eps_from_cells(c[:64], c[64:]))

    def nodt_cell_loss(c):
        return _l1_stop_dt(c[:64], e4.dual_eps_from_cells(c[:64], c[64:]))

    loss = jax.jit(lambda q: true_cell_loss(stack_cells(q)))
    p0 = jnp.asarray(P0, jnp.float32)
    g_true = np.asarray(jax.jit(jax.grad(lambda q: true_cell_loss(stack_cells(q))))(p0), float)
    g_nodt = np.asarray(jax.jit(jax.grad(lambda q: nodt_cell_loss(stack_cells(q))))(p0), float)
    fwd_equal = float(loss(p0)) == float(jax.jit(lambda q: nodt_cell_loss(stack_cells(q)))(p0))
    loss0 = float(loss(p0))
    out = {'loss0': loss0, 'forward_values_identical': bool(fwd_equal),
           'g_param_true': g_true.tolist(), 'g_param_nodt': g_nodt.tolist(),
           'dt_path_share': [float((t - n) / t) if t else None for t, n in zip(g_true, g_nodt)],
           'directions': {}}
    for i, name in enumerate(NAMES[:4]):
        v = np.eye(8)[i] * P0[i]
        points = losses_along(loss, P0, v, HS)
        ad = g_nodt[i] * P0[i]
        out['directions'][name] = {'ad_relative_nodt': float(ad),
                                   'ad_relative_true': float(g_true[i] * P0[i]),
                                   'order': fit_order(points, loss0, ad),
                                   'fd': fd_budget(points, ad, P0[i])}
        print('revert', name, out['directions'][name]['order']['verdict'],
              out['directions'][name]['fd']['verdict'], flush=True)
    return out


FLOOR_HS = tuple(np.linspace(1e-5, 1e-4, 64))


def measure_l2_floor():
    """Second attempt, L2 thickness only (note: 'Second attempt ... declared BEFORE').

    Measures the L2 loss's float32 noise floor per thickness control on a fresh
    fine grid, then re-judges the STORED first-attempt ladders with that floor in
    place of one ulp. Bands unchanged. First attempt stays recorded as is.
    """
    first = json.loads((ROOT / 'validation/research/multiband_nu/results/adq_stack_l2.json').read_text())
    assert 'instrument_error' not in first
    f_nom = e4.f_res(e4.PARAMS0)

    def cell_loss(c):
        dz, ec = c[:64], c[64:]
        en = e4.dual_eps_from_cells(dz, ec)
        ts, dt = e4._run(dz, en, e4.N2_STEPS, e4._drive(f_nom), e4.K_PRB_L2,
                         e4.DOMAIN_XY, e4.DXY, e4.CHECKPOINT_EVERY)
        phase = 2 * jnp.pi * jnp.float32(f_nom) * jnp.arange(e4.N2_STEPS, dtype=jnp.float32) * dt
        return (dt * jnp.sum(ts * jnp.cos(phase))) ** 2 + (dt * jnp.sum(ts * jnp.sin(phase))) ** 2

    loss = jax.jit(lambda q: cell_loss(stack_cells(q)))
    loss0 = first['loss0']
    assert float(loss(jnp.asarray(P0, jnp.float32))) == loss0, 'L2 not reproducible bit for bit'
    out = {'first_attempt_key': 'adq_stack_l2', 'loss0': loss0, 'floor_hs': list(FLOOR_HS),
           'directions': {}}
    for i, name in enumerate(NAMES[:4]):
        v = np.eye(8)[i] * P0[i]
        ys = np.array([float(loss(jnp.asarray(P0 + h * v, jnp.float32))) for h in FLOOR_HS])
        hs = np.asarray(FLOOR_HS)
        res2 = ys - np.polyval(np.polyfit(hs, ys, 2), hs)
        res3 = ys - np.polyval(np.polyfit(hs, ys, 3), hs)
        sigma = float(np.sqrt(np.sum(res2 ** 2) / (len(hs) - 3)))
        rms2, rms3 = float(np.sqrt(np.mean(res2 ** 2))), float(np.sqrt(np.mean(res3 ** 2)))
        reliable = rms3 >= 0.9 * rms2
        r1 = first['directions'][name]
        pts = [{'h': s['h'], 'loss_plus': s['loss_plus'], 'loss_minus': s['loss_minus']}
               for s in r1['fd']['steps']]
        ad = r1['ad_relative']
        order = fit_order(pts, loss0, ad, sigma=sigma)
        fd = fd_budget(pts, ad, P0[i], sigma=sigma)
        elig = [k for k, e in enumerate(order['eligible']) if e]
        bound = min((order['r1'][k] / (pts[k]['h'] * abs(ad)) for k in elig), default=None)
        out['directions'][name] = {
            'sigma': sigma, 'sigma_in_ulp': sigma / ulp(loss0), 'rms_quadratic': rms2,
            'rms_cubic': rms3, 'sigma_reliable': bool(reliable), 'floor_losses': ys.tolist(),
            'order': order, 'fd': fd,
            'lower_side_diagnostic': (None if 'R1' not in order else bool(order['R1']['slope'] >= 1.8)),
            'empirical_rel_error_bound': bound,
            'first_attempt_order': r1['order']['verdict'], 'first_attempt_fd': r1['fd']['verdict']}
        print('l2_floor', name, f"sigma={sigma/ulp(loss0):.1f}ulp reliable={reliable}",
              order['verdict'], order.get('R1', {}).get('slope'), fd['verdict'], f"bound={bound}", flush=True)
    return out


def measure_resolution():
    base = [np.asarray(d, float) for d in w7.a3_profiles()]
    sizes = [len(d) for d in base]
    cuts = np.cumsum(sizes)[:-1]
    raw = w7._ad3_loss(*base, w7.AD_N_STEPS)
    loss = jax.jit(lambda flat: raw(tuple(jnp.split(flat, cuts.tolist()))))
    # Preserve host builder coordinates for FD, as in the original instrument.
    x = np.concatenate(base)
    g = np.asarray(jax.jit(jax.grad(lambda flat: raw(tuple(jnp.split(flat, cuts.tolist())))))(jnp.asarray(x, jnp.float32)), float)
    l0 = float(loss(jnp.asarray(x, jnp.float32)))
    out = {'loss0': l0, 'loss_ulp': ulp(l0), 'profiles': [d.tolist() for d in base],
           'g_cell': g.tolist(), 'axes': {}, 'legacy_view': '5% dominance, 15% smoke; equal-split CFL subgradient'}
    start = 0
    for ai, axis in enumerate('xyz'):
        d, grad = base[ai], g[start:start + sizes[ai]]
        d32 = d.astype(np.float32)
        tied = np.isclose(d32, d32.min(), rtol=w7.AD5_TIE_REL, atol=0.)
        per_cell = []
        for k in range(sizes[ai]):
            v = np.zeros_like(x)
            v[start + k] = d[k]
            per_cell.append(losses_along(loss, x, v, RES_HS))
        rows = []
        for hi, h in enumerate(RES_HS):
            lp = np.array([ps[hi]['loss_plus'] for ps in per_cell])
            lm = np.array([ps[hi]['loss_minus'] for ps in per_cell])
            fd = (lp - lm) / (2 * h * d)
            plus, minus = (lp - l0) / (h * d), (l0 - lm) / (h * d)
            q = ulp(l0) / (2 * h * d)
            quanta = abs(fd) / q
            dom = ~tied & (abs(fd) > .05 * abs(fd[~tied]).max())
            rel = abs(grad - fd) / np.maximum(abs(fd), 1e-300)
            worst = int(np.flatnonzero(dom)[np.argmax(rel[dom])])
            resolved = dom & (quanta >= 50)
            rows.append({'h': h, 'loss_plus': lp.tolist(), 'loss_minus': lm.tolist(), 'g_fd': fd.tolist(),
                         'fd_plus': plus.tolist(), 'fd_minus': minus.tolist(), 'quantum': q.tolist(),
                         'quanta': quanta.tolist(), 'dominant_cells': np.flatnonzero(dom).tolist(),
                         'worst_cell': worst, 'worst_relative_error': float(rel[worst]),
                         'worst_quanta': float(quanta[worst]), 'worst_error_quanta': float(abs(grad[worst] - fd[worst]) / q[worst]),
                         'worst_resolved_error': float(rel[resolved].max()) if resolved.any() else None,
                         'legacy_smoke_held': bool(rel[dom].max() <= .15 and np.all(np.sign(fd[dom]) == np.sign(grad[dom]))),
                         'split_model': (plus + (minus - plus) / tied.sum()).tolist()})
        v = np.zeros_like(x)
        v[start:start + sizes[ai]] = d
        points = losses_along(loss, x, v, HS) if axis in 'yz' else []
        orders = {str(k): fit_order(per_cell[k], l0, grad[k] * d[k]) for k in range(len(d)) if not tied[k]} if axis in 'yz' else {}
        out['axes'][axis] = {'g_ad': grad.tolist(), 'n_tied': int(tied.sum()),
                             'tied_cells': np.flatnonzero(tied).tolist(), 'gmax_all': float(abs(grad).max()),
                             'gmax_nontied': float(abs(grad[~tied]).max()), 'per_h': rows,
                             'cell_orders': orders, 'axis_points': points,
                             'axis_ad': float(g @ v),
                             'axis_order': fit_order(points, l0, float(g @ v)) if points else None}
        print('resolution', axis, [(r['h'], r['worst_relative_error'], r['worst_quanta']) for r in rows], flush=True)
        start += sizes[ai]
    old = json.loads((ROOT / 'validation/research/multiband_nu/results/w7_accuracy_ad.json').read_text())
    conditions = {}
    for axis in 'yz':
        a = out['axes'][axis]
        rows = {p['h']: p for p in a['per_h']}
        conditions[axis] = {'attempt1_old': old['ad3']['axes'][axis]['worst_dominant_rel_err'],
                            'attempt1_new': rows[.001]['worst_relative_error'],
                            'under_50_quanta': rows[.001]['worst_quanta'] < 50,
                            'improves_at_01': rows[.01]['worst_relative_error'] < rows[.001]['worst_relative_error'],
                            'improves_at_04': rows[.04]['worst_relative_error'] < rows[.001]['worst_relative_error'],
                            'axis_taylor_holds': a['axis_order']['verdict'] == 'HELD'}
    out['resolution_conditions'] = conditions
    out['resolution_verdict'] = 'SUPPORTED' if all(all(v[k] for k in ('under_50_quanta', 'improves_at_01', 'improves_at_04', 'axis_taylor_holds')) for v in conditions.values()) else 'UNCONFIRMED'
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('stack_l1', 'stack_l2', 'resolution', 'revert_l1_nodt', 'stack_l2_floor'), required=True)
    args = parser.parse_args()
    assert str(Path(rfx.__file__).resolve()).startswith(str(ROOT) + '/')
    assert all(d.platform == 'cpu' for d in jax.devices())
    out = ROOT / f'validation/research/multiband_nu/results/adq_{args.arm}.json'
    # Refuse to overwrite evidence or rerun an already started arm.
    if out.exists():
        raise FileExistsError(out)
    claim = out.with_suffix('.started')
    with claim.open('x') as f:
        f.write(f'{time.time()}\n')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    provenance = {'rfx_file': rfx.__file__, 'git_sha': sha, 'platform': 'cpu',
                  'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  'arm': args.arm, 'hs': HS, 'resolution_hs': RES_HS, 'n_quanta': N_QUANTA,
                  'rho_band': RHO_BAND}
    t = time.monotonic()
    try:
        sc = selfcheck()
        provenance['selfcheck'] = sc
        if not sc['all_pass']:
            raise RuntimeError(f'judge selfcheck failed: {sc}')
        if args.arm == 'resolution':
            result = measure_resolution()
        elif args.arm == 'revert_l1_nodt':
            result = measure_revert()
        elif args.arm == 'stack_l2_floor':
            result = measure_l2_floor()
        else:
            result = measure_stack(args.arm)
        provenance.update(result)
    except Exception as exc:
        provenance['instrument_error'] = repr(exc)
        raise
    finally:
        provenance['elapsed_seconds'] = time.monotonic() - t
        out.write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
