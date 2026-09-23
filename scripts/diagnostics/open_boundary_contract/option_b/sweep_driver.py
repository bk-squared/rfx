"""Option B measurements from the Addendum 1 source export."""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mpl_config')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
sys.path.insert(0, str(SRC))

import numpy as np
import jax
import jax.numpy as jnp
from rfx import Simulation, Box
from rfx.boundaries import cpml
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx import simulation as lowlevel
from rfx.sources.sources import GaussianPulse
from rfx.sources.msl_port import msl_port_from_entry, msl_cross_section_span, msl_probe_x_coords_n
from rfx.geometry.rasterize_grid import coords_from_uniform_grid

SCALES = (0, 0.01, 0.03, 0.1, 0.3, 1, 3)
LAYERS = (4, 8, 16)
EXPECTED_SHA = 'e7f7e02704fd46ea7e21f127b19fc81cb66d6148'
ORIGINAL_PROFILE = cpml._cpml_profile
ORIGINAL_RUN = lowlevel.run
ORIGINAL_CORE = lowlevel.make_core_step
FACES = tuple(a + '_' + side for a in 'xyz' for side in ('lo', 'hi'))
EPS0, MU0 = 8.8541878128e-12, 1.25663706212e-6


def serial(x):
    if dataclasses.is_dataclass(x):
        return {f.name: serial(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if hasattr(x, '_asdict'):
        return serial(x._asdict())
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serial(v) for v in x]
    if isinstance(x, (np.ndarray, jax.Array)):
        return serial(np.asarray(x).tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def write_json(path, data):
    Path(path).write_text(json.dumps(serial(data), indent=2, allow_nan=False) + '\n')


def load_module(rel):
    spec = importlib.util.spec_from_file_location('_option_b_' + Path(rel).stem, SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def scaled_profile(scale):
    tree = ast.parse(inspect.getsource(ORIGINAL_PROFILE))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'alpha' for t in node.targets):
            assert ast.unparse(node.value) == '0.05 * (1.0 - rho)'
            node.value = ast.BinOp(node.value, ast.Mult(), ast.Constant(float(scale)))
            changed += 1
    assert changed == 1
    ast.fix_missing_locations(tree)
    ns = dict(ORIGINAL_PROFILE.__globals__)
    exec(compile(tree, str(__file__) + '::scaled_profile', 'exec'), ns)
    # Preserve the shipped np.where evaluation at s=0, including its warning.
    return ns['_cpml_profile']


def grid_record(grid):
    coords = coords_from_uniform_grid(grid)
    pads = {f: int(getattr(grid, 'pad_' + f)) for f in FACES}
    axes = {}
    for a in 'xyz':
        v = np.asarray(getattr(coords, a), dtype=float)
        p, q = pads[a + '_lo'], pads[a + '_hi']
        axes[a] = dict(nodes=len(v), interior_cells=len(v)-p-q-1,
                       face_lo_m=float(v[p]), face_hi_m=float(v[len(v)-1-q]),
                       allocated_lo_m=float(v[0]), allocated_hi_m=float(v[-1]))
    return dict(shape=list(grid.shape), dx_m=float(grid.dx), dt_s=float(grid.dt), pads=pads, axes=axes)


class DryCaptured(Exception):
    pass


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def preflight(sim, out):
    stream = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
        issues = sim.preflight(strict=False)
        for issue in issues:
            print(str(issue))
    text = stream.getvalue()
    (out / 'preflight.txt').write_text(text)
    return text


def energy_witness(energy, post_start, dt):
    energy = np.asarray(energy, dtype=float)
    if post_start >= len(energy):
        return dict(status='truncation-suspect', reason='source has not ended', post_source_start_step=post_start)
    peak = float(np.max(energy[post_start:]))
    end = float(energy[-1])
    tail = float(np.max(energy[max(post_start, int(.95*len(energy))):]))
    ratio = 10*np.log10(max(end, 1e-300)/max(peak, 1e-300))
    tail_ratio = 10*np.log10(max(tail, 1e-300)/max(peak, 1e-300))
    finite = bool(np.isfinite(energy).all())
    return dict(definition='0.5*dx^3*sum_interior(eps0*eps_r*sum(E_i^2)+mu0*mu_r*sum(H_i^2)); same-step Yee samples',
                source_end_definition='last source-table sample > 1e-6 of that source peak, then one step',
                post_source_start_step=post_start, post_source_start_s=post_start*dt,
                end_energy_J=end, post_source_peak_energy_J=peak, end_vs_post_peak_db=ratio,
                last_5pct_max_energy_J=tail, last_5pct_vs_post_peak_db=tail_ratio,
                status='below -40 dB' if finite and peak > 0 and ratio < -40 else 'truncation-suspect')


class Recorder:
    def __init__(self, out, scale, dry):
        self.out, self.scale, self.dry = out, scale, dry
        self.calls = []
        self.active = None
    def install(self):
        lowlevel.run = self.run
        lowlevel.make_core_step = self.core
    def core(self, ctx):
        record = self.active
        record['received_grid'] = grid_record(ctx.grid)
        record['field_dtype'] = str(ctx.materials.eps_r.dtype)
        if ctx.use_cpml:
            params = ctx.cpml_params
            arrays = {face + '_' + field: np.asarray(getattr(getattr(params, face), field))
                      for face in FACES for field in cpml.CPMLParams._fields}
            np.savez_compressed(self.out / f'cpml_{len(self.calls)-1:02d}.npz', **arrays)
            record['cpml'] = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                      min=float(v.min()) if v.size else None,
                                      max=float(v.max()) if v.size else None,
                                      finite=bool(np.isfinite(v).all()),
                                      sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert all(np.all(v == 1) for k, v in arrays.items() if k.endswith('_kappa'))
        if ctx.pec_edge_masks is not None:
            edges = [np.asarray(e) for e in ctx.pec_edge_masks]
            record['pec_edges'] = {a: dict(total=int(e.sum()),
                x_lo=int(e[:ctx.grid.pad_x_lo].sum()),
                x_hi=int(e[-ctx.grid.pad_x_hi:].sum()) if ctx.grid.pad_x_hi else 0)
                for a, e in zip('xyz', edges)}
            if getattr(self, 'msl_trace_sample', None) is not None:
                j, k = self.msl_trace_sample
                record['trace_x_edge_line'] = edges[0][:, j, k].astype(int).tolist()
                assert edges[0][:ctx.grid.pad_x_lo, j, k].all()
                assert edges[0][-ctx.grid.pad_x_hi:-1, j, k].all()
        record['realized_source_apertures'] = serial(ctx.src_meta)
        record['realized_waveguide_apertures'] = [serial({k: v for k, v in cfg._asdict().items()
            if k in ('normal_axis', 'direction', 'u_lo', 'u_hi', 'v_lo', 'v_hi', 'x_index', 'ref_x', 'probe_x', 'a', 'b', 'f_cutoff', 'source_x_m', 'reference_x_m', 'probe_x_m')}) for cfg in ctx.waveguide_meta]
        print('REALIZED ' + json.dumps(serial(record)), flush=True)
        if self.dry:
            raise DryCaptured()
        kernel = ORIGINAL_CORE(ctx)
        sl = tuple(slice(getattr(ctx.grid, 'pad_'+a+'_lo'),
                         getattr(ctx.grid, 'n'+a)-getattr(ctx.grid, 'pad_'+a+'_hi')) for a in 'xyz')
        eps = ctx.materials.eps_r[sl] * EPS0
        mu = ctx.materials.mu_r[sl] * MU0
        dv = float(ctx.grid.dx)**3 * .5
        def measured(carry, step, sources, magnetic):
            nxt, probe, extras = kernel(carry, step, sources, magnetic)
            state = nxt['fdtd']
            e2 = sum(getattr(state, 'e'+a)[sl]**2 for a in 'xyz')
            h2 = sum(getattr(state, 'h'+a)[sl]**2 for a in 'xyz')
            energy = dv * jnp.sum(eps*e2 + mu*h2)
            return nxt, jnp.concatenate((probe, jnp.reshape(energy, (1,)))), extras
        return measured
    def run(self, grid, materials, n_steps, *args, **kwargs):
        row = dict(n_steps=int(n_steps), source_end_step=0)
        sources = kwargs.get('sources') or []
        for src in sources:
            v = np.asarray(src.waveform)
            active = np.flatnonzero(abs(v) > 1e-6 * np.max(abs(v)))
            if len(active):
                row['source_end_step'] = max(row['source_end_step'], int(active[-1]+1))
        for cfg in kwargs.get('waveguide_ports') or []:
            for v in (np.asarray(cfg.e_inc_table), np.asarray(cfg.h_inc_table)):
                active = np.flatnonzero(abs(v) > 1e-6*np.max(abs(v)))
                if len(active):
                    row['source_end_step'] = max(row['source_end_step'], int(active[-1]+1))
        self.calls.append(row)
        self.active = row
        start = time.monotonic()
        result = ORIGINAL_RUN(grid, materials, n_steps, *args, **kwargs)
        self.last_result = result
        data = np.asarray(result.time_series)
        ts, energy = data[:, :-1], data[:, -1]
        np.savez_compressed(self.out / f'witness_{len(self.calls)-1:02d}.npz',
                            time_series=ts, energy_J=energy, dt_s=np.asarray(grid.dt))
        row.update(energy_witness=energy_witness(energy, row['source_end_step'], grid.dt), wall_s=time.monotonic()-start)
        return result._replace(time_series=result.time_series[:, :-1])


def build_msl(layers, single, low):
    source = (SRC / 'scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py').read_text()
    names = {'EPS_R','H_SUB','W_TRACE','L_LINE','PORT_MARGIN','DX','F_MAX','LX','LY','LZ','N_FREQS','NUM_PERIODS'}
    nodes = [n for n in ast.parse(source).body if
             (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in names for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == '_build_sim')]
    class Variant(Simulation):
        def __init__(self, **kw):
            kw['cpml_layers'] = layers
            super().__init__(**kw)
        def add_msl_port(self, **kw):
            if single and self._msl_ports:
                return self
            kw['terminates'] = ()
            if low:
                kw['waveform'] = GaussianPulse(f0=2.5e9, bandwidth=.8, cutoff=5)
            return super().add_msl_port(**kw)
    ns = dict(Simulation=Variant, Box=Box, Boundary=Boundary, BoundarySpec=BoundarySpec)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(__file__)+'::msl_builder', 'exec'), ns)
    return ns['_build_sim']()


def msl_geometry(sim):
    grid = sim._build_grid()
    ports = []
    for pe in sim._resolve_msl_probe_entries(grid):
        mp = msl_port_from_entry(pe)
        span = msl_cross_section_span(grid, mp)
        ports.append(dict(name=pe.name, terminates=pe.terminates, span=span,
                          probe_x_m=msl_probe_x_coords_n(grid, mp, n_probes=pe.n_probes,
                            n_offset_cells=pe.n_probe_offset, n_spacing_cells=pe.n_probe_spacing)))
    record = dict(grid=grid_record(grid), ports=ports, declared_domain_m=sim._domain)
    return record


def measure_msl(args, out):
    result = {}
    freqs = np.linspace(.5e9 if args.rig == 'msl_low' else 3e9, 4.5e9, 81)
    for label, single in [('two_port', False), ('one_port', True)]:
        dest = out / label
        dest.mkdir()
        sim = build_msl(args.layers, single, args.rig == 'msl_low')
        geom = msl_geometry(sim)
        rec = Recorder(dest, args.scale, args.dry)
        span = geom['ports'][0]['span']
        rec.msl_trace_sample = (span['w_centre'], span['n_hi'])
        rec.install()
        row = dict(realized=geom, preflight_text=preflight(sim, dest), far_port='removed' if single else 'present')
        try:
            s = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs), num_periods=12,
                                         enforce_passivity=False, report_every=None)
            smat = np.asarray(s.S)
            np.savez_compressed(dest / 'sparams.npz', S=smat, freqs_Hz=np.asarray(s.freqs))
            row.update(freqs_Hz=np.asarray(s.freqs), S=smat, settling_db=serial(s.settling_db),
                       max_abs_S=float(abs(smat).max()), max_column_power=float((abs(smat)**2).sum(axis=0).max()),
                       S11_db=20*np.log10(np.maximum(abs(smat[0, 0]), 1e-30)),
                       max_S11_db=float(20*np.log10(max(float(abs(smat[0, 0]).max()), 1e-30))))
            if not single:
                row['max_abs_S12_minus_S21'] = float(abs(smat[0, 1]-smat[1, 0]).max())
            row['status'] = 'complete'
        except DryCaptured:
            row['status'] = 'build-only: stopped before stepping'
        row['solves'] = rec.calls
        write_json(dest / 'result.json', row)
        result[label] = row
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rig', choices=['msl', 'msl_low', 'waveguide', 'patch', 'plane'])
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    jax.config.update('jax_enable_x64', False)
    if not args.dry:
        assert all(d.platform == 'gpu' for d in jax.devices()), jax.devices()
    from verify_job_sources import verify
    verify(ROOT)
    sha = (SRC / 'PROVENANCE.txt').read_text().strip()
    assert sha == EXPECTED_SHA, sha
    stem = f'{args.scale:g}_{args.layers}'
    out = ROOT / ('dry_2' if args.dry else 'raw') / args.rig / stem
    out.mkdir(parents=True)
    run_file = ROOT / 'jobs_2' / ('msl' if args.rig.startswith('msl') else args.rig) / 'run_id.txt'
    row = dict(rig=args.rig, scale=args.scale, layers=args.layers, commit=sha,
               run_id=None if args.dry else run_file.read_text().strip(), dtype='float32',
               devices=list(map(str, jax.devices())), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               alpha_max_S_per_m=.05*args.scale, f_alpha_formula_Hz=.05*args.scale/(2*np.pi*EPS0))
    cpml._cpml_profile = scaled_profile(args.scale)
    start = time.monotonic()
    with (out / 'run.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                if args.rig.startswith('msl'):
                    row['measurements'] = measure_msl(args, out)
                else:
                    from extra_rigs import measure
                    row['measurements'] = measure(args, out)
                row['status'] = 'build-only' if args.dry else 'complete'
            except Exception as exc:
                row.update(status='STOP', exception=repr(exc), traceback=traceback.format_exc())
                print(row['traceback'], flush=True)
            row['wall_s'] = time.monotonic()-start
            write_json(out / 'result.json', row)
            if not args.dry:
                dest = ROOT / 'results' / args.rig
                dest.mkdir(parents=True, exist_ok=True)
                write_json(dest / (stem+'.json'), row)
            print('ARM_FINAL ' + json.dumps(serial({k: v for k, v in row.items() if k != 'measurements'})), flush=True)
    return 0 if row['status'] != 'STOP' else 1


if __name__ == '__main__':
    raise SystemExit(main())
