"""Local variants for the explicitly named physical structures."""
from __future__ import annotations
import json
from pathlib import Path
import sys

# The executable owns the monkeypatch state; do not import a second copy.
m = sys.modules['__main__'] if hasattr(sys.modules['__main__'], 'Recorder') else __import__('sweep_driver')
np, jnp = m.np, m.jnp


def measure_patch(args, out):
    oracle = m.load_module('tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py')
    sim = oracle._build(n=3, pad_h=10, cpml=args.layers)
    row = dict(num_periods=150, n_cells_per_h=3, pad_h=10,
               grid=m.grid_record(sim._build_grid()), preflight_text=m.preflight(sim, out),
               port_apertures=[], source='committed interior Ez dipole')
    rec = m.Recorder(out, args.scale, args.dry)
    rec.install()
    try:
        result = sim.run(num_periods=150, skip_preflight=True)
        ts = np.asarray(result.time_series)
        row.update(settling_db=oracle._settling_db(ts),
                   worst_rate_per_step=max(oracle._late_time_log_rate_per_step(ts)),
                   rates_per_step=oracle._late_time_log_rate_per_step(ts), status='complete')
    except m.DryCaptured:
        row['status'] = 'build-only: stopped before stepping'
    row['solves'] = rec.calls
    return row


def build_waveguide(layers):
    battery = m.load_module('tests/oracle/test_waveguide_port_validation_battery.py')
    # DESIGN.md names WR-90 explicitly; the cited battery has a 40x20 mm default.
    # 0.254 mm divides both requested WR-90 dimensions (90 by 40 cells).
    battery.DOMAIN = (.12, .02286, .01016)
    sim = battery._build_sim(np.linspace(7e9, 12e9, 101), dx=.254e-3,
                             cpml_layers=layers, waveform='modulated_gaussian')
    sim._waveguide_ports = sim._waveguide_ports[:1]
    return sim


def measure_waveguide(args, out):
    from rfx.sources.waveguide_port import extract_waveguide_s11, settling_db_from_port_records
    sim = build_waveguide(args.layers)
    grid = sim._build_grid()
    row = dict(grid=m.grid_record(grid), preflight_text=m.preflight(sim, out),
               far_port='removed', extraction='extract_waveguide_s11 on returned single-drive modal records; no clean-guide subtraction',
               num_periods=40, declared_cross_section_m=[.02286, .01016])
    rec = m.Recorder(out, args.scale, args.dry)
    rec.install()
    try:
        sim.run(num_periods=40, compute_s_params=False, skip_preflight=True)
        cfg = rec.last_result.waveguide_ports[0]
        s11 = np.asarray(extract_waveguide_s11(cfg))
        db = 20*np.log10(np.maximum(abs(s11), 1e-30))
        row.update(freqs_Hz=np.asarray(cfg.freqs), S11=s11, S11_db=db,
                   max_S11_db=float(db.max()), max_abs_S=float(abs(s11).max()),
                   settling_db=settling_db_from_port_records([cfg]),
                   aperture_m=[float(cfg.a), float(cfg.b)], cutoff_Hz=float(cfg.f_cutoff), status='complete')
        np.savez_compressed(out / 'modal_records.npz',
                            **{k: np.asarray(getattr(cfg, k)) for k in ('v_probe_t','v_ref_t','i_probe_t','i_ref_t','v_inc_t')})
    except m.DryCaptured:
        row['status'] = 'build-only: stopped before stepping'
    row['solves'] = rec.calls
    return row


PLANE_STEPS = 4096
PLANE_DX = .0025
PLANE_FREQS = np.linspace(.2e9, 6e9, 291)


def build_plane(layers, ref_extra=None):
    # Explicit plane source, same soft-Ez injection and clean-reference subtraction
    # as the named oracle. Every transverse node receives the same waveform.
    if ref_extra is None:
        length = .06
        bx = 'cpml'
    else:
        # Causal stencil bound, stronger than the continuum group-delay bound:
        # the shortest wall round trip in cells is greater than PLANE_STEPS.
        length = (PLANE_STEPS + 64 + ref_extra)*PLANE_DX
        bx = 'pec'
        layers = 0
    sim = m.Simulation(freq_max=6e9, dx=PLANE_DX,
        domain=(length, 4*PLANE_DX, 4*PLANE_DX), cpml_layers=layers,
        boundary=m.BoundarySpec(x=bx, y='periodic', z='periodic'))
    grid = sim._build_grid()
    coords = m.coords_from_uniform_grid(grid)
    cx, cy, cz = [(n-1)//2 for n in grid.shape]
    pulse = m.GaussianPulse(f0=4e9, bandwidth=1.0, cutoff=5)
    for j in range(grid.ny):
        for k in range(grid.nz):
            sim.add_source((float(coords.x[cx]),float(coords.y[j]),float(coords.z[k])),
                           component='ez', amplitude_kind='field', waveform=pulse)
    for i in (cx+3, cx-3):
        sim.add_probe((float(coords.x[i]),float(coords.y[cy]),float(coords.z[cz])), component='ez')
    meta = dict(grid=m.grid_record(grid), source_node=cx, source_plane_m=float(coords.x[cx]),
                source_aperture_nodes=[grid.ny, grid.nz], probes_x_nodes=[cx+3,cx-3],
                pulse=m.serial(pulse), record_steps=PLANE_STEPS, window_s=[0, (PLANE_STEPS-1)*grid.dt],
                window_definition='full-record rectangular DFT', transverse_boundaries='periodic',
                shortest_reference_wall_roundtrip_steps=min(2*cx-3, 2*(grid.nx-1-cx)-3) if ref_extra is not None else None)
    if ref_extra is not None:
        assert meta['shortest_reference_wall_roundtrip_steps'] > PLANE_STEPS
    return sim, meta


def spectrum(ts, dt):
    t = np.arange(len(ts))*dt
    return np.exp(-2j*np.pi*PLANE_FREQS[:,None]*t[None,:]) @ np.asarray(ts, dtype=float)*dt


def crossing_frequencies(db, threshold):
    cross = []
    for i in range(len(db)-1):
        a, b = db[i]-threshold, db[i+1]-threshold
        if a == 0:
            cross.append(float(PLANE_FREQS[i]))
        elif a*b < 0:
            cross.append(float(PLANE_FREQS[i]+(PLANE_FREQS[i+1]-PLANE_FREQS[i])*(-a)/(b-a)))
    return cross


def plane_run(sim, meta, args, dest):
    dest.mkdir()
    meta['preflight_text'] = m.preflight(sim, dest)
    rec = m.Recorder(dest, args.scale, args.dry)
    rec.install()
    try:
        result = sim.run(n_steps=PLANE_STEPS, compute_s_params=False, skip_preflight=True)
        ts = np.asarray(result.time_series)
        meta['status'] = 'complete'
    except m.DryCaptured:
        ts = None
        meta['status'] = 'build-only: stopped before stepping'
    meta['solves'] = rec.calls
    m.write_json(dest/'result.json', meta)
    return ts, meta


def measure_plane(args, out):
    sim, meta = build_plane(args.layers)
    if args.dry:
        _, row = plane_run(sim, meta, args, out/'small')
        # Reference build only: no solve or compilation is needed here.
        _, row['reference_grid'] = build_plane(0, 0)
        _, row['reference_check_grid'] = build_plane(0, 128)
        return row
    refdir = m.ROOT/'raw'/'plane_reference'
    if not refdir.exists():
        refdir.mkdir()
        refs, metas = [], []
        for extra in (0, 128):
            ref, rmeta = build_plane(0, extra)
            ts, rmeta = plane_run(ref, rmeta, args, refdir/str(extra))
            refs.append(ts)
            metas.append(rmeta)
        dt = metas[0]['grid']['dt_s']
        fref = spectrum(refs[0][:,0], dt)
        fdiff = spectrum(refs[1][:,0]-refs[0][:,0], dt)
        null_db = 20*np.log10(np.maximum(abs(fdiff/fref), 1e-30))
        # Independent time-domain known echo, measured through the same DFT window.
        delay = 64
        echo = np.zeros(len(refs[0]))
        echo[delay:] = .001*refs[0][:-delay,0]
        known = spectrum(echo, dt)/fref
        known_db = 20*np.log10(np.maximum(abs(known), 1e-30))
        calibration = dict(reference=metas, frequency_Hz=PLANE_FREQS,
            null_db=null_db, max_null_db=float(null_db.max()), known_echo_amplitude=.001,
            known_echo_delay_steps=delay, known_echo_db=known_db,
            known_echo_max_abs_error_db=float(abs(known_db+60).max()),
            min_incident_spectrum_relative=float((abs(fref)/abs(fref).max()).min()))
        np.savez_compressed(refdir/'reference.npz', time_series=refs[0], dt_s=dt)
        m.write_json(refdir/'calibration.json', calibration)
    calibration = json.loads((refdir/'calibration.json').read_text())
    with np.load(refdir/'reference.npz') as z:
        ref_ts, dt = z['time_series'], float(z['dt_s'])
    ts, row = plane_run(sim, meta, args, out/'small')
    assert dt == row['grid']['dt_s']
    r = spectrum(ts[:,0]-ref_ts[:,0], dt)/spectrum(ref_ts[:,0], dt)
    rmirror = spectrum(ts[:,1]-ref_ts[:,1], dt)/spectrum(ref_ts[:,1], dt)
    db = 20*np.log10(np.maximum(abs(r), 1e-30))
    row.update(freqs_Hz=PLANE_FREQS, R=r, R_db=db, max_R_db=float(db.max()),
               reflection_definition='small two-ended normal-incidence plane-wave box minus clean reference, divided by incident reference at probe +3',
               crossings_minus40_Hz=crossing_frequencies(db, -40),
               crossings_minus60_Hz=crossing_frequencies(db, -60),
               reference_calibration=calibration,
               mirror_probe_max_abs_R_difference=float(abs(r-rmirror).max()),
               mirror_probe_R=rmirror)
    csv = m.ROOT/'results'/'plane'
    csv.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv/f'{args.scale:g}_{args.layers}.csv',
               np.column_stack([PLANE_FREQS, r.real, r.imag, db, rmirror.real, rmirror.imag]),
               delimiter=',', header='frequency_Hz,R_real,R_imag,R_dB,R_mirror_real,R_mirror_imag', comments='')
    return row


def measure(args, out):
    return {'patch': measure_patch, 'waveguide': measure_waveguide, 'plane': measure_plane}[args.rig](args, out)
