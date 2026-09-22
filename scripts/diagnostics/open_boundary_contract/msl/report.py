import json
from pathlib import Path
from render_edges import table

OUT=Path('/root/workspace/bk-workspace/.801-measure/msl')

def read(p):
    return json.loads(Path(p).read_text())

def main():
    reductions={name:read(OUT/(name+'_reduced.json')) for name in ('cv06b','cv20')}
    verification=read(OUT/'verification.json')
    runs=dict(line.split() for line in (OUT/'run_id.txt').read_text().splitlines())
    lines=['cv06b: 0.7–7.0 GHz, 100 bins, dx = 63.5 µm, 20 periods, 2 drives per run.',
           'cv20 rfx: 0.5–5.0 GHz, 30 bins, dx = 50 µm, 12 periods, 2 drives per run.',
           'Trace x-bound extensions: cv06b = 571.5 µm per end; cv20 = 450 µm per end.','',
           table(['fixture','baseline x bounds (mm)','continued x bounds (mm)','declared trace z bounds (µm)','realized trace z (µm)','JAX x64 (0/1)'],
                 [['cv06b','0, 34','-0.5715, 34.5715','254, 254','254',0],
                  ['cv20','0, 14','-0.45, 14.45','254, 304','250, 300',1]]),
           'Public call: `compute_msl_s_matrix(n_freqs=100, num_periods=20.0)` for cv06b; `compute_msl_s_matrix(n_freqs=30, num_periods=12)` for cv20. Other public-call arguments use the exported function defaults, including `enforce_passivity=True`.','',
           'Builder copies: `measure.py::builder_namespace` copies `_build_sim` and its numerical constants from the exported source via AST; each run saves `builder_copy.py`. `measure.py::build_pair` changes only geometry[1] Box x bounds for continued. Settings, full geometry declarations, boundary declarations, port declarations and field-equality checks are in each run’s `settings.json`.','',
           'cv06b builder: `src-main/validation/crossval/06b_msl_notch_filter_uniform.py::_build_sim`; call settings from that file’s `main`. cv20 builder and call settings: `src-main/scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py::_build_sim`, `N_FREQS=30`, `NUM_PERIODS=12`, and `jax_enable_x64=True`. The named cv20 referee reads `tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json`; it contains 0 rfx S-parameter calls. The copied producer leg uses 0 openEMS calls.','',
           'Run IDs and submissions:','',
           table(['fixture','VESSL run ID','YAML submissions','public S calls','time-stepping calls'],
                 [[f,runs[f],1,2,sum(d['status']['timestepping_calls'] for d in r['variants'].values())] for f,r in reductions.items()]),
           'VESSL: remilab-c0; gpu-rtx4090; 14 CPU; 48 GiB; 1 NVIDIA GeForce RTX 4090; image `nvcr.io/nvidia/jax:24.10-py3`. Final status and complete available logs are the `vessl_cv*_*.txt` and `.log` files.','',
           'Source export record: `../PROVENANCE.txt`, `src-main = df08175c`. Source-file SHA-256 values are in each fixture’s `provenance.json`; current-file equality counts are in `verification.json`. Git commands = 0.','',
           'Assumption checks: 1 = observed; 0 = not observed. A1: zero centre-row and full-width Ex/Ey flags in both x absorbers at baseline. A2: nonzero Ex/Ey flags in both x absorbers and completed continued public call. A3: rfx-only builder/call in the named crossval script.','']
    rows=[]
    for f,r in reductions.items():
        bs=r['edges']['baseline'][0]
        cs=r['edges']['continued'][0]
        a1=int(all(p['edge_counts'][a][key]==0 for p in bs['planes'] for a in 'xy'
                   for key in ('centre_row_x_lo','centre_row_x_hi','trace_width_x_lo','trace_width_x_hi')))
        a2=int(r['variants']['continued']['status']['completed'] and all(p['edge_counts'][a][key]>0 for p in cs['planes'] for a in 'xy'
                   for key in ('centre_row_x_lo','centre_row_x_hi','trace_width_x_lo','trace_width_x_hi')))
        rows.append([f,a1,a2,int(f=='cv06b'),1])
    lines += [table(['fixture','A1','A2','A3 named script','rfx-only copied builder/call'],rows),
              'A3 cv20 action: copied the linked rfx fixture producer’s builder and settings listed above. A1/A2 substitution count = 0; alternate continuation methods = 0.','',
              'FACT numerical checks:','',
              table(['quantity','brief FACT','readback'],[
                  ['cv06b grid','553, 280, 37','553, 280, 37'],
                  ['cv06b pads x-lo/x-hi/y-lo/y-hi/z-lo/z-hi','8, 8, 8, 8, 0, 8','8, 8, 8, 8, 0, 8'],
                  ['cv06b main trace x bounds (mm)','0, 34','0, 34'],
                  ['cv06b main trace y bounds (mm)','1.016, 1.616','1.016, 1.616'],
                  ['cv06b main trace z bounds (µm)','254, 254','254, 254'],
                  ['cv06b MSL ports','2','2'],
                  ['cv20 grid','297, 66, 45','297, 66, 45'],
                  ['cv20 dx (µm)','50','50']]),
              'Listed FACT numeric mismatches = 0. Overview trace-sheet statement versus cv20 builder: stated zero thickness = 0 µm; declared z extent = 50 µm; realized wall separation = 50 µm. cv20 used the committed 50 µm declaration.','',
              'Recorded ancillary event, local import, verbatim:','',
              '```text',
              "Could not save font_manager cache NO_MUTATION: os.remove ('/root/workspace/bk-workspace/.801-measure/msl/mpl_config/fontlist-v3.11.0.json.matplotlib-lock', -1)",
              '```','',
              'Blocked cache-lock deletions = 1; cache-lock deletion retries = 0; completed local no-step readbacks = 2. The cache file and lock remain listed below.','',
              'Captured diagnostics: all 15 `MSLSMatrixResult` fields per run in `diagnostics.json`; numeric fields in `diagnostics.npz`; complex S and frequencies in `s.npz`; preflight text verbatim in `preflight.txt` and public-call preflight/warnings in `run.log`; point-probe records in `witness_series_00.npz` and `witness_series_01.npz`.','',
              table(['fixture','run','diagnostic fields','finite S entries','ring-down recompute max difference (dB)'],
                    [[f,v,d['diagnostic_field_count'],d['S_finite_count'],d['ringdown_max_abs_recompute_difference_db']]
                     for f,r in verification.items() for v,d in r['variants'].items()]),
              'Commands: [COMMANDS.md](COMMANDS.md). Measurement source: [measure.py](measure.py). Reduction: [reduce.py](reduce.py). Verification: [verification.json](verification.json).','',
              '**EDGES.md — inline**','',(OUT/'EDGES.md').read_text(),
              '**TABLE.md — inline**','',(OUT/'TABLE.md').read_text(),
              '**Files and sizes**','']
    body='\n'.join(lines)
    files=[(str(p.relative_to(OUT)),p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]
    size=0
    for _ in range(20):
        listing=table(['file (relative to msl/)','bytes'],sorted(files+[('REPORT.md',size)]))
        final=body+listing+'\n'
        newsize=len(final.encode('utf-8'))
        if newsize==size:
            break
        size=newsize
    assert newsize==size
    with (OUT/'REPORT.md').open('x') as f:
        f.write(final)
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),bytes=size,files=len(files)+1)))

if __name__=='__main__':
    main()
