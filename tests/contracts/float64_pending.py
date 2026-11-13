"""Exact observed names and item-1 reasons; no patterns or dtype-wide exemptions."""

# Each pending entry holds (census reason, per-scene violation counts); omitted scenes have zero.
# Every observed float32 leaf belongs to FIX, including A2e's empty magnetic table.
FLOAT32_BY_CONTRACT = {}

FLOAT32_PENDING_2613 = {
    'xs.2': ('found at step 0: magnetic waveform table, uniform_loop.py:362 (empty in these scenes)',
        {'U-mat': 1, 'U-kerr': 1, 'U-smooth': 1, 'U-conformal': 1}),
}

# FIX rows no read point shows, with measured or admission-based reasons.
FLOAT32_UNSEEN_2613 = {
    'TFSF auxiliary CPML profile': 'tfsf.py:514-519,537; tfsf_2d.py:276-280; tfsf_oblique_open.py:255-259: U/G-tfsf refuse before setup',
    'waveguide reference eps/mu': 'core/yee.py:327,330: no waveguide calculator in A2d scenes',
    'topology occupancy': 'occupancy.py:218: no topology design in A2d scenes',
    'Kottke inverse baseline': 'occupancy.py:243: no design occupancy or Kottke-PEC mode',
    'staircase smoothing': 'smoothing.py:1256: U-smooth uses a sphere and full node tensors',
    'dielectric inverse': 'smoothing.py:1812: no Kottke-PEC mode in A2d scenes',
    'Kottke occupancy tensor': 'smoothing.py:1934-1946: no design occupancy in A2d scenes',
    'TFSF correction table': 'tfsf_discrete_plane_wave.py:125: U/G-tfsf and their single-object splits refuse transverse invariance before setup',
    'TFSF delayed time': 'tfsf_discrete_plane_wave.py:316: same U/G-tfsf refusal before setup',
    'TFSF window weights': 'tfsf_discrete_plane_wave.py:320: same U/G-tfsf refusal; no custom source windows',
    'Method-B interpolation': 'tfsf_oblique_open.py:163-168: no Method-B scene in A2d',
    'MSL eigenmode amplitudes': 'msl_port.py:1236-1264: no MSL port in A2d scenes',
    'Python-float dt time axes': 'source_coefficients.py:138,161,170; port_drive.py:49; msl_port.py:1182,1184: no cut_cell_floor pin or ports; realized source table passes P1/P2',
    'traced smoothing': 'smoothing.py:1503,1603,1802: scene coordinates are concrete',
    'traced wire mask': 'wire_mask.py:49-51: no wire or traced coordinate in A2d scenes',
}

PENDING_CEILING = 16
