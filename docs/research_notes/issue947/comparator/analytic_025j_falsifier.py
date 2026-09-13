import numpy as np
from tests import _transverse_resonance_o3 as tr
from tests.oracle import test_leontovich_alpha_oracle as o
f=8e9
modes=tr.find_modes(f,o.B_PLATE,o.G_STUB,o.RS0,o.ETA_0)
k0=2*np.pi*f/tr.C0
ks=max(modes,key=lambda k:-k.imag)
x=np.arange(201)*o.DX
# A prescribed exact continuum mixture at the gap midplane: no FDTD
# data and no parameter search. Carrier exp(-j*k0*x) cancels from |.|.
q=.25j
hy=1+q*np.exp(-1j*(ks-k0)*x)
ez_same_x=1+(ks/k0)*q*np.exp(-1j*(ks-k0)*x)
ez_at_yee_x=1+(ks/k0)*q*np.exp(1j*(ks-k0)*o.DX/2)*np.exp(-1j*(ks-k0)*x)
a_h=tr._fit_alpha_loglin(x,abs(hy)); a_e=tr._fit_alpha_loglin(x,abs(ez_same_x)); a_es=tr._fit_alpha_loglin(x,abs(ez_at_yee_x))
print({'frequency_hz':f,'lossy_kx':str(ks),'prescribed_midplane_lossy_over_tem':str(q),'alpha_Hy':a_h,'alpha_Ez_same_x':a_e,'relative_split_same_x':abs(a_e/a_h-1),'alpha_Ez_with_half_x_offset':a_es,'relative_split_actual_x':abs(a_es/a_h-1),'existing_gate':o.O3_MODEL_GATE})
