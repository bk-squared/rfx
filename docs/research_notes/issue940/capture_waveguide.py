from pathlib import Path
import hashlib,json,sys,warnings
from unittest.mock import patch
import jax,numpy as np,pytest
from rfx import Simulation
root=Path('/root/rfx-codex/.git/issue940')
real=Simulation.compute_waveguide_s_matrix
rows=[]; arrays={}
def capture(sim,*args,**kwargs):
 result=real(sim,*args,**kwargs)
 n=kwargs['n_steps'];S=np.asarray(result.s_params);grid=sim._build_grid()
 arrays[f'S_{n}']=S; arrays[f'eps_{n}']=np.asarray(kwargs['eps_override'])
 rows.append({'n_steps':n,'max_column_power':float(np.max(np.sum(abs(S)**2,axis=0))),'grid_shape':grid.shape,'pad_x_lo':grid.pad_x_lo,'dx':grid.dx})
 return result
with patch.object(Simulation,'compute_waveguide_s_matrix',capture):
 code=pytest.main(['tests/unit/ports/test_waveguide_port_spectrum_guard.py::test_issue150_two_slab_toy_run_length_invariant_and_passive','-q','-s','-o','addopts='])
np.savez_compressed(root/'corrected_waveguide.npz',**arrays)
r={'pytest_exit_code':int(code),'jax':jax.__version__,'backend':jax.default_backend(),'x64':bool(jax.config.jax_enable_x64),'rows':rows,'source_sha256':hashlib.sha256(Path('tests/unit/ports/test_waveguide_port_spectrum_guard.py').read_bytes()).hexdigest()}
(root/'corrected_waveguide.json').write_text(json.dumps(r,indent=2)+'\n')
print(r);sys.exit(int(code))
