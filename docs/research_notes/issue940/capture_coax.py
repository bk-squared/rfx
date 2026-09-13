from pathlib import Path
import hashlib,json,sys
from unittest.mock import patch
import jax,numpy as np,pytest
import rfx.simulation as simulation
from rfx import Simulation
root=Path('/root/rfx-codex/.git/issue940');real_run=simulation.run;real_s=Simulation.compute_coaxial_two_port
arrays={};rows=[]
def capture_run(*args,**kwargs):
 result=real_run(*args,**kwargs);i=len([k for k in arrays if k.startswith('time_series_')]);arrays[f'time_series_{i}']=np.asarray(result.time_series);return result
def capture_s(sim,*args,**kwargs):
 result=real_s(sim,*args,**kwargs);arrays['S']=np.asarray(result.s_params);arrays['settling_db']=np.asarray(result.settling_db);arrays['freqs']=np.asarray(result.freqs);return result
with patch.object(simulation,'run',capture_run),patch.object(Simulation,'compute_coaxial_two_port',capture_s):
 code=pytest.main(['tests/unit/sparams/test_settling_witness.py::test_settled_coax_two_port_stays_silent','-q','-s','-o','addopts='])
np.savez_compressed(root/'coax_baseline.npz',**arrays)
r={'pytest_exit_code':int(code),'jax':jax.__version__,'x64':bool(jax.config.jax_enable_x64),'settling_db':arrays.get('settling_db',np.array([])).tolist(),'captured_fields':list(arrays),'source_sha256':hashlib.sha256(Path('tests/unit/sparams/test_settling_witness.py').read_bytes()).hexdigest()}
(root/'coax_baseline.json').write_text(json.dumps(r,indent=2)+'\n');print(r);sys.exit(int(code))
