"""Single serial test entry for reciprocal field additions; CPU smoke only."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
import design_lens as lens
import _lens_filter_common as common
from lens_overlap_check import cosine,compare,map_gradient,main as overlap_main
from lens_reciprocal_field import build,physical_bounds,run
OUT=HERE/'out/reciprocal_tests';OUT.mkdir(parents=True,exist_ok=True)

class ReciprocalTests(unittest.TestCase):
 def test_argument_validation(self):
  for case,args in [(lens,['--stage','timing','--drive','reciprocal']),
    (SimpleNamespace(CASE='filter',__doc__='filter'),['--stage','field','--drive','reciprocal']),
    (lens,['--stage','field','--empty']), (lens,['--stage','field','--drive','reciprocal'])]:
   with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):common.main(case,[*args,'--out',str(OUT/'unused')])
 def test_known_overlap_maps(self):
  with tempfile.TemporaryDirectory(dir=OUT) as d:
   d=Path(d);u=np.arange(.018,.108,.003);v=np.arange(.072,.102,.003);grad=np.arange(2250,dtype=float).reshape(15,15,10)+1
   arrays=dict(iterate=0,freqs_hz=np.array([1e10]),planes=np.array(['xz','yz']),drive=np.array('TFSF reciprocal'))
   for plane in ('xz','yz'):
    mask,g=map_gradient(grad,plane,u,v,.063);self.assertTrue(mask.all())
    E=np.zeros((1,3,len(u),len(v)),np.complex64);E[0,0]=g.reshape(len(u),len(v))
    arrays.update({plane+'_axes':np.array(plane),plane+'_u_m':u,plane+'_v_m':v,plane+'_offset_m':.063,plane+'_eps':np.ones((len(u),len(v))),plane+'_E':E})
   np.savez(d/'main.npz',iteration=[0],grad_eps=grad[None],freqs_hz=[1e10],boresight_E_x=np.array([[1+0j]]))
   np.savez(d/'f.npz',**dict(arrays,drive=np.array('Ex current dipole')))
   for sign in (1,-1):
    adj=dict(arrays)
    for plane in ('xz','yz'):
     E=np.zeros_like(arrays[plane+'_E']);E[:,0]=sign;adj[plane+'_E']=E
    np.savez(d/'a.npz',**adj)
    result,_=compare(d/'main.npz',d/'f.npz',d/'a.npz',0)
    for value in result['cosine_similarity'].values():self.assertAlmostEqual(value,sign)
   from unittest.mock import patch
   with patch.object(sys,'argv',['lens_overlap_check.py','--record',str(d/'main.npz'),'--forward',str(d/'f.npz'),'--adjoint',str(d/'a.npz'),'--out',str(d/'plots')]),contextlib.redirect_stdout(io.StringIO()):overlap_main()
   for plane in ('xz','yz'):self.assertTrue((d/'plots'/f'overlap_{plane}.png').exists())
   np.savez(d/'fit.npz',iteration=[0],grad_eps=grad[None],freqs_hz=[1e10])
   result,_=compare(d/'fit.npz',d/'f.npz',d/'a.npz',0)
   self.assertIn('fitted jointly',result['constant_mode'])
   for value in result['cosine_similarity'].values():self.assertAlmostEqual(value,1)
   print('Synthetic fixed-constant cosine: identical +1, sign flip -1; joint fitted constant: +1',flush=True)
 def test_real_forward_coordinate_mapping(self):
  root=Path('/Users/byungkwankim/mnt/remilab-fs/personal-workspaces/byungkwan-workspace/rfx-showcase-runs/20261004-showcase-lens')
  f=root/'20261007T102225Z-975f3a08-field/field_0.npz'
  if not f.exists():self.skipTest('real forward fields unavailable')
  with np.load(root/'20261004T182324Z-b184fc8e-main/iterations.npz') as z:gradient=z['grad_eps'][0]
  with np.load(f) as z:
   for plane in ('xz','yz'):
    mask,g=map_gradient(gradient,plane,z[plane+'_u_m'],z[plane+'_v_m'],float(z[plane+'_offset_m']))
    np.testing.assert_array_equal(mask,z[plane+'_eps']>1)
    self.assertEqual(len(g),2700)
  print('Real forward coordinate mapping: 2700 block-plane samples per plane',flush=True)
 def test_main_geometry_build_only(self):
  sim,grid=build(.001,empty=False);bounds=physical_bounds(sim,grid)
  np.testing.assert_allclose([bounds['x'],bounds['y'],bounds['z']],[[.003,.123],[.003,.123],[.003,.117]],atol=1e-12)
  self.assertEqual(tuple(np.asarray(grid.shape)[[1,2,0]]),(157,157,151))
  self.assertAlmostEqual(grid.dt,1.9065748695310057e-12,places=23)
  print('Main mesh build only: physical shape (157,157,151); TFSF bounds',bounds,flush=True)
 def test_cpu_empty_smoke(self):
  # Actual lens CPML/domain at the declared 3 mm smoke mesh, not main measurement.
  with tempfile.TemporaryDirectory(dir=OUT) as d:
   d=Path(d);source=d/'source';source.mkdir();output=d/'output';output.mkdir()
   sim,grid=build(.003,empty=True)
   self.assertEqual(sim._tfsf.direction,'-x');self.assertEqual(sim._tfsf.polarization,'ey');self.assertTrue(sim._tfsf.closed_box)
   bounds=physical_bounds(sim,grid)
   self.assertLess(bounds['z'][0],.018);self.assertGreater(bounds['z'][1],.102)
   saved=dict(dx_m=.003,precision='float32',smoke=True,grid_shape=np.asarray(grid.shape)[[1,2,0]].tolist(),freqs_hz=lens.FREQS.tolist())
   (source/'result.json').write_text(json.dumps(dict(id='design-lens',model=saved)))
   (source/'record_length.json').write_text(json.dumps(dict(n_steps=200,dt_s=grid.dt)))
   np.savez(source/'iterations.npz',eps=lens.starts()['primary'][None])
   args=SimpleNamespace(precision='float32',smoke=True,iterates=[0],empty=True,out=output)
   with contextlib.redirect_stdout(io.StringIO()):run(lens,args,source)
   with np.load(output/'field_reciprocal_0.npz') as z:
    for plane in ('xz','yz'):
     self.assertEqual(z[plane+'_axes'].shape,());self.assertEqual(str(z[plane+'_axes']),plane)
     self.assertEqual(z[plane+'_E'].shape,(1,3,len(z[plane+'_u_m']),len(z[plane+'_v_m'])))
     self.assertTrue(np.isfinite(z[plane+'_E']).all());self.assertTrue((z[plane+'_eps']==1).all())
     E=z[plane+'_E'][0];self.assertGreater(np.linalg.norm(E[0]),1e4*np.linalg.norm(E[1:]))
     i=int(np.argmin(abs(z[plane+'_u_m']-.063)));j=int(np.argmin(abs(z[plane+'_v_m']-.027)))
     # With exp(-jwt) DFT, a wave propagating -z has positive spatial phase slope.
     self.assertGreater(float(np.angle(np.conj(E[0,i,j])*E[0,i,j+1])),0)
   report=json.loads((output/'field_reciprocal.json').read_text());print('CPU smoke only:',json.dumps(report['iterates']['0']['witnesses']),flush=True)
   # Retained-plate construction uses the identical injector and grid.
   full,fullgrid=build(.003,empty=False);self.assertEqual(fullgrid.shape,grid.shape)
   self.assertEqual(full._tfsf.direction,sim._tfsf.direction)
   self.assertEqual(len(full._geometry),1);self.assertEqual(len(sim._geometry),0)
   args.empty=False;args.out=d/'with_plate';args.out.mkdir()
   with contextlib.redirect_stdout(io.StringIO()):run(lens,args,source)
   with np.load(args.out/'field_reciprocal_0.npz') as z:
    self.assertTrue(np.any(z['xz_eps']>1));self.assertTrue(np.isfinite(z['xz_E']).all())
   print('CPU plate+dielectric smoke: 200 steps, both DFT planes finite',flush=True)

if __name__=='__main__':unittest.main()
