"""Serial timing-helper and driver JSON tests; small stubs, no FDTD solves."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]))
import _lens_filter_common as common
OUT=HERE/'out/timing_tests';OUT.mkdir(parents=True,exist_ok=True)

class TimingTests(unittest.TestCase):
 def test_warmup_and_synchronization(self):
  events=[];argument=object();counter=iter([10,11,20,22,30,33])
  def function(x):self.assertIs(x,argument);events.append('call');return 'result'
  def block(x):events.append('sync_input' if x is argument else 'sync_result')
  def clock():self.assertTrue(events[-1].startswith('sync_'));events.append('clock');return next(counter)
  self.assertEqual(common.synchronized_wall_times(function,argument,block=block,clock=clock),[1.,2.,3.])
  self.assertEqual(events[:3],['sync_input','call','sync_result'])
  self.assertEqual(events.count('call'),4);self.assertEqual(events.count('sync_result'),4)
 def test_stage_json_both_drivers(self):
  import jax.numpy as jnp
  for kind in ('lens','filter'):
   with self.subTest(case=kind),tempfile.TemporaryDirectory(dir=OUT) as tmp:
    out=Path(tmp)
    case=SimpleNamespace(CASE=kind,__doc__='timing stub',SMOKE_DX=.003,DESIGN_DX=.001,
                         starts=lambda:{'primary':np.ones(2),'S1':np.ones(2)},loss=lambda r:jnp.sum(r*r))
    case.with_objective=lambda objective,case=case:case  # the lens driver binds its objective first
    model=SimpleNamespace(dx=.003,grid=SimpleNamespace(shape=(3,4,5),dt=1e-12),default_steps=200,
                          response_fn=lambda n:lambda eps:2*eps)
    def optimize(*args):
     np.savez(out/'iterations.npz',eps=np.ones((3,2)),objective=[3.,2.,1.],wall_s=[21.4,11.283,11.275])
     return SimpleNamespace(rows={'wall_s':[21.4,11.283,11.275]})
    def record(out,case,model,stage,details):common.dc.save_json(out/'stage.json',dict(stage=stage,**details))
    with patch.object(common,'setup',return_value=model),patch.object(common,'optimize',side_effect=optimize),patch.object(common,'evaluate',return_value={'reported':{'wall_s':99.}}),patch.object(common,'record',side_effect=record):
     self.assertEqual(common.main(case,['--stage','timing','--smoke','--out',str(out)]),0)
    summary=json.loads((out/'timing_summary.json').read_text());stage=json.loads((out/'stage.json').read_text())
    self.assertTrue(all(stage[k]==v for k,v in summary.items()))
    self.assertEqual(len(summary['forward_wall_s']),3);self.assertTrue(all(x>=0 for x in summary['forward_wall_s']))
    self.assertEqual(summary['value_and_grad_wall_s'],[11.283,11.275])
    self.assertAlmostEqual(summary['value_and_grad_wall_s_median'],11.279)
    self.assertEqual(summary['forward_wall_s_median'],float(np.median(summary['forward_wall_s'])))
    self.assertEqual(summary['n_steps'],200);self.assertEqual(summary['mesh']['grid_shape'],[3,4,5])
    self.assertEqual(summary['checkpointed_reverse_mode'],'yes' if kind=='filter' else 'no')
    self.assertEqual(summary['checkpoint_segments'],10 if kind=='filter' else None)
    self.assertTrue(summary['device_name']);self.assertEqual(stage['eager_check']['reported']['wall_s'],99.)
    self.assertIn('wall_s',stage)
    with np.load(out/'iterations.npz') as saved:np.testing.assert_array_equal(saved['wall_s'],[21.4,11.283,11.275])
    print(kind,': three forward samples; two existing gradient samples; summary/stage keys agree',flush=True)
 def test_arguments_unchanged(self):
  case=SimpleNamespace(CASE='lens',__doc__='timing stub')
  case.with_objective=lambda objective:case
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
   common.main(case,['--stage','timing','--iterations','3','--out',str(OUT/'unused')])

if __name__=='__main__':unittest.main()
