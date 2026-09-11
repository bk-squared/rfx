import sys,importlib.util,runpy
sys.path.insert(0,'/root/rfx-codex')
import rfx.sources.msl_port as m
sp=importlib.util.spec_from_file_location('candidate_msl','/tmp/rfx729-count-only.py');c=importlib.util.module_from_spec(sp);sys.modules[sp.name]=c;sp.loader.exec_module(c)
m.compute_msl_mode_profile=c.compute_msl_mode_profile
runpy.run_path('/tmp/rfx729-local.py',run_name='__main__')
