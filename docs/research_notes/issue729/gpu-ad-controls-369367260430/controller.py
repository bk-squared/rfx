import json,os,subprocess,sys,time
from pathlib import Path
out=Path(os.environ['RFX_OUT']);arms=json.loads((out/'arms.json').read_text());results=[]
for arm in arms:
    work=Path('/tmp/rfx-i729-ad-'+arm['name'])
    for safe in [arm['checkout'],str(Path(arm['checkout'])/'.git')]:
        subprocess.run(['git','config','--global','--add','safe.directory',safe],check=True)
    subprocess.run(['git','clone','--quiet','--no-checkout',arm['checkout'],str(work)],check=True)
    subprocess.run(['git','-C',str(work),'checkout','--quiet','--detach',arm['source_sha']],check=True)
    armout=out/arm['name'];armout.mkdir()
    env=os.environ.copy();env.update(RFX_OUT=str(armout),RFX_SHA=arm['source_sha'],PYTHONPATH=str(work))
    env.pop('JAX_DEFAULT_MATMUL_PRECISION',None)
    if arm['precision'] is not None:env['JAX_DEFAULT_MATMUL_PRECISION']=arm['precision']
    print('START_ARM',json.dumps(arm),flush=True);start=time.monotonic()
    with (out/(arm['name']+'.log')).open('w') as log:
        child=subprocess.Popen([sys.executable,'-u',str(out/'driver.py'),'run'],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        rc=child.wait()
    results.append(dict(arm=arm['name'],exit_code=rc,wall_s=time.monotonic()-start))
    (out/'outcomes.json').write_text(json.dumps(results,indent=2)+'\n')
    print('END_ARM',json.dumps(results[-1]),flush=True)
raise SystemExit(1 if any(x['exit_code'] for x in results) else 0)
