import json
import pathlib
import subprocess

root=pathlib.Path('/root/workspace/bk-workspace/.801-measure/ports')
rows=[]
for run_id in (369367262622,369367262623):
    log_path=root/f'vessl_{run_id}.log'
    with log_path.open('x') as f:
        result=subprocess.run(['vessl','run','logs',str(run_id),'--tail','100000'],stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,result.returncode
    assert log_path.stat().st_size>0
    result=subprocess.run(['vessl','run','read',str(run_id)],capture_output=True,text=True,check=True)
    with (root/f'vessl_{run_id}_status.txt').open('x') as f:
        f.write(result.stdout)
        f.write(result.stderr)
    rows.append(dict(run_id=run_id,logs_bytes=log_path.stat().st_size,read_returncode=result.returncode))
with (root/'runs.json').open('x') as f:
    json.dump(rows,f,indent=2)
    f.write('\n')
print(json.dumps(rows))
