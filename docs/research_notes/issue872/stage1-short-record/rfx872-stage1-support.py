import sys,json
import numpy as np
from scipy.signal import decimate
sys.path.insert(0,'/root/rfx-codex')
from rfx.harminv import harminv
from rfx.grid import C0
x=np.load('/tmp/rfx872-stage1/input.npz'); y=x['signal']; dt=float(x['dt']); lo=float(x['fmin']); hi=float(x['fmax']); a=.04; dx=.001
ref=C0/2*np.sqrt(2)/a
fyee=np.arcsin(C0*dt*np.sqrt(2)*np.sin(np.pi*dx/(2*a))/dx)/(np.pi*dt)
print('analytic',ref,'discrete',fyee,'error',100*(fyee/ref-1))
y9=decimate(y,9,n=180,ftype='fir',zero_phase=True)[10:-10]
y45=decimate(y9,5,n=100,ftype='fir',zero_phase=True)[10:-10]
for name,z,step in [('raw',y,1),('stage9',y9,9),('stage45',y45,45),('short-native',y[540:540+18*45],1)]:
 l=int(len(z)*.33); s=np.linalg.svd(np.lib.stride_tricks.sliding_window_view(z,l)[:-1],compute_uv=False);s/=s[0]
 modes=harminv(z,dt*step,lo,hi,decimate=False)
 print(name,len(z),'rank',sum(s>1e-3),'sv',s[:12].tolist(),'modes',[dict(m._asdict(),lattice_error_pct=100*(m.freq/fyee-1)) for m in modes])
 # exact single mode at known discrete pole, both conjugates and an offset
 t=np.arange(len(z))*dt*step
 basis=np.stack([np.cos(2*np.pi*fyee*t),np.sin(2*np.pi*fyee*t),np.ones(len(z))],axis=1)
 c=np.linalg.lstsq(basis,z,rcond=None)[0]
 print('one_mode_unexplained',np.linalg.norm(z-basis@c)/np.linalg.norm(z))
