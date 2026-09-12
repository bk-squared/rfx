"""Power-conjugate source-model diagnostic; NOT a replacement for MSL S.

The existing shaped source is J=e*u and its scalar added conductivity is
sigma_port=1/(R*N), N=sum(m*e^2). Vproj=<e,E_mid>/N, Isource=N*u.
The matched resistor component is external to this diagnostic port; orthogonal
field dissipation stays inside its network. All samples use the same midpoint
clock. Finite-window and floating-point uncertainty remain separate questions.
"""
from __future__ import annotations
import numpy as np


def dft(values, dt, freqs):
    values=np.asarray(values,dtype=np.float64)
    t=(np.arange(len(values),dtype=float)+.5)*dt
    return np.concatenate([np.exp(-2j*np.pi*np.asarray(freqs)[a:a+16,None]*t[None,:])@values*dt
                           for a in range(0,len(freqs),16)],axis=0)


def waves(post_e, source_u, profiles, volumes, sigma_port, dt, freqs):
    """One drive's ordinary point samples E^{n+1}, initial E=0, and u^n."""
    columns=[]; norms=[]
    for post,e,m in zip(post_e,profiles,volumes):
        post=np.asarray(post,dtype=np.float64);e=np.asarray(e);m=np.asarray(m)
        assert post.ndim==2 and post.shape[1]==len(e)==len(m)
        norm=float(np.sum(m*e*e));assert norm>0
        previous=np.concatenate([np.zeros_like(post[:1]),post[:-1]],axis=0)
        columns.append(((previous+post)*.5)@(m*e)/norm)
        norms.append(norm)
    norm=np.asarray(norms);ref=1/(np.asarray(sigma_port)*norm)
    assert np.all(np.isfinite(ref)&(ref>0))
    phi=np.column_stack(columns)
    source_u=np.asarray(source_u,dtype=np.float64)
    assert phi.shape==source_u.shape
    voltage=dft(phi,dt,freqs).T
    current=dft(source_u*norm[None,:],dt,freqs).T
    a=np.sqrt(ref)[:,None]*current/2
    b=voltage/np.sqrt(ref)[:,None]-a
    return dict(a=a,b=b,voltage=voltage,source_current=current,
                reference_impedances=ref,profile_norm=norm)


def assemble(records):
    a=np.stack([r['a'] for r in records],axis=1).transpose(2,0,1)
    b=np.stack([r['b'] for r in records],axis=1).transpose(2,0,1)
    for r in records:
        np.testing.assert_array_equal(r['reference_impedances'],records[0]['reference_impedances'])
    s=np.linalg.solve(a.transpose(0,2,1),b.transpose(0,2,1)).transpose(0,2,1)
    return s.transpose(1,2,0),np.linalg.cond(a)


def self_test(*, gain=1.0):
    refs=np.array([25.,100.]);sigma=[];profiles=[np.array([.2,.1,.5,.5]),np.array([.3,.1,.25,.25])]
    volumes=[np.array([.2,.3,.4,.5]),np.array([.3,.2,.4,.1])]
    norms=np.array([np.sum(m*g*g) for g,m in zip(profiles,volumes)])
    sigma=1/(refs*norms)
    dt=.01;n=256;t=(np.arange(n)+.5)*dt;f=np.array([2.,4.,7.])
    expected=gain*np.array([[.2,.6],[.6,-.2]])
    records=[]
    for drive in range(2):
        a=np.zeros((n,2));a[:,drive]=np.exp(-((t-.8)/.2)**2)*np.cos(2*np.pi*4*t)
        b=a@expected.T;phi=(a+b)*np.sqrt(refs)[None,:]
        current=2*a/np.sqrt(refs)[None,:]
        post=[]
        for p,(g,m) in enumerate(zip(profiles,volumes)):
            # An independently prescribed orthogonal field does not change
            # the work-conjugate projection, unlike a centre-line sample.
            h=np.array([m[1]*g[1],-m[0]*g[0],0.,0.]);assert abs(np.dot(m*g,h))<1e-15
            midpoint=phi[:,p,None]*g[None,:]+np.sin(3*t)[:,None]*h[None,:]
            end=np.empty_like(midpoint);last=np.zeros_like(g)
            for k,row in enumerate(midpoint):
                end[k]=2*row-last;last=end[k]
            post.append(end)
        r=waves(post,current/norms[None,:],profiles,volumes,sigma,dt,f)
        np.testing.assert_allclose(r['reference_impedances'],refs,rtol=1e-14)
        records.append(r)
    observed,cond=assemble(records)
    np.testing.assert_allclose(observed,np.repeat(expected[:,:,None],len(f),axis=2),atol=5e-12)
    assert np.max(cond)<1.000001
    if gain==2:
        assert np.max(np.linalg.svd(observed.transpose(2,0,1),compute_uv=False))>1.2
    print('PASS: weighted source-work projection, midpoint clock, unequal references, complex DFT and full wave assembly')


if __name__=='__main__':
    self_test()
    self_test(gain=2.0)
