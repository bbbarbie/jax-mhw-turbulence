"""Task 2 extra: Lyapunov spectrum with M exponents in the archive box (L=64).
Usage: python experiments/task2_spectrum_M.py <res> <alpha> <nu> <M> <T> [diffop]
"""
import sys, time, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
from sens.mhw_system import MHWSystem, get_spun_state
from sens.lyapunov import lyapunov_spectrum, exponents_from_logR, batched_se, kaplan_yorke
res=int(sys.argv[1]); alpha=float(sys.argv[2]); nu=float(sys.argv[3]); M=int(sys.argv[4]); T=float(sys.argv[5]); diffop=int(sys.argv[6]) if len(sys.argv)>6 else 6
S=MHWSystem(res=res,alpha=alpha,nu=nu,diffop=diffop); u0,fl,en=get_spun_state(S,0,300.0)
print(f"{S.tag()}: flux last100 {fl[-100:].mean():.3f}+-{fl[-100:].std():.3f} E {en[-100:].mean():.2f}", flush=True)
t=time.time(); nR=int(T/(40*S.dt))
logR,_=lyapunov_spectrum(S,u0,alpha,M=M,renorm_every=40,n_renorm=nR,key=jax.random.PRNGKey(0))
lam,se=exponents_from_logR(logR,S.dt,40,discard=nR//10); seb=batched_se(logR,S.dt,40,discard=nR//10)
Np=int((lam>0).sum()); print(f"M={M} T={T}: N+={Np} lambda1={lam[0]:.4f} lam_M={lam[-1]:.4f} sum={lam.sum():.3f} D_KY={kaplan_yorke(lam) if lam[-1]<0 else None} ({time.time()-t:.0f}s)")
print("lam:",np.round(lam,4).tolist()); print("se_b:",np.round(seb,4).tolist())
json.dump(dict(params=dict(res=res,alpha=alpha,nu=nu,diffop=diffop,dt=S.dt),M=M,T=T,lam=lam.tolist(),se=se.tolist(),se_batched=seb.tolist(),N_plus=Np,sum_M=float(lam.sum()),wall_s=time.time()-t,
  kaplan_yorke=(kaplan_yorke(lam) if lam[-1]<0 else None)), open(f"results/sens/spectrum_{S.tag()}_M{M}_T{T:g}.json","w"), indent=1)
