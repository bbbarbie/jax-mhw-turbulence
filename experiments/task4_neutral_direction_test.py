"""Task 4 diagnostic: NILSS on the same trajectory/tangent seeds with three neutral directions
(rhs | discrete | central).  Usage: python experiments/task4_neutral_direction_test.py <rhs|discrete|central> <T> <M> <DT>
"""
import sys, time, json, numpy as np; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
from sens.mhw_system import MHWSystem, get_spun_state
from sens.nilss import nilss_segments, nilss_solve
which=sys.argv[1]; T=float(sys.argv[2]); M=int(sys.argv[3]); DT=float(sys.argv[4])
S=MHWSystem(res=32, alpha=0.2, L=16.0); u0,_,_=get_spun_state(S,0,300.0)
f={"rhs":S.rhs,"discrete":None,"central":S.f_central}[which]
nseg=int(DT/S.dt); K=int(T/DT); t=time.time()
segs,_=nilss_segments(S.step,u0,0.2,S.flux,M=M,nseg_steps=nseg,K=K,dt=S.dt,key=jax.random.PRNGKey(1),rec_every=nseg//20,K_pre=5,rhs=f)
sol=nilss_solve(segs)
conv=[(k*DT, nilss_solve(segs,K_use=k)['dJds']) for k in [5,10,20,40,80,160,320] if k<=K]
print(f"f={which} M={M} DT={DT} T={T}: dJds={sol['dJds']:.4f} (ibp {sol['dJds_ibp']:.4f}) perp {sol['term_perp']:.4f} dil {sol['term_dilB']:.4f} max|a|={np.abs(sol['a']).max():.3e} max|xi|={np.abs(sol['xi']).max():.2f} mean|vperp|={sol['vperp_norm'].mean():.3e} max|vperp|={sol['vperp_norm'].max():.3e} conv={[(round(a),round(b,3)) for a,b in conv]} ({time.time()-t:.0f}s)", flush=True)
json.dump(dict(f=which,M=M,DT=DT,T=T,dJds=sol['dJds'],dJds_ibp=sol['dJds_ibp'],max_a=float(np.abs(sol['a']).max()),max_xi=float(np.abs(sol['xi']).max()),mean_vperp=float(sol['vperp_norm'].mean()),max_vperp=float(sol['vperp_norm'].max()),conv=conv),
  open(f"results/sens/task4_ftest_{which}_M{M}_DT{DT:g}_T{T:g}.json","w"),indent=1)
