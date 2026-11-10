import solver, numpy as np, time, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h=0.03
t=time.time(); vals, vecs, inside, (x,y,z) = solver.modes(h,(1,1,1),30,60.0); print(time.time()-t, vals)
np.savez("scan_even.npz", vals=vals, vecs=vecs, inside=inside, x=x, y=y, z=z, h=h)
n=len(vals); fig,axs=plt.subplots(n//6,6*3,figsize=(36,n//6*3))
for m in range(n):
    F=np.zeros(inside.shape); F[inside]=vecs[:,m]
    r,c=divmod(m,6)
    for j,sl in enumerate([F[0,:,:],F[:,0,:],F[:,:,0]]):
        a=axs[r,c*3+j]; vmax=np.abs(F).max(); a.imshow(sl.T,origin='lower',cmap='RdBu',vmin=-vmax,vmax=vmax); a.set_xticks([]);a.set_yticks([])
        if j==0: a.set_title(f"{m}: {vals[m]:.2f}",fontsize=14)
plt.tight_layout(); plt.savefig("scan_even.png",dpi=40)
