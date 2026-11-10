import solver, numpy as np, sys, time, itertools
h=0.035; classes=list(itertools.product((1,-1),repeat=3))
for ci in map(int, sys.argv[1:]):
    t=time.time(); v,_,_,_=solver.modes(h,classes[ci],24,0.0)
    np.save(f"spec_c{ci}.npy", v); print(ci, classes[ci], round(time.time()-t,1), v[0].round(2), v[-1].round(2), flush=True)

import os
if all(os.path.exists(f"spec_c{i}.npy") for i in range(8)):
    vs = [np.load(f"spec_c{i}.npy") for i in range(8)]
    top = min(v.max() for v in vs); allv = np.sort(np.concatenate(vs)); lev = allv[allv <= top]
    np.save("fd_spectrum.npy", lev); print("merged", len(lev), "levels up to", round(top, 2))
