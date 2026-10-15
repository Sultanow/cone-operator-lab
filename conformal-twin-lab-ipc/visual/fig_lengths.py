import os, sys
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'; os.makedirs(OUT, exist_ok=True)
# Figure 5: lengths of test geodesics in h_cp vs h_P 
# Usage: python visual/fig_lengths.py OUTDIR_of_wp_c/length_compare.npz OUTDIR
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':8.5})
Z=np.load(sys.argv[1] if len(sys.argv) > 1 else 'length_compare.npz'); x=Z['lcp'][Z['keep']]; y=Z['P3'][Z['keep']]
pairs=np.unique(np.round(np.c_[x,y],5),axis=0); r=y/x
fig,ax=plt.subplots(figsize=(3.4,3.2),dpi=220); ax.scatter(pairs[:,0],pairs[:,1],s=3,c='#4E79A7',alpha=.55,lw=0)
t=np.linspace(0,pairs.max()*1.03,10); ax.plot(t,t,color='#1b2733',lw=.8,label='equal length')
ax.plot(t,r.max()*t,'--',color='#E15759',lw=.8,label=f'ratio {r.max():.3f}'); ax.plot(t,r.min()*t,'--',color='#F28E2B',lw=.8,label=f'ratio {r.min():.3f}')
ax.set_xlim(0,pairs[:,0].max()*1.03); ax.set_ylim(0,pairs[:,1].max()*1.03)
ax.set_xlabel(r'length in the circle-packing twin $h_{\mathrm{cp}}$'); ax.set_ylabel(r'length in $h_P$ (level 3)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax.set_aspect('equal'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_lengths.pdf'),bbox_inches='tight',pad_inches=.02)
