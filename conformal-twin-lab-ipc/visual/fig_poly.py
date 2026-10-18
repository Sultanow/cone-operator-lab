import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
import numpy as np,json,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})   # embed TrueType, not Type 3 fonts (publisher requirement)
from render3d import render,R
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9})
V=np.array(R['vertices'],float)
fig,axs=plt.subplots(1,2,figsize=(6.4,3.1),dpi=220)
for ax,(e,a),t in zip(axs,[(25,45),(89,45)],['oblique view','view along the $z$-axis']):
    ax.imshow(render(V,e,a,N=900)); ax.axis('off'); ax.set_title(t,fontsize=8.5)
fig.subplots_adjust(wspace=.02,left=0,right=1,top=.92,bottom=0)
fig.savefig(os.path.join(OUT, 'fig_poly.pdf'),bbox_inches='tight',pad_inches=.02,dpi=400);fig.savefig(os.path.join(OUT, 'fig_poly.png'),bbox_inches='tight',pad_inches=.02)
