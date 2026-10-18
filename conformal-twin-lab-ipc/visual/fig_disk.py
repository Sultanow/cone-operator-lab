import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
import json,numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})   # embed TrueType, not Type 3 fonts (publisher requirement)
from matplotlib.path import Path; from matplotlib.patches import PathPatch, Circle
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9})
D=json.load(open(os.path.join(SRC, 'development.json'))); R=json.load(open(os.path.join(SRC, 'results.json')))
rr=np.array(R['radii']); face_of=D['face_of']
pal=['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F','#EDC948','#B07AA1','#FF9DA7']
def mob(P): return lambda z:(z+P)/(1+np.conj(P)*z)
def geod(p,q,n=40):
    f=mob(-p); qq=f(q); t=np.linspace(0,1,n); return mob(p)(t*qq)
def hcirc(c,r,n=120):
    s=np.tanh(r/2)*np.exp(1j*np.linspace(0,2*np.pi,n)); return mob(c)(s)
fig,ax=plt.subplots(figsize=(5.6,5.6),dpi=220)
ax.add_patch(Circle((0,0),1,fc='#f4f6f8',ec='#2b3a47',lw=1.0,zorder=0))
verts=[];codes=[]
for n,t in D['orient'].items():
    p={int(k):complex(*v) for k,v in D['pos'][n].items()}
    a,b,c=t; pts=np.concatenate([geod(p[a],p[b]),geod(p[b],p[c]),geod(p[c],p[a])])
    xy=np.c_[pts.real,pts.imag]
    ax.fill(xy[:,0],xy[:,1],fc=pal[face_of[int(n)]],alpha=.55,ec='none',zorder=1,rasterized=True)
    verts+=list(xy)+[xy[0]]; codes+=[Path.MOVETO]+[Path.LINETO]*(len(xy)-1)+[Path.CLOSEPOLY]
    for u,v in [(a,b),(b,c),(c,a)]:
        g=geod(p[u],p[v]); ax.plot(g.real,g.imag,color='#2b3a47',lw=.35,zorder=2,rasterized=True)
clip=PathPatch(Path(verts,codes),fc='none',ec='none'); ax.add_patch(clip)
done=set()
for n,t in D['orient'].items():
    for k,v in D['pos'][n].items():
        key=(int(k),round(v[0],9),round(v[1],9))
        if key in done: continue
        done.add(key); z=hcirc(complex(*v),rr[int(k)])
        ln,=ax.plot(z.real,z.imag,color='#1a1a1a',lw=.55,zorder=3,rasterized=True); ln.set_clip_path(clip)
# label each nonagon at hyperbolic-ish centroid of its developed triangles
cent={}
for n,t in D['orient'].items():
    p=[complex(*v) for v in D['pos'][n].values()]; cent.setdefault(face_of[int(n)],[]).append(np.mean(p))
for fidx,zs in cent.items():
    z=np.mean(zs); ax.text(z.real,z.imag,str(fidx+1),ha='center',va='center',fontsize=10,weight='bold',color='#14212b',zorder=5,
        bbox=dict(boxstyle='circle,pad=0.18',fc='white',ec='#14212b',lw=.6,alpha=.9))
# label one copy of each vertex orbit representative
reps={5:'5',6:'6',7:'7',2:'2',1:'1',18:'18'}
ax.set_xlim(-1.02,1.02);ax.set_ylim(-1.02,1.02);ax.set_aspect('equal');ax.axis('off')
fig.savefig(os.path.join(OUT, 'fig_disk.pdf'),bbox_inches='tight',pad_inches=.02,dpi=330); fig.savefig(os.path.join(OUT, 'fig_disk.png'),bbox_inches='tight',pad_inches=.02)
print('ok',len(done))
