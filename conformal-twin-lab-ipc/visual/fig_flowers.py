import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
import json,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})   # embed TrueType, not Type 3 fonts (publisher requirement)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9})
D=json.load(open(os.path.join(SRC, 'development.json')));R=json.load(open(os.path.join(SRC, 'results.json')))
V=np.array(R['vertices'],float);rr=np.array(R['radii'])
orient=[tuple(t) for t in D['orient'].values()]
def mob(P): return lambda z:(z+P)/(1+np.conj(P)*z)
def geod(p,q,n=40): f=mob(-p);qq=f(q);t=np.linspace(0,1,n);return mob(p)(t*qq)
def hcirc(c,r,n=150): return mob(c)(np.tanh(r/2)*np.exp(1j*np.linspace(0,2*np.pi,n)))
def ang(a,b,c): return np.arccos(np.clip((np.cosh(b)*np.cosh(c)-np.cosh(a))/(np.sinh(b)*np.sinh(c)),-1,1))
def link(v):
    ts=[t for t in orient if v in t]; ts=[t[t.index(v):]+t[:t.index(v)] for t in ts]
    nxt={t[1]:t[2] for t in ts}; a=ts[0][1]; cyc=[a]
    while nxt[cyc[-1]]!=a: cyc.append(nxt[cyc[-1]])
    return cyc
def euclid(ax,v,col):
    cyc=link(v); th=0; scale=max(np.linalg.norm(V[u]-V[v]) for u in cyc)
    pts=[]
    for k,a in enumerate(cyc):
        b=cyc[(k+1)%len(cyc)]; ea,eb=V[a]-V[v],V[b]-V[v]
        al=np.arccos(ea@eb/np.linalg.norm(ea)/np.linalg.norm(eb))
        pa=np.linalg.norm(ea)/scale*np.exp(1j*th); pb=np.linalg.norm(eb)/scale*np.exp(1j*(th+al))
        ax.fill([0,pa.real,pb.real],[0,pa.imag,pb.imag],fc=col,alpha=.28,ec='#2b3a47',lw=.5)
        th+=al
    s=np.linspace(0,th,300); rad=0.18+0.035*s/(2*np.pi)
    ax.plot(rad*np.cos(s),rad*np.sin(s),color='#c0392b',lw=1.3)
    ax.plot(0,0,'o',ms=3,color='k'); return th
def hyper(ax,v,col):
    cyc=link(v); th=0; P={}
    for k,a in enumerate(cyc):
        b=cyc[(k+1)%len(cyc)]; la,lb,lab=rr[v]+rr[a],rr[v]+rr[b],rr[a]+rr[b]
        al=ang(lab,la,lb); P[a]=np.tanh(la/2)*np.exp(1j*th); th+=al
    for k,a in enumerate(cyc):
        b=cyc[(k+1)%len(cyc)]; g=np.concatenate([geod(0j,P[a]),geod(P[a],P[b]),geod(P[b],0j)])
        ax.fill(g.real,g.imag,fc=col,alpha=.28,ec='#2b3a47',lw=.5)
    for u in [v]+cyc:
        z=hcirc(0j if u==v else P[u],rr[u]); ax.plot(z.real,z.imag,color='k',lw=.6)
    return th
fig,axs=plt.subplots(1,4,figsize=(6.4,1.95),dpi=220)
for (v,col),k in zip([(4,'#4E79A7'),(5,'#F28E2B')],[0,2]):
    te=euclid(axs[k],v,col); thh=hyper(axs[k+1],v,col)
    axs[k].set_title('vertex %d, polyhedron\n%d triangles, angle sum $%.1f^{\\circ}$'%(v+1,len(link(v)),np.degrees(te)),fontsize=8)
    axs[k+1].set_title('vertex %d, hyperbolic\n$r_{%d}=%.4f$, angle sum $360^{\\circ}$'%(v+1,v+1,rr[v]),fontsize=8)
    print(v+1,np.degrees(te),np.degrees(thh)-360)
for a in axs:
    a.set_aspect('equal');a.axis('off')
for a in axs[[1,3]]: a.add_patch(plt.Circle((0,0),1,fc='none',ec='#9aa7b1',lw=.6,ls=':'))
for a in axs: a.set_xlim(-1.05,1.05);a.set_ylim(-1.05,1.05)
fig.subplots_adjust(wspace=.05,left=.01,right=.99,top=.8,bottom=.02)
fig.savefig(os.path.join(OUT, 'fig_flowers.pdf'),bbox_inches='tight',pad_inches=.02);fig.savefig(os.path.join(OUT, 'fig_flowers.png'),bbox_inches='tight',pad_inches=.02)
