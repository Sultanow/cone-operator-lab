"""Tiny z-buffer renderer (numpy) so that the self-interlocking genus-3 surface is drawn with correct occlusion."""
import numpy as np, json
import os
R=json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'results.json')))
tri=np.array(R['triangles'])-1; F=[[x-1 for x in f] for f in R['faces']]
face_of=np.array([[f for f,face in enumerate(F) if set(t)<=set(face)][0] for t in tri])
pal=np.array([[0x4E,0x79,0xA7],[0xF2,0x8E,0x2B],[0xE1,0x57,0x59],[0x76,0xB7,0xB2],[0x59,0xA1,0x4F],[0xED,0xC9,0x48],[0xB0,0x7A,0xA1],[0xFF,0x9D,0xA7]])/255
orig_edges=set()
for f in F:
    for a,b in zip(f,f[1:]+f[:1]): orig_edges.add(frozenset((a,b)))
def rot(elev,azim):
    a,e=np.radians(azim),np.radians(elev)
    Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    return Rx@Rz
def render(X,elev=25,azim=30,N=700,span=None,light=(0.35,0.5,0.8),bg=1.0,edges=True,hidden=True):
    Rm=rot(elev,azim); P=X@Rm.T  # x right, y "into"? we use (x, z) screen, depth = y
    sx,sy,dep=-P[:,0],P[:,2],-P[:,1]   # a real camera on the +y side sees -x to the right (no mirror image)
    span=span or 1.08*np.abs(np.c_[sx,sy]).max()
    to=lambda u:(u/span+1)*N/2
    px,py=to(sx),N-to(sy)
    img=np.full((N,N,3),bg); zb=np.full((N,N),np.inf)
    l=np.array(light);l=l/np.linalg.norm(l)
    for n,(i,j,k) in enumerate(tri):
        nrm=np.cross(P[j]-P[i],P[k]-P[i]);nrm/=np.linalg.norm(nrm)
        # screen-facing normal: flip so it points to viewer (viewer along -y in P)
        if nrm[1]>0: nrm=-nrm
        shade=0.62+0.38*abs(nrm@np.array([l[0],-l[2],l[1]]))
        col=pal[face_of[n]]*shade+0.15*(1-shade)
        x0,x1=int(max(0,np.floor(min(px[[i,j,k]])))),int(min(N-1,np.ceil(max(px[[i,j,k]]))))
        y0,y1=int(max(0,np.floor(min(py[[i,j,k]])))),int(min(N-1,np.ceil(max(py[[i,j,k]]))))
        if x1<x0 or y1<y0: continue
        gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
        A=np.array([[px[i],px[j],px[k]],[py[i],py[j],py[k]],[1,1,1]])
        if abs(np.linalg.det(A))<1e-9: continue
        bc=np.linalg.solve(A,np.stack([gx.ravel(),gy.ravel(),np.ones(gx.size)]))
        ins=(bc>=-1e-9).all(0)
        d=(bc*dep[[i,j,k]][:,None]).sum(0)
        yy,xx=gy.ravel().astype(int),gx.ravel().astype(int)
        m=ins&(d<zb[yy,xx])
        zb[yy[m],xx[m]]=d[m]; img[yy[m],xx[m]]=col
    if edges:
        E=np.array(R['edges'])-1
        for a,b in E:
            orig=frozenset((a,b)) in orig_edges
            t=np.linspace(0,1,int(3*N))
            x=px[a]+(px[b]-px[a])*t; y=py[a]+(py[b]-py[a])*t; d=dep[a]+(dep[b]-dep[a])*t
            xi,yi=x.astype(int),y.astype(int); ok=(xi>=0)&(xi<N)&(yi>=0)&(yi<N)
            xi,yi,d=xi[ok],yi[ok],d[ok]
            vis=d<=zb[yi,xi]+0.012*span
            c=np.array([0.10,0.13,0.17]) if orig else np.array([0.30,0.33,0.37])
            w=0.95 if orig else 0.45
            for dx,dy in ([(0,0),(1,0),(0,1)] if orig else [(0,0)]):
                X_=np.clip(xi[vis]+dx,0,N-1);Y_=np.clip(yi[vis]+dy,0,N-1)
                img[Y_,X_]=img[Y_,X_]*(1-w)+c*w
            if orig and hidden:
                hid=~vis; dash=((np.arange(len(xi))//6)%2==0)
                X_=xi[hid&dash];Y_=yi[hid&dash]
                img[Y_,X_]=img[Y_,X_]*0.55+np.array([1,1,1])*0.45
    alpha=(np.abs(img-bg).sum(-1)>1e-6).astype(float)
    return np.dstack([img,alpha])
