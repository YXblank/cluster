import numpy as np
import cv2
from itertools import combinations, permutations
from scipy.spatial.distance import cdist

def dihedral(a,b,t,threshold=.15):
    a,b,t=[np.asarray(x,float)/np.linalg.norm(x) for x in (a,b,t)]
    d=np.arctan2(t@np.cross(a,b),a@b)
    return float(d), int(d>threshold)-int(d< -threshold)

def backproject(depth,K):
    y,x=np.indices(depth.shape); rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
    return rays*depth[...,None]

def extract(depth,K,instances,categories,patches,threshold=.15,min_pixels=8):
    xyz=backproject(depth,K); valid=np.isfinite(depth)&(depth>0)
    nodes=[]
    def node(mask,code):
        y,x=np.where(mask&valid)
        if len(x)>=min_pixels: nodes.append(dict(xyz=xyz[y,x].mean(0).tolist(),uv=[float(x.mean()),float(y.mean())],code=code))
    for i in np.unique(instances):
        if i>0:
            m=instances==i; c=int(np.bincount(categories[m].astype(int)).argmax()); node(m,['s',c])
    normals={}; centers={}
    for i in np.unique(patches):
        p=xyz[(patches==i)&valid]
        if i<=0 or len(p)<min_pixels: continue
        c=p.mean(0); _,s,v=np.linalg.svd(p-c,full_matrices=False)
        if s[1]<1e-8: continue
        n=v[-1]; normals[i]=n if n@c<0 else -n; centers[i]=c
    for a,b in combinations(normals,2):
        ma=(patches==a).astype('uint8'); mb=(patches==b).astype('uint8')
        boundary=((cv2.dilate(ma,np.ones((3,3),np.uint8))>0)&(mb>0))&valid
        p=xyz[boundary]
        if len(p)<min_pixels: continue
        _,s,v=np.linalg.svd(p-p.mean(0),full_matrices=False)
        if s[0]<1e-8: continue
        t=v[0]; orient=np.cross(normals[a],centers[b]-centers[a]);
        if abs(t@orient)<1e-8: continue
        if t@orient<0: t=-t
        _,k=dihedral(normals[a],normals[b],t,threshold)
        if k: node(boundary,['g',int(np.bincount(categories[boundary].astype(int)).argmax()),k])
    return nodes

def descriptor(p):
    p=np.asarray(p,float); lengths=np.array([np.linalg.norm(p[i]-p[j]) for i,j in [(0,1),(1,2),(2,0)]])
    if min(lengths)<1e-9 or np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))<1e-8*max(lengths)**2: raise ValueError('degenerate triangle')
    angles=[]
    for i in range(3):
        a=p[(i+1)%3]-p[i]; b=p[(i+2)%3]-p[i]; angles.append(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    return np.r_[lengths/max(lengths),angles]

def similarity(a,b,lambda_phi=1.,lambda_r=1.):
    return float(np.exp(-lambda_phi*np.abs(a[3:]-b[3:]).sum()-lambda_r*np.abs(a[:3]-b[:3]).sum()))

def topology(nodes,dc,T):
    p=np.array([n['xyz'] for n in nodes]); d=cdist(p,p); np.fill_diagonal(d,np.inf)
    k=int(np.ceil(T*(len(p)-1))); edges=set()
    for i in range(len(p)):
        for j in np.argsort(d[i])[:k]:
            if d[i,j]<dc: edges.add(tuple(sorted((i,int(j)))))
    # bounded greedy clusters (complete-link distance criterion)
    clusters=[]
    for i in range(len(p)):
        for c in clusters:
            if all(d[i,j]<dc for j in c): c.append(i); break
        else: clusters.append([i])
    adj={i:set() for i in range(len(clusters))}
    for i,j in combinations(adj,2):
        if min(d[a,b] for a in clusters[i] for b in clusters[j])<2*dc: adj[i].add(j); adj[j].add(i)
    tris=[]
    for c in clusters:
        valid=[]
        for t in combinations(c,3):
            if all(tuple(sorted(e)) in edges for e in combinations(t,2)):
                try: valid.append((t,descriptor(p[list(t)])))
                except ValueError: pass
        tris.append(valid)
    return p,clusters,adj,tris

def match(query,reference,dc=3.,T=1.,h=5,minimum=.7,lambda_phi=1.,lambda_r=1.):
    if not query or not reference: return []
    q,qc,qa,qt=topology(query,dc,T); r,rc,ra,rt=topology(reference,dc,T)
    votes=[]; anchors={}
    for ci,ts in enumerate(qt):
        by_cluster=[]
        for cj,rs in enumerate(rt):
            candidates=[]
            for a,ad in ts:
                local=[]
                for b,_ in rs:
                    for bp in permutations(b):
                        if any(query[i]['code']!=reference[j]['code'] for i,j in zip(a,bp)): continue
                        score=similarity(ad,descriptor(r[list(bp)]),lambda_phi,lambda_r)
                        if score>=minimum: local.append((score,a,bp))
                candidates.extend(sorted(local,reverse=True)[:h])
            if candidates: by_cluster.append((sum(x[0] for x in candidates),cj,candidates))
        if by_cluster:
            _,cj,cs=max(by_cluster,key=lambda x:x[0]); anchors[ci]=cj
            for score,a,b in cs:
                votes.extend((score,i,j) for i,j in zip(a,b))
    chosen={}; used=set()
    for score,i,j in sorted(votes,reverse=True):
        if i not in chosen and j not in used: chosen[i]=j; used.add(j)
    # Local cross-cluster refinement using distance consistency to matched anchors.
    for ci,c in enumerate(qc):
        neighbors=[ci]+sorted(qa[ci]); allowed=set()
        for n in neighbors:
            if n in anchors: allowed.update([anchors[n],*ra[anchors[n]]])
        for i in c:
            if i in chosen: continue
            support=[a for a in chosen if any(a in qc[n] for n in neighbors)]
            if not support: continue
            opts=[]
            for cj in allowed:
                for j in rc[cj]:
                    if j in used or query[i]['code']!=reference[j]['code']: continue
                    score=np.exp(-np.mean([abs(np.linalg.norm(q[i]-q[a])-np.linalg.norm(r[j]-r[chosen[a]])) for a in support])/dc)
                    if score>=minimum: opts.append((score,j))
            if opts: _,j=max(opts); chosen[i]=j; used.add(j)
    return sorted(chosen.items())

def pose(query,reference,matches,K):
    if len(matches)<6: raise ValueError('At least six correspondences required by this reference implementation')
    X=np.array([reference[j]['xyz'] for i,j in matches],np.float64); u=np.array([query[i]['uv'] for i,j in matches],np.float64)
    ok,v,t,inliers=cv2.solvePnPRansac(X,u,np.asarray(K,float),None,iterationsCount=1000,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<6: raise ValueError('PnP rejected')
    idx=inliers.ravel(); v,t=cv2.solvePnPRefineLM(X[idx],u[idx],np.asarray(K,float),None,v,t)
    R=cv2.Rodrigues(v)[0]; P=np.eye(4); P[:3,:3]=R; P[:3,3]=t.ravel()
    return dict(T_cw=P.tolist(),inliers=idx.tolist(),matches=matches)
