"""Runnable SSG Core reference implementation.

Inputs are registered metric depth, camera intrinsics, instance/category maps,
and pre-segmented continuous surface-patch labels. This module does not invent
an instance or surface segmentation model; those are explicit upstream inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations, permutations
from collections import defaultdict, deque
import json
import numpy as np
import cv2

@dataclass
class Node:
    xyz: list[float]
    uv: list[float]
    layer: str                 # 'semantic' or 'structural'
    category: int
    sign: int = 0              # structural: -1 concave, +1 convex, 0 unknown
    node_id: int = -1
    @property
    def code(self):
        return (self.layer, int(self.category), int(self.sign) if self.layer == 'structural' else 0)

@dataclass
class Triangle:
    vertices: tuple[int,int,int]
    descriptor: list[float]


def _unit(v):
    v=np.asarray(v,dtype=float); n=np.linalg.norm(v)
    if n < 1e-12: raise ValueError('zero-length direction vector')
    return v/n


def signed_dihedral(n_a, n_b, tangent, threshold=0.15):
    """Eq.(3): signed angle in radians and {-1,0,+1} convexity code.
    Sign convention depends on consistently oriented normals and tangent.
    """
    a,b,t=map(_unit,(n_a,n_b,tangent))
    delta=float(np.arctan2(np.dot(t,np.cross(a,b)),np.dot(a,b)))
    sign=1 if delta>threshold else (-1 if delta < -threshold else 0)
    return delta,sign


def backproject(depth, K):
    depth=np.asarray(depth,dtype=float); K=np.asarray(K,dtype=float)
    if depth.ndim!=2 or K.shape!=(3,3): raise ValueError('depth must be HxW and K must be 3x3')
    y,x=np.indices(depth.shape); pix=np.stack((x,y,np.ones_like(x)),axis=-1)
    return (pix @ np.linalg.inv(K).T)*depth[...,None]


def _majority(values):
    values=np.asarray(values,dtype=int)
    values=values[values>=0]
    if not len(values): return -1
    return int(np.bincount(values).argmax())


def extract_semantic_nodes(depth,K,instances,categories,min_pixels=8):
    xyz=backproject(depth,K); depth=np.asarray(depth)
    instances=np.asarray(instances); categories=np.asarray(categories)
    if instances.shape!=depth.shape or categories.shape!=depth.shape: raise ValueError('all maps must have same registered image dimensions')
    valid=np.isfinite(depth)&(depth>0); out=[]
    for iid in np.unique(instances):
        if iid<=0: continue
        mask=(instances==iid)&valid; yy,xx=np.where(mask)
        if len(xx)<min_pixels: continue
        out.append(Node(xyz[yy,xx].mean(0).tolist(),[float(xx.mean()),float(yy.mean())], 'semantic',_majority(categories[mask]),0,len(out)))
    return out


def extract_structural_nodes(depth,K,patches,categories,threshold=.15,min_pixels=8,merge_regions=True):
    """Fit local patch planes, estimate shared-boundary tangent and Eq.(3).
    Patch labels are expected from a real surface segmentation stage.
    Adjacent boundary segments with the same category/sign are optionally
    merged by connected image-space components.
    """
    depth=np.asarray(depth,dtype=float); patches=np.asarray(patches); categories=np.asarray(categories)
    if patches.shape!=depth.shape or categories.shape!=depth.shape: raise ValueError('patch/category maps must match depth')
    xyz=backproject(depth,K); valid=np.isfinite(depth)&(depth>0)
    normals={}; centers={};
    for pid in np.unique(patches):
        if pid<=0: continue
        pts=xyz[(patches==pid)&valid]
        if len(pts)<max(min_pixels,3): continue
        c=pts.mean(0); _,s,vh=np.linalg.svd(pts-c,full_matrices=False)
        if len(s)<2 or s[1]<1e-8: continue
        n=_unit(vh[-1]); # orient toward camera origin
        if np.dot(n,c)>0: n=-n
        normals[int(pid)]=n; centers[int(pid)]=c
    raw=[]; kernel=np.ones((3,3),np.uint8)
    for a,b in combinations(normals,2):
        ma=(patches==a).astype(np.uint8); mb=(patches==b).astype(np.uint8)
        boundary=(cv2.dilate(ma,kernel)>0)&(mb>0)&valid
        pts=xyz[boundary]
        if len(pts)<min_pixels: continue
        _,s,vh=np.linalg.svd(pts-pts.mean(0),full_matrices=False)
        if s[0]<1e-8: continue
        tangent=_unit(vh[0]); orient=np.cross(normals[a],centers[b]-centers[a])
        if abs(np.dot(tangent,orient))<1e-8: continue
        if np.dot(tangent,orient)<0: tangent=-tangent
        _,sgn=signed_dihedral(normals[a],normals[b],tangent,threshold)
        if sgn==0: continue
        yy,xx=np.where(boundary)
        raw.append((boundary,Node(pts.mean(0).tolist(),[float(xx.mean()),float(yy.mean())],'structural',_majority(categories[boundary]),sgn,len(raw))))
    if not merge_regions: return [n for _,n in raw]
    # Merge touching boundary masks only when category and convexity code agree.
    result=[]; used=set()
    for i,(mi,ni) in enumerate(raw):
        if i in used: continue
        group=[i]; used.add(i); changed=True
        while changed:
            changed=False
            for j,(mj,nj) in enumerate(raw):
                if j in used or ni.category!=nj.category or ni.sign!=nj.sign: continue
                if np.any((cv2.dilate(np.logical_or.reduce([raw[k][0] for k in group]).astype(np.uint8),kernel)>0) & mj):
                    group.append(j); used.add(j); changed=True
        pts=np.concatenate([xyz[raw[k][0]] for k in group],axis=0)
        coords=np.concatenate([np.column_stack(np.where(raw[k][0])) for k in group],axis=0)
        yy,xx=coords[:,0],coords[:,1]
        result.append(Node(pts.mean(0).tolist(),[float(xx.mean()),float(yy.mean())],'structural',ni.category,ni.sign,len(result)))
    return result


def triangle_descriptor(points):
    """Eq.(6) descriptor: normalized three edge ratios + three angles.
    Edge order is (01,12,20); vertex ordering must be consistent when comparing.
    """
    p=np.asarray(points,dtype=float)
    if p.shape!=(3,3): raise ValueError('triangle points must be 3x3')
    edges=[(0,1),(1,2),(2,0)]
    lengths=np.array([np.linalg.norm(p[i]-p[j]) for i,j in edges])
    if lengths.min()<1e-9 or np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))<1e-8*lengths.max()**2:
        raise ValueError('degenerate triangle')
    angles=[]
    for i in range(3):
        a=p[(i+1)%3]-p[i]; b=p[(i+2)%3]-p[i]
        angles.append(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    return np.r_[lengths/lengths.max(),np.asarray(angles)]


def triangle_discrepancy(d1,d2):
    d1=np.asarray(d1); d2=np.asarray(d2)
    # L1 discrepancies stated in Section III-C; exact weighting/normalization
    # follows Eq.(6) coefficients supplied by the user at runtime.
    return float(np.abs(d1[3:]-d2[3:]).sum()),float(np.abs(d1[:3]-d2[:3]).sum())


def triangle_similarity(d1,d2,lambda_angle=1.,lambda_ratio=1.):
    da,dr=triangle_discrepancy(d1,d2)
    return float(np.exp(-lambda_angle*da-lambda_ratio*dr))


def _cluster_graph(nodes,dc,neighbor_ratio):
    xyz=np.asarray([n.xyz for n in nodes],float); N=len(nodes)
    if N==0:return xyz,[],{},set()
    dist=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=2); np.fill_diagonal(dist,np.inf)
    edges=set(); k=max(1,int(np.ceil(neighbor_ratio*max(1,N-1))))
    for i in range(N):
        for j in np.argsort(dist[i])[:k]:
            if dist[i,j]<dc: edges.add(tuple(sorted((i,int(j)))))
    # connected components of the retained graph
    adj={i:set() for i in range(N)}
    for i,j in edges: adj[i].add(j); adj[j].add(i)
    comps=[]; seen=set()
    for i in range(N):
        if i in seen: continue
        q=[i]; seen.add(i); comp=[]
        while q:
            u=q.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); q.append(v)
        comps.append(sorted(comp))
    # Connect isolated node to nearest node in nearest non-singleton cluster.
    for ci,c in enumerate(comps):
        if len(c)==1 and len(comps)>1:
            others=[j for j,x in enumerate(comps) if j!=ci and len(x)>1]
            if others:
                j=min(others,key=lambda z:min(dist[c[0],v] for v in comps[z]))
                v=min(comps[j],key=lambda v:dist[c[0],v]); edges.add(tuple(sorted((c[0],v))))
    # Inter-cluster links between nearest compatible nodes in spatially nearby clusters.
    inter=defaultdict(set)
    for ci,c1 in enumerate(comps):
        for cj in range(ci+1,len(comps)):
            c2=comps[cj]; pairs=[(dist[a,b],a,b) for a in c1 for b in c2]
            d,a,b=min(pairs)
            if d<2*dc and nodes[a].code==nodes[b].code:
                inter[ci].add(cj); inter[cj].add(ci); edges.add(tuple(sorted((a,b))))
    return xyz,comps,inter,edges


def _triangles(nodes,clusters,edges):
    out=[]
    for c in clusters:
        ts=[]
        for tri in combinations(c,3):
            if all(tuple(sorted(e)) in edges for e in combinations(tri,2)):
                try: ts.append(Triangle(tri,triangle_descriptor([nodes[i].xyz for i in tri]).tolist()))
                except ValueError: pass
        out.append(ts)
    return out


def hierarchical_match(query,reference,dc=3.,neighbor_ratio=.85,top_h=12,
                        min_similarity=.5,lambda_angle=1.,lambda_ratio=1.):
    """Apply the manuscript's two-layer hierarchical matching independently.
    Returns (query_index, reference_index, score, layer) correspondences.
    Sparse two-node/isolated structures are retained as edges/anchors and can
    participate in the local inter-cluster refinement stage.
    """
    final=[]
    for layer in ('semantic','structural'):
        q=[n for n in query if n.layer==layer]; r=[n for n in reference if n.layer==layer]
        if len(q)<1 or len(r)<1: continue
        _,qc,qi,qe=_cluster_graph(q,dc,neighbor_ratio); _,rc,ri,re=_cluster_graph(r,dc,neighbor_ratio)
        qt=_triangles(q,qc,qe); rt=_triangles(r,rc,re)
        votes=[]; cluster_anchor={}
        # Complete search of compatible map triangles within each reference cluster.
        for qci,qtris in enumerate(qt):
            cluster_scores=[]
            for rci,rtris in enumerate(rt):
                cand=[]
                for qtri in qtris:
                    qids=qtri.vertices
                    for rtri in rtris:
                        for perm in permutations(rtri.vertices):
                            if not all(q[i].code==r[j].code for i,j in zip(qids,perm)): continue
                            rd=triangle_descriptor([r[j].xyz for j in perm])
                            score=triangle_similarity(qtri.descriptor,rd,lambda_angle,lambda_ratio)
                            if score>=min_similarity: cand.append((score,qids,perm))
                cand=sorted(cand,key=lambda x:x[0],reverse=True)[:top_h]
                if cand: cluster_scores.append((sum(x[0] for x in cand),rci,cand))
            if cluster_scores:
                _,best,cands=max(cluster_scores,key=lambda x:x[0]); cluster_anchor[qci]=best
                for score,qi0,ri0 in cands:
                    votes.extend((score,int(a),int(b),layer) for a,b in zip(qi0,ri0))
        # one-to-one correspondence constraint
        chosen={}; used=set()
        for score,i,j,l in sorted(votes,key=lambda x:x[0],reverse=True):
            if i not in chosen and j not in used: chosen[i]=(j,score,l); used.add(j)
        # Local inter-cluster search using distance consistency to established anchors.
        for qci,c in enumerate(qc):
            allowed_ref_clusters=set()
            for adj in {qci,*qi.get(qci,set())}:
                if adj in cluster_anchor:
                    a=cluster_anchor[adj]; allowed_ref_clusters.add(a); allowed_ref_clusters.update(ri.get(a,set()))
            support=[i for i in chosen if i in c or any(i in qc[x] for x in qi.get(qci,set()))]
            for i in c:
                if i in chosen or not support or not allowed_ref_clusters: continue
                opts=[]
                for rci in allowed_ref_clusters:
                    for j in rc[rci]:
                        if j in used or q[i].code!=r[j].code: continue
                        errs=[]
                        for a in support:
                            ja=chosen[a][0]
                            errs.append(abs(np.linalg.norm(np.asarray(q[i].xyz)-q[a].xyz)-np.linalg.norm(np.asarray(r[j].xyz)-r[ja].xyz)))
                        score=float(np.exp(-np.mean(errs)/max(dc,1e-6)))
                        if score>=min_similarity: opts.append((score,j))
                if opts:
                    score,j=max(opts); chosen[i]=(j,score,layer); used.add(j)
        final.extend((i,j,s,l) for i,(j,s,l) in chosen.items())
    # Deduplicate by layer and query index; retain highest score.
    best={}
    for i,j,s,l in final:
        key=(l,i)
        if key not in best or s>best[key][2]: best[key]=(i,j,s,l)
    return sorted(best.values(),key=lambda x:(x[3],x[0]))


def estimate_pose_pnp(query_nodes,reference_nodes,matches,K,reprojection_error=3.,min_inliers=6):
    if len(matches)<min_inliers: raise ValueError(f'need >= {min_inliers} node correspondences; got {len(matches)}')
    X=[]; uv=[]
    for qi,ri,score,layer in matches:
        X.append(reference_nodes[ri].xyz); uv.append(query_nodes[qi].uv)
    X=np.asarray(X,dtype=np.float64); uv=np.asarray(uv,dtype=np.float64); K=np.asarray(K,dtype=np.float64)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(X,uv,K,None,iterationsCount=1000,reprojectionError=reprojection_error,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<min_inliers: raise RuntimeError('PnP-RANSAC rejected the graph correspondences')
    ids=inliers.ravel(); rvec,tvec=cv2.solvePnPRefineLM(X[ids],uv[ids],K,None,rvec,tvec)
    R,_=cv2.Rodrigues(rvec); T_cw=np.eye(4); T_cw[:3,:3]=R; T_cw[:3,3]=tvec.ravel()
    return {'T_cw':T_cw.tolist(),'inlier_indices':ids.tolist(),'inlier_count':int(len(ids)),'match_count':len(matches)}


def nodes_from_json(path):
    data=json.load(open(path)); return [Node(**x) for x in data]
def save_nodes(path,nodes):
    json.dump([asdict(n) for n in nodes],open(path,'w'),indent=2)
