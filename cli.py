import argparse,json,numpy as np
from pathlib import Path
from ssg_core import extract_semantic_nodes,extract_structural_nodes,hierarchical_match,estimate_pose_pnp,nodes_from_json,save_nodes

def main():
 p=argparse.ArgumentParser(description='SSG Core: real registered depth/labels -> dual-layer graph -> hierarchical matches -> PnP')
 sub=p.add_subparsers(dest='cmd',required=True)
 e=sub.add_parser('extract'); e.add_argument('npz'); e.add_argument('out')
 m=sub.add_parser('match'); m.add_argument('query'); m.add_argument('reference'); m.add_argument('--dc',type=float,default=3.); m.add_argument('--T',type=float,default=.85); m.add_argument('--h',type=int,default=12); m.add_argument('--threshold',type=float,default=.5); m.add_argument('--out',default='matches.json')
 z=sub.add_parser('localize'); z.add_argument('query'); z.add_argument('reference'); z.add_argument('K'); z.add_argument('--matches',default='matches.json'); z.add_argument('--out',default='pose.json')
 a=p.parse_args()
 if a.cmd=='extract':
  d=np.load(a.npz); K=d['K']; sem=extract_semantic_nodes(d['depth'],K,d['instances'],d['categories']); st=extract_structural_nodes(d['depth'],K,d['patches'],d['categories']); save_nodes(a.out,sem+st); print(f'wrote {len(sem)} semantic and {len(st)} structural nodes: {a.out}')
 elif a.cmd=='match':
  q=nodes_from_json(a.query); r=nodes_from_json(a.reference); matches=hierarchical_match(q,r,a.dc,a.T,a.h,a.threshold); json.dump([{'query':i,'reference':j,'score':s,'layer':l} for i,j,s,l in matches],open(a.out,'w'),indent=2); print(f'{len(matches)} node correspondences written to {a.out}')
 else:
  q=nodes_from_json(a.query); r=nodes_from_json(a.reference); raw=json.load(open(a.matches)); matches=[(x['query'],x['reference'],x['score'],x['layer']) for x in raw]; K=np.loadtxt(a.K).reshape(3,3); result=estimate_pose_pnp(q,r,matches,K); json.dump(result,open(a.out,'w'),indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
