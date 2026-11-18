import argparse,json
from pathlib import Path
import numpy as np
from core import extract, match, pose
p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
a=sub.add_parser('extract'); a.add_argument('input'); a.add_argument('output')
a=sub.add_parser('localize'); a.add_argument('query'); a.add_argument('reference'); a.add_argument('intrinsics'); a.add_argument('output'); a.add_argument('--dc',type=float,default=3); a.add_argument('--T',type=float,default=1); a.add_argument('--h',type=int,default=5)
a=sub.add_parser('demo'); a.add_argument('--output',default='demo_result.json')
a=p.parse_args()
if a.cmd=='extract':
    z=np.load(a.input); result=extract(*(z[k] for k in ['depth','K','instances','categories','patches']))
elif a.cmd=='localize':
    q=json.loads(Path(a.query).read_text()); r=json.loads(Path(a.reference).read_text()); K=np.loadtxt(a.intrinsics); result=pose(q,r,match(q,r,a.dc,a.T,a.h),K)
else:
    K=np.array([[500,0,320],[0,500,240],[0,0,1.]])
    X=np.array([[0,0,5],[1,0,5],[0,1,5.2],[1,1,5.5],[.3,.7,6],[.8,.2,6.2]])
    r=[dict(xyz=x.tolist(),code=['s',i]) for i,x in enumerate(X)]; Y=X+np.array([.2,-.1,.4]); u=Y@K.T; u=u[:,:2]/u[:,2:]; q=[dict(xyz=x.tolist(),uv=v.tolist(),code=['s',i]) for i,(x,v) in enumerate(zip(Y,u))]; result=pose(q,r,match(q,r),K); assert np.allclose(np.array(result['T_cw'])[:3,3],[.2,-.1,.4],atol=1e-4); print('Synthetic pose test passed')
Path(a.output).write_text(json.dumps(result,indent=2))
