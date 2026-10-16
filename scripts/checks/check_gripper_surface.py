"""Offline visible-CAD/contact-position comparison from recorded evaluation traces."""
import json,numpy as np,trimesh
from pxr import Usd,UsdGeom
import argparse
parser=argparse.ArgumentParser(description="Measure mean contact positions against visible CAD triangles; this is not an exact surface-gap measurement.")
parser.add_argument('--asset', required=True)
parser.add_argument('--trace', required=True)
parser.add_argument('--output', required=True)
args=parser.parse_args()
s=Usd.Stage.Open(args.asset);cache=UsdGeom.XformCache();rows=json.load(open(args.trace));report={}

for name,meshname in [('static_pad','wrist_roll_follower_so101_v1'),('moving_pad','moving_jaw_so101_v1')]:
 pad=next(p for p in s.Traverse() if p.GetName()==name);body=pad.GetParent();parts=[]
 for p in Usd.PrimRange(body):
  if p.GetTypeName()!='Mesh' or p.GetName()!=meshname or UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
  m=UsdGeom.Mesh(p);points=np.array(m.GetPointsAttr().Get());mat=np.array(cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(body).GetInverse()).T;points=(np.c_[points,np.ones(len(points))]@mat.T)[:,:3];faces=[];idx=list(m.GetFaceVertexIndicesAttr().Get());off=0
  for count in m.GetFaceVertexCountsAttr().Get():
   poly=idx[off:off+count];off+=count
   faces.extend([[poly[0],poly[i],poly[i+1]] for i in range(1,count-1)])
  parts.append(trimesh.Trimesh(points,faces,process=False))
 mesh=trimesh.util.concatenate(parts);pts=[];selected=[r for r in rows if r['cube_height_m']>.47][::10]
 for r in selected:
  d=r['collision_pads'][name];c=np.array(d['corners_world_m']);axes=np.column_stack([c[4]-c[0],c[2]-c[0],c[1]-c[0]]);R=axes/np.linalg.norm(axes,axis=0);center=c.mean(0);offset=np.array(pad.GetAttribute('xformOp:translate').Get())
  pts.extend([(np.array(pt)-center)@R+offset for pt in d['mean_contact_points_world_m']])
 if not pts:
  report[name]={'samples':0};continue
 nearest,dist,_=trimesh.proximity.closest_point_naive(mesh,np.array(pts))
 report[name]={'samples':len(pts),'minimum_mm':float(dist.min()*1000),'median_mm':float(np.median(dist)*1000),'maximum_mm':float(dist.max()*1000)}
print(json.dumps(report,indent=2))
from pathlib import Path
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
