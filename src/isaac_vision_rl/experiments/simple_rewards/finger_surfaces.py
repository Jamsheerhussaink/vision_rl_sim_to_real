"""Independent distances from real inward finger mesh patches to opposite cube faces.

Geometry is loaded from the configured USD, in each rigid body's coordinates.
The contact-region boxes select mesh patches, never replace them with box centers.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch


def closest_surface_point(point, triangles):
    """Exact nearest point on triangle interiors or edges (batch of query points)."""
    a,b,c=triangles.unbind(1)
    ab=b-a;ac=c-a
    n=torch.linalg.cross(ab,ac)
    nn=n.square().sum(-1).clamp_min(1e-20)
    p=point[:,None,:]
    projection=p-(((p-a)*n).sum(-1)/nn)[...,None]*n
    v=projection-a
    d00=ab.square().sum(-1);d01=(ab*ac).sum(-1);d11=ac.square().sum(-1)
    denom=(d00*d11-d01.square()).clamp_min(1e-20)
    d20=(v*ab).sum(-1);d21=(v*ac).sum(-1)
    u=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
    inside=(u>=0)&(w>=0)&(u+w<=1)&(nn>1e-18)
    candidates=[projection]
    distances=[torch.where(inside,(projection-p).square().sum(-1),torch.inf)]
    for x,y in ((a,b),(b,c),(c,a)):
        edge=y-x
        t=(((p-x)*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-20)).clamp(0,1)
        q=x+t[...,None]*edge
        candidates.append(q);distances.append((q-p).square().sum(-1))
    d=torch.stack(distances,dim=-1)
    q=torch.stack(candidates,dim=-2)
    flat=d.flatten(1).argmin(-1)
    nearest=q.flatten(1,2)[torch.arange(len(point),device=point.device),flat]
    return torch.linalg.vector_norm(nearest-point,dim=-1),nearest


def choose_opposite_sides(fixed,moving,cube,cube_rotation,fixed_inward=None):
    """Choose horizontal-ish opposite faces once, before a grasp attempt."""
    outward=fixed-moving if fixed_inward is None else -fixed_inward
    direction=torch.bmm(cube_rotation.transpose(1,2),outward.unsqueeze(-1)).squeeze(-1)
    horizontal=cube_rotation[:,2,:].abs()<.7
    axis=torch.where(horizontal,direction.abs(),-torch.ones_like(direction)).argmax(-1)
    sign=torch.where(direction.gather(1,axis[:,None]).squeeze(1)>=0,1.,-1.)
    return axis,sign


def opposite_targets(cube,rotation,axis,sign,half_size=.025):
    offset=torch.nn.functional.one_hot(axis,3).to(cube.dtype)*(sign*half_size)[:,None]
    delta=torch.bmm(rotation,offset.unsqueeze(-1)).squeeze(-1)
    return cube+delta,cube-delta


def patch_penetration(body_position,body_rotation,triangles,cube,cube_rotation,half_size=.025):
    """Conservative sampled intrusion depth of the existing inner mesh patch."""
    samples=torch.cat((triangles.reshape(-1,3),triangles.mean(1)),dim=0)
    world=torch.matmul(samples,body_rotation.transpose(1,2))+body_position[:,None,:]
    rel=torch.bmm(world-cube[:,None,:],cube_rotation)
    return (half_size-rel.abs()).min(-1).values.clamp_min(0).max(-1).values


def distance_to_face(target,body_position,body_rotation,triangles,cube,cube_rotation,half_size=.025):
    local=torch.bmm(body_rotation.transpose(1,2),(target-body_position).unsqueeze(-1)).squeeze(-1)
    distance,nearest=closest_surface_point(local,triangles)
    nearest_world=torch.bmm(body_rotation,nearest.unsqueeze(-1)).squeeze(-1)+body_position
    penetration=patch_penetration(body_position,body_rotation,triangles,cube,cube_rotation,half_size)
    return torch.maximum(distance,penetration),nearest_world,penetration


def face_region_targets(center,rotation,axis,half_extent=.006):
    """Nine fixed surface targets in a central 12 x 12 mm cube-side region.

    The samples stay exactly on the side plane: there is no free air-gap tolerance.
    Taking the minimum of their distances is continuous, even if the nearest sample changes.
    """
    coords=torch.tensor([-half_extent,0.,half_extent],device=center.device,dtype=center.dtype)
    grid=torch.cartesian_prod(coords,coords)
    tangent1=torch.nn.functional.one_hot((axis+1)%3,3).to(center.dtype)
    tangent2=torch.nn.functional.one_hot((axis+2)%3,3).to(center.dtype)
    offsets=grid[None,:,0,None]*tangent1[:,None,:]+grid[None,:,1,None]*tangent2[:,None,:]
    return center[:,None,:]+torch.bmm(offsets,rotation.transpose(1,2))


def distance_to_region(targets,body_position,body_rotation,triangles,cube,cube_rotation):
    local=torch.bmm(targets-body_position[:,None,:],body_rotation)
    distances,nearest=closest_surface_point(local.flatten(0,1),triangles)
    distances=distances.reshape(len(targets),-1)
    nearest=nearest.reshape(len(targets),-1,3)
    index=distances.argmin(-1);batch=torch.arange(len(targets),device=targets.device)
    point=nearest[batch,index]
    point_world=torch.bmm(body_rotation,point.unsqueeze(-1)).squeeze(-1)+body_position
    penetration=patch_penetration(body_position,body_rotation,triangles,cube,cube_rotation)
    center_distance=torch.maximum(distances[:,4],penetration)
    return (torch.maximum(distances[batch,index],penetration),point_world,penetration,
            targets[batch,index],center_distance)


class FingerSurfaces:
    """USD-backed inward patches. Fixed +X and moving -X match source face normals."""
    def __init__(self,usd_path,device,dtype):
        from pxr import Usd,UsdGeom,UsdPhysics
        from isaac_vision_rl.tasks.pickplace.mdp.rewards import FINGER_CONTACT_REGIONS
        stage=Usd.Stage.Open(str(usd_path));cache=UsdGeom.XformCache()
        if stage is None:raise ValueError('Cannot open finger geometry: '+str(usd_path))
        self.triangles={};self.audit={'target_region_half_extent_m':.006,'target_samples_per_side':9,'asset':str(usd_path),'sha256':hashlib.sha256(Path(usd_path).read_bytes()).hexdigest(),'patches':{}}
        for i,(_,name,lower,upper) in enumerate(FINGER_CONTACT_REGIONS):
            collected=[]
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                body=prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
                if not body or body.GetName()!=name:continue
                mesh=UsdGeom.Mesh(prim)
                points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
                relative=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
                points=(np.c_[points,np.ones(len(points))]@relative.T)[:,:3]
                indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());offset=0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    face=indices[offset:offset+count];offset+=count
                    for j in range(1,count-1):collected.append(points[face[[0,j,j+1]]])
            if not collected:raise ValueError('Missing collision mesh for '+name)
            tri=np.asarray(collected);center=tri.mean(1)
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            lengths=np.linalg.norm(normal,axis=-1)
            normal/=np.maximum(lengths[:,None],1e-20)
            inner=np.array([1. if i==0 else -1.,0.,0.])
            mask=((center>=lower)&(center<=upper)).all(-1)&((normal@inner)>.90)&(lengths>1e-12)
            tri=tri[mask]
            if len(tri)<10:raise ValueError('Cannot verify inward finger patch: '+name)
            self.triangles[name]=torch.tensor(tri,device=device,dtype=dtype)
            self.audit['patches'][name]={'triangles':len(tri),'bounds_min':tri.min((0,1)).tolist(),'bounds_max':tri.max((0,1)).tolist(),'inward_normal_local':inner.tolist()}

    def measure(self,robot,cube,rotation,state):
        from isaaclab.utils.math import matrix_from_quat
        bodies=[];centers=[]
        for name,tri in self.triangles.items():
            i=robot.find_bodies(name)[0][0]
            position=robot.data.body_pos_w.torch[:,i]
            rot=matrix_from_quat(robot.data.body_quat_w.torch[:,i])
            centers.append(torch.matmul(rot,tri.mean((0,1)))+position)
            bodies.append((position,rot,tri))
        axis,sign=choose_opposite_sides(*centers,cube,rotation,fixed_inward=bodies[0][1][:,:,0])
        latch=(state.stage==1)&~state.finger_sides_valid
        state.finger_axis[:]=torch.where(latch,axis,state.finger_axis)
        state.finger_sign[:]=torch.where(latch,sign,state.finger_sign)
        state.finger_sides_valid|=latch
        axis=torch.where(state.finger_sides_valid,state.finger_axis,axis)
        sign=torch.where(state.finger_sides_valid,state.finger_sign,sign)
        targets=opposite_targets(cube,rotation,axis,sign)
        result={'finger_target_axis':axis,'fixed_finger_side_sign':sign}
        for label,target,body in zip(('fixed','moving'),targets,bodies,strict=True):
            region=face_region_targets(target,rotation,axis)
            distance,nearest,penetration,selected,center_distance=distance_to_region(region,*body,cube,rotation)
            result[label+'_finger_distance']=distance
            result[label+'_finger_target_center']=target
            result[label+'_finger_target']=selected
            result[label+'_finger_center_distance']=center_distance
            result[label+'_finger_surface_point']=nearest
            result[label+'_finger_penetration']=penetration
        return result
