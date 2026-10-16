#!/usr/bin/env python3
"""Build a separate PhysX robot asset with anchored-root support and calibrated pads."""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
source = Usd.Stage.Open(str(ROOT / "usd/robot.usd"))
output = ROOT / "usd/robot_grasp_surface_v3.usda"
source.Flatten().Export(str(output))
stage = Usd.Stage.Open(str(output))
# Replace the imported root joint with the one authored by fix_root_link.
# Keeping both gives the Newton viewer two connections to the same base body.
obsolete = [str(p.GetPath()) for p in stage.Traverse() if p.GetName() == "root_joint"]
for path in obsolete:
    stage.RemovePrim(path)
for prim in stage.Traverse():
    relation = prim.GetRelationship("isaac:physics:robotJoints")
    if relation:
        relation.SetTargets([p for p in relation.GetTargets() if str(p) not in obsolete])
# Expose instance meshes before authoring collision approximation overrides.
instances = [str(p.GetPath()) for p in stage.Traverse() if p.IsInstance()]
for path in instances:
    stage.GetPrimAtPath(path).SetInstanceable(False)
for prim in stage.Traverse():
    for prop in list(prim.GetProperties()):
        if prop.GetName().startswith("newton:"):
            prim.RemoveProperty(prop.GetName())
    schemas = [name for name in prim.GetAppliedSchemas() if not name.startswith("Newton")]
    if schemas:
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
    elif prim.HasMetadata("apiSchemas"):
        prim.ClearMetadata("apiSchemas")
    if prim.GetName() == "moving_pad":
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.002, -0.065, 0.0188))
    if prim.GetName() in ("moving_pad", "static_pad"):
        # Historical helper boxes extended beyond the visible CAD surfaces.
        # Keep their outlines for comparison, but they must not carry the cube.
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveProperty("physics:collisionEnabled")
    if prim.GetName() == "gripper_frame_link":
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.032, 0.0, -0.076))
        # A fixed virtual tool marker must still have positive inertial values in PhysX.
        # 1 gram is negligible relative to the arm, unlike the ill-conditioned 1e-9 kg/zero inertia source.
        prim.GetAttribute("physics:mass").Set(0.001)
        prim.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(1e-7, 1e-7, 1e-7))
    if prim.GetName() == "gripper_frame_joint":
        prim.GetAttribute("physics:localPos0").Set(Gf.Vec3f(0.032, 0.0, -0.076))
    if prim.GetName() == "wrist_roll" and prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.GetLowerLimitAttr().Set(-154.6986)
        joint.GetUpperLimitAttr().Set(84.7966)
    if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetTypeName() == "Mesh" and "gripper_link" in str(prim.GetPath()):
        prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexDecomposition")
        if prim.GetName() in ("moving_jaw_so101_v1", "wrist_roll_follower_so101_v1"):
            # Fit the visible finger surfaces instead of coarse outward hulls.
            prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
            for name, dtype, value in (
                ("errorPercentage", Sdf.ValueTypeNames.Float, 0.1),
                ("hullVertexLimit", Sdf.ValueTypeNames.Int, 64),
                ("maxConvexHulls", Sdf.ValueTypeNames.Int, 128),
                ("minThickness", Sdf.ValueTypeNames.Float, 0.0001),
                ("shrinkWrap", Sdf.ValueTypeNames.Bool, True),
                ("voxelResolution", Sdf.ValueTypeNames.Int, 2000000),
            ):
                prim.CreateAttribute(f"physxConvexDecompositionCollision:{name}", dtype).Set(value)
stage.GetRootLayer().Save()
print(output)
