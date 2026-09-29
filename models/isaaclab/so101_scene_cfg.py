"""
Isaac Lab (Isaac Sim 6.1.0 / Isaac Lab v3.00 EA) Scene Configuration:
Defines the equivalent pick-and-place scene for SO-101 in Isaac Lab:
- Light green table with black legs
- Yellow cube (rigid body with PhysX contact material)
- Small cardboard box
- Eye-to-Hand Raytraced Depth Camera sensor
- Room lighting (dome + key directional)
"""
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


@configclass
class SO101PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-101 Pick and Place workcell in Isaac Lab."""

    # --------------------------------------------------------------------------
    # 1. Studio Ground Plane & Room Lighting
    # --------------------------------------------------------------------------
    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.88, 0.88, 0.88),
            size=(5.0, 5.0),
        ),
    )

    # Overhead ceiling room light (soft white LED panel)
    ceiling_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CeilingLight",
        spawn=sim_utils.DistantLightCfg(
            color=(0.85, 0.85, 0.82),
            intensity=2500.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.10, 1.60), rot=(0.92, 0.0, 0.38, 0.0)),
    )

    # Dome ambient room fill
    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.7, 0.7, 0.75),
            intensity=800.0,
        ),
    )

    # --------------------------------------------------------------------------
    # 2. Table: Green Top with Black Legs (Matched from physical setup)
    # --------------------------------------------------------------------------
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.64, 0.60, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.58, 0.14), roughness=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.385)),
    )

    # --------------------------------------------------------------------------
    # 3. Yellow Cube (5cm, 55g mass, high friction)
    # --------------------------------------------------------------------------
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/YellowCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.88, 0.02), roughness=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.4,
                dynamic_friction=1.2,
                restitution=0.0, # no-bounce
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.055),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.06, 0.425)),
    )

    # --------------------------------------------------------------------------
    # 4. Calibrated Depth Camera (Eye-to-Hand)
    # --------------------------------------------------------------------------
    # Mounting position: [0.42, 0.30, 0.68], looking at table center [0.28, -0.03, 0.41]
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WorkspaceCamera",
        update_period=0.02, # 50 Hz capture
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.15, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.42, 0.30, 0.68),
            rot=(0.18, 0.42, -0.85, 0.25),
            convention="ros",
        ),
    )
