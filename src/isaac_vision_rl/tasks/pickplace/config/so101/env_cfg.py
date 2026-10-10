# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claovementse

import os
from typing import Any, cast

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp

# Paths to assets
_ISAAC_VISION_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
USD_DIR = os.path.join(_ISAAC_VISION_RL_ROOT, "usd")


@configclass
class PickplacePhysicsCfg(PresetCfg):
    """Physics presets for SO-101 pick and place task."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default: PhysxAutoCfg = physx


##
# Scene definition
##


@configclass
class PickplaceSceneCfg(InteractiveSceneCfg):
    """Physical workcell styled after env_final_sim.png."""

    # 1. Hall lighting: ambient fill plus three local overhead panels.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.94, 0.96, 1.0), intensity=600.0),
    )
    overhead_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OverheadFront",
        spawn=sim_utils.DiskLightCfg(color=(1.0, 0.98, 0.95), intensity=5000.0, radius=0.30),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.05, -0.30, 1.35),
            rot=(0.173648, 0.0, 0.0, 0.984808),  # Downward, tilted toward workspace.
        ),
    )
    overhead_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OverheadLeft",
        spawn=sim_utils.DiskLightCfg(color=(0.94, 0.97, 1.0), intensity=2500.0, radius=0.35),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.40, 0.20, 1.30),
            rot=(0.0, -0.173648, 0.0, 0.984808),
        ),
    )
    overhead_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OverheadTable",
        spawn=sim_utils.DiskLightCfg(color=(1.0, 1.0, 0.98), intensity=6000.0, radius=0.22),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.25, 0.18, 1.20),
            rot=(0.0, 0.0, 0.0, 1.0),  # Directly illuminates the tabletop and cube.
        ),
    )

    # 2. Dark Green Table: keep Z=0.40m; extend XY to fill square policy-camera FOV.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(USD_DIR, "table_reference.usda"),
            scale=(2.0, 2.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.15, 0.0)),
    )

    # 3. SO-101 6-DOF Robot Arm (seated on tabletop at Z = 0.405m)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(USD_DIR, "robot_grasp_calibrated.usda"),
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,  # The tabletop arm must be anchored, not free-floating.
            ),
            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.26, 0.08, 0.405),  # 2 cm farther left and 3 cm farther back.
            rot=(0.0, 0.0, -0.258819045, 0.965925826),  # Isaac Lab 3 xyzw: -30 deg yaw
            joint_pos={
                "shoulder_pan": -0.5,  # Open ready pose leaves the cube visible at reset.
                "shoulder_lift": -1.05,  # Folded home stance, clear of the tabletop.
                "elbow_flex": 1.35,  # Retract the forearm toward the base.
                "wrist_flex": 1.10,  # Point the open fingers down, as in the reference.
                "wrist_roll": 0.0,
                "gripper": 1.20,  # OPEN JAWS ready to grasp cube!
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                effort_limit=2.94,
                velocity_limit=1.5,
                armature=0.001,
                stiffness=80.0,
                damping=6.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit=0.20,
                velocity_limit=0.8,
                armature=0.0001,
                stiffness=20.0,
                damping=0.5,
            ),
        },
    )

    fixed_finger_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link/"
            "lower_arm_link/wrist_link/gripper_link"
        ),
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    moving_finger_contact = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link/"
            "lower_arm_link/wrist_link/gripper_link/moving_jaw_so101_v1_link"
        ),
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )

    # 4. Yellow Cube (5cm) resting flush on tabletop (table surface Z=0.40m, center Z=0.425m)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.5,
                angular_damping=0.5,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
                max_depenetration_velocity=4.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.055),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PbrMdlCfg(
                diffuse_color_constant=(0.8, 0.60, 0.008),
                reflection_roughness_constant=0.65,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.02, 0.426)),
    )

    # 5. Target Cardboard Box placed on top of table within comfortable reach of arm
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(USD_DIR, "box_reference.usda"),
            rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PbrMdlCfg(
                diffuse_color_constant=(0.35, 0.12, 0.055),
                reflection_roughness_constant=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, 0.26, 0.401),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # 6. Tabletop-only camera framing excludes table edges and room background.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=512,
        width=512,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.5,
            horizontal_aperture=24.88,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.055, -0.30, 0.98),
            rot=(0.953717, 0.0, 0.0, -0.300706),
            convention="ros",
        ),
    )

    # 7. Hide the physical housing so an optical viewpoint cannot see its own body.
    camera_model = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CameraModel",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD_DIR, "camera.usd"), visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.055, -0.30, 0.98),
            rot=(0.953717, 0.0, 0.0, -0.300706),
        ),
    )


@configclass
class PickLiftSceneCfg(PickplaceSceneCfg):
    """Reference workcell with a physical box behind the lift workspace."""

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(USD_DIR, "box_reference.usda"),
            rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PbrMdlCfg(
                diffuse_color_constant=(0.35, 0.12, 0.055),
                reflection_roughness_constant=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, 0.26, 0.401),  # Keep the physical box behind the initial grasp workspace.
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for smooth, robust SO-101 pick and place control."""

    arm_action = mdp.SlewLimitedArmActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        # [-1, 1] spans the robot's soft joint limits instead of a tiny home-pose window.
        scale=1.0,
        alpha=0.2,
        preserve_order=True,
        rescale_to_limits=True,
    )
    gripper_action = mdp.CameraGripperActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # Keep the zero-mean initial policy fully open. Negative actions close
        # progressively: 0 = open (1.2 rad), -1 = closed (0 rad).
        scale=1.20,
        offset=1.20,
        use_default_offset=False,
        preserve_order=True,
        clip={"gripper": (0.0, 1.20)},
        xy_tolerance=0.020,
        z_tolerance=0.025,
        min_confidence=0.65,
        confirm_steps=2,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for vision RL (asymmetric actor-critic)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """1D Proprioceptive observations for the actor policy."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        filtered_targets = ObsTerm(func=mdp.filtered_joint_targets)
        assist_state = ObsTerm(func=mdp.grasp_assist_state)
        camera_cube = ObsTerm(func=mdp.camera_cube_observation)
        time_remaining = ObsTerm(func=mdp.time_remaining)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImageCfg(ObsGroup):
        """Global simulated RGB image (B, C, H, W for CNN)."""

        rgb = ObsTerm(func=mdp.global_camera_rgb)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the critic during training."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        filtered_targets = ObsTerm(func=mdp.filtered_joint_targets)
        assist_state = ObsTerm(func=mdp.grasp_assist_state)
        task_state = ObsTerm(func=mdp.critic_task_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    image: ImageCfg = ImageCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for reset events and domain randomization (friction, mass, position)."""

    # Select the sensor's actual USD camera after every reset; headless is a no-op.
    camera_view = EventTerm(func=mdp.select_task_camera_view, mode="reset")

    # 1. Reset to the reference home pose, with zero position/velocity noise.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Always restore cube pose/velocity; fixed curricula use zero-range offsets.
    reset_cube_pose: EventTerm | None = EventTerm(
        func=cast(Any, mdp.reset_root_state_uniform),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.02),
                "z": (0.0, 0.0),  # flush on table surface
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
        },
    )

    # Box pose randomization for sim-to-real placement robustness
    reset_box_pose: EventTerm | None = EventTerm(
        func=cast(Any, mdp.reset_root_state_uniform),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "yaw": (-0.26, 0.26),
            },
            "velocity_range": {},
        },
    )

    # 3. Domain Randomization: Physics materials & friction coefficients (tabletop & cube)
    randomize_cube_friction: EventTerm | None = EventTerm(
        func=cast(Any, mdp.randomize_rigid_body_material),
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.8, 1.6),
            "dynamic_friction_range": (0.6, 1.4),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    # 4. Domain Randomization: Cube mass variation (+/- 20%)
    randomize_cube_mass: EventTerm | None = EventTerm(
        func=cast(Any, mdp.randomize_rigid_body_mass),
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.045, 0.065),
            "operation": "abs",
        },
    )


@configclass
class PickLiftRewardsCfg:
    """Reward terms focused purely on picking and lifting the cube vertically (anti-drag prioritized)."""

    # Direct cube approach, without an overhead waypoint or staged descent.
    reach_cube = None  # Progress feedback replaces repeated stationary proximity credit.
    reach_progress = RewTerm(func=cast(Any, mdp.ReachCubeProgress), weight=3.0)

    # Closing before the tool is centered turns the fingers into a pusher.
    premature_close = RewTerm(func=mdp.premature_gripper_closure_penalty, weight=-3.0)

    # 4. Grasping: Reward closing fingers when cube is centered between fingertips
    grasp_cube = RewTerm(func=cast(Any, mdp.GraspCubeProgress), weight=4.0)

    # 5. Vertical lifting, gated on cube proximity and gripper closure:
    # Continuous elevation progress + lift-off bonus + target height hold bonus
    lift_cube = RewTerm(func=cast(Any, mdp.LiftCubeProgress), weight=8.0)
    # At dt=0.04, success awards 240 once before termination. This exceeds the
    # discounted return from hovering just below the success height for 6 seconds.
    lift_completed = RewTerm(func=cast(Any, mdp.LiftSuccess), weight=6000.0)

    # 6. Anti-Dragging Penalty: Strongly penalize sliding, dragging, or shoving the cube across the table
    drag_penalty = RewTerm(func=mdp.cube_table_drag_penalty, weight=-5.0)
    displacement_penalty = RewTerm(func=cast(Any, mdp.CubeTableDisplacementPenalty), weight=-12.0)

    # 7. Table collision penalty: Penalize gripper pushing down or scraping into tabletop
    table_collision = RewTerm(func=mdp.gripper_table_collision_penalty, weight=-0.5)

    # 8. Action smoothness and jerk penalty (enforces smooth, graceful trajectories)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # 9. Joint velocity penalty (strictly clamped to prevent physics explosion spikes and fast jerky motion)
    joint_vel = RewTerm(func=mdp.safe_joint_vel_penalty, weight=-0.01)

    # 10. Joint limits penalty (prevents clipping against mechanical stops)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)


@configclass
class PickplaceRewardsCfg(PickLiftRewardsCfg):
    """Full pick-and-place rewards with gated placement reward active only after verified vertical lift."""

    # 5. Lower stationary lift hold weight so box transport dominates once lifted
    lift_cube = RewTerm(func=cast(Any, mdp.LiftCubeProgress), weight=3.0)

    # 11. Placing: Transport cube to target box once securely lifted (> 0.440m) off table
    place_cube = RewTerm(func=mdp.object_goal_distance, weight=12.0, params={"minimal_height": 0.440})
    release_cube = RewTerm(func=mdp.release_cube_in_box, weight=12.0)
    place_completed = RewTerm(func=cast(Any, mdp.PlaceSuccess), weight=6000.0)
    # Placement must proceed past the lift milestone without repeated lift bonuses.
    lift_completed = None


# Backward compatibility alias
RewardsCfg = PickplaceRewardsCfg


@configclass
class TerminationsCfg:
    """Termination conditions for the MDP."""

    # Fail with a physical-state diagnostic before NaNs can enter rewards or PPO.
    physics_health = DoneTerm(func=mdp.assert_finite_physics)

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Cube fell off the table (table surface is 0.40m, threshold is 0.35m)
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("cube"), "minimum_height": 0.35},
    )

    # Reset immediately if robot arm suffers a physics singularity / drops below table
    robot_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.20},
    )


@configclass
class PickLiftTerminationsCfg(TerminationsCfg):
    lift_success = DoneTerm(func=cast(Any, mdp.LiftSuccess))


@configclass
class PickplaceTerminationsCfg(TerminationsCfg):
    placed = DoneTerm(func=cast(Any, mdp.PlaceSuccess))


##
# Environment configurations
##


@configclass
class PickLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: Dedicated Pick-and-Lift environment focused 100% on picking and elevating vertically."""

    # Scene settings: physical box behind the lift workspace.
    scene: PickLiftSceneCfg = PickLiftSceneCfg(num_envs=32, env_spacing=2.5)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: PickLiftRewardsCfg = PickLiftRewardsCfg()
    terminations: PickLiftTerminationsCfg = PickLiftTerminationsCfg()
    is_finite_horizon = True  # The 6-second deadline is part of the observed task.

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 6.0

        # Viewport camera default looking from RealSense camera perspective
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.055, -0.30, 0.98),
            lookat=(0.055, 0.11, 0.40),
        )

        # Simulation physics rate (100 Hz physics, 25 Hz policy)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        from isaac_vision_rl.utils.pipeline import configure_curriculum

        configure_curriculum(self, os.environ.get("SO101_CURRICULUM", "lift"))
        if os.environ.get("SO101_WRIST_ONLY_CONTROL", "0") == "1":
            if os.environ.get("SO101_SMOOTH_CONTROL", "0") == "1":
                raise ValueError("Choose wrist-only or smooth control, not both")
            self.actions.arm_action.hold_wrist_roll_after_capture = True
        if os.environ.get("SO101_SMOOTH_CONTROL", "0") == "1":
            self.actions.arm_action.max_target_speed = 0.75
            self.actions.arm_action.max_target_acceleration = 2.0
            self.actions.arm_action.wrist_roll_speed = 0.25
            self.actions.arm_action.hold_wrist_roll_after_capture = True
            self.actions.gripper_action.close_target_speed = 1.2
            self.actions.gripper_action.closure_hold_s = 1.2


@configclass
class PickplaceEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 2: Full Pick-and-Place environment (pick, lift, transport to box, place)."""

    # Scene settings
    scene: PickplaceSceneCfg = PickplaceSceneCfg(num_envs=32, env_spacing=2.5)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: PickplaceRewardsCfg = PickplaceRewardsCfg()
    terminations: PickplaceTerminationsCfg = PickplaceTerminationsCfg()
    is_finite_horizon = True

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 6.0

        # Viewport camera default looking from RealSense camera perspective
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.055, -0.30, 0.98),
            lookat=(0.055, 0.11, 0.40),
        )

        # Simulation physics rate (100 Hz physics, 25 Hz policy)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        from isaac_vision_rl.utils.pipeline import configure_curriculum

        configure_curriculum(self, os.environ.get("SO101_CURRICULUM", "place"))
