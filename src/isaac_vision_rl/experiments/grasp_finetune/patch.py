"""Grasp fine-tuning rewards, installed only by the dedicated experiment launcher."""

import torch

VERSION = "four_stage_grasp_close_v5"
PREGRASP_CLEARANCE_M = 0.045
TERMS = (
    ("overhead_approach", "table_clearance", "descend_to_pinch_progress"),
    ("grasp_target_progress", "jaw_closing_progress", "table_slide_speed"),
    ("supported_lift_progress", "held_lift_dwell", "object_speed"),
    ("box_approach_progress", "lowering_progress", "settled_release"),
)
DISPLAY_NAMES = {
    "overhead_approach": "Move above cube",
    "table_clearance": "Table clearance",
    "descend_to_pinch_progress": "Open and descend (centered, wrist down)",
    "grasp_target_progress": "Move toward grasp position",
    "jaw_closing_progress": "Jaw closing progress",
    "bilateral_contact": "Valid two-finger contact",
    "table_slide_speed": "Cube sliding on table",
    "supported_lift_progress": "Lift toward target height",
    "held_lift_dwell": "Stable held lift",
    "object_speed": "Cube movement during lift",
    "box_approach_progress": "Carry toward box",
    "lowering_progress": "Lower into box",
    "settled_release": "Successful placement",
}


def descent_preparation_potential(measurements):
    """Score a safe, useful pose above the cube for signed Reach feedback.

    This is one graph, not separate bonuses: the hand must be centered, point
    down, and open, then gets more credit as it descends toward the pinch
    height. Taking the change in this score gives positive/negative/zero
    feedback for improving, worsening, or holding still.
    """
    jaw = measurements["jaw"]
    xy = measurements.get("xy_distance", torch.full_like(jaw, float("inf")))
    downward = measurements.get("downward_cosine", torch.zeros_like(jaw)).clamp(-1.0, 1.0)
    height = measurements.get("tool_height_above_cube", torch.full_like(jaw, 0.045))
    clearance = measurements.get("clearance", torch.zeros_like(jaw))

    centered = torch.exp(-torch.square(xy / 0.06))
    wrist_down = ((downward - 0.50) / 0.50).clamp(0.0, 1.0)
    jaws_open = (jaw / 1.20).clamp(0.0, 1.0)
    # A one-way score: it grows while descending from the pregrasp height to
    # the pinch height, then stays level. A symmetric target-distance score
    # would reward rising again after the hand descended slightly too far.
    descent_quality = torch.exp(-((height - 0.0095).clamp(min=0.0) / 0.035))
    potential = centered * wrist_down * jaws_open * (0.25 + 0.75 * descent_quality)
    return torch.where(clearance >= 0.004, potential, torch.zeros_like(potential))


def current_stage_only(stage):
    """Show just the reward family for the robot's current task stage."""
    return (int(stage),)


def jaw_closing_score(jaw_position):
    """Map the open-to-closed jaw range to 0..1; lower jaw angle means closed."""
    return ((1.20 - jaw_position) / 0.25).clamp(0.0, 1.0)


def closing_gate(measurements):
    """Only train jaw closure near the cube, with the gripper pointing down."""
    distance = measurements.get("grasp_pose_distance")
    downward = measurements.get("downward_cosine")
    jaw = measurements["jaw"]
    if distance is None or downward is None:
        return torch.zeros_like(jaw, dtype=torch.bool)
    return (distance < 0.045) & (downward > 0.80) & (jaw > 0.90)


def grasp_stage_entry_gate(measurements):
    """Enter Grasp only after descent and a measured fit between open fingers."""
    reference = measurements["reach_ready"]
    distance = measurements.get("pinch_center_distance")
    between = measurements.get("cube_between_fingers")
    downward = measurements.get("downward_cosine")
    jaw = measurements["jaw"]
    clearance = measurements["clearance"]
    if any(value is None for value in (distance, between, downward)):
        return torch.zeros_like(reference, dtype=torch.bool)
    # `cube_between_fingers` already tests whether the rotated cube fits inside
    # the measured gap between the calibrated inner-pad centers. Requiring a
    # second gap > projected-width + 5 mm check rejected some valid centered
    # poses, so keep the fit test in one place only.
    gate = (
        (distance < 0.035)
        & between.bool()
        & (downward > 0.75)
        # The measured cube-between-fingers geometry confirms that the jaws
        # are open enough to enter Grasp. Do not add a stricter angle-only gate.
        & (jaw > 0.90)
        & (clearance >= 0.004)
    )
    return gate


def reach_gate_checks(measurements):
    """Return the actual Reach -> Grasp transition checks separately."""
    reference = measurements["reach_ready"]
    distance = measurements.get("pinch_center_distance")
    between = measurements.get("cube_between_fingers")
    downward = measurements.get("downward_cosine")
    if any(value is None for value in (distance, between, downward)):
        false = torch.zeros_like(reference, dtype=torch.bool)
        return {name: false for name in (
            "descended_to_pinch", "cube_between_fingers", "wrist_down", "jaws_open", "safe_clearance"
        )}
    return {
        "descended_to_pinch": distance < 0.035,
        "cube_between_fingers": between.bool(),
        "wrist_down": downward > 0.75,
        "jaws_open": measurements["jaw"] > 0.90,
        "safe_clearance": measurements["clearance"] >= 0.004,
    }


def install():
    """Patch the imported reward module in this process only; original files stay intact."""
    from isaac_vision_rl.utils import four_stage
    global PREGRASP_CLEARANCE_M
    PREGRASP_CLEARANCE_M = four_stage.PREGRASP_CLEARANCE_M

    parent = four_stage.StageMachine

    class JawClosingStageMachine(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Reach has three simple terms: signed progress to the overhead
            # waypoint, table safety, then signed descent progress. Keep the
            # descent line as strong as the approach line, but do not reward
            # opening or wrist angle by themselves.
            if self.reward_style == "continuous":
                self.weights = (
                    (0.30, -1.00, 0.30),
                    (0.10, 0.50, -2.00),
                    *self.weights[2:],
                )
            self._approach_previous = torch.zeros_like(self.dwell)
            self._descent_previous = torch.zeros_like(self.dwell)
            self._descent_previous_height = torch.zeros_like(self.dwell)
            self._reach_initialized = torch.zeros_like(self.started)
            self._close_best = torch.zeros_like(self.dwell)
            self._close_initialized = torch.zeros_like(self.started)
            self._target_previous = torch.zeros_like(self.dwell)
            self._target_initialized = torch.zeros_like(self.started)

        def reset(self, ids=None):
            super().reset(ids)
            ids = slice(None) if ids is None else ids
            self._approach_previous[ids] = 0.0
            self._descent_previous[ids] = 0.0
            self._descent_previous_height[ids] = 0.0
            self._reach_initialized[ids] = False
            self._close_best[ids] = 0.0
            self._close_initialized[ids] = False
            self._target_previous[ids] = 0.0
            self._target_initialized[ids] = False

        def update(self, measurements, dt, inspection=False):
            stage_before = self.stage.clone()
            route_before = self.reach_route.clone()
            descent_before = self.descent_ready.clone()
            # Stage 2 is where descent and jaw-closing are learned. Do not
            # require the historical overhead latch as well as the measured
            # pinch pose; that duplicated checkpoint kept all runs in Reach.
            # The transition still requires descent, cube-between-fingers,
            # downward wrist, open-enough jaws, and safe table clearance.
            entry_ready = grasp_stage_entry_gate(measurements)
            gated_measurements = dict(measurements)
            gated_measurements["reach_ready"] = entry_ready
            gated_measurements["bilateral"] = measurements["bilateral"] & (
                (stage_before != 0) | entry_ready
            )
            super().update(gated_measurements, dt, inspection=inspection)

            # The shared state machine resets Reach to its raise subgoal if the
            # hand is low and still off-center. That can command an upward
            # recovery after the descent phase has already begun. In this
            # experiment, keep the descent route once its overhead gate opened;
            # the table-clearance safety checks remain active.
            descent_route_locked = (stage_before == 0) & descent_before & (route_before == 1)
            self.reach_route[:] = torch.where(
                descent_route_locked, torch.ones_like(self.reach_route), self.reach_route
            )

            # Split Reach into two understandable targets: raise/center above
            # the cube, then prepare and descend toward the pinch pose. A first
            # sample, target switch, or reset only establishes a baseline.
            reach = stage_before == 0
            tool_height = measurements.get(
                "tool_height_above_cube", torch.full_like(measurements["pregrasp_distance"], PREGRASP_CLEARANCE_M)
            )
            raise_distance = (PREGRASP_CLEARANCE_M - tool_height).abs()
            overhead_distance = torch.where(
                self.reach_route == 0, raise_distance, measurements["pregrasp_distance"]
            )
            pinch_potential = descent_preparation_potential(measurements)
            route_changed = self.reach_route != route_before
            descent_just_unlocked = self.descent_ready & ~descent_before
            target_changed = route_changed | descent_just_unlocked
            initialized = self._reach_initialized & ~target_changed
            approach_delta = (self._approach_previous - overhead_distance) / 0.05
            descent_delta = pinch_potential - self._descent_previous
            rising_during_descent = self.descent_ready & (
                tool_height > self._descent_previous_height + 1.0e-5
            )
            # Pose refinement may improve the score while the arm rises. Make
            # any upward movement during the unlocked descent a clear negative
            # signal, so it cannot be hidden by simultaneous pose improvement.
            descent_delta = torch.where(
                rising_during_descent,
                -(tool_height - self._descent_previous_height) / 0.05,
                descent_delta,
            )
            downward = measurements.get("downward_cosine", torch.zeros_like(tool_height))
            # Give the policy a gradient before the exact pinch pose. This lets
            # it learn to open and align over the cube, then descend; stage
            # progression still requires the measured pinch and cube-fit checks.
            descent_context = self.reach_route == 1
            self.group_raw[:, 0, 0] = torch.where(
                reach & initialized & ~self.descent_ready,
                approach_delta,
                torch.zeros_like(approach_delta),
            )
            self.group_raw[:, 0, 2] = torch.where(
                reach & initialized & descent_context,
                descent_delta,
                torch.zeros_like(descent_delta),
            )
            self._approach_previous[:] = torch.where(reach, overhead_distance, 0.0)
            self._descent_previous[:] = torch.where(reach, pinch_potential, 0.0)
            self._descent_previous_height[:] = torch.where(reach, tool_height, 0.0)
            self._reach_initialized[:] = reach
            self.grasp_gate_checks = reach_gate_checks(measurements)

            in_grasp = stage_before == 1
            # The Reach distance/readiness/clearance family has served its
            # purpose once Grasp begins. Safety terminations remain enabled.
            self.group_raw[:, 0, :] = torch.where(
                in_grasp[:, None],
                torch.zeros_like(self.group_raw[:, 0, :]),
                self.group_raw[:, 0, :],
            )
            score = jaw_closing_score(measurements["jaw"])
            target_distance = measurements["grasp_pose_distance"]
            target_delta = (self._target_previous - target_distance) / 0.05
            target_delta *= measurements["downward_cosine"].clamp(0.0, 1.0)
            target_progress = torch.where(
                in_grasp & self._target_initialized, target_delta,
                torch.zeros_like(target_delta),
            )
            self._target_previous[:] = torch.where(in_grasp, target_distance, 0.0)
            self._target_initialized[:] = in_grasp
            gate = closing_gate(measurements)
            # A real opposing contact ends jaw-shaping immediately. Contact
            # validity and the existing 0.2 s dwell still control stage change.
            allowed = in_grasp & gate & ~measurements["bilateral"]
            # Reward only newly achieved closing progress. Opening has no
            # positive or negative reward, and reopening cannot refill the
            # reward because the best closing score is kept for the whole
            # Grasp stage. Track the score even when the pose is invalid, so
            # closing outside the pinch area cannot be collected later.
            first = in_grasp & ~self._close_initialized
            closing_delta = (score - self._close_best).clamp(min=0.0)
            shaped = torch.where(
                in_grasp & self._close_initialized & allowed,
                closing_delta,
                torch.zeros_like(closing_delta),
            )
            self._close_best[:] = torch.where(
                in_grasp, torch.maximum(self._close_best, score), 0.0
            )
            self._close_best[:] = torch.where(first, score, self._close_best)
            self._close_initialized[:] = in_grasp

            # Stage 2 now gets specific guidance to the pinch target, then
            # gated jaw-closing progress, plus the existing table-slide cost.
            # Bilateral contact still gates the stage transition in the parent
            # state machine, but is not a fourth reward term.
            self.group_raw[:, 1, 0] = target_progress
            self.group_raw[:, 1, 1] = shaped
            self.group_contribution[:] = self.group_raw * torch.tensor(
                self.weights, device=self.stage.device
            )
            self.raw[:] = self.group_raw.sum(1)
            self.contribution[:] = self.group_contribution.sum(1)
            return self.contribution.sum(-1)

    four_stage.StageMachine = JawClosingStageMachine
    four_stage.VERSION = VERSION
    four_stage.TERMS = TERMS
    four_stage.active_reward_groups = current_stage_only
    # RewardPanel imports this helper once. Update it too if another component
    # imported the panel before this experiment launcher installed the patch.
    from isaac_vision_rl.utils import reward_panel as panel_module
    panel_module.active_reward_groups = current_stage_only
    # Keep chart names and term-to-slot mapping aligned with the actual
    # StageMachine outputs, even when the panel had not been imported yet.
    panel_module.TERMS = TERMS
    panel_module.DISPLAY_NAMES = DISPLAY_NAMES
    panel_module.CONDITIONS = (
        "Descend until the cube fits between the open fingers at the pinch height for 0.08 s",
        *four_stage.CONDITIONS[1:],
    )
    panel_module.CUMULATIVE_TERMS["grasp_target_progress"] = (
        "grasp_target_progress_total", "grasp_target_running_total"
    )
    # The live panel is created lazily during training. Reset this experiment's
    # cumulative grasp-distance chart at each episode boundary.
    from isaac_vision_rl.utils import training_reward_charts
    original_chart_update = training_reward_charts.update_training_charts

    def update_experiment_charts(env, state):
        panel = getattr(env, "_training_reward_panel", None)
        if panel is not None:
            age = int(env.episode_length_buf[0])
            if age < getattr(panel, "last_episode_step", -1):
                panel.grasp_target_running_total = 0.0
        original_chart_update(env, state)

    training_reward_charts.update_training_charts = update_experiment_charts
    return four_stage
