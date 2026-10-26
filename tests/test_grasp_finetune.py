import torch
import sys

from isaac_vision_rl.experiments.grasp_finetune.patch import (
    TERMS,
    closing_gate,
    current_stage_only,
    descent_preparation_potential,
    grasp_stage_entry_gate,
    install,
    jaw_closing_score,
    reach_gate_checks,
)
from isaac_vision_rl.utils.four_stage import StageMachine as OriginalStageMachine
from isaac_vision_rl.utils import reward_panel


def test_stage_two_has_three_separate_terms_and_original_class_is_preserved(monkeypatch):
    assert TERMS[0] == (
        "overhead_approach",
        "table_clearance",
        "descend_to_pinch_progress",
    )
    assert TERMS[1] == (
        "grasp_target_progress",
        "jaw_closing_progress",
        "table_slide_speed",
    )
    assert all(len(stage_terms) == 3 for stage_terms in TERMS)
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original_version, original_terms = module.VERSION, module.TERMS
    original_groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", OriginalStageMachine)
    monkeypatch.setattr(module, "VERSION", original_version)
    monkeypatch.setattr(module, "TERMS", original_terms)
    monkeypatch.setattr(module, "active_reward_groups", original_groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        monkeypatch.setattr(panel, "active_reward_groups", panel.active_reward_groups)
        monkeypatch.setattr(panel, "TERMS", panel.TERMS)
        monkeypatch.setattr(panel, "DISPLAY_NAMES", panel.DISPLAY_NAMES)
        monkeypatch.setattr(panel, "CUMULATIVE_TERMS", dict(panel.CUMULATIVE_TERMS))
    installed = install()
    assert installed.StageMachine is not OriginalStageMachine
    assert OriginalStageMachine.__module__ == "isaac_vision_rl.utils.four_stage"
    assert reward_panel.TERMS == TERMS
    assert reward_panel.DISPLAY_NAMES["grasp_target_progress"] == "Move toward grasp position"
    assert current_stage_only(1) == (1,)


def test_reach_gate_diagnostics_name_each_transition_condition():
    values = {
        "reach_ready": torch.tensor([False]),
        "pinch_center_distance": torch.tensor([0.02]),
        "cube_between_fingers": torch.tensor([True]),
        "downward_cosine": torch.tensor([0.90]),
        "jaw": torch.tensor([1.15]),
        "clearance": torch.tensor([0.01]),
    }
    assert reach_gate_checks(values) == {
        "descended_to_pinch": torch.tensor([True]),
        "cube_between_fingers": torch.tensor([True]),
        "wrist_down": torch.tensor([True]),
        "jaws_open": torch.tensor([True]),
        "safe_clearance": torch.tensor([True]),
    }


def test_graph_selects_only_the_current_stage():
    assert [current_stage_only(stage) for stage in range(4)] == [(0,), (1,), (2,), (3,)]


def test_reach_has_separate_signed_overhead_and_descent_progress(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, version, terms = module.StageMachine, module.VERSION, module.TERMS
    groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", version)
    monkeypatch.setattr(module, "TERMS", terms)
    monkeypatch.setattr(module, "active_reward_groups", groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        for name in ("active_reward_groups", "TERMS", "DISPLAY_NAMES", "CONDITIONS"):
            monkeypatch.setattr(panel, name, getattr(panel, name))
        monkeypatch.setattr(panel, "CUMULATIVE_TERMS", dict(panel.CUMULATIVE_TERMS))
    state = install().StageMachine(1, "cpu", reward_mode="overlap", reward_style="continuous")
    assert state.weights[0] == (0.30, -1.00, 0.30)
    assert state.weights[1] == (0.10, 0.50, -2.00)

    def values(*, pregrasp, pinch, jaw=1.20, down=0.95, overhead=False, height=0.045):
        return {
            "jaw": torch.tensor([jaw]), "bilateral": torch.tensor([False]),
            "enclosed": torch.tensor([False]), "unsafe": torch.tensor([False]),
            "height": torch.tensor([0.425]), "cube_start_height": torch.tensor([0.425]),
            "speed": torch.tensor([0.0]), "reach_ready": torch.tensor([False]),
            "pregrasp_distance": torch.tensor([pregrasp]), "alignment": torch.tensor([1.0]),
            "enclosure": torch.tensor([0.0]), "clearance": torch.tensor([0.01]),
            "slide_speed": torch.tensor([0.0]), "box_distance": torch.tensor([0.15]),
            "lowering": torch.tensor([0.0]), "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]), "timed_out": torch.tensor([False]),
            "pinch_center_distance": torch.tensor([pinch]),
            "cube_between_fingers": torch.tensor([False]),
            "downward_cosine": torch.tensor([down]),
            "grasp_pose_distance": torch.tensor([pinch]),
            "xy_distance": torch.tensor([0.01 if overhead else pregrasp]),
            "tool_height_above_cube": torch.tensor([height]),
        }

    # Baselines are free: neither target gives a reset spike.
    state.reach_route[:] = 1
    state.update(values(pregrasp=0.06, pinch=0.08), 1 / 30)
    assert state.group_raw[0, 0, 0].item() == 0.0
    assert state.group_raw[0, 0, 2].item() == 0.0

    # Moving closer to the overhead waypoint raises its own chart; moving
    # farther lowers it; holding still adds nothing.
    state.update(values(pregrasp=0.04, pinch=0.08), 1 / 30)
    assert state.group_raw[0, 0, 0].item() > 0.0
    state.update(values(pregrasp=0.06, pinch=0.08), 1 / 30)
    assert state.group_raw[0, 0, 0].item() < 0.0
    state.update(values(pregrasp=0.06, pinch=0.08), 1 / 30)
    assert state.group_raw[0, 0, 0].item() == 0.0

    # Reach a safe overhead pose to unlock descent; the latch step establishes
    # a fresh baseline so changing targets cannot create reward by itself.
    state.update(values(pregrasp=0.02, pinch=0.08, overhead=True), 1 / 30)
    assert state.descent_ready.item()
    assert state.group_raw[0, 0, 2].item() == 0.0
    state.update(values(pregrasp=0.02, pinch=0.06, overhead=True, height=0.025), 1 / 30)
    assert state.group_raw[0, 0, 2].item() > 0.0
    state.update(values(pregrasp=0.02, pinch=0.06, overhead=True, height=0.025), 1 / 30)
    assert state.group_raw[0, 0, 2].item() == 0.0
    state.update(values(pregrasp=0.02, pinch=0.08, overhead=True, height=0.05), 1 / 30)
    assert state.group_raw[0, 0, 2].item() < 0.0

    # Moving away from a good open, downward-facing pose lowers the same signed
    # potential; table-clearance protection remains separate.
    state.update(values(pregrasp=0.02, pinch=0.06, jaw=0.80, overhead=True), 1 / 30)
    assert state.group_raw[0, 0, 2].item() < 0.0
    state.update(values(pregrasp=0.02, pinch=0.06, down=0.20, overhead=True), 1 / 30)
    assert state.group_raw[0, 0, 2].item() < 0.0
    assert len(TERMS[0]) == 3

    state.reset()
    state.reach_route[:] = 1
    state.update(values(pregrasp=0.01, pinch=0.01, overhead=True), 1 / 30)
    assert state.group_raw[0, 0, 0].item() == 0.0
    assert state.group_raw[0, 0, 2].item() == 0.0


def test_descent_preparation_reward_teaches_open_align_and_descend_without_hover_bonus(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, version, terms = module.StageMachine, module.VERSION, module.TERMS
    groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", version)
    monkeypatch.setattr(module, "TERMS", terms)
    monkeypatch.setattr(module, "active_reward_groups", groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        for name in ("active_reward_groups", "TERMS", "DISPLAY_NAMES", "CONDITIONS"):
            monkeypatch.setattr(panel, name, getattr(panel, name))
        monkeypatch.setattr(panel, "CUMULATIVE_TERMS", dict(panel.CUMULATIVE_TERMS))
    state = install().StageMachine(1, "cpu", reward_mode="overlap", reward_style="continuous")
    state.reach_route[:] = 1

    def values(jaw, height=.10, xy=.02, down=.90, clearance=.01):
        return {
            "jaw": torch.tensor([jaw]), "bilateral": torch.tensor([False]),
            "enclosed": torch.tensor([False]), "unsafe": torch.tensor([False]),
            "height": torch.tensor([.425]), "cube_start_height": torch.tensor([.425]),
            "speed": torch.tensor([0.]), "reach_ready": torch.tensor([False]),
            "pregrasp_distance": torch.tensor([.10]), "alignment": torch.tensor([1.]),
            "enclosure": torch.tensor([0.]), "clearance": torch.tensor([clearance]),
            "slide_speed": torch.tensor([0.]), "box_distance": torch.tensor([.15]),
            "lowering": torch.tensor([0.]), "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]), "timed_out": torch.tensor([False]),
            "pinch_center_distance": torch.tensor([.10]),
            "cube_between_fingers": torch.tensor([False]),
            "downward_cosine": torch.tensor([down]),
            "grasp_pose_distance": torch.tensor([.10]), "xy_distance": torch.tensor([xy]),
            "tool_height_above_cube": torch.tensor([height]),
        }

    # First observation only sets the baseline. Opening and descending in the
    # centered, wrist-down corridor then raise the same simple graph.
    state.update(values(.50), 1 / 30)
    assert state.group_raw[0, 0, 2].item() == 0.0
    state.update(values(.80), 1 / 30)
    assert state.group_raw[0, 0, 2].item() > 0.0
    state.update(values(.80), 1 / 30)
    assert state.group_raw[0, 0, 2].item() == 0.0
    state.update(values(.80, height=.05), 1 / 30)
    assert state.group_raw[0, 0, 2].item() > 0.0
    state.update(values(.80, height=.08), 1 / 30)
    assert state.group_raw[0, 0, 2].item() < 0.0

    # A bad pose cannot create a positive opening reward; changing away from a
    # good pose can lower the signed score.
    state.update(values(.70, xy=.20), 1 / 30)
    assert state.group_raw[0, 0, 2].item() <= 0.0
    state.update(values(.90, down=.20), 1 / 30)
    assert state.group_raw[0, 0, 2].item() <= 0.0
    state.update(values(1.10, clearance=.002), 1 / 30)
    assert state.group_raw[0, 0, 2].item() <= 0.0

    # From a stationary off-center or wrist-up baseline, opening alone is flat.
    assert descent_preparation_potential(values(.50, xy=1.0, down=.90)).item() == 0.0
    assert descent_preparation_potential(values(.80, xy=1.0, down=.90)).item() == 0.0
    assert descent_preparation_potential(values(.50, xy=.02, down=.20)).item() == 0.0
    assert descent_preparation_potential(values(.80, xy=.02, down=.20)).item() == 0.0


def test_descent_height_score_is_one_way_and_never_rewards_rising():
    def pose(height):
        return {
            "jaw": torch.tensor([1.20]),
            "xy_distance": torch.tensor([0.01]),
            "downward_cosine": torch.tensor([0.95]),
            "tool_height_above_cube": torch.tensor([height]),
            "clearance": torch.tensor([0.02]),
        }

    high = descent_preparation_potential(pose(0.045))
    pinching = descent_preparation_potential(pose(0.0095))
    slightly_low = descent_preparation_potential(pose(0.004))
    assert pinching > high
    assert torch.allclose(slightly_low, pinching)  # no incentive to rise after passing target


def test_descent_stage_cannot_reset_to_raise_or_gain_reward_by_rising(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, version, terms = module.StageMachine, module.VERSION, module.TERMS
    groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", version)
    monkeypatch.setattr(module, "TERMS", terms)
    monkeypatch.setattr(module, "active_reward_groups", groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        for name in ("active_reward_groups", "TERMS", "DISPLAY_NAMES", "CONDITIONS"):
            monkeypatch.setattr(panel, name, getattr(panel, name))
        monkeypatch.setattr(panel, "CUMULATIVE_TERMS", dict(panel.CUMULATIVE_TERMS))
    state = install().StageMachine(1, "cpu", reward_mode="overlap", reward_style="continuous")
    state.reach_route[:] = 1
    state.descent_ready[:] = True

    def sample(height, xy):
        return {
            "jaw": torch.tensor([1.20]), "bilateral": torch.tensor([False]),
            "enclosed": torch.tensor([False]), "unsafe": torch.tensor([False]),
            "height": torch.tensor([0.425]), "cube_start_height": torch.tensor([0.425]),
            "speed": torch.tensor([0.0]), "reach_ready": torch.tensor([False]),
            "pregrasp_distance": torch.tensor([0.05]), "alignment": torch.tensor([1.0]),
            "enclosure": torch.tensor([0.0]), "clearance": torch.tensor([0.02]),
            "slide_speed": torch.tensor([0.0]), "box_distance": torch.tensor([0.15]),
            "lowering": torch.tensor([0.0]), "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]), "timed_out": torch.tensor([False]),
            "pinch_center_distance": torch.tensor([0.02]),
            "cube_between_fingers": torch.tensor([False]),
            "downward_cosine": torch.tensor([0.95]),
            "grasp_pose_distance": torch.tensor([0.02]),
            "xy_distance": torch.tensor([xy]),
            "tool_height_above_cube": torch.tensor([height]),
        }

    state.update(sample(0.020, 0.040), 1 / 30)  # parent logic would request a raise here
    assert state.reach_route.item() == 1
    state.update(sample(0.030, 0.010), 1 / 30)  # moves up while becoming better centered
    assert state.reach_route.item() == 1
    assert state.group_raw[0, 0, 2].item() < 0.0

def test_reach_to_grasp_gate_requires_descent_and_cube_between_open_fingers():
    def measurements(distance, between=True, gap=0.075, width=0.050, jaw=1.15):
        return {
            "reach_ready": torch.tensor([True]),
            "pinch_center_distance": torch.tensor([distance]),
            "cube_between_fingers": torch.tensor([between]),
            "finger_gap": torch.tensor([gap]),
            "cube_width_on_jaw_axis": torch.tensor([width]),
            "downward_cosine": torch.tensor([0.95]),
            "jaw": torch.tensor([jaw]),
            "clearance": torch.tensor([0.01]),
        }

    assert not grasp_stage_entry_gate(measurements(0.08)).item()  # still above
    assert not grasp_stage_entry_gate(measurements(0.02, between=False)).item()
    assert not grasp_stage_entry_gate(measurements(0.02, between=False, gap=0.052)).item()  # geometry says cannot fit
    assert not grasp_stage_entry_gate(measurements(0.02, jaw=0.80)).item()
    assert grasp_stage_entry_gate(measurements(0.02, jaw=1.00)).item()  # geometry confirms enough opening
    assert grasp_stage_entry_gate(measurements(0.02)).item()


def test_fit_check_is_not_repeated_with_an_extra_five_mm_gap_margin():
    # The calibrated pad geometry has already certified the cube-between flag.
    # A 2 mm total fit margin should not be rejected by a second redundant test.
    values = {
        "reach_ready": torch.tensor([True]),
        "pinch_center_distance": torch.tensor([0.02]),
        "cube_between_fingers": torch.tensor([True]),
        "finger_gap": torch.tensor([0.052]),
        "cube_width_on_jaw_axis": torch.tensor([0.050]),
        "downward_cosine": torch.tensor([0.95]),
        "jaw": torch.tensor([1.10]),
        "clearance": torch.tensor([0.01]),
    }
    assert grasp_stage_entry_gate(values).item()


def test_overhead_pose_cannot_advance_but_descended_between_pose_can(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, version, terms = module.StageMachine, module.VERSION, module.TERMS
    groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", version)
    monkeypatch.setattr(module, "TERMS", terms)
    monkeypatch.setattr(module, "active_reward_groups", groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        for name in ("active_reward_groups", "TERMS", "DISPLAY_NAMES", "CONDITIONS"):
            monkeypatch.setattr(panel, name, getattr(panel, name))
    state = install().StageMachine(1, "cpu", reward_mode="overlap", reward_style="continuous")

    def update(distance, between):
        values = {
            # A cube-fit pinch pose with open-enough jaws can pass without the
            # separate historical overhead latch.
            "jaw": torch.tensor([1.0]), "bilateral": torch.tensor([False]),
            "enclosed": torch.tensor([False]), "unsafe": torch.tensor([False]),
            "height": torch.tensor([0.425]), "cube_start_height": torch.tensor([0.425]),
            "speed": torch.tensor([0.0]), "reach_ready": torch.tensor([True]),
            "pregrasp_distance": torch.tensor([0.01]), "alignment": torch.tensor([1.0]),
            "enclosure": torch.tensor([0.0]), "clearance": torch.tensor([0.01]),
            "slide_speed": torch.tensor([0.0]), "box_distance": torch.tensor([0.15]),
            "lowering": torch.tensor([0.0]), "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]), "timed_out": torch.tensor([False]),
            "pinch_center_distance": torch.tensor([distance]),
            "cube_between_fingers": torch.tensor([between]),
            "finger_gap": torch.tensor([0.075]), "cube_width_on_jaw_axis": torch.tensor([0.050]),
                "downward_cosine": torch.tensor([0.95]),
                "grasp_pose_distance": torch.tensor([distance]),
                "xy_distance": torch.tensor([0.01]),
                "tool_height_above_cube": torch.tensor([0.045]),
        }
        state.update(values, 0.05)

    for _ in range(4):
        update(0.08, True)
    assert not state.descent_ready.item()  # the removed historical latch stays false
    assert state.stage.item() == 0  # overhead hover does not pass
    update(0.02, False)
    update(0.02, False)
    assert state.stage.item() == 0  # descended, but cube is outside the jaw gap
    update(0.02, True)
    assert state.stage.item() == 0  # needs 0.08 s continuous readiness
    update(0.02, True)
    assert not state.descent_ready.item()
    assert state.stage.item() == 1


def test_jaw_score_grows_only_as_jaw_closes():
    values = jaw_closing_score(torch.tensor([1.20, 1.075, 0.95, 0.50]))
    assert torch.allclose(values, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_gate_requires_nearby_cube_downward_pose_and_open_enough_jaws():
    def measurement(distance, down, jaw):
        return {
            "grasp_pose_distance": torch.tensor([distance]),
            "downward_cosine": torch.tensor([down]),
            "jaw": torch.tensor([jaw]),
        }

    assert closing_gate(measurement(0.02, 0.95, 1.15)).item()
    assert not closing_gate(measurement(0.08, 0.95, 1.15)).item()
    assert not closing_gate(measurement(0.02, 0.50, 1.15)).item()
    assert not closing_gate(measurement(0.02, 0.95, 0.80)).item()


def test_reward_increments_only_for_gated_closing_and_stops_on_bilateral_contact(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, original_version, original_terms = module.StageMachine, module.VERSION, module.TERMS
    original_groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", original_version)
    monkeypatch.setattr(module, "TERMS", original_terms)
    monkeypatch.setattr(module, "active_reward_groups", original_groups)
    panel = sys.modules.get("isaac_vision_rl.utils.reward_panel")
    if panel is not None:
        monkeypatch.setattr(panel, "active_reward_groups", panel.active_reward_groups)
        monkeypatch.setattr(panel, "TERMS", panel.TERMS)
        monkeypatch.setattr(panel, "DISPLAY_NAMES", panel.DISPLAY_NAMES)
        monkeypatch.setattr(panel, "CUMULATIVE_TERMS", dict(panel.CUMULATIVE_TERMS))
    module = install()
    installed = module.StageMachine
    state = installed(1, "cpu", reward_mode="overlap", reward_style="continuous")
    state.stage[:] = 1

    def update(jaw, *, gate=True, bilateral=False):
        values = {
            "jaw": torch.tensor([jaw]),
            "bilateral": torch.tensor([bilateral]),
            "grasp_pose_distance": torch.tensor([0.02 if gate else 0.08]),
            "downward_cosine": torch.tensor([0.95]),
            "enclosed": torch.tensor([True]),
            "unsafe": torch.tensor([False]),
            "height": torch.tensor([0.425]),
            "cube_start_height": torch.tensor([0.425]),
            "speed": torch.tensor([0.0]),
            "reach_ready": torch.tensor([False]),
            "pregrasp_distance": torch.tensor([0.10]),
            "alignment": torch.tensor([1.0]),
            "enclosure": torch.tensor([0.0]),
            "clearance": torch.tensor([0.02]),
            "slide_speed": torch.tensor([0.0]),
            "box_distance": torch.tensor([0.15]),
            "lowering": torch.tensor([0.0]),
            "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]),
            "timed_out": torch.tensor([False]),
        }
        state.update(values, 1 / 30)
        assert state.group_raw[0, 0].abs().sum().item() == 0.0
        return state.group_raw[0, 1, 1].item()

    assert update(1.20) == 0.0  # first Stage-2 sample establishes baseline
    assert update(1.10) > 0.0  # closing near cube earns credit
    assert update(1.10) == 0.0  # holding still earns nothing further
    assert update(1.15) == 0.0  # reopening earns no reward and no penalty
    assert update(1.10) == 0.0  # reclosing cannot farm already earned progress
    assert update(1.00) > 0.0  # a new best closing position earns once
    assert update(1.15) == 0.0  # opening remains neutral
    assert update(1.00) == 0.0  # repeated closing cannot refill the reward
    assert update(1.05, gate=False) == 0.0  # wrong pose cannot earn closing credit
    assert update(1.00, bilateral=True) == 0.0  # contact stops additional squeeze reward


def test_grasp_target_progress_is_signed_and_flat_when_stationary(monkeypatch):
    module = __import__("isaac_vision_rl.utils.four_stage", fromlist=["StageMachine"])
    original, original_version, original_terms = module.StageMachine, module.VERSION, module.TERMS
    original_groups = module.active_reward_groups
    monkeypatch.setattr(module, "StageMachine", original)
    monkeypatch.setattr(module, "VERSION", original_version)
    monkeypatch.setattr(module, "TERMS", original_terms)
    monkeypatch.setattr(module, "active_reward_groups", original_groups)
    monkeypatch.setattr(reward_panel, "active_reward_groups", reward_panel.active_reward_groups)
    monkeypatch.setattr(reward_panel, "TERMS", reward_panel.TERMS)
    monkeypatch.setattr(reward_panel, "DISPLAY_NAMES", reward_panel.DISPLAY_NAMES)
    monkeypatch.setattr(reward_panel, "CUMULATIVE_TERMS", dict(reward_panel.CUMULATIVE_TERMS))
    state = install().StageMachine(1, "cpu", reward_mode="overlap", reward_style="continuous")
    state.stage[:] = 1

    def update(distance):
        values = {
            "jaw": torch.tensor([1.20]), "bilateral": torch.tensor([False]),
            "grasp_pose_distance": torch.tensor([distance]), "downward_cosine": torch.tensor([1.0]),
            "enclosed": torch.tensor([False]), "unsafe": torch.tensor([False]),
            "height": torch.tensor([0.425]), "cube_start_height": torch.tensor([0.425]),
            "speed": torch.tensor([0.0]), "reach_ready": torch.tensor([False]),
            "pregrasp_distance": torch.tensor([0.10]), "alignment": torch.tensor([1.0]),
            "enclosure": torch.tensor([0.0]), "clearance": torch.tensor([0.02]),
            "slide_speed": torch.tensor([0.0]), "box_distance": torch.tensor([0.15]),
            "lowering": torch.tensor([0.0]), "inside_box": torch.tensor([False]),
            "placed": torch.tensor([False]), "timed_out": torch.tensor([False]),
        }
        state.update(values, 1 / 30)
        return state.group_raw[0, 1, 0].item()

    assert update(0.10) == 0.0  # establish a no-free-reward baseline
    assert update(0.08) > 0.0  # closer to the pinch target
    assert update(0.08) == 0.0  # stationary
    assert update(0.10) < 0.0  # farther from the pinch target
