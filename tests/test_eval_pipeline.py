from scripts.eval_pipeline import active_four_stage


def test_active_four_stage_reads_one_environment_from_reward_log():
    assert active_four_stage({"FourStage/REACH_fraction": 0.0, "FourStage/GRASP_fraction": 1.0}) == "GRASP"
    assert active_four_stage({"FourStage/LIFT_fraction": 1.0}) == "LIFT"


def test_active_four_stage_handles_missing_or_invalid_logs():
    assert active_four_stage({}) is None
    assert active_four_stage(None) is None
    assert active_four_stage({"FourStage/REACH_fraction": "bad"}) is None
