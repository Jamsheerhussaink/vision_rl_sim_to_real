import pytest

from isaac_vision_rl.utils.dagger import should_use_teacher


def test_dagger_switches_after_configured_short_policy_probe():
    assert not should_use_teacher(
        policy_steps=4,
        clearance_m=0.006,
        takeover_after_steps=5,
        minimum_clearance_m=0.0055,
    )
    assert should_use_teacher(
        policy_steps=5,
        clearance_m=0.006,
        takeover_after_steps=5,
        minimum_clearance_m=0.0055,
    )


def test_dagger_switches_before_table_margin_and_stays_with_teacher():
    assert should_use_teacher(
        policy_steps=1,
        clearance_m=0.005,
        takeover_after_steps=5,
        minimum_clearance_m=0.0055,
    )
    assert should_use_teacher(
        policy_steps=1,
        clearance_m=0.02,
        takeover_after_steps=5,
        minimum_clearance_m=0.0055,
        teacher_already_in_control=True,
    )


@pytest.mark.parametrize("clearance", [float("nan"), float("inf")])
def test_dagger_rejects_invalid_clearance(clearance):
    with pytest.raises(ValueError, match="finite"):
        should_use_teacher(
            policy_steps=0,
            clearance_m=clearance,
            takeover_after_steps=5,
            minimum_clearance_m=0.0055,
        )
