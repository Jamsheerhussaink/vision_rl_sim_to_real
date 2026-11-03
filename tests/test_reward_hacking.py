"""The pick reward passes the hack audit; Madhan's original logic still shows its known hacks."""
import pytest

from vision_rl.rewards.audit.checks import check_all
from vision_rl.rewards.audit.world import Camera
from vision_rl.rewards.pick import PickRewardConfig


@pytest.fixture(scope="module")
def camera():
    return Camera()


def test_ours_has_no_known_hacks(camera):
    _, verdicts = check_all(PickRewardConfig(), camera, loop_trials=40)
    bad = [f"{v.kind} {v.name}: {v.detail}" for v in verdicts if not v.ok]
    assert not bad, "\n".join(bad)


def test_audit_still_catches_madhans_hacks(camera):
    # If these stop failing, the audit got weaker, not the old reward better.
    _, verdicts = check_all(PickRewardConfig.madhan(), camera, loop_trials=40)
    failed = {v.name for v in verdicts if not v.ok}
    assert {"fake_grasp", "ledge_park", "side_swipe", "align_dither"} <= failed
