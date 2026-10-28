"""Motor safety limits shared by the simulator and the real SO-101 arm, so both obey the same rules."""

# Lowest jaw target (radians, 0 = fully closed, 1.2 = open on the real arm) that the policy may command.
#
# Evidence: the model 900 mirror run (outputs/mirror/1793161843229080388.json) commanded 0.0 rad (2029 ticks)
# while the cube stopped the real jaw at about 2430-2450 ticks (about 0.56 rad). The jaw could not move, the
# servo kept pushing, the gap grew past the 350 tick tracking limit, and the program stopped.
#
# 0.45 rad = 2359 ticks: about 80 ticks under the cube stall, so the jaw still squeezes the cube firmly, but the
# gap stays far below the 350 tick limit (it holds for a stall anywhere up to about 0.93 rad, even a tipped corner).
GRIPPER_MIN_CLOSE_RAD = 0.45
