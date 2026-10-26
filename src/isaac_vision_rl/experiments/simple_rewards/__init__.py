"""Isolated simple-reward experiment. Activate only in its own process."""
def activate():
    import sys
    from . import stages
    sys.modules['isaac_vision_rl.utils.four_stage'] = stages
