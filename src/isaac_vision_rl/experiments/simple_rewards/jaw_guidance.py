"""Simulation-only closing guidance, independent of rewards and actor inputs."""
import torch


def closing_permission(stage, distance, downward, between):
    """Require a nearby centered cube before closing; retain learned carry/release."""
    return (stage >= 2) | ((stage == 1) & (distance < .04) & (downward > .75) & between)
