"""Reward-hacking audit: a toy world, honest + exploit strategies, and checks.

The toy world is a few lines of kinematics (gripper point, cube, jaw), not
MuJoCo. It is fast and deterministic, and it lets an exploit be scripted
exactly. It cannot find hacks that need real contact physics; those need
rollouts of a trained policy in the real env (watch the per-term metrics).
"""
