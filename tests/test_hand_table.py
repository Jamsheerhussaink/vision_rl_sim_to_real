"""Reject fingertip scraping even when the tool reference stays above the table."""
from types import SimpleNamespace as NS
import torch
from isaac_vision_rl.utils.hand_table import hand_table_metrics, TABLE_SENSORS
from isaac_vision_rl.tasks.pickplace.mdp.terminations import unsafe_hand_table_contact
from isaac_vision_rl.tasks.pickplace.mdp.rewards import gripper_table_collision_penalty


class Scene(dict):
    env_origins = torch.zeros(2, 3)


def environment():
    scene = Scene(robot=NS(data=NS(
        body_pos_w=NS(torch=torch.tensor([[[0., 0., .43]], [[0., 0., .45]]])),
        body_quat_w=NS(torch=torch.tensor([[[0., 0., 0., 1.]], [[0., 0., 0., 1.]]])),
    )))
    for name in TABLE_SENSORS:
        scene[name] = NS(data=NS(normal_force_matrix_w=torch.zeros(2, 1, 1, 3)))
    return NS(scene=scene, num_envs=2, device="cpu", extras={},
              _hand_table_vertices=[(0, torch.tensor([[0., 0., -.032]]))])


def test_low_finger_is_detected_despite_high_reference():
    env = environment()
    clearance, _ = hand_table_metrics(env)
    torch.testing.assert_close(clearance, torch.tensor([-.002, .018]))
    assert unsafe_hand_table_contact(env).tolist() == [True, False]
    assert gripper_table_collision_penalty(env).tolist() == [1., 0.]


def test_filtered_table_force_ends_episode_and_adds_penalty():
    env = environment()
    env.scene["moving_table_contact"].data.normal_force_matrix_w[1, 0, 0, 2] = 1.
    assert unsafe_hand_table_contact(env).tolist() == [True, True]
    assert gripper_table_collision_penalty(env)[1] > 0


def test_environment_origin_does_not_change_clearance():
    env = environment()
    env.scene.env_origins = torch.tensor([[0., 0., 2.], [0., 0., 3.]])
    env.scene["robot"].data.body_pos_w.torch[:, 0, 2] += torch.tensor([2., 3.])
    torch.testing.assert_close(hand_table_metrics(env)[0], torch.tensor([-.002, .018]))
