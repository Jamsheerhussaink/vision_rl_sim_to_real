from test_opposite_side_grasp import sample
from isaac_vision_rl.utils.four_stage import StageMachine


def test_descent_retains_small_errors_but_rejects_large_sideways_error():
    s=StageMachine(1,'cpu',reward_mode='overlap',reward_style='continuous')
    s.descent_ready[:]=True
    s.using_descent_target[:]=True
    def run(xy,cos):
        s.update(sample(side_push=False,top_push=False,top_approach_push=False,grasp_target_xy_distance=xy,downward_cosine=cos),1/30)
    run(.02,.93)
    assert s.using_descent_target.item()
    run(.04,.93)
    assert not s.using_descent_target.item()
    run(.02,.93)
    assert not s.using_descent_target.item()
    run(.01,.99)
    assert s.using_descent_target.item()
