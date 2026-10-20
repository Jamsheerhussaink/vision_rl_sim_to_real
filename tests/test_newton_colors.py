import numpy as np
from types import SimpleNamespace as NS
from isaac_vision_rl.utils.newton_colors import apply_newton_colors,PALETTE

class Colors:
    def __init__(self):self.values=np.ones((4,3))
    def numpy(self):return self.values.copy()
    def assign(self,values):self.values=values


def test_parent_object_color_wins_over_child_box_primitive_name():
    colors=Colors()
    labels=['/World/envs/env_0/Table/Geometry/table_top_link/box',
            '/World/envs/env_0/Box/Geometry/floor/box',
            '/World/envs/env_0/Robot/Geometry/gripper_link/box',
            '/World/envs/env_0/Cube/geometry/box']
    viewer=NS(model=NS(shape_color=colors,shape_label=labels))
    counts=apply_newton_colors(viewer)
    assert counts==dict(table=1,robot=1,cube=1,box=1)
    np.testing.assert_allclose(colors.values,[PALETTE[n] for n in ('table','box','robot','cube')])
