"""Visible, simulation-only comparison of wrist angles at one tool position."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def plan_poses(cfg):
    import numpy as np
    from scipy.optimize import least_squares
    from isaac_vision_rl.utils.grasp_geometry import GraspGeometry
    from isaac_vision_rl.utils.table_clearance import HandTableClearance
    from isaac_vision_rl.utils.four_stage import PREGRASP_CLEARANCE_M
    robot=cfg.scene.robot
    geometry=GraspGeometry(robot.spawn.usd_path,robot.init_state.pos,robot.init_state.rot)
    guard=HandTableClearance(robot.spawn.usd_path,robot.init_state.pos,robot.init_state.rot)
    position=np.array([*cfg.scene.cube.init_state.pos[:2],.425+PREGRASP_CLEARANCE_M])
    radial=position-np.array(robot.init_state.pos);radial[2]=0.;radial/=np.linalg.norm(radial)
    initial=geometry.solve(position,position_weight=5.)
    rng=np.random.default_rng(101)
    poses=[]
    for degrees in (90,60,30,0):
        angle=np.deg2rad(degrees)
        axis=radial*np.sin(angle)+np.array([0.,0.,-1.])*np.cos(angle)
        def residual(q):
            pose=geometry.forward(q)
            return np.r_[5.*(pose[:3,3]-position),.2*(pose[:3,2]-axis),.02*q[4]]
        starts=[initial]+[rng.uniform(geometry.lower,geometry.upper) for _ in range(8)]
        solved=[least_squares(residual,np.clip(q,geometry.lower,geometry.upper),
                              bounds=(geometry.lower,geometry.upper),max_nfev=300) for q in starts]
        best=min(solved,key=lambda s:np.linalg.norm(s.fun))
        pose=geometry.forward(best.x)
        error=float(np.linalg.norm(pose[:3,3]-position))
        axis_error=float(np.rad2deg(np.arccos(np.clip(pose[:3,2]@axis,-1.,1.))))
        q=np.r_[best.x,1.2]
        clearance=guard.minimum(q)
        if error>.002 or axis_error>5. or clearance<.005:
            raise RuntimeError(f'Pose {degrees}: position error {error}, axis error {axis_error}, clearance {clearance}')
        poses.append(dict(degrees_from_down=degrees,joints=q.tolist(),
                          planned_position_m=position.tolist(),planned_error_m=error,
                          planned_axis_error_deg=axis_error,planned_clearance_m=clearance))
        initial=best.x
    return poses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--keep-open',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--compare-hover',help='Recorded rollout JSON: compare its final arm pose with the overhead target, jaws closed/open')
    parser.add_argument('--hold-seconds',type=float,default=4.)
    parser.add_argument('--output',default='outputs/four_stage/wrist_orientation_check')
    args=parser.parse_args()
    os.environ.update(SO101_POLICY='rgb',SO101_CURRICULUM='place',SO101_REWARD_PROFILE='four-stage',
                      SO101_GRASP_REWARD_MODE='bounded',SO101_REWARD_STYLE='continuous',
                      SO101_RGB_FPS='30',SO101_CAMERA_WIDTH='640',SO101_CAMERA_HEIGHT='360',
                      SO101_CUBE_JITTER='0',SO101_BOX_JITTER='0',SO101_APPROACH_CURRICULUM='0')
    import numpy as np
    import torch
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    from isaac_vision_rl.utils.manual_inspection import configure_manual_inspection
    from isaac_vision_rl.utils.four_stage import measurements,StageMachine
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES
    from isaac_vision_rl.utils.reward_panel import RewardPanel
    cfg=PickplaceEnvCfg();cfg.scene.num_envs=1;cfg.sim.device='cuda:0'
    configure_manual_inspection(cfg)
    def make_plan():
        poses=plan_poses(cfg)
        if not args.compare_hover:return poses
        from isaac_vision_rl.utils.grasp_geometry import GraspGeometry
        from isaac_vision_rl.utils.table_clearance import HandTableClearance
        robot=cfg.scene.robot
        g=GraspGeometry(robot.spawn.usd_path,robot.init_state.pos,robot.init_state.rot)
        guard=HandTableClearance(robot.spawn.usd_path,robot.init_state.pos,robot.init_state.rot)
        trace=json.loads(Path(args.compare_hover).read_text())['trace']
        far=np.asarray(trace[-1]['joint_position_rad'],dtype=float)
        near=np.asarray(poses[-1]['joints'],dtype=float)
        result=[]
        for label,source,jaw in [('Far / closed jaws',far,0.),('Far / open jaws',far,1.2),
                                 ('Near / closed jaws',near,0.),('Near / open jaws',near,1.2)]:
            q=source.copy();q[5]=jaw;pose=g.forward(q[:5]);clearance=guard.minimum(q)
            if clearance<.005:raise RuntimeError('Comparison pose fails table clearance')
            result.append(dict(label=label,joints=q.tolist(),planned_position_m=pose[:3,3].tolist(),
                               degrees_from_down=float(np.rad2deg(np.arccos(np.clip(-pose[2,2],-1.,1.)))),
                               planned_clearance_m=clearance))
        return result
    if args.plan_only:
        print(json.dumps(make_plan(),indent=2));return
    from isaaclab.app import launch_simulation
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[];images=[]
    with launch_simulation(cfg,{'device':'cuda:0','visualizer':None if args.headless else 'newton'}):
        # Loading USD geometry before SimulationApp can break Kit extensions.
        poses=make_plan()
        from isaaclab.envs import ManagerBasedRLEnv
        env=ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot=env.scene['robot'];names=ARM_NAMES+['gripper']
            ids=[robot.find_joints(n)[0][0] for n in names]
            limits=robot.data.soft_joint_pos_limits.torch[0,ids]
            panel=RewardPanel([]);viewer=None
            panel.snapshot=dict(stage=0,reward_stage=0,step_reward=0.,
                                group_contribution=[[0.,0.,0.] for _ in range(4)],
                                raw=[0.,0.,0.],contribution=[0.,0.,0.],reward_style='continuous')
            panel.chart_note='AUTOMATIC WRIST POSE CHECK | Fixed tool position | SIMULATION ONLY'
            if not args.headless:
                env.sim.render()
                viewer=next(v._viewer for v in env.sim.visualizers if hasattr(v,'_viewer') and v._viewer is not None)
                from isaac_vision_rl.utils.newton_colors import apply_newton_colors
                apply_newton_colors(viewer)
                viewer.register_ui_callback(panel.draw_charts,position='free')
            state=StageMachine(1,env.device,reward_mode='bounded',reward_style='continuous')
            for pose in poses:
                # Static diagnostic placements, not policy actions or a motion plan.
                q=torch.tensor([pose['joints']],device=env.device,dtype=torch.float32)
                robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=ids)
                env.action_manager.reset(env_ids=torch.arange(env.num_envs,device=env.device))
                action=torch.zeros((1,6),device=env.device)
                action[0,:5]=2*(q[0,:5]-limits[:5,0])/(limits[:5,1]-limits[:5,0])-1
                action[0,5]=(q[0,5]-.6)/.6
                last=None
                for step in range(max(30,round(args.hold_seconds/env.step_dt))):
                    if viewer is not None and not viewer.is_running():return
                    env.step(action)
                    m=measurements(env)
                    if bool(m['unsafe'][0]):raise RuntimeError('Unsafe pose in orientation diagnostic')
                    m['reach_ready']=torch.zeros_like(m['reach_ready'])
                    reward=state.update(m,env.step_dt)
                    last=dict(stage=0,reward_stage=0,step_reward=float(reward[0]),
                              group_contribution=state.group_contribution[0].cpu().tolist(),
                              raw=state.raw[0].cpu().tolist(),contribution=state.contribution[0].cpu().tolist(),
                              reward_style='continuous',reward_weights=state.weights)
                    label=pose.get('label',f"{pose['degrees_from_down']:.1f} degrees from DOWN")
                    panel.chart_note=f"STATIC POSE CHECK | {label} | No training"
                    panel.snapshot=last;panel.record(last)
                    if viewer is not None:
                        env.sim.render();time.sleep(.015)
                body=robot.find_bodies('gripper_frame_link')[0][0]
                position=robot.data.body_pos_w.torch[0,body].cpu().numpy()
                report={**pose,'measured_position_m':position.tolist(),
                        'measured_position_error_m':float(np.linalg.norm(position-np.array(pose['planned_position_m']))),
                        'measured_downward_cosine':float(m['downward_cosine'][0]),
                        'measured_angle_from_down_deg':float(np.rad2deg(np.arccos(np.clip(float(m['downward_cosine'][0]),-1.,1.)))),
                        'clearance_m':float(m['clearance'][0]),
                        'measured_jaw_rad':float(m['jaw'][0]),
                        'orientation_reward':float(state.group_contribution[0,0,1]),
                        'proximity_reward':float(state.group_contribution[0,0,0]),
                        'approach_reward':float(state.group_contribution[0,0,:2].sum())}
                rows.append(report)
                images.append(env.scene['camera'].data.output['rgb'].torch[0,:,:,:3].cpu().numpy().astype('uint8'))
                print(json.dumps(report),flush=True)
            ordered=all(rows[i]['orientation_reward']<rows[i+1]['orientation_reward'] for i in range(len(rows)-1))
            if args.compare_hover:
                ordered=(rows[0]['orientation_reward']<0<rows[1]['orientation_reward']
                         and rows[0]['approach_reward']<rows[1]['approach_reward']<rows[3]['approach_reward']
                         and rows[2]['approach_reward']<rows[3]['approach_reward'])
            report=dict(passed=ordered and max(r['measured_position_error_m'] for r in rows)<.01,
                        position_tolerance_m=.01,
                        check='Measured PhysX poses and actual reward terms; no learned policy or grasp evaluation',poses=rows)
            output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,2,figsize=(14,9))
            for ax,frame,row in zip(axes.flat,images,rows):
                ax.imshow(frame);ax.axis('off')
                label=row.get('label',f"{row['measured_angle_from_down_deg']:.1f} degrees from down")
                ax.set_title(f"{label} | approach reward {row['approach_reward']:+.5f}")
            fig.suptitle('Measured approach reward: recorded hover vs overhead target' if args.compare_hover
                         else 'Actual simulated wrist poses at the same target position')
            fig.tight_layout();fig.savefig(output.with_suffix('.png'),dpi=140);plt.close(fig)
            print(json.dumps({'passed':report['passed'],'report':str(output.with_suffix('.json'))}),flush=True)
            while args.keep_open and viewer is not None and viewer.is_running():
                env.sim.render();time.sleep(.03)
        finally:env.close()


if __name__=='__main__':main()
