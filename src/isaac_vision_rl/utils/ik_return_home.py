"""Post-attempt simulation recovery, separate from rewards and demonstrations."""
import numpy as np


def can_return(actual,m):
    if not np.isfinite(actual).all():return False,'non-finite joint feedback'
    if m['unsafe']:return False,'unsafe measured state'
    if not np.isfinite([m['height'],m['speed']]).all():return False,'non-finite cube feedback'
    if m['height']>.445 or m['speed']>.05:return False,'cube may still be held or moving'
    return True,'cube resting; recovery may open and withdraw'


def make_home_plan(geometry,clearance,actual,home):
    actual=np.asarray(actual);home=np.asarray(home)
    tool=geometry.forward(actual[:5])[:3,3]
    home_tool=geometry.forward(home[:5])[:3,3]
    opened=actual.copy();opened[5]=1.2
    lifted=geometry.solve([*tool[:2],max(.55,tool[2])],opened[:5],position_weight=5.)
    overhead=geometry.solve([*home_tool[:2],.52],lifted,position_weight=5.)
    waypoints=[('open',opened),('clear_box',np.r_[lifted,1.2]),
               ('above_home',np.r_[overhead,1.2]),('fold_home',np.r_[home[:5],1.2]),
               ('close_home',np.r_[home[:5],0.])]
    start=actual
    for _,end in waypoints:
        clearance.check_segment(start,end,margin=.005,samples=121)
        start=end
    return waypoints


def return_home(env,teacher,joint_ids,video_path=None,viewer=None):
    import torch
    from isaac_vision_rl.utils.four_stage import measurements
    robot=env.scene['robot'];device=env.device
    actual=robot.data.joint_pos.torch[0,joint_ids].cpu().numpy()
    m={k:float(v[0]) for k,v in measurements(env).items() if v.ndim==1}
    allowed,reason=can_return(actual,m)
    if not allowed:return dict(status='blocked',reason=reason)
    try:plan=make_home_plan(teacher.geometry,teacher.clearance,actual,teacher.home)
    except ValueError as e:return dict(status='blocked',reason=str(e))
    limits=robot.data.soft_joint_pos_limits.torch[0,joint_ids]
    reference=actual.copy();count=0;minimum=float('inf');peak_force=0.;writer=None
    report=dict(status='incomplete',steps=0)
    if video_path:
        import cv2
        camera=env.scene['camera'].cfg
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),1/env.step_dt,(camera.width,camera.height))
        if not writer.isOpened():raise RuntimeError('Return-home video encoder failed')
    try:
        for phase,goal in plan:
            stable=0.;done=False
            for _ in range(round(15/env.step_dt)):
                if viewer is not None and not viewer.is_running():
                    return dict(status='interrupted',phase=phase,steps=count)
                q=robot.data.joint_pos.torch[0,joint_ids].cpu().numpy()
                m={k:float(v[0]) for k,v in measurements(env).items() if v.ndim==1}
                # Do not continue a recovery after a new contact or uncertain feedback.
                allowed,reason=can_return(q,m)
                if not allowed:return dict(status='blocked',phase=phase,steps=count,reason=reason)
                reference+=np.clip(goal-reference,-np.array([.3,.3,.3,.3,.15,.4])*env.step_dt,
                                   np.array([.3,.3,.3,.3,.15,.4])*env.step_dt)
                target=torch.tensor(reference,device=device,dtype=torch.float32)
                target=torch.clamp(target,limits[:,0],limits[:,1]);target[5].clamp_(0.,1.2)
                action=torch.zeros((1,6),device=device)
                action[0,:5]=2*(target[:5]-limits[:5,0])/(limits[:5,1]-limits[:5,0])-1
                action[0,5]=(target[5]-.6)/.6
                env.action_manager.process_action(action)
                # Step only physics/controller, not the terminated RL episode.
                for _ in range(env.cfg.decimation):
                    env.action_manager.apply_action();env.scene.write_data_to_sim()
                    env.sim.step(render=False);env.scene.update(env.physics_dt)
                env.sim.render()
                count+=1
                q=robot.data.joint_pos.torch[0,joint_ids].cpu().numpy()
                mm=measurements(env);minimum=min(minimum,float(mm['clearance'][0]));peak_force=max(peak_force,float(mm['table_force'][0]))
                if writer is not None:
                    frame=env.scene['camera'].data.output['rgb'].torch[0,:,:,:3].cpu().numpy().astype('uint8')
                    writer.write(frame[:,:,::-1])
                stable=stable+env.step_dt if np.max(np.abs(q-goal))<.025 else 0.
                if stable>=.25:done=True;break
            if not done:return dict(status='timeout',phase=phase,steps=count)
        report=dict(status='home',steps=count,gripper='closed',
                    final_joint_error_rad=float(np.max(np.abs(q-np.r_[teacher.home[:5],0.]))),
                    minimum_table_clearance_m=minimum,maximum_table_force_N=peak_force)
        return report
    finally:
        if writer is not None:writer.release()
