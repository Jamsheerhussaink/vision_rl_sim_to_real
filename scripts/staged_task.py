"""Four-stage manual control, bounded scripted diagnostics and policy evaluation."""
import argparse,json,math,os,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['manual','hold','scripted','evaluate','controller-smoke','ik-teacher','dagger'],default='manual')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--reward-style',choices=['continuous','progress'],default='continuous')
    p.add_argument('--keep-open',action='store_true',help='Keep Newton and final reward graphs visible after the bounded attempt')
    p.add_argument('--checkpoint')
    p.add_argument('--lift-hold-seconds',type=float,default=None,help='Override saved lift hold duration for a transition diagnostic')
    p.add_argument('--lift-rise-min',type=float,default=None)
    p.add_argument('--lift-rise-max',type=float,default=None)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--output',default='outputs/four_stage/session.json')
    p.add_argument('--record-demo',action='store_true',help='Save aligned actor observations and expert actions; only complete safe attempts are eligible for cloning')
    p.add_argument('--dagger-policy-steps',type=int,default=5,help='Number of learner actions before the IK teacher takes control in DAgger mode')
    p.add_argument('--dagger-safe-clearance-mm',type=float,default=5.5,help='Hand-table margin that triggers teacher takeover in DAgger mode')
    p.add_argument('--record-video',action='store_true')
    p.add_argument('--only-success-video',action='store_true',help='Discard recorded video if the rollout does not achieve success')
    p.add_argument('--reward-window',action='store_true',help='Open standalone live reward charts window (supports headless mode)')
    p.add_argument('--ui-smoke',action='store_true',help='Render a bounded manual panel check without jogging')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--cube-offset-x',type=float,default=0.,help='Fixed demonstration cube X offset in metres, at most 1 cm')
    p.add_argument('--cube-offset-y',type=float,default=0.,help='Fixed demonstration cube Y offset in metres, at most 1 cm')
    args=p.parse_args()
    if not all(abs(v)<=.01 for v in (args.cube_offset_x,args.cube_offset_y)):
        p.error('Demonstration cube offsets must be finite and within +/-0.01 m')
    os.environ["SO101_REWARD_STYLE"]=args.reward_style
    control_hz=30
    if args.checkpoint:
        import torch
        metadata=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        os.environ["SO101_GRASP_REWARD_MODE"]=("bounded" if metadata.get("so101_agent_schema")=="four_stage_v2_bounded" else "overlap")
        os.environ["SO101_REWARD_STYLE"]=metadata.get("so101_reward_style","progress")
        os.environ["SO101_TRAINING_GOAL"]=metadata.get("so101_training_goal","place")
        for key,default in (('rise_min',.15),('rise_max',.20),('hold_seconds',.5)):
            os.environ['SO101_LIFT_'+key.upper()]=str(metadata.get('so101_lift_'+key,default))
        profile=metadata.get('so101_rgb_profile','1280x720_letterbox128_v1')
        resolution=re.fullmatch(r'(\d+)x(\d+)_letterbox128_v1',profile)
        if resolution is None:p.error(f'Unsupported checkpoint image profile: {profile}')
        os.environ['SO101_CAMERA_WIDTH'],os.environ['SO101_CAMERA_HEIGHT']=resolution.groups()
        control_hz=int(metadata.get('so101_control_hz',30))
        if control_hz not in (25,30,60):p.error(f'Unsupported checkpoint control rate: {control_hz}')
        del metadata
    for key in ('rise_min','rise_max','hold_seconds'):
        value=getattr(args,'lift_'+key)
        if value is not None:
            if not math.isfinite(value) or value<=0:p.error('Lift overrides must be finite and positive')
            os.environ['SO101_LIFT_'+key.upper()]=str(value)
    if float(os.environ.get('SO101_LIFT_RISE_MAX','.20'))<=float(os.environ.get('SO101_LIFT_RISE_MIN','.15')):
        p.error('Maximum lift rise must exceed minimum')
    if args.keep_open and args.headless and not args.reward_window:p.error('--keep-open requires the viewer or --reward-window')
    if args.record_demo and args.mode not in ('ik-teacher','dagger'):p.error('--record-demo requires --mode ik-teacher or dagger')
    if args.mode=='dagger' and not args.checkpoint:p.error('--mode dagger requires --checkpoint')
    if args.mode=='dagger' and not args.record_demo:p.error('--mode dagger requires --record-demo')
    if args.dagger_policy_steps<0 or not math.isfinite(args.dagger_safe_clearance_mm) or args.dagger_safe_clearance_mm<0:
        p.error('DAgger action count and clearance must be non-negative')

    if args.mode=='manual' and args.headless:p.error('Manual control requires Newton Viewer')
    if args.mode=='evaluate' and not args.checkpoint:p.error('Evaluation requires a four-stage checkpoint')
    os.environ.update(SO101_POLICY='rgb',SO101_CURRICULUM='place',SO101_REWARD_PROFILE='four-stage',
                     SO101_APPROACH_CURRICULUM='0',SO101_CUBE_JITTER='0',SO101_BOX_JITTER='0',SO101_RGB_FPS=str(control_hz))
    import numpy as np
    import torch
    from isaaclab.app import launch_simulation
    from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
    from isaac_vision_rl.utils.four_stage import (
        tick,machine,VERSION,LIFT_RISE_MIN_M,LIFT_RISE_MAX_M,
    )
    from isaac_vision_rl.utils.grasp_geometry import ARM_NAMES
    from isaac_vision_rl.utils.reward_panel import RewardPanel
    cfg=PickplaceEnvCfg();cfg.scene.num_envs=1;cfg.sim.device=args.device;cfg.seed=args.seed
    cube_position=list(cfg.scene.cube.init_state.pos)
    cube_position[0]+=args.cube_offset_x;cube_position[1]+=args.cube_offset_y
    cfg.scene.cube.init_state.pos=tuple(cube_position)
    if args.mode=='manual':
        from isaac_vision_rl.utils.manual_inspection import configure_manual_inspection
        configure_manual_inspection(cfg)
    cfg.events.randomize_cube_mass=None;cfg.events.randomize_cube_friction=None
    cfg.events.reset_cube_pose.params['pose_range']={}
    cfg.events.reset_box_pose.params['pose_range']={}
    cfg.sim.default_visualizer_cfg.streaming_view=True
    cfg.sim.default_visualizer_cfg.streaming_envs=[0]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if args.mode in ("ik-teacher", "dagger") and (output.exists() or output.with_suffix(".mp4").exists()):
        p.error("Use a new output path to preserve existing teacher/DAgger evidence")
    rows=[];video=None;demo=None;teacher=None;demo_success=False;ui_frames=0
    with launch_simulation(cfg,{'device':args.device,'visualizer':None if args.headless else 'newton'}):
        from isaaclab.envs import ManagerBasedRLEnv
        class InspectionEnv(ManagerBasedRLEnv):
            block_autoreset=False
            def _reset_idx(self,env_ids):
                # Authoring/evaluation mode freezes the terminal scene. Training keeps normal autoreset.
                if not self.block_autoreset:super()._reset_idx(env_ids)
        env=InspectionEnv(cfg=cfg)
        try:
            obs,_=env.reset();env.block_autoreset=True
            robot=env.scene['robot'];names=ARM_NAMES+['gripper']
            ids=[robot.find_joints(n)[0][0] for n in names]
            panel=RewardPanel(names)
            reward_window=None
            if args.reward_window:
                from isaac_vision_rl.utils.reward_window import RewardWindow
                reward_window=RewardWindow()
            if args.mode=='manual':
                panel.reason='MANUAL INSPECTION: all episode terminations OFF | 1 environment | Run to jog'
            initial=tick(env)
            panel.joints=robot.data.joint_pos.torch[0,ids].cpu().tolist()
            panel.snapshot=dict(stage=int(initial.stage[0]),reward_stage=int(initial.reward_stage[0]),
                group_contribution=initial.group_contribution[0].cpu().tolist(),
                raw=initial.raw[0].cpu().tolist(),contribution=initial.contribution[0].cpu().tolist(),
                step_reward=0.,measurements={k:float(initial.measurements[k][0])
                    if k in initial.measurements else 0.
                    for k in ('height','clearance','table_force','bilateral','slide_distance')})
            viewer=None
            if not args.headless:
                env.sim.render()
                viewer=next(v._viewer for v in env.sim.visualizers if hasattr(v,'_viewer') and v._viewer is not None)
                from isaac_vision_rl.utils.newton_colors import apply_newton_colors
                print('Newton fallback color counts:',apply_newton_colors(viewer),flush=True)
                viewer.register_ui_callback(panel.draw if args.mode=='manual' else panel.draw_charts,position='free')
            limits=robot.data.soft_joint_pos_limits.torch[0,ids]
            target=robot.data.joint_pos.torch[0,ids].clone()
            panel.joint_limits=limits.cpu().tolist()
            def hold():
                nonlocal target
                target=robot.data.joint_pos.torch[0,ids].clone()
                env.action_manager.reset(env_ids=torch.arange(env.num_envs,device=args.device))
                robot.set_joint_position_target(robot.data.joint_pos.torch.clone())
            policy=None
            if args.mode in ('evaluate','dagger'):
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
                from isaaclab.utils.dict import class_to_dict
                from rsl_rl.runners import OnPolicyRunner
                from isaac_vision_rl.tasks.pickplace.config.so101.agents.rsl_rl_ppo_cfg import PPORunnerCfg
                import importlib.metadata
                agent=handle_deprecated_rsl_rl_cfg(PPORunnerCfg(),importlib.metadata.version('rsl-rl-lib'))
                wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
                runner=OnPolicyRunner(wrapped,class_to_dict(agent),log_dir=None,device=args.device)
                runner.load(str(Path(args.checkpoint).resolve()));policy=runner.get_inference_policy()
            if args.mode in ('ik-teacher','dagger'):
                from isaac_vision_rl.utils.ik_teacher import IKTeacher
                from isaac_vision_rl.utils.grasp_geometry import GraspGeometry
                from isaac_vision_rl.utils.table_clearance import HandTableClearance
                asset=cfg.scene.robot
                g=GraspGeometry(asset.spawn.usd_path,asset.init_state.pos,asset.init_state.rot)
                guard=HandTableClearance(asset.spawn.usd_path,asset.init_state.pos,asset.init_state.rot)
                teacher=IKTeacher(g,guard,target.detach().cpu().numpy(),
                    env.scene['cube'].data.root_pos_w.torch[0].cpu().numpy(),
                    env.scene['box'].data.root_pos_w.torch[0].cpu().numpy(),env.step_dt)
                if args.record_demo:
                    from isaac_vision_rl.utils.demonstrations import Demonstration
                    demo=Demonstration(output.with_suffix('').with_name(output.stem+'_demo'))
            plan=[]
            if args.mode=='scripted':
                from isaac_vision_rl.utils.grasp_geometry import GraspGeometry
                g=GraspGeometry(cfg.scene.robot.spawn.usd_path,cfg.scene.robot.init_state.pos,cfg.scene.robot.init_state.rot)
                cp=np.array(cfg.scene.cube.init_state.pos);cp[2]=.425
                bp=np.array(cfg.scene.box.init_state.pos)
                # Place inside the near part of the box, within the arm's verified reach.
                bp[:2]-=.02
                q=target[:5].cpu().numpy()
                plan.append((q.copy(),1.2,.5))
                for pos,jaw,duration in ((cp+[0,0,.04],1.2,3.),(cp+[0,0,.012],1.2,2.),
                        (cp+[0,0,.012],0.,2.),(cp+[0,0,.10],0.,3.),
                        (bp+[0,0,.10],0.,3.),(bp+[0,0,.043],0.,3.),
                        (bp+[0,0,.043],1.2,2.),(bp+[0,0,.10],1.2,3.)):
                    q=g.solve(pos,q);plan.append((q.copy(),jaw,duration))
                plan.append((q.copy(),1.2,1.))
            from isaac_vision_rl.utils.dagger import should_use_teacher
            phase=0;phase_time=0.;start_target=target.clone()
            dagger_policy_steps=0;dagger_teacher_control=False
            if args.record_video:
                import cv2
                video=cv2.VideoWriter(str(output.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30.,
                                      (cfg.scene.camera.width,cfg.scene.camera.height))
                if not video.isOpened():raise RuntimeError('MP4 encoder failed')
            steps=0;loops=0;was_jogging=False
            while args.max_steps==0 or (loops if args.ui_smoke else steps)<args.max_steps:
                if viewer is not None and not viewer.is_running():break
                loops+=1
                if panel.reset_requested:
                    env.block_autoreset=False;obs,_=env.reset();env.block_autoreset=True
                    panel.reset_requested=False;panel.locked=False;panel.reason='Reset to HOME';panel.total=0.
                    panel.clear_history()
                    hold()
                    fresh=tick(env)
                    panel.snapshot=dict(stage=int(fresh.stage[0]),reward_stage=int(fresh.reward_stage[0]),
                        group_contribution=[[0.,0.,0.] for _ in range(4)],
                        raw=[0.,0.,0.],contribution=[0.,0.,0.],step_reward=0.,
                        measurements={k:float(fresh.measurements[k][0])
                            if k in fresh.measurements else 0.
                            for k in ('height','clearance','table_force','bilateral','slide_distance')})
                if panel.stop_requested:
                    hold();panel.stop_requested=False
                if args.ui_smoke or (args.mode=='manual' and (panel.paused or panel.locked)):
                    env.sim.render();time.sleep(.01)
                    continue
                if args.mode=='manual':
                    if panel.hold_gripper and panel.gripper_hold_target is None:
                        panel.gripper_hold_target=float(target[5])
                    if panel.pending_pose is not None:
                        target=torch.tensor(panel.pending_pose,device=args.device,dtype=target.dtype)
                        panel.pending_pose=None
                    jogging=any(panel.velocity)
                    if was_jogging and not jogging:hold()
                    was_jogging=jogging
                    target+=torch.tensor(panel.velocity,device=args.device)*env.step_dt
                    if panel.hold_gripper:
                        target[5]=panel.gripper_hold_target
                elif args.mode in ('ik-teacher','dagger'):
                    state=tick(env)
                    body=robot.find_bodies('gripper_frame_link')[0][0]
                    proposed=teacher.step(robot.data.joint_pos.torch[0,ids].cpu().numpy(),
                        robot.data.body_pos_w.torch[0,body].cpu().numpy(),
                        env.scene['cube'].data.root_pos_w.torch[0].cpu().numpy(),
                        {k:float(v[0]) for k,v in state.measurements.items() if v.ndim==1})
                    if teacher.failure or teacher.phase_name=='done':break
                    target=torch.tensor(proposed,device=args.device,dtype=torch.float32)
                elif args.mode=='hold':
                    pass
                elif args.mode=='controller-smoke':
                    target[5]=.5 if (steps//15)%2==0 else 1.2
                elif args.mode=='scripted':
                    if phase>=len(plan):break
                    goal,jaw,duration=plan[phase];end=torch.tensor([*goal,jaw],device=args.device,dtype=torch.float32)
                    phase_time+=env.step_dt
                    target=start_target+(end-start_target)*min(1.,phase_time/duration)
                    if phase_time>=duration:
                        phase+=1;phase_time=0.;start_target=target.clone()
                target=torch.maximum(torch.minimum(target,limits[:,1]),limits[:,0])
                target[5].clamp_(0.,1.2)
                action=torch.zeros(1,6,device=args.device)
                action[0,:5]=2*(target[:5]-limits[:5,0])/(limits[:5,1]-limits[:5,0])-1
                action[0,5]=(target[5]-.6)/.6
                expert_action=None
                if policy is not None:
                    with torch.inference_mode():action=policy(wrapped.get_observations())
                    if agent.clip_actions is not None:
                        action=action.clamp(-agent.clip_actions,agent.clip_actions)
                    if args.mode=='dagger':
                        expert_action=torch.zeros_like(action)
                        expert_action[0,:5]=2*(target[:5]-limits[:5,0])/(limits[:5,1]-limits[:5,0])-1
                        expert_action[0,5]=(target[5]-.6)/.6
                        clearance_m=float(state.measurements['clearance'][0])
                        dagger_teacher_control=should_use_teacher(
                            policy_steps=dagger_policy_steps,
                            clearance_m=clearance_m,
                            takeover_after_steps=args.dagger_policy_steps,
                            minimum_clearance_m=args.dagger_safe_clearance_mm/1000.,
                            teacher_already_in_control=dagger_teacher_control,
                        )
                        if dagger_teacher_control:
                            action=expert_action
                        else:
                            dagger_policy_steps+=1
                if demo is not None:
                    # Snapshot BEFORE stepping: post-step observations would mislabel actions.
                    pre_policy=obs['policy'][0].detach().cpu().numpy().copy()
                    pre_image=obs['image'][0].detach().cpu().numpy().copy()
                    # DAgger stores the teacher's correction for the exact
                    # pre-action observation, even while the learner acts.
                    sent_action=(expert_action[0] if args.mode=='dagger' and expert_action is not None
                                 else action[0]).detach().cpu().numpy().copy()
                    teacher_phase=teacher.phase_name
                obs,reward,terminated,truncated,info=env.step(action)
                if demo is not None:
                    demo.add(pre_policy,pre_image,sent_action,float(reward[0]),
                             bool(terminated[0] or truncated[0]),teacher_phase)
                steps+=1
                s=tick(env);m=s.measurements
                panel.total+=float(reward[0]);panel.joints=robot.data.joint_pos.torch[0,ids].cpu().tolist()
                diagnostic_names=('jaw','height','lift_rise','speed','clearance','table_force','slide_speed','slide_distance','bilateral','enclosed','reach_ready','unsafe','inside_box','placed','opening_for_release','top_push','top_approach_push','side_push','tool_down_speed','pregrasp_distance','alignment','enclosure','box_distance','lowering','fixed_contact_in_pad','moving_contact_in_pad','fixed_contact_force_N','moving_contact_force_N','fixed_contact_force_x_N','fixed_contact_force_y_N','fixed_contact_force_z_N','moving_contact_force_x_N','moving_contact_force_y_N','moving_contact_force_z_N','opposing_contact_forces','contact_force_cosine')
                diagnostics={k:round(float(m[k][0]),5) for k in diagnostic_names if k in m}
                for name in ('fixed_finger_contact','moving_finger_contact'):
                    f=env.scene[name].data.normal_force_matrix_w.torch
                    diagnostics[name+'_N']=float(torch.linalg.vector_norm(f[0].reshape(-1,3).sum(0)))
                row=dict(step=steps,stage=int(s.stage[0]),reward_stage=int(s.reward_stage[0]),
                    cube_position_m=env.scene['cube'].data.root_pos_w.torch[0].cpu().tolist(),
                    tool_position_m=robot.data.body_pos_w.torch[0,robot.find_bodies('gripper_frame_link')[0][0]].cpu().tolist(),
                    joint_position_rad=robot.data.joint_pos.torch[0,ids].cpu().tolist(),
                    joint_reference_rad=torch.cat([env.action_manager.get_term(n).processed_actions[0]
                        for n in ('arm_action','gripper_action')]).detach().cpu().tolist(),
                    dwell_fraction=min(1.,float(s.dwell[0])/(.08,.2,s.lift_hold_seconds,.4)[int(s.stage[0])]),
                    reward_weights=s.weights,reward_style=s.reward_style,
                    group_contribution=s.group_contribution[0].detach().cpu().tolist(),raw=s.raw[0].cpu().tolist(),contribution=s.contribution[0].cpu().tolist(),
                    step_reward=float(reward[0]),episode_return=panel.total,measurements=diagnostics)
                if teacher is not None:row["teacher_phase"]=teacher.phase_name
                if teacher is not None:
                    row['teacher_elapsed_s']=round(float(teacher.elapsed),4)
                    row['teacher_goal_position_m']=(None if teacher.position is None
                        else np.asarray(teacher.position,dtype=float).tolist())
                    row['teacher_tcp_error_m']=(None if teacher.position is None
                        else float(np.linalg.norm(np.asarray(row['tool_position_m'])-teacher.position)))
                    row['teacher_joint_goal_error_rad']=float(np.max(np.abs(
                        robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy()-teacher.goal[:5])))
                if args.mode=='dagger':
                    row['dagger_teacher_control']=dagger_teacher_control
                    row['dagger_learner_actions']=dagger_policy_steps
                panel.snapshot=row;rows.append(row)
                panel.record(row)
                if reward_window is not None:reward_window.send(row)
                panel.history=panel.history[1:]+[float(reward[0])]
                if video is not None:
                    frame=env.scene['camera'].data.output['rgb'].torch[0,:,:,:3].cpu().numpy().astype('uint8')
                    video.write(frame[:,:,::-1])
                if bool(terminated[0]) or bool(truncated[0]):
                    reasons=[n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])]
                    panel.reason='Ended: '+', '.join(reasons);panel.locked=True;panel.paused=True;hold()
                    row['termination']=reasons
                    if args.mode!='manual':break
                if viewer is not None:env.sim.render()
            ui_frames=panel.draw_count
            if viewer is not None and args.ui_smoke:
                frame=viewer.get_frame()
                if frame is not None:
                    from PIL import Image
                    if hasattr(frame,'numpy'):frame=frame.numpy()
                    Image.fromarray(np.asarray(frame).astype('uint8')).save(output.with_suffix('.png'))
            result=dict(version=VERSION,mode=args.mode,steps=steps,ui_draw_count=ui_frames,
                        checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                        lift_rise_min_m=LIFT_RISE_MIN_M,
                        lift_rise_max_m=LIFT_RISE_MAX_M,
                        lift_hold_seconds=machine(env).lift_hold_seconds,
                        terminations_disabled=args.mode=='manual',num_envs=cfg.scene.num_envs,
                        seed=args.seed,cube_start_m=cube_position,success=bool(machine(env).success[0]),
                        end_reason=(rows[-1].get('termination') if rows and 'termination' in rows[-1]
                                    else (('teacher_failure: '+teacher.failure) if teacher is not None and teacher.failure
                                          else 'diagnostic_budget_or_window_closed')),trace=rows)
            if teacher is not None:
                result["teacher_failure"]=teacher.failure
                result["teacher_phase"]=teacher.phase_name
            demo_success=result["success"] and (teacher is None or teacher.failure is None)
            if teacher is not None:
                if video is not None:video.release();video=None
                from isaac_vision_rl.utils.ik_return_home import return_home
                result['return_home']=return_home(env,teacher,ids,
                    output.with_name(output.stem+'_return_home.mp4') if args.record_video else None,viewer)
            output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
            if args.keep_open and viewer is not None:
                if video is not None:video.release();video=None
                if demo is not None:
                    demo.finish(demo_success,teacher.failure if teacher and teacher.failure else "see rollout trace")
                    demo=None
                while viewer.is_running():
                    env.sim.render();time.sleep(.02)
        finally:
            if demo is not None:demo.finish(demo_success,teacher.failure if teacher and teacher.failure else "see rollout trace")
            if video is not None:
                video.release()
                if args.only_success_video and not bool(result.get("success", False)):
                    output.with_suffix('.mp4').unlink(missing_ok=True)
            env.close()

if __name__=='__main__':main()
