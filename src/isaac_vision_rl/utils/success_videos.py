"""Disk-backed RGB episode capture; retain only successful full placements."""
import atexit
import json
from pathlib import Path


class SuccessVideos:
    def __init__(self, directory, count, fps, size=(640,360)):
        import cv2
        self.cv2=cv2
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.count=count;self.fps=fps;self.size=size
        import os
        self.goal=os.environ.get('SO101_TRAINING_GOAL','place')
        self.writers={};self.paths={};self.frames={};self.finished=set()
        self.episodes=[0]*count
        import os
        self.events=StageEventVideos(self.directory,count,fps,size) if os.environ.get('SO101_STAGE_VIDEOS','0')=='1' else None
        atexit.register(self.close)

    def reset(self,ids=None):
        if self.events is not None:self.events.reset(ids)
        ids=range(self.count) if ids is None else ids
        for index in ids:
            i=int(index)
            # RewardManager invokes reset once per slot; count each episode once.
            new_episode=i in self.writers or i in self.finished or self.episodes[i]==0
            self.discard(i)
            self.finished.discard(i)
            if new_episode:self.episodes[i]+=1

    def discard(self,i):
        writer=self.writers.pop(i,None)
        if writer is not None:writer.release()
        path=self.paths.pop(i,None)
        if path is not None:path.unlink(missing_ok=True)
        self.frames.pop(i,None)

    def capture(self,rgb,success,failure,step):
        for i in range(self.count):
            if i in self.finished:continue
            if i not in self.writers:
                path=self.directory/f'.attempt_env{i:03d}_episode{self.episodes[i]:06d}.mp4'
                writer=self.cv2.VideoWriter(str(path),self.cv2.VideoWriter_fourcc(*'mp4v'),self.fps,self.size)
                if not writer.isOpened():raise RuntimeError(f'Cannot record training video: {path}')
                self.writers[i]=writer;self.paths[i]=path;self.frames[i]=0
            image=self.cv2.resize(rgb[i,:,:,:3],self.size)
            self.writers[i].write(self.cv2.cvtColor(image,self.cv2.COLOR_RGB2BGR))
            self.frames[i]+=1
            if bool(success[i]) and not bool(failure[i]):
                self.writers.pop(i).release()
                path=self.paths.pop(i)
                target=self.directory/f'success_step{step:09d}_env{i:03d}_episode{self.episodes[i]:06d}.mp4'
                path.rename(target)
                metadata=dict(environment=i,episode=self.episodes[i],simulation_step=int(step),
                              frames=self.frames.pop(i),fps=self.fps,success='stable_held_lift' if self.goal=='lift' else 'four_stage_placement',
                              source='training rollout, not independent evaluation')
                target.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
                self.finished.add(i)
            elif bool(failure[i]):
                self.discard(i);self.finished.add(i)

    def close(self):
        for i in list(self.writers):self.discard(i)
        if self.events is not None:self.events.close()


def record_training_step(env,state):
    import os
    if not hasattr(env,'_success_videos'):
        env._success_videos=SuccessVideos(os.environ['SO101_SUCCESS_VIDEO_DIR'],env.num_envs,1/env.step_dt)
        print(f'[INFO] Successful episode videos: {env._success_videos.directory}',flush=True)
    rgb=env.scene['camera'].data.output['rgb']
    rgb=rgb.torch if hasattr(rgb,'torch') else rgb
    images=rgb.detach().cpu().numpy()
    success=state.success.cpu().tolist();failure=state.failure.cpu().tolist()
    env._success_videos.capture(images,success,failure,env.common_step_counter)
    if env._success_videos.events is not None:
        # Copy diagnostics once; avoid many tiny CUDA synchronizations per frame.
        m={k:v.detach().cpu().tolist() for k,v in state.measurements.items()
           if k in ("clearance","table_force","height","gripper_out_of_view","cube_out_of_view","robot_dropped")}
        lost=state.lost.cpu().tolist();before=state.reward_stage.cpu().tolist()
        reasons=[]
        for i in range(env.num_envs):
            why=[]
            if float(m['clearance'][i])<.003 or float(m['table_force'][i])>.1:why.append('unsafe_table_contact')
            if float(m['height'][i])<.35:why.append('cube_dropped')
            for name in ('gripper_out_of_view','cube_out_of_view','robot_dropped'):
                if bool(m.get(name,[False]*env.num_envs)[i]):why.append(name)
            if float(lost[i])>=.2 and int(before[i])>=2:why.append('lost_grasp')
            if failure[i] and not why:why.append('stage_failure')
            reasons.append(why)
        env._success_videos.events.capture(images,before,state.stage.cpu().tolist(),
                                           success,reasons,env.common_step_counter)



class StageEventVideos:
    """Finalize each stage separately so later failures preserve earlier milestones."""
    labels=('reach','grasp','lift','place')

    def __init__(self,directory,count,fps,size=(640,360)):
        import cv2
        self.cv2=cv2;self.root=Path(directory);self.count=count;self.fps=fps;self.size=size
        self.writers={};self.paths={};self.stages={};self.frames={};self.finished=set()
        self.episodes=[0]*count
        atexit.register(self.close)

    def discard(self,i):
        writer=self.writers.pop(i,None)
        if writer is not None:writer.release()
        path=self.paths.pop(i,None)
        if path is not None:path.unlink(missing_ok=True)
        self.stages.pop(i,None)
        self.frames.pop(i,None)

    def reset(self,ids=None):
        import os
        save_failures=os.environ.get('SO101_STAGE_VIDEOS_FAILURES','0')=='1'
        for index in (range(self.count) if ids is None else ids):
            i=int(index)
            new=i in self.writers or i in self.finished or self.episodes[i]==0
            if i in self.writers:
                if save_failures:self.finish(i,'failures',['episode_ended_without_milestone'],None)
                else:self.discard(i)
            self.finished.discard(i)
            if new:self.episodes[i]+=1

    def finish(self,i,category,reasons,step):
        self.writers.pop(i).release()
        folder=self.root/category;folder.mkdir(parents=True,exist_ok=True)
        stage=self.stages.pop(i)
        target=folder/f'{self.labels[stage]}_env{i:03d}_episode{self.episodes[i]:06d}_step{step if step is not None else "end"}.mp4'
        self.paths.pop(i).rename(target)
        metadata=dict(environment=i,episode=self.episodes[i],stage=self.labels[stage],
                      outcome=category,reasons=reasons,simulation_step=step,
                      frames=self.frames.pop(i),fps=self.fps,
                      source='training stage segment, not independent evaluation or full-task success')
        target.with_suffix('.json').write_text(json.dumps(metadata,indent=2))

    def capture(self,rgb,before,after,success,reasons,step):
        import os
        save_failures=os.environ.get('SO101_STAGE_VIDEOS_FAILURES','0')=='1'
        self.root.mkdir(parents=True,exist_ok=True)
        for i in range(self.count):
            if i in self.finished:continue
            if i not in self.writers:
                path=self.root/f'.stage_env{i:03d}_episode{self.episodes[i]:06d}.mp4'
                w=self.cv2.VideoWriter(str(path),self.cv2.VideoWriter_fourcc(*'mp4v'),self.fps,self.size)
                if not w.isOpened():raise RuntimeError(f'Cannot record stage video: {path}')
                self.writers[i]=w;self.paths[i]=path;self.stages[i]=int(before[i]);self.frames[i]=0
            frame=self.cv2.resize(rgb[i,:,:,:3],self.size)
            self.writers[i].write(self.cv2.cvtColor(frame,self.cv2.COLOR_RGB2BGR));self.frames[i]+=1
            if reasons[i]:
                if save_failures:self.finish(i,'failures',reasons[i],int(step))
                else:self.discard(i)
                self.finished.add(i)
            elif int(after[i])>int(before[i]) or bool(success[i]):
                label=self.labels[self.stages[i]]
                self.finish(i,f'stage_success/{label}',[],int(step))
                if bool(success[i]):self.finished.add(i)

    def close(self):
        import os
        save_failures=os.environ.get('SO101_STAGE_VIDEOS_FAILURES','0')=='1'
        for i in list(self.writers):
            if save_failures:self.finish(i,'failures',['recording_interrupted'],None)
            else:self.discard(i)
