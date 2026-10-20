import json
import numpy as np
import cv2
from isaac_vision_rl.utils.success_videos import SuccessVideos


def test_saves_every_success_and_discards_failed_attempts(tmp_path):
    recorder=SuccessVideos(tmp_path,2,30,size=(64,48))
    frames=np.zeros((2,48,64,3),dtype=np.uint8)
    frames[0,:,:,0]=255
    recorder.reset();recorder.reset()
    assert recorder.episodes==[1,1]
    for step in range(4):recorder.capture(frames,[False,False],[False,False],step)
    recorder.capture(frames,[True,False],[False,True],4)
    recorder.capture(frames,[True,False],[False,True],5)
    paths=list(tmp_path.glob('success*.mp4'));assert len(paths)==1
    cap=cv2.VideoCapture(str(paths[0]));ok,image=cap.read()
    assert ok and image[:,:,2].mean()>240  # RGB red encoded correctly as BGR.
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==5
    cap.release()
    assert json.loads(paths[0].with_suffix('.json').read_text())['environment']==0
    recorder.reset([0,1]);recorder.reset([0,1])
    assert recorder.episodes==[2,2]
    recorder.capture(frames,[True,True],[False,False],6)
    assert len(list(tmp_path.glob('success*.mp4')))==3
    recorder.close()
    assert not list(tmp_path.glob('.attempt*'))


def test_failure_never_retained_as_success_and_shutdown_cleans_pending(tmp_path):
    r=SuccessVideos(tmp_path,1,30,size=(64,48))
    rgb=np.zeros((1,48,64,3),dtype=np.uint8)
    r.capture(rgb,[True],[True],1)
    assert not list(tmp_path.glob('success*.mp4'))
    r.reset();r.capture(rgb,[False],[False],2)
    r.close()
    assert not list(tmp_path.glob('*.mp4'))


def test_stage_successes_and_later_failure_are_separate_readable_clips(tmp_path):
    from isaac_vision_rl.utils.success_videos import StageEventVideos
    r=StageEventVideos(tmp_path,1,30,size=(64,48));r.reset();r.reset()
    assert r.episodes==[1]
    rgb=np.zeros((1,48,64,3),dtype=np.uint8)
    r.capture(rgb,[0],[0],[False],[[]],1)
    r.capture(rgb,[0],[1],[False],[[]],2)
    r.capture(rgb,[1],[1],[False],[[]],3)
    r.capture(rgb,[1],[2],[False],[[]],4)
    r.capture(rgb,[2],[2],[False],[['unsafe_table_contact','lost_grasp']],5)
    r.capture(rgb,[2],[2],[False],[['lost_grasp']],6)
    assert len(list((tmp_path/'stage_success/reach').glob('*.mp4')))==1
    assert len(list((tmp_path/'stage_success/grasp').glob('*.mp4')))==1
    failures=list((tmp_path/'failures').glob('*.mp4'));assert len(failures)==1
    data=json.loads(failures[0].with_suffix('.json').read_text())
    assert data['stage']=='lift' and 'unsafe_table_contact' in data['reasons']
    for video in tmp_path.rglob('*.mp4'):
        cap=cv2.VideoCapture(str(video));ok,_=cap.read();assert ok;cap.release()
    r.reset();r.reset();assert r.episodes==[2]
    r.capture(rgb,[3],[3],[True],[[]],7)
    assert len(list((tmp_path/'stage_success/place').glob('*.mp4')))==1
    r.close();assert not list(tmp_path.glob('.stage*'))


def test_unfinished_stage_preserved_as_interrupted_not_success(tmp_path):
    from isaac_vision_rl.utils.success_videos import StageEventVideos
    r=StageEventVideos(tmp_path,1,30,size=(64,48))
    r.capture(np.zeros((1,48,64,3),dtype=np.uint8),[0],[0],[False],[[]],1)
    r.close();r.close()
    reports=list((tmp_path/'failures').glob('*.json'))
    assert len(reports)==1
    assert json.loads(reports[0].read_text())['reasons']==['recording_interrupted']


def test_full_success_and_stage_recorders_coexist_when_enabled(tmp_path,monkeypatch):
    monkeypatch.setenv('SO101_STAGE_VIDEOS','1')
    r=SuccessVideos(tmp_path,1,30,size=(64,48));r.reset()
    rgb=np.zeros((1,48,64,3),dtype=np.uint8)
    r.capture(rgb,[True],[False],1)
    r.events.capture(rgb,[3],[3],[True],[[]],1)
    assert len(list(tmp_path.glob('success*.mp4')))==1
    assert len(list((tmp_path/'stage_success/place').glob('*.mp4')))==1
    r.close()
