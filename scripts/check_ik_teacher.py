"""CPU-only IK preflight; no simulator, GPU allocation or motor commands."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.update(SO101_POLICY='rgb',SO101_CURRICULUM='place',SO101_REWARD_PROFILE='four-stage',SO101_RGB_FPS='30')
from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickplaceEnvCfg
from isaac_vision_rl.utils.ik_teacher import offline_check
if __name__=='__main__':
    result=offline_check(PickplaceEnvCfg())
    path=Path('outputs/four_stage/ik_teacher_preflight.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
