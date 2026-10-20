"""Simulation-only manual inspection settings; never applied to training/evaluation."""

def configure_manual_inspection(cfg):
    from isaaclab.managers import TerminationTermCfg
    cfg.scene.num_envs=1
    for name,value in vars(cfg.terminations).items():
        if isinstance(value,TerminationTermCfg):setattr(cfg.terminations,name,None)
    cfg.manual_no_termination=True
