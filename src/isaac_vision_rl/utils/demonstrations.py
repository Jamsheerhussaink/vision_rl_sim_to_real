"""Bounded-memory, aligned actor observations/actions for imitation learning."""
import json
from pathlib import Path
import numpy as np


class Demonstration:
    def __init__(self,directory,chunk_size=64):
        self.path=Path(directory);self.path.mkdir(parents=True,exist_ok=False)
        self.chunk_size=chunk_size;self.rows=[];self.chunks=0;self.steps=0
        import os
        self.metadata=dict(reward_style=os.environ.get("SO101_REWARD_STYLE","progress"),
                           control_hz=int(os.environ.get("SO101_RGB_FPS","30")),schema='so101_rgb_ik_demo_v1',eligible_for_cloning=False,
                           status='incomplete',alignment='observation_t -> action_t -> outcome_t_plus_1',
                           actor_inputs=['policy','image'],privileged_teacher_inputs_not_actor_inputs=True)
        self._write()

    def _write(self):
        (self.path/'metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    def add(self,policy,image,action,reward,terminated,phase):
        row=dict(policy=np.array(policy,dtype=np.float32,copy=True),
                 image=np.array(image,dtype=np.float32,copy=True),
                 action=np.array(action,dtype=np.float32,copy=True),reward=np.float32(reward),
                 terminated=np.bool_(terminated),phase=np.asarray(phase),step=np.int64(self.steps))
        if not all(np.isfinite(row[k]).all() for k in ('policy','image','action')):raise ValueError('Non-finite demonstration')
        if np.abs(row['action']).max()>1.00001:raise ValueError('Action outside normalized policy limits')
        self.rows.append(row);self.steps+=1
        if len(self.rows)>=self.chunk_size:self.flush()

    def flush(self):
        if not self.rows:return
        np.savez_compressed(self.path/f'chunk_{self.chunks:05d}.npz',
                            **{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
        self.rows=[];self.chunks+=1

    def finish(self,success=False,reason='interrupted',checkpoint_schema='four_stage_v2_bounded'):
        self.flush()
        self.metadata.update(status='complete',eligible_for_cloning=bool(success),
                             success=bool(success),reason=reason,steps=self.steps,chunks=self.chunks,
                             compatible_policy_schema=checkpoint_schema)
        self._write()


def find_eligible_demonstrations(root_dir='outputs/four_stage'):
    """Search root_dir for complete demonstration directories marked eligible for cloning."""
    root = Path(root_dir)
    eligible = []
    if not root.exists():
        return eligible
    for meta_path in sorted(root.glob('*/metadata.json')):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get('eligible_for_cloning') is True and meta.get('status') == 'complete':
                eligible.append(meta_path.parent)
        except Exception:
            continue
    return eligible


class DemonstrationDataset:
    """PyTorch-compatible dataset for imitation learning from eligible SO-101 demonstrations."""

    def __init__(self, demo_dirs, expected_schema='so101_rgb_ik_demo_v1', expected_policy_schema='four_stage_v2_bounded'):
        import torch
        if isinstance(demo_dirs, (str, Path)):
            demo_dirs = [demo_dirs]
        self.demo_dirs = [Path(d) for d in demo_dirs]
        if not self.demo_dirs:
            raise ValueError("No demonstration directories provided")

        policies = []
        images = []
        actions = []
        self.total_steps = 0

        for d in self.demo_dirs:
            meta_file = d / 'metadata.json'
            if not meta_file.exists():
                raise FileNotFoundError(f"Missing metadata.json in {d}")
            meta = json.loads(meta_file.read_text())
            if not meta.get('eligible_for_cloning', False):
                raise ValueError(f"Demonstration in {d} is not marked eligible_for_cloning")
            if meta.get('schema') != expected_schema:
                raise ValueError(f"Demonstration in {d} schema {meta.get('schema')!r} != expected {expected_schema!r}")
            if meta.get('compatible_policy_schema') != expected_policy_schema:
                raise ValueError(
                    f"Demonstration in {d} policy schema {meta.get('compatible_policy_schema')!r} != expected {expected_policy_schema!r}"
                )

            chunks = sorted(d.glob('chunk_*.npz'))
            if not chunks:
                raise FileNotFoundError(f"No chunk_*.npz files found in {d}")

            for chunk_path in chunks:
                with np.load(chunk_path) as chunk:
                    pol = chunk['policy']
                    img = chunk['image']
                    act = chunk['action']

                    if not (np.isfinite(pol).all() and np.isfinite(img).all() and np.isfinite(act).all()):
                        raise ValueError(f"Non-finite values found in {chunk_path}")
                    if pol.ndim != 2 or pol.shape[1] != 29:
                        raise ValueError(f"Invalid policy shape {pol.shape} in {chunk_path}, expected (N, 29)")
                    if img.ndim != 4 or img.shape[1:] != (3, 128, 128):
                        raise ValueError(f"Invalid image shape {img.shape} in {chunk_path}, expected (N, 3, 128, 128)")
                    if act.ndim != 2 or act.shape[1] != 6:
                        raise ValueError(f"Invalid action shape {act.shape} in {chunk_path}, expected (N, 6)")
                    if np.abs(act).max() > 1.00001:
                        raise ValueError(f"Action out of [-1, 1] bounds in {chunk_path}")

                    policies.append(pol.astype(np.float32))
                    images.append(img.astype(np.float32))
                    actions.append(np.clip(act, -1.0, 1.0).astype(np.float32))
                    self.total_steps += len(pol)

        if not policies:
            raise ValueError("No demonstration samples loaded")

        self.policy_array = np.concatenate(policies, axis=0)
        self.image_array = np.concatenate(images, axis=0)
        self.action_array = np.concatenate(actions, axis=0)

    def __len__(self):
        return len(self.policy_array)

    def __getitem__(self, idx):
        import torch
        return {
            'policy': torch.from_numpy(self.policy_array[idx]),
            'image': torch.from_numpy(self.image_array[idx]),
            'action': torch.from_numpy(self.action_array[idx]),
        }

