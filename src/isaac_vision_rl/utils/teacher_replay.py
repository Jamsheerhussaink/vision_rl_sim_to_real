"""Small, opt-in demonstration rehearsal batches without loading the full RGB dataset."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict


class TeacherReplay:
    def __init__(self, checkpoint, batch_size=32):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        directories = saved.get('infos', {}).get('train_demos', [])
        if not directories:
            raise ValueError('Teacher replay requires a BC checkpoint with train_demos metadata')
        self.chunks = []
        for directory in directories:
            directory = Path(directory)
            meta = json.loads((directory / 'metadata.json').read_text())
            if not meta.get('eligible_for_cloning'):
                raise ValueError(f'Ineligible teacher demonstration: {directory}')
            self.chunks.extend(sorted(directory.glob('chunk_*.npz')))
        if not self.chunks:
            raise ValueError('No teacher demonstration chunks found')
        self.batch_size = batch_size

    def sample(self, device):
        path = self.chunks[int(torch.randint(len(self.chunks), ()).item())]
        with np.load(path) as data:
            index = torch.randint(len(data['action']), (self.batch_size,)).numpy()
            policy, image, action = [torch.as_tensor(data[key][index], device=device)
                                     for key in ('policy', 'image', 'action')]
        return TensorDict({'policy': policy, 'image': image}, batch_size=[self.batch_size]), action


def rehearsal_loss(predicted, target):
    return torch.nn.functional.smooth_l1_loss(predicted, target, beta=0.05)
