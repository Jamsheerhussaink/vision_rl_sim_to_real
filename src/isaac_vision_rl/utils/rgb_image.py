"""Shared RGB preprocessing for simulation and real-camera input."""
import torch.nn.functional as F


def actor_rgb(image, size=128):
    """BHWC uint8 RGB to BCHW floats, with aspect-preserving padding."""
    rgb = image[..., :3].permute(0, 3, 1, 2).float() / 255.0
    height, width = rgb.shape[-2:]
    scale = size / max(height, width)
    h, w = round(height * scale), round(width * scale)
    rgb = F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False)
    top, left = (size - h) // 2, (size - w) // 2
    return F.pad(rgb, (left, size - w - left, top, size - h - top))
