import numpy as np
from PIL import Image


def create_rgbd_composite(rgb: np.ndarray, depth: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """
    Creates a side-by-side composite of RGB and colormapped Depth image.
    """
    # Normalize depth for visualization (0.3m to 1.0m workspace range)
    d_norm = np.clip((depth - 0.30) / (0.90 - 0.30), 0.0, 1.0)
    
    # Simple turbo-like false color mapping without heavy dependencies
    r = np.clip(1.5 - np.abs(4.0 * d_norm - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * d_norm - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * d_norm - 1.0), 0, 1)
    depth_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # Set invalid / 0 depth to black
    depth_rgb[depth <= 0.05] = 0

    composite = np.concatenate([rgb, depth_rgb], axis=1)
    return composite


def save_rgbd_composite(rgb: np.ndarray, depth: np.ndarray, output_path: str):
    """Saves RGB-D composite image to disk."""
    comp = create_rgbd_composite(rgb, depth)
    Image.fromarray(comp).save(output_path)
