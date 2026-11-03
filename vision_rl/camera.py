"""Camera settings shared by sim and the real RealSense D455.

The real camera streams RGB at 424x240 @ 30 fps. The policy sees 224x224,
made by squashing the full 424x240 frame (no crop). Sim renders 424x240 too,
then goes through the same `to_policy_image`, so both sides look identical.
"""
import numpy as np

CAMERA_NAME = "realsense_d455_rgb"   # camera in scene.xml
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 424, 240, 30
OBS_SIZE = 224                       # policy input is OBS_SIZE x OBS_SIZE


def area_resize_matrix(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] matrix that averages input pixels into output pixels.

    Each output pixel covers n_in / n_out input pixels; partly covered input
    pixels count by the covered fraction (area averaging, like cv2.INTER_AREA).
    Rows sum to 1.
    """
    edges = np.arange(n_out + 1) * (n_in / n_out)
    lo, hi = edges[:-1, None], edges[1:, None]
    px = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, px + 1) - np.maximum(lo, px), 0.0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


_RESIZE_ROWS = area_resize_matrix(CAM_HEIGHT, OBS_SIZE)   # 240 -> 224
_RESIZE_COLS = area_resize_matrix(CAM_WIDTH, OBS_SIZE)    # 424 -> 224


def to_policy_image(rgb, xp=np, size=OBS_SIZE):
    """Squash uint8 [..., 240, 424, 3] camera frames to uint8 [..., size, size, 3].

    Pass `xp=jax.numpy` to run it inside jitted training code; the math is the
    same two matrix products, so sim and real stay pixel-identical. `size` must
    match the policy's training image size (saved in its checkpoint).
    """
    if size == OBS_SIZE:
        rows, cols = xp.asarray(_RESIZE_ROWS), xp.asarray(_RESIZE_COLS)
    else:
        rows = xp.asarray(area_resize_matrix(CAM_HEIGHT, size))
        cols = xp.asarray(area_resize_matrix(CAM_WIDTH, size))
    x = rgb.astype(xp.float32)
    x = xp.einsum("ih,...hwc->...iwc", rows, x)
    x = xp.einsum("jw,...iwc->...ijc", cols, x)
    return xp.clip(xp.round(x), 0, 255).astype(xp.uint8)
