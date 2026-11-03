"""The 424x240 -> 224x224 squash is the same in numpy and JAX and keeps colours."""
import jax.numpy as jnp
import numpy as np

from vision_rl.camera import CAM_HEIGHT, CAM_WIDTH, OBS_SIZE, area_resize_matrix, to_policy_image


def test_resize_matrix_rows_sum_to_one():
    for n_in in (CAM_HEIGHT, CAM_WIDTH):
        m = area_resize_matrix(n_in, OBS_SIZE)
        assert m.shape == (OBS_SIZE, n_in)
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-6)


def test_shape_and_flat_colour():
    img = np.full((CAM_HEIGHT, CAM_WIDTH, 3), (200, 90, 30), np.uint8)
    out = to_policy_image(img)
    assert out.shape == (OBS_SIZE, OBS_SIZE, 3) and out.dtype == np.uint8
    assert (out == img[0, 0]).all()


def test_numpy_matches_jax_with_batch():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (2, CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    a = to_policy_image(imgs)
    b = np.asarray(to_policy_image(jnp.asarray(imgs), xp=jnp))
    assert a.shape == (2, OBS_SIZE, OBS_SIZE, 3)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1   # float rounding only
