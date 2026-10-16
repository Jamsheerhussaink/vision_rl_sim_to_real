import unittest
import torch
from isaac_vision_rl.utils.rgb_image import actor_rgb


class RGBImageTest(unittest.TestCase):
    def test_letterbox_and_channels(self):
        image = torch.zeros(2, 240, 424, 4, dtype=torch.uint8)
        image[..., 0] = 255
        image[..., 3] = 255
        result = actor_rgb(image)
        self.assertEqual(tuple(result.shape), (2, 3, 128, 128))
        self.assertTrue(torch.all(result[:, 0, 28:100] == 1))
        self.assertTrue(torch.all(result[:, :, :28] == 0))
        self.assertTrue(torch.all(result[:, :, 100:] == 0))
        self.assertTrue(torch.all(result[:, 1:] == 0))
