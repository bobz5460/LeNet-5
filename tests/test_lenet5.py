import unittest
import torch
from PIL import Image

from data import emnist_preprocessing_metadata
from lenet5 import LeNet5, LeNetLarge, architecture_metadata, model_from_architecture
from webui_preprocess import preprocess


class LeNet5Tests(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(tuple(LeNet5(26)(torch.zeros(3, 1, 32, 32)).shape), (3, 26))

    def test_manifest_describes_all_layers(self):
        meta = architecture_metadata(10)
        self.assertEqual(meta["input"]["shape"], [1, 1, 32, 32])
        self.assertEqual(meta["layers"][-1]["out_features"], 10)
        self.assertEqual([x["op"] for x in meta["layers"]].count("tanh"), 4)

    def test_large_lenet_preserves_lenet_methods(self):
        self.assertEqual(tuple(LeNetLarge(62)(torch.zeros(3, 1, 32, 32)).shape), (3, 62))
        meta = architecture_metadata(62, "large")
        self.assertEqual([x["op"] for x in meta["layers"]].count("tanh"), 4)
        self.assertEqual([x["op"] for x in meta["layers"]].count("avg_pool2d"), 2)
        self.assertIsInstance(model_from_architecture(meta["id"], 62), LeNetLarge)

    def test_web_input_skips_emnist_storage_orientation_correction(self):
        image = Image.new("L", (2, 3))
        image.putpixel((0, 0), 255)
        _, preview = preprocess(image, emnist_preprocessing_metadata())
        # Pixel stays in the upper-left after resize/pad; it is not rotated for web input.
        self.assertGreater(preview.getpixel((2, 2)), 0)


if __name__ == "__main__": unittest.main()
