import unittest
import torch
from PIL import Image

from data import emnist_preprocessing_metadata
from lenet5 import LeNet5, LeNetLarge, LeNetMax, ConfigurableLeNet, architecture_metadata, make_config, model_from_architecture
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

    def test_configurable_max_model_and_manifest_round_trip(self):
        config = make_config("max", activation="gelu", pooling="max", channels=(40, 120, 480), hidden_dim=640)
        model = ConfigurableLeNet(62, config)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 32, 32)).shape), (2, 62))
        meta = architecture_metadata(62, config=config)
        self.assertEqual(meta["config"]["activation"], "gelu")
        self.assertEqual([layer["op"] for layer in meta["layers"]].count("max_pool2d"), 2)
        restored = model_from_architecture(meta, 62)
        self.assertEqual(restored.config, config)
        self.assertEqual(tuple(LeNetMax(62)(torch.zeros(1, 1, 32, 32)).shape), (1, 62))

    def test_regularization_is_exported_without_changing_original_lenet5(self):
        original = LeNet5(10)
        self.assertEqual(list(original.state_dict()), [
            "features.0.weight", "features.0.bias", "features.3.weight", "features.3.bias",
            "features.6.weight", "features.6.bias", "classifier.1.weight", "classifier.1.bias",
            "classifier.3.weight", "classifier.3.bias",
        ])
        config = make_config("max", batch_norm=True, dropout=0.2)
        meta = architecture_metadata(62, config=config)
        self.assertTrue(any(layer["op"] == "batch_norm2d" for layer in meta["layers"]))
        self.assertTrue(any(layer["op"] == "dropout" for layer in meta["layers"]))
        self.assertEqual(model_from_architecture(meta, 62).config, config)

    def test_web_input_skips_emnist_storage_orientation_correction(self):
        image = Image.new("L", (2, 3))
        image.putpixel((0, 0), 255)
        _, preview = preprocess(image, emnist_preprocessing_metadata())
        # Pixel stays in the upper-left after resize/pad; it is not rotated for web input.
        self.assertGreater(preview.getpixel((2, 2)), 0)
        operations = emnist_preprocessing_metadata()["operations"]
        self.assertEqual([operation["op"] for operation in operations if operation.get("apply_to") == "dataset"], ["transpose"])

    def test_web_input_translates_legacy_emnist_orientation(self):
        metadata = emnist_preprocessing_metadata()
        metadata["operations"] = [
            {"op": "transpose", "reason": "correct EMNIST storage orientation"},
            {"op": "flip_horizontal", "reason": "correct EMNIST storage orientation"},
            *metadata["operations"][1:],
        ]
        image = Image.new("L", (2, 3)); image.putpixel((0, 0), 255)
        _, preview = preprocess(image, metadata)
        # A legacy model expects the mirrored, but not quarter-turned, drawing.
        self.assertGreater(preview.getpixel((29, 2)), 0)


if __name__ == "__main__": unittest.main()
