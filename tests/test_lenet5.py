import unittest
import torch

from lenet5 import LeNet5, architecture_metadata


class LeNet5Tests(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(tuple(LeNet5(26)(torch.zeros(3, 1, 32, 32)).shape), (3, 26))

    def test_manifest_describes_all_layers(self):
        meta = architecture_metadata(10)
        self.assertEqual(meta["input"]["shape"], [1, 1, 32, 32])
        self.assertEqual(meta["layers"][-1]["out_features"], 10)
        self.assertEqual([x["op"] for x in meta["layers"]].count("tanh"), 4)


if __name__ == "__main__": unittest.main()
