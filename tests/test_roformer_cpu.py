import unittest

try:
    import torch
    from utils.config import Config
    from models.roformer.roformer import RoFormerDecoderLayer
except ModuleNotFoundError:
    torch = None


class RoformerCpuTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_decoder_layer_cpu_raises(self):
        config = Config(vocab_size=100, d_model=16, num_heads=4, num_layers=1)
        layer = RoFormerDecoderLayer(config)
        x = torch.randn(1, 1, 16)
        with self.assertRaises(RuntimeError):
            layer(x)


if __name__ == "__main__":
    unittest.main()
