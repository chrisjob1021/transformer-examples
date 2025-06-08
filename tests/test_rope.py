import unittest

try:
    import torch
    from utils.rope import apply_rope
except ModuleNotFoundError:
    torch = None


class RopeTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_apply_rope_shape(self):
        x = torch.randn(2, 2, 3, 4)
        out = apply_rope(x, past_seq_len=0, visualize=False)
        self.assertEqual(out.shape, x.shape)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_apply_rope_rotation_simple(self):
        x = torch.zeros(1, 1, 1, 4)
        x[..., 0] = 1.0
        x[..., 1] = 0.0
        x[..., 2] = 0.0
        x[..., 3] = 1.0
        out = apply_rope(x, past_seq_len=0, freq=1.0, visualize=False)
        torch.testing.assert_close(out, x)


if __name__ == "__main__":
    unittest.main()
