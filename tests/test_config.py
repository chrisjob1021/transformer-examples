import unittest
import importlib.util
import os

# Load Config from utils/config.py without importing utils package
spec = importlib.util.spec_from_file_location(
    "config", os.path.join(os.path.dirname(__file__), "..", "utils", "config.py")
)
config_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_mod)
Config = config_mod.Config


class ConfigTest(unittest.TestCase):
    def test_config_defaults(self):
        cfg = Config(vocab_size=10, d_model=32, num_heads=4)
        self.assertEqual(cfg.per_head_dim, 8)
        self.assertEqual(cfg.ffn_dim, 32)
        self.assertEqual(cfg.vocab_size, 10)


if __name__ == "__main__":
    unittest.main()
