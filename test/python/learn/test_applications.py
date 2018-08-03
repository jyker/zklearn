import unittest
from zk.learn.base import CONFIG
from zk.learn.applications import VGG16, VGG16_default_config

class TestImageBasedMalNet(unittest.TestCase):
    def get_config(self):
        return VGG16_default_config()

    def test_ImageBasedMalNet(self):
        CK = CONFIG.KEYS

        cfg = self.get_config()
        cfg[CK.INPUT_SHAPE] = (6400, 64, 1)

        net = VGG16(cfg)
        print(net.summary())


if __name__ == "__main__":
    unittest.main()