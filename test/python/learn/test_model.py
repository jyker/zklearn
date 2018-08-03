import unittest
from zk.learn.base import CONFIG
from zk.learn import ImageBasedMalNet, ImageBasedMalNet_default_config


class TestImageBasedMalNet(unittest.TestCase):
    def get_config(self):
        return ImageBasedMalNet_default_config()

    def test_ImageBasedMalNet(self):
        CK = CONFIG.KEYS

        cfg = self.get_config()
        cfg[CK.INPUT_SHAPE] = (1600, 64, 4)
        cfg[CK.KERNEL_SIZE] = (8, 2)
        cfg[CK.STRIDES] = (2, 1)
        cfg[CK.POOL_STRIDE] = (2, 1)
        cfg[CK.BATCHSIZE] = 64

        net = ImageBasedMalNet(cfg)
        print(net.summary())


if __name__ == "__main__":
    unittest.main()