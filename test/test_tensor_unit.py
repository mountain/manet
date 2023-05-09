import unittest

import torch as th

from manet.mac import MLP, MacTensorUnit


class TensorUnitCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mlp = MLP(3, [3], mac_unit=MacTensorUnit)

    def test_sanity(self):
        mlp = MLP(3, [3], mac_unit=MacTensorUnit)
        result = mlp(th.rand(4, 3))
        dims = result.size()
        self.assertEqual(dims[0], 4)
        self.assertEqual(dims[1], 3)

    def test_dimension(self):
        mlp = MLP(3, [5], mac_unit=MacTensorUnit)
        result = mlp(th.rand(4, 3))
        dims = result.size()
        self.assertEqual(dims[0], 4)
        self.assertEqual(dims[1], 5)


if __name__ == '__main__':
    unittest.main()
