"""Structural checks for every generator, independent of estimator details."""
import sys
from pathlib import Path
import unittest

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import DGPs


class GeneratorKeys(unittest.TestCase):
    def test_each_structural_noise_has_a_fresh_key(self):
        for name in dir(DGPs):
            if not name.startswith('DGP_'):
                continue
            generator = getattr(DGPs, name)
            for intervention in [False, True]:
                with self.subTest(generator=name, intervention=intervention):
                    with jax.debug_key_reuse(True):
                        data = generator(20, seed=13, dim=3, gt=intervention)
                    for array in data:
                        self.assertEqual(array.shape[0], 20)
                        self.assertTrue(np.all(np.isfinite(array)))


if __name__ == '__main__':
    unittest.main()
