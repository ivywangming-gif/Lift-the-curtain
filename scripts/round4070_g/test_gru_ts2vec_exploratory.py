"""Contract tests for the isolated GRU/TS2Vec study."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gru_ts2vec_exploratory as study


class ExploratoryContracts(unittest.TestCase):
    def test_timestamp_masking_does_not_reorder_or_change_command_columns(self):
        x = np.arange(2 * 12 * 5, dtype=np.float32).reshape(2, 12, 5)
        lengths = np.asarray([12, 10])
        viewed = study.augmentation_view(x, lengths, 40700, np.asarray([], dtype=np.int64))
        unmasked = np.any(viewed[0] != 0.0, axis=1)
        np.testing.assert_array_equal(viewed[0, unmasked, 3:], x[0, unmasked, 3:])
        np.testing.assert_array_equal(viewed[0, unmasked, :3], x[0, unmasked, :3])
        np.testing.assert_array_equal(viewed[1, 10:, :], x[1, 10:, :])
        self.assertEqual(viewed.shape, x.shape)

    def test_hierarchical_loss_is_finite_and_has_gradients(self):
        torch.manual_seed(1)
        z1 = torch.randn(3, 16, 8, requires_grad=True)
        z2 = torch.randn(3, 16, 8, requires_grad=True)
        lengths = torch.tensor([16, 15, 13])
        loss = study.hierarchical_contrastive_loss(z1, z2, lengths)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z1.grad)
        self.assertTrue(torch.isfinite(z1.grad).all())

    def test_matched_loss_never_invents_pairs(self):
        z = torch.randn(4, 8)
        loss, positive_n, negative_n = study.matched_supcon(z, [])
        self.assertEqual((positive_n, negative_n), (0, 0))
        self.assertEqual(float(loss), 0.0)

    def test_offline_gate_is_fail_closed_and_pause_hashes_are_present(self):
        gate = study.ensure_offline_gate()
        self.assertTrue(gate["global_pause"])
        self.assertTrue(gate["g1_pause"])
        self.assertFalse(gate["training_window_allow_training"])
        self.assertTrue(len(gate["pause_sha256"]) == 64)
        self.assertTrue(len(gate["pause_g1_sha256"]) == 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
