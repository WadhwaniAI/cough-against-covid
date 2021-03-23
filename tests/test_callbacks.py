"""Tests cac.callbacks.ModelCheckpoint"""
import unittest
from cac.callbacks import ModelCheckpoint


class ModelCheckpointTestCase(unittest.TestCase):
    """Class to run tests on ModelCheckpoint"""
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_model_ckpt(self):
        # define a checkpoint object that will save checkpoint as `/tmp/`
        # when `auc-roc` improves (increases) after every `1` epochs
        ckpt = ModelCheckpoint(
            ckpt_dir='/tmp/', period=1, monitor='auc-roc', monitor_mode='max'
        )

        # suppose the metric for 1st epoch gets updated to 0.6
        metric_dict = {
            'auc-roc': 0.6
        }
        save_status = ckpt.update_best_metric(
            epoch_counter=1, epoch_metric_dict=metric_dict
        )

        self.assertTrue(save_status['save'])
        self.assertEqual(save_status['path'], '/tmp/1_ckpt.pth.tar')
        self.assertIn("auc-roc improved from -inf to 0.6", save_status['info'])


if __name__ == "__main__":
    unittest.main()
