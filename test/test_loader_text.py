import unittest
import numpy as np

from universal_datagen.loader.loader_text import AM2018TxtLoader, ORIG_SHAPE

data_paths = [
    'universal_datagen/test/data/text/sim', 
    'universal_datagen/test/data/text/exp'
]
txtfiles = [
    'universal_datagen/test/data/text/exp/00000.xyz',  # exp before sim (alphabetically)
    'universal_datagen/test/data/text/exp/00001.xyz', 
    'universal_datagen/test/data/text/sim/1.xyz', 
    'universal_datagen/test/data/text/sim/2.xyz'
]
num_classes = 4 # with unknown

def flatten(array):
    out = []
    for sl in array:
        for v in sl:
            out.append(v)
    return out

class TestTxtLoader(unittest.TestCase):
    def setUp(self):
        self.loader = AM2018TxtLoader(num_classes, None)
        self.default_height = ORIG_SHAPE[0]
        self.default_width = ORIG_SHAPE[1]

        self.input_shape = (200, 100)
        self.output_shape = (150, 250)
    
    def test_discover_data(self):
        data = self.loader.discover_data(data_paths)
        # Found the 2 paths
        self.assertEqual(len(data), 2)
        # Found 2 files per path
        for p in range(len(data)):
            self.assertEqual(len(data[p]), 2)
        
        # See if ordering/files match
        data_flat = flatten(data)
        self.assertListEqual(data_flat, txtfiles)

    def test_discover_data_max_data(self):
        data = self.loader.discover_data(data_paths, num_data=1)

        # Found the 2 paths
        self.assertEqual(len(data), 2)
        # Only kept 1 file per path
        for p in range(len(data)):
            self.assertEqual(len(data[p]), 1)

        data_flat = flatten(data)
        self.assertListEqual(data_flat, txtfiles[0:-1:2])

    def test_get_labeled_exp_data(self):
        img, lbl = self.loader._get_labeled(txtfiles[0], source='experiment')
        self.assertIsNotNone(lbl) # labeled -> label exists

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(lbl.shape[:2], (self.default_height, self.default_width))

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)
    
    def test_get_labeled_exp_data_shape(self):
        img, lbl = self.loader._get_labeled(txtfiles[0], source='experiment', input_shape=self.input_shape, output_shape=self.output_shape)
        self.assertIsNotNone(lbl) # labeled -> label exists

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)

        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)
        self.assertEqual(lbl.shape[:2], self.output_shape)

    def test_get_unlabeled_exp_data(self):
        img, lbl = self.loader._get_unlabeled(txtfiles[0], source='experiment')
        self.assertIsNone(lbl) # unlabeled
        
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))

    def test_get_unlabeled_exp_data_shape(self):
        img, lbl = self.loader._get_unlabeled(txtfiles[0], source='experiment', input_shape=self.input_shape)
        self.assertIsNone(lbl) # unlabeled
        
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        
        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], self.input_shape)

    def test_get_labeled_sim_data(self):
        img, lbl = self.loader._get_labeled(txtfiles[2], source='simulation')
        self.assertIsNotNone(lbl) # labeled -> label exists

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(lbl.shape[:2], (self.default_height, self.default_width))

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)
    
    def test_get_labeled_sim_data_shape(self):
        img, lbl = self.loader._get_labeled(txtfiles[2], source='simulation', input_shape=self.input_shape, output_shape=self.output_shape)
        self.assertIsNotNone(lbl) # labeled -> label exists

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)

        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)
        self.assertEqual(lbl.shape[:2], self.output_shape)

    def test_get_unlabeled_sim_data(self):
        img, lbl = self.loader._get_unlabeled(txtfiles[2], source='simulation')
        self.assertIsNone(lbl) # unlabeled

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))

    def test_get_unlabeled_sim_data_shape(self):
        img, lbl = self.loader._get_unlabeled(txtfiles[2], source='simulation', input_shape=self.input_shape)
        self.assertIsNone(lbl) # unlabeled

        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)

    def test_get_labeled_auto_data(self):
        e_img, e_lbl = self.loader._get_labeled(txtfiles[1])
        s_img, s_lbl = self.loader._get_labeled(txtfiles[3])
        
        self.assertIsNotNone(e_lbl) # labeled -> label exists
        self.assertIsNotNone(s_lbl) # labeled -> label exists

        # default dimensions when not providing shape
        self.assertEqual(e_img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(e_lbl.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(s_img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(s_lbl.shape[:2], (self.default_height, self.default_width))        

        # not empty
        self.assertGreater(np.count_nonzero(e_img), 0)
        self.assertGreater(np.count_nonzero(e_lbl), 0)
        self.assertGreater(np.count_nonzero(s_img), 0)
        self.assertGreater(np.count_nonzero(s_lbl), 0)
    
    def test_get_labeled_auto_data_shape(self):
        e_img, e_lbl = self.loader._get_labeled(txtfiles[1], input_shape=self.input_shape, output_shape=self.output_shape)
        s_img, s_lbl = self.loader._get_labeled(txtfiles[3], input_shape=self.input_shape, output_shape=self.output_shape)
        
        self.assertIsNotNone(e_lbl) # labeled -> label exists
        self.assertIsNotNone(s_lbl) # labeled -> label exists

        # not empty
        self.assertGreater(np.count_nonzero(e_img), 0)
        self.assertGreater(np.count_nonzero(e_lbl), 0)
        self.assertGreater(np.count_nonzero(s_img), 0)
        self.assertGreater(np.count_nonzero(s_lbl), 0)
        
        # right dimensions
        self.assertEqual(e_img.shape[:2], self.input_shape)
        self.assertEqual(e_lbl.shape[:2], self.output_shape)
        self.assertEqual(s_img.shape[:2], self.input_shape)
        self.assertEqual(s_lbl.shape[:2], self.output_shape)

    def test_get_unlabeled_auto_data(self):
        e_img, e_lbl = self.loader._get_unlabeled(txtfiles[1])
        s_img, s_lbl = self.loader._get_unlabeled(txtfiles[3])

        self.assertIsNone(e_lbl) # unlabeled
        self.assertIsNone(s_lbl) # unlabeled

        # default dimensions when not providing shape
        self.assertEqual(e_img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(s_img.shape[:2], (self.default_height, self.default_width))

    def test_get_unlabeled_auto_data_shape(self):
        e_img, e_lbl = self.loader._get_unlabeled(txtfiles[1], input_shape=self.input_shape)
        s_img, s_lbl = self.loader._get_unlabeled(txtfiles[3], input_shape=self.input_shape)
        
        self.assertIsNone(e_lbl) # unlabeled
        self.assertIsNone(s_lbl) # unlabeled

        # right dimensions
        self.assertEqual(e_img.shape[:2], self.input_shape)
        self.assertEqual(s_img.shape[:2], self.input_shape)
