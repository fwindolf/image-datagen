import unittest
import numpy as np

from image_datagen.loader.loader_text import AM2018TxtLoader, ORIG_SHAPE

data_paths = [
    'image_datagen/test/data/text/sim', 
    'image_datagen/test/data/text/exp'
]
txtfiles = [
    'image_datagen/test/data/text/exp/00000.xyz',  # exp before sim (alphabetically)
    'image_datagen/test/data/text/exp/00001.xyz', 
    'image_datagen/test/data/text/sim/1.xyz', 
    'image_datagen/test/data/text/sim/2.xyz'
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

    def test_get_image_exp_data(self):
        img = self.loader._get_image(txtfiles[0], shape=None, source='experiment')        

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)

    def test_get_image_exp_data_shape(self):
        img = self.loader._get_image(txtfiles[0], shape=self.input_shape, source='experiment')
        
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)        
        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)

    def test_get_image_sim_data(self):
        img = self.loader._get_image(txtfiles[2], shape=None, source='simulation')        

        # default dimensions when not providing shape
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)

    def test_get_image_sim_data_shape(self):
        img = self.loader._get_image(txtfiles[2], shape=self.input_shape, source='simulation')
        
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)        
        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)

    def test_get_label_exp_data(self):
        lbl = self.loader._get_label(txtfiles[0], shape=None, source='experiment')

        self.assertIsNotNone(lbl)
        # default dimensions when not providing shape
        self.assertEqual(lbl.shape[:2], (self.default_height, self.default_width))
        # not empty
        self.assertGreater(np.count_nonzero(lbl), 0)

    def test_get_label_exp_data_shape(self):
        lbl = self.loader._get_label(txtfiles[0], shape=self.input_shape, source='experiment')
        
        # not empty
        self.assertGreater(np.count_nonzero(lbl), 0)        
        # right dimensions
        self.assertEqual(lbl.shape[:2], self.input_shape)

    def test_get_label_sim_data(self):
        lbl = self.loader._get_label(txtfiles[2], shape=None, source='simulation')

        self.assertIsNotNone(lbl)
        # default dimensions when not providing shape
        self.assertEqual(lbl.shape[:2], (self.default_height, self.default_width))
        # not empty
        self.assertGreater(np.count_nonzero(lbl), 0)

    def test_get_label_sim_data_shape(self):
        lbl = self.loader._get_label(txtfiles[2], shape=self.input_shape, source='simulation')
        
        # not empty
        self.assertGreater(np.count_nonzero(lbl), 0)        
        # right dimensions
        self.assertEqual(lbl.shape[:2], self.input_shape)

    