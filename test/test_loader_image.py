import unittest
import numpy as np

from loader.loader_image import AM2018ImageLoader, ORIG_SHAPE

data_paths = ['test/data/image/', ]
files = ['test/data/image/%s/00000.%s', 'test/data/image/%s/00001.%s',]
num_classes = 4

class TestImageLoader(unittest.TestCase):
    def setUp(self):
        self.loader = AM2018ImageLoader()
        self.default_height = ORIG_SHAPE[0]
        self.default_width = ORIG_SHAPE[1]

        self.input_shape = (80, 70)
        self.output_shape = (60, 90)

    def test_discover_only_pairs_of_data(self):
        data = self.loader.discover_data(data_paths)

        # Found one path
        self.assertEqual(len(data), 1)
        # 00003 and 00002 skipped as they dont have both image and target
        self.assertListEqual(data[0], files)

    def test_discover_data_max_data(self):
        num_data = 1
        data = self.loader.discover_data(data_paths, num_data=num_data)
        # Found one path
        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0]), num_data)
        self.assertEqual(data[0][0], files[0])

    def test_get_labeled_wrong_source(self):
        with  self.assertRaises(AttributeError):
            self.loader._get_labeled(files[0], source='wrong')

    def test_get_labeled_data(self):
        img, lbl = self.loader._get_labeled(files[0])

        # default dimensions
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))
        self.assertEqual(lbl.shape[:2], (self.default_height, self.default_width))

        # right datatype
        self.assertIn(img.dtype, [np.float, np.uint8])
        self.assertIn(lbl.dtype, [np.float, np.uint8])

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)
    
    def test_get_labeled_data_shape(self):
        img, lbl = self.loader._get_labeled(files[0], input_shape=self.input_shape, output_shape=self.output_shape)

        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)
        self.assertEqual(lbl.shape[:2], self.output_shape)

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)
        self.assertGreater(np.count_nonzero(lbl), 0)

    def test_get_unlabeled_data(self):
        img, lbl = self.loader._get_unlabeled(files[1])

        # no label
        self.assertIsNone(lbl)

        # default dimensions
        self.assertEqual(img.shape[:2], (self.default_height, self.default_width))

        # right datatype
        self.assertIn(img.dtype, [np.float, np.uint8])
        
        # not empty
        self.assertGreater(np.count_nonzero(img), 0)

    def test_get_unlabeled_data_shape(self):
        img, lbl = self.loader._get_unlabeled(files[1], input_shape=self.input_shape)

        # no label
        self.assertIsNone(lbl)

        # right dimensions
        self.assertEqual(img.shape[:2], self.input_shape)

        # not empty
        self.assertGreater(np.count_nonzero(img), 0)