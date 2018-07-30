import unittest
import numpy as np

from loader.loader_base import LoaderBase

class TestLoaderBaseResize(unittest.TestCase):
    def setUp(self):
        num_classes = 2
        crop_scale = None
        self.loader_base = LoaderBase(num_classes, crop_scale)

    def test_dimensions_height(self):
        in_shape = (10, 10, 2)
        out_shape = (20, 10)
        img = np.ones(in_shape)
        
        img_resized = self.loader_base._resize(img, out_shape)
        # Same dimension
        self.assertEqual(img_resized.shape[:2], out_shape)

        # Same channels as input
        self.assertEqual(img_resized.shape[2], in_shape[2])
    
    def test_dimensions_width(self):
        in_shape = (10, 10, 2)
        out_shape = (10, 20)
        img = np.ones(in_shape)
        
        img_resized = self.loader_base._resize(img, out_shape)
        # Same dimension
        self.assertEqual(img_resized.shape[:2], out_shape)

        # Same channels as input
        self.assertEqual(img_resized.shape[2], in_shape[2])

    def test_restore_channels(self):
        in_shape = (10, 10, 1)
        out_shape = (10, 30)
        img = np.ones(in_shape)
        
        img_resized = self.loader_base._resize(img, out_shape)
        # Same dimension
        self.assertEqual(img_resized.shape[:2], out_shape)

        # Lost channel gets restored
        self.assertEqual(len(img_resized.shape), len(in_shape))

class TestLoaderBaseCrop(unittest.TestCase):
    def setUp(self):
        num_classes = 2
        crop_scale = None
        self.loader_base = LoaderBase(num_classes, crop_scale)

    def test_crop_works(self):
        im_shape = (90, 100, 1)
        lb_shape = (70, 80, 3)
        
        im_shape_crop = (20, 30)
        lb_shape_crop = (40, 50)
        
        im = np.ones(im_shape)
        lb = np.ones(lb_shape)

        im_crop, lb_crop = self.loader_base._get_crop(im, lb, im_shape_crop, lb_shape_crop)

        # im crop dimensions
        self.assertEqual(im_crop.shape[:2], im_shape_crop[:2])
        # im crop channels
        self.assertEqual(im_crop.shape[2], im_shape[2])

        # lb crop dimensions
        self.assertEqual(lb_crop.shape[:2], lb_shape_crop[:2])
        # im crop channels
        self.assertEqual(lb_crop.shape[2], lb_shape[2])
    
    def test_crop_seed(self):
        im_shape = (100, 100, 1)
        im_crop_shape = (30, 30)
        img = np.random.randint(10, size=im_shape, dtype=np.uint8)

        seed = 1
        im_crop, _ = self.loader_base._get_crop(img, None, im_crop_shape, None, seed=seed)
        im_crop_diff, _ = self.loader_base._get_crop(img, None, im_crop_shape, None)
        im_crop_same, _ = self.loader_base._get_crop(img, None, im_crop_shape, None, seed=seed)

        # different seed -> different result
        self.assertFalse(np.array_equal(im_crop, im_crop_diff)) 
        # same seed ->  same result
        self.assertTrue(np.array_equal(im_crop,im_crop_same))
    
    def test_crop_same_content(self):
        im_shape = (100, 80, 1)
        im_crop_shape = (40, 40)

        img = np.random.randint(10, size=im_shape, dtype=np.uint8)

        im_crop1, im_crop2 = self.loader_base._get_crop(img, img, im_crop_shape, im_crop_shape)

        self.assertTrue(np.array_equal(im_crop1, im_crop2))

class TestLoaderBaseCropScale(unittest.TestCase):
    pass # TODO: Add tests for crop scale
