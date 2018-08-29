import unittest
import numpy as np

from image_datagen.loader.loader_base import LoaderBase
from image_datagen.generator.generator_base import GeneratorBase

data_paths = []

default_height = 400
default_width = 300
n_classes = 4
n_channels = 1
stack_size = 3

class MockLoader(LoaderBase):
    def __init__(self, n_channels):
        super().__init__(n_classes, None)
        self.c = n_channels

    def _get_image(self, file, shape, source='auto'):
        if shape is None:
            shape = (default_height, default_width, self.c)
        elif len(shape) == 2:
            shape = (*shape, self.c)

        return np.random.randint(2, size=shape, dtype=np.uint8)

    def _get_label(self, file, shape, source='auto'):
        if shape is None:
            shape = (default_height, default_width, n_classes)
        elif len(shape) == 2:
            shape = (*shape, n_classes)

        return np.random.randint(n_classes, size=shape, dtype=np.uint8)

class TestGeneratorBaseCreation(unittest.TestCase):
    def setUp(self):
        self.ih = 30
        self.iw = 20
        self.sz = stack_size

        self.oh = 50
        self.ow = 40
        self.nc = n_classes

        self.gen = GeneratorBase(data_paths, (self.ih, self.iw, self.sz), (self.oh, self.ow, self.nc))
    
    def test_parameters(self):
        self.assertIsNotNone(self.gen.data_paths)

        self.assertIsNone(self.gen.loader)
        self.assertIsNone(self.gen.data)
        
        self.assertEqual(self.ih, self.gen.input_height)
        self.assertEqual(self.iw, self.gen.input_width)
        self.assertEqual(self.sz, self.gen.stack_size)

        self.assertEqual(self.oh, self.gen.output_height)
        self.assertEqual(self.ow, self.gen.output_width)
        self.assertEqual(self.nc, self.gen.n_classes)

class TestGeneratorBaseDataAquisitionChannelLastColor(unittest.TestCase):
    def setUp(self):
        self.ih = 30
        self.iw = 20
        self.c = 2
        self.sz = stack_size

        self.oh = 50
        self.ow = 40
        self.nc = n_classes

        self.gen = GeneratorBase(data_paths, (self.ih, self.iw, self.sz), (self.oh, self.ow, self.nc))
        self.gen.loader = MockLoader(n_channels=self.c)

        self.files = [None] * (self.sz + 1) # <stack_size + 1> files for <stack_size> images
    
    def test_get_stacked_labeled_format(self):
        imgs, lbl = self.gen._get_stacked(self.files, labeled=True)

        # stacked format will not collapse in the last dimension
        self.assertEqual(imgs.shape, (self.sz, default_height, default_width, self.c))
        self.assertEqual(lbl.shape, (default_height, default_width, self.nc))

    def test_get_stacked_unlabeled_format(self):
        imgs, lbl = self.gen._get_stacked(self.files, labeled=False)
       
        # stacked format will not collapse in the last dimension
        self.assertEqual(imgs.shape, (self.sz, default_height, default_width, self.c))
        self.assertEqual(lbl.shape, (default_height, default_width, self.c))

    def test_get_stacked_labeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)

        imgs, lbl = self.gen._get_stacked(self.files, labeled=True, input_shape=input_shape, output_shape=output_shape)
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (self.sz, self.ih, self.iw, self.c))
        self.assertEqual(lbl.shape, (self.oh, self.ow, self.nc))

    def test_get_stacked_unlabeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)

        imgs, lbl = self.gen._get_stacked(self.files, labeled=False, input_shape=input_shape, output_shape=output_shape)
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (self.sz, self.ih, self.iw, self.c))
        self.assertEqual(lbl.shape, (self.oh, self.ow, self.c))

class TestGeneratorBaseDataAquisitionChannelLastGreyscale(unittest.TestCase):
    def setUp(self):
        self.ih = 30
        self.iw = 20
        self.sz = stack_size

        self.oh = 50
        self.ow = 40
        self.nc = n_classes

        self.gen = GeneratorBase(data_paths, (self.ih, self.iw, self.sz), (self.oh, self.ow, self.nc))
        self.gen.loader = MockLoader(n_channels=1)

        self.files = [None] * (self.sz + 1) # <stack_size + 1> files for <stack_size> images
    
    def test_get_stacked_labeled_format(self):
        imgs, lbl = self.gen._get_stacked(self.files, labeled=True)
        
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (default_height, default_width, n_channels * self.sz))
        self.assertEqual(lbl.shape, (default_height, default_width, self.nc))

    def test_get_stacked_unlabeled_format(self):
        imgs, lbl = self.gen._get_stacked(self.files, labeled=False)
        
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (default_height, default_width, n_channels * self.sz))
        self.assertEqual(lbl.shape, (default_height, default_width, n_channels))

    def test_get_stacked_labeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)

        imgs, lbl = self.gen._get_stacked(self.files, labeled=True, input_shape=input_shape, output_shape=output_shape)
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (self.ih, self.iw, n_channels * self.sz))
        self.assertEqual(lbl.shape, (self.oh, self.ow, self.nc))

    def test_get_stacked_unlabeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)

        imgs, lbl = self.gen._get_stacked(self.files, labeled=False, input_shape=input_shape, output_shape=output_shape)
        # stacked format will collapse in the last dimension, only works on greyscale images
        self.assertEqual(imgs.shape, (self.ih, self.iw, n_channels * self.sz))
        self.assertEqual(lbl.shape, (self.oh, self.ow, n_channels))

    def test_get_sequence_labeled_format(self):
        imgs, lbl = self.gen._get_sequence(self.files, labeled=True)
        
        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(imgs.shape, (self.sz, default_height, default_width, n_channels))
        self.assertEqual(lbl.shape, (self.sz, default_height, default_width, self.nc))

    def test_get_sequence_unlabeled_format(self):
        imgs, lbl = self.gen._get_sequence(self.files, labeled=False)

        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(imgs.shape, (self.sz, default_height, default_width, n_channels))
        self.assertEqual(lbl.shape, (self.sz, default_height, default_width, n_channels))
    
    def test_get_sequence_labeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)
        imgs, lbl = self.gen._get_sequence(self.files, labeled=True, input_shape=input_shape, output_shape=output_shape)

        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(imgs.shape, (self.sz, self.ih, self.iw,  n_channels))
        self.assertEqual(lbl.shape, (self.sz, self.oh, self.ow, self.nc))
    
    def test_get_sequence_unlabeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)
        imgs, lbl = self.gen._get_sequence(self.files, labeled=False, input_shape=input_shape, output_shape=output_shape)

        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(imgs.shape, (self.sz, self.ih, self.iw,  n_channels))
        self.assertEqual(lbl.shape, (self.sz, self.oh, self.ow, n_channels))
    
    def test_get_pair_labeled_format(self):
        img, lbl = self.gen._get_pair(self.files[:2], labeled=True)

        # pair format will create a pair of images
        self.assertEqual(img.shape, (default_height, default_width, n_channels))
        self.assertEqual(lbl.shape, (default_height, default_width, self.nc))

    def test_get_pair_labeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)
        img, lbl = self.gen._get_pair(self.files[:2], labeled=True, input_shape=input_shape, output_shape=output_shape)

        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(img.shape, (self.ih, self.iw,  n_channels))
        self.assertEqual(lbl.shape, (self.oh, self.ow, self.nc))

    def test_get_pair_unlabeled_format(self):
        img, lbl = self.gen._get_pair(self.files[:2], labeled=False)

        # pair format will create a pair of images
        self.assertEqual(img.shape, (default_height, default_width, n_channels))
        self.assertEqual(lbl.shape, (default_height, default_width, n_channels))

    def test_get_pair_unlabeled_shape(self):
        input_shape = (self.ih, self.iw)
        output_shape = (self.oh, self.ow)
        img, lbl = self.gen._get_pair(self.files[:2], labeled=False, input_shape=input_shape, output_shape=output_shape)

        # sequence format will create stack_size images stacked ontop of each other
        self.assertEqual(img.shape, (self.ih, self.iw,  n_channels))
        self.assertEqual(lbl.shape, (self.oh, self.ow, n_channels))