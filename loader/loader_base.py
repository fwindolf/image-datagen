import numpy as np 
import os
import cv2
import glob

class LoaderBase():
    """
    Base class for loaders.
    It provides methods and interfaces for loader implementations that handle real data.

    All images are generated or handled in channel_last ordering.
    """

    def __init__(self, num_classes, crop_scale):
        """
        Args:
            num_classes: Number of classes used (includes unknown class)
            crop_scale: Factor with which the images are scaled before cropping. (< 1 for more area covered)
        """
        self.n_classes = num_classes
        self.crop_scale = crop_scale

    def _resize(self, img, output_shape):
        """
        Resize an image to output_shape.
        Input and output image should be channel_last ordered.
        Args:
            img: The image to be resized
            output_shape: (height, width) of the resulting image
        """
        assert(len(output_shape) == 2) # exactly h, w
        assert(img.dtype == np.uint8 or img.dtype == np.float)
        
        s_before = len(img.shape)       
        new_height = output_shape[0]
        new_width = output_shape[1]
        # resize takes (new_width, new_height) as shape        
        img = cv2.resize(img, (new_width, new_height))
        # check if shape got collapsed, reinflate if necessary
        if s_before > len(img.shape):
            img = img[:, :, np.newaxis]

        return img 
    
    def _get_crop(self, x, y, x_shape, y_shape, unlabeled=False, seed=None):
        """
        Get a crop for the pair of image and label x, y.

        If the desired shape is bigger than the actual shape, the image will be resized to fit that shape, but not cropped.
        If the desired shape is smaller, a crop in the desired shape will be returned with matching crops for input and 
        output images, even when the shape is different.
        If the caller provides no label data, the data will be handeled as if unlabeled was set to True

        Args:
            x: Image data of any resolution
            y: Label data of any resolution
            x_shape: The shape that the returned image data should have
            y_shape: The shape that the returned label data should have
            unlabeled: Indicates that the y component remains untouched
            seed: In case that more images are cropped at the same place to the sam dimension use the same seed
        Returns:
            A tuple of x, y cropped and witht he respecive shapes        
        """
        if y_shape is None:
            y_shape = x_shape

        if y is None:
            y = np.zeros_like(x)
            unlabeled = True
        
        assert(len(y.shape) >= 3)        
        assert(len(x.shape) == 3)

        xc, yc = None, None
        h_new, w_new = x_shape
        
        if y_shape:
            h_out, w_out = y_shape
        else:
            h_out, w_out = h_new, w_new
        h, w = x.shape[:2]

        # rescale the whole image to make the crops cover more area (eg. even with little input_shape)
        if self.crop_scale is not None:
            # calculate the scaled dimensions, but only scale so far that the crop covers all image at max.
            hs, ws = max(int(h * self.crop_scale), h_new), max(int(h * self.crop_scale), w_new)
            x = self._resize(x, (hs, ws))
            h, w = x.shape[:2]

        hy, wy = y.shape[:2]

        # make same dimensions so cropping is easier
        if hy != h or wy != w:
            y = self._resize(y, (h, w))

        # If the current height is smaller than the desired, dont crop...
        if h_new >= h:
            xc = self._resize(x, (h_new, w))
            yc = self._resize(y, (h_out, wy))
        else:
            if seed is not None:
                np.random.seed(seed)
            ch = np.random.randint(0, h - h_new)
            xc = x[ch:ch + h_new, :, :]
            yc = y[ch:ch + h_new, :, :]

        # same procedure for the width
        if w_new >= w:
            xc = self._resize(xc, (h, w_new))
            yc = self._resize(yc, (hy, w_out))
        else:
            if seed is not None:
                np.random.seed(seed)
            cw = np.random.randint(0, w - w_new)
            xc = xc[:, cw:cw + w_new, :]
            yc = yc[:, cw:cw + w_new, :]

        # bring y crop to output dimensions
        yc = self._resize(yc, y_shape)

        if unlabeled:
            return xc, None
        
        return xc, yc

    def _get_labeled(self, file, input_shape=None, output_shape=None, source='auto'):
        raise NotImplementedError("Implement this method in a base class")
    
    def _get_unlabeled(self, file, input_shape=None, source='auto'):
        raise NotImplementedError("Implement this method in a base class")