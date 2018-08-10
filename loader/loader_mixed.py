import numpy as np
import glob
import os
from PIL import Image, ImageDraw

from universal_datagen.loader.loader_base import LoaderBase
from universal_datagen.loader.loader_text import AM2018TxtLoader
from universal_datagen.loader.loader_image import AM2018ImageLoader

TXT_FOLDER = "txt"

class AM2018MixedLoader(LoaderBase):
    """
    Load images and targets of painted images

    """
    def __init__(self, num_classes, crop_scale, txt_folder=TXT_FOLDER):
        """
        """
        super().__init__(num_classes, crop_scale)

        self.txt_loader = AM2018TxtLoader(num_classes, crop_scale)
        self.img_loader = AM2018ImageLoader(crop_scale)

        self.txt_folder = txt_folder

    def discover_data(self, data_paths, num_data=None):
        """
        Discover the data in the data_paths.
        The paths should contain two folder. One with the original images called images as .jpg files and the other 
        called txt containing .xyz files.

        The discovered file format will be path_to_data/%s/name.%s with placeholders for image/txt folder 
        and for the file endings of images/simulation files.
        """        
        data = []
        for path in sorted(data_paths):
            images = sorted(glob.glob(os.path.join(path, "%s/*.%s" % (self.img_loader.img_folder, self.img_loader.img_format))))
            txtfiles = sorted(glob.glob(os.path.join(path, "%s/*.%s" % (self.txt_folder, self.txt_loader.format))))

            # compare names
            img_names = set([img[img.rfind('/') + 1:img.rfind('.')] for img in images])
            txt_names = set([txt[txt.rfind('/') + 1:txt.rfind('.')] for txt in txtfiles])
            
            valid = img_names & txt_names # intersection of lists
            
            # create some placeholder path for the image/txtfile
            files = [path + "/%s/" + str(name) + ".%s" for name in sorted(list(valid))] 

            if num_data is not None:
                assert(num_data > 0)
                max_idx = max(1, int(num_data/ len(data_paths))) # make it so we have num_data files evenly distributed among data_paths
                files = files[:max_idx]

            data.append(files) # new array for each sequence, so data is like this [[...], [...], ...]
        
        return data

    def __crop_image(self, image, shape):
        """
        Crop the input image to shape
        """
        return image[0:shape[0], 0:shape[0], np.newaxis]

    def _get_labeled(self, file, input_shape=None, output_shape=None, source='auto'):
        """
        Get a real image as input and an abstracted (generated) image with classes as label.
        """
        img, _ = self.img_loader._get_unlabeled(file, input_shape, source)
        _, trg = self.txt_loader._get_labeled(file % (self.txt_folder, self.txt_loader.format), input_shape, output_shape, source) 

        img = self.__crop_image(img, self.txt_loader.shape)
        
        return img, trg

    def _get_unlabeled(self, file, input_shape=None, source='auto'):
        """
        Get a real image as input and an abstracted image without classification information as label.
        """
        img, _ = self.img_loader._get_unlabeled(file, input_shape, source)
        trg, _ = self.txt_loader._get_unlabeled(file % (self.txt_folder, self.txt_loader.format), input_shape, source) 

        img = self.__crop_image(img, self.txt_loader.shape)

        return img, trg