import numpy as np 
import os
import glob

from PIL import Image

from image_datagen.loader.loader_base import LoaderBase

class ImageLoader(LoaderBase):
    """
    Base DataLoader for image data
    """
    def __init__(self, classes, crop_scale, img_folder, img_format, trg_folder, trg_format):
        """
        Args:
            classes: The number of classes present in the target data
            crop_scale: Factor with which the images are scaled before cropping. (< 1 for more area covered)
            img_folder: The name of the folder in which the images are found in the data_paths
            img_format: The file endings of valid images
            trg_folder: The name of the folder in which the target images are found in the data_paths
            trg_format: The file endings of valid target images
        """
        super().__init__(classes, crop_scale)

        self.img_folder = img_folder
        self.img_format = img_format
        self.trg_folder = trg_folder
        self.trg_format = trg_format

        self.shape = ORIG_SHAPE

    def __clean_format(self, img, rescale=False):
        """
        Clean the format of an image to be either np.uint8 or np.float
        """
        dtype = img.dtype
        imin = np.min(img)
        imax = np.max(img)
        
        if dtype == np.uint8:
            return img
            
        if  dtype == np.float and imin >= 0 and imax <= 1:
            return img
        
        # np.uint8 from np.integer related
        if issubclass(dtype, np.integer) and imin >= 0 and imax <= 255:
            if imax <= 1 and rescale:
                assert(np.max(img) == 1)
                assert(np.min(img) == 0)

            return img.astype(np.uint8)
        
        # np.float from np.floating related
        if issubclass(dtype, np.floating) and imin >= 0 and imax <= 1: # [pylint: ignore E1101]
            if imax > 1 and rescale:
                img = img / 255.
            
            return img.astype(np.float)

        raise AttributeError("Unable to discover format for image: ", img.shape, dtype, "[", imin, ",", imax, "]")
    
    def __load_image(self, file, shape=None):
        """
        Load an image from disk
        Args:
            file: The file path of the image to load
            shape: The desired shape of the image. If None, the default shape will be used
        Returns:
            An greyscale image with the desired shape
        """
        img = np.asarray(Image.open(file).convert('L')) # 0: greyscale
        if img is None:
            return img
        
        if shape is None:
            shape = self.shape
        
        # clean format
        img = self.__clean_format(img, rescale=True)

        # resize
        return self._resize(img, shape)

    def __load_target(self, file, shape=None):
        """
        Load a target image from disk
        Args:
            file: The file path of the image to load
            shape: The desired shape of the image. If None, the default shape will be used
        Returns:
            An image with n_classes channels and the desired shape
        """
        trg = Image.open(file)

        if trg is None:
            return trg
        
        if shape is None:
            shape = self.shape

        trg = np.array(trg)

        # to classes
        if(len(trg.shape) == 2 or trg.shape[-1] == 1):
            trg = np.squeeze(trg)
            out = np.zeros((self.n_classes, *trg.shape), dtype=np.uint8)
            for c in range(self.n_classes):
                out[c] = (trg == c).astype(np.uint8)
            
            trg = np.moveaxis(out, 0, -1)

        # clean format
        trg = self.__clean_format(trg) # no rescale to not mess with classes
        
        # resize
        return self._resize(trg, shape)

    def discover_data(self, data_paths, num_data=None):
        """
        Discover all usable data in the provided data paths
        Args:
            data_paths: List of paths where data files can be found.
            num_data: Limit the number of files that are used
        Returns:
            A nested array with all the sequences of data files found.
        """
        data = []
        for path in sorted(data_paths):                        
            # get images and targets in folder            
            images = sorted(glob.glob(os.path.join(path, "%s/*.%s" % (self.img_folder, self.img_format))))
            targets = sorted(glob.glob(os.path.join(path, "%s/*.%s" % (self.trg_folder, self.trg_format))))
            
            # sort out all the images/targets that dont have a partner
            img_names = set([img[img.rfind('/') + 1:img.rfind('.')] for img in images])
            trg_names = set([trg[trg.rfind('/') + 1:trg.rfind('.')] for trg in targets])

            valid = img_names & trg_names # intersection of lists

            # create some placeholder path for the image/target
            valid = sorted(valid, key=lambda x: int(x))
            files = [path + "%s/" + str(name) + ".%s" for name in valid] 
                        
            if num_data is not None:
                assert(num_data > 0)
                max_idx = max(1, int(num_data/ len(data_paths))) # make it so we have num_data files evenly distributed among data_paths
                files = files[:max_idx]

            data.append(files) # new array for each sequence, so data is like this [[...], [...], ...]
        
        return data

    def _get_image(self, file, shape, source='auto'):
        """
        Get the files image content.

        Args:
            file: File containing the data
            shape: Shape of the data
            source: Data source modifier
        Return:
            An image of <shape> dimension
        """
        img_path = file % (self.img_folder, self.img_format)
        if source == 'auto':
            img = self.__load_image(img_path, shape)
        else:
            raise AttributeError("Unknown source: %s" % source)

        return img

    def _get_label(self, file, shape, source='auto'):
        """
        Get the files label content

        Args:
            file: File containing the data
            shape: Shape of the data
            source: Data source modifier
        Return:
            An label of <shape> dimension
        """
        lbl_path = file % (self.trg_folder, self.trg_format)
        if source == 'auto':
            lbl = self.__load_target(lbl_path, shape)
        else:
            raise AttributeError("Unknown source: %s" % source)

        return lbl

ORIG_SHAPE = (1600, 1600)

IMG_FOLDER = "images"
IMG_FORMAT = "jpg"
TRG_FOLDER = "targets"
TRG_FORMAT = "png"
TRG_CLASSES = 4 # unknown, solid, liquid, gas

class AM2018ImageLoader(ImageLoader):
    """
    A loader that can handle AM2018 based image data
    """
    def __init__(self, classes=TRG_CLASSES, crop_scale=None, img_folder=IMG_FOLDER, img_format=IMG_FORMAT, trg_folder=TRG_FOLDER, trg_format=TRG_FORMAT):
        """
        Args:
            crop_scale: Factor with which the images are scaled before cropping. (< 1 for more area covered)
            img_folder: The name of the folder in which the images are found in the data_paths
            img_format: The file endings of valid images
            trg_folder: The name of the folder in which the target images are found in the data_paths
            trg_format: The file endings of valid target images
        """
        super().__init__(classes, crop_scale, img_folder, img_format, trg_folder, trg_format)



class BubblesImageLoader(ImageLoader):
    """
    A loader that can handle AM2018 based image data
    """
    def __init__(self, classes=1, crop_scale=None, img_folder='.', img_format='png', trg_folder='.', trg_format='png'):
        """
        Args:
            crop_scale: Factor with which the images are scaled before cropping. (< 1 for more area covered)
            img_folder: The name of the folder in which the images are found in the data_paths
            img_format: The file endings of valid images
            trg_folder: The name of the folder in which the target images are found in the data_paths
            trg_format: The file endings of valid target images
        """
        super().__init__(classes, crop_scale, img_folder, img_format, trg_folder, trg_format)