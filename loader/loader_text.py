import numpy as np
import glob
import os
import cv2

from universal_datagen.loader.loader_base import LoaderBase

ORIG_SHAPE = (1600, 1600)
RADIUS = 10
SIM_SCALE = 1000
SIM_ORIGIN = (800, 1000)
SIM_FORMAT = "xyz"

class AM2018TxtLoader(LoaderBase):
    """
    Base class for AM2018 data that is generated from text.
    It can handle experimental and simulation data and will visualize both in equal ways.

    Classes:
     - 0: unknown
     - 1: solid
     - 2: liquid
     - 3: gas
    
    Note that this implementation does not differentiate between unknown particles and background.
    All images are generated on channel_last ordering.
    """

    def __init__(self, num_classes, crop_scale, data_format=SIM_FORMAT):
        """
        Args:
            num_classes: Number of classes used (includes unknown class)
            crop_scale: Factor with which the images are scaled before cropping. (< 1 for more area covered)
        """
        super().__init__(num_classes, crop_scale)

        self.scale = SIM_SCALE
        self.origin = SIM_ORIGIN
        self.radius = RADIUS
        self.shape = ORIG_SHAPE
        self.format = data_format

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
            # sort files by filename interpreted as int
            files = sorted(glob.glob(os.path.join(path, "*.%s" % self.format)), key=lambda x: int(x[x.rfind('/') + 1:x.rfind('.')]))
            if num_data is not None:
                assert(num_data > 0)
                max_idx = max(1, int(num_data/ len(data_paths))) # make it so we have num_data files evenly distributed among data_paths
                files = files[:max_idx]

            data.append(files) # new array for each sequence, so data is like this [[...], [...], ...]
        
        return data
    
    def __generate_from_experimental(self, txtfile):
        """
        Generate an image from experimental data.
        Data extracted from experiments' format is:
        [0]  # of particles
        [1]  header
        [2:] particles in format idx, x, y, z, label
        
        Args:
            txtfile: File that contains the data
        Returns:
            An pair of image, label with image dimensions of (1600, 1600, 1)
        """
        with open(txtfile, 'r') as f:
            lines = [l.split() for l in f.readlines()[2:]]

        points = [l[1:3] for l in lines]
        classes = [l[-1] for l in lines]

        img = np.zeros((*self.shape, 1), dtype=np.uint8)
        lbl = np.zeros((self.n_classes, *self.shape), dtype=np.uint8)
            
        for i, p in enumerate(points):
            px, py = int(float(p[0])), int(float(p[1]))
            cv2.circle(img, (px, py), self.radius, (1, 1, 1), -1)
            cv2.circle(lbl[int(classes[i])], (px, py), self.radius, (1, 1, 1), -1)

        # add unknown where there are no pixels
        lbl[0] = 1
        for c in range(1, self.n_classes):
            lbl[0] = np.clip(lbl[0] - lbl[c], 0, 1) # clip in case something overlaps

        # make label channel_last
        lbl = np.moveaxis(lbl, 0, -1)

        # check if there is labels (-> other classes than 0:unknown)
        if np.count_nonzero(classes) > 0:
            return img, lbl

        return img, None

    def __generate_from_simulation(self, txtfile):
        """
        Generate an image from simulation data.
        Simulation data format is:
        [0]  # of particles
        [1]  size of lattice 
        [2:] particles in format idx, x, y, z, ..., label

        Args:
            txtfile: File that contains the data
        Returns:
            An pair of image, label with dimensions of (1600, 1600, 1/n_classes)
        """
        with open(txtfile, 'r') as f:
            lines = f.readlines()

        lattice_line = lines[1]
        lattice = lattice_line[lattice_line.find('\"') +1:lattice_line.rfind('\"')].split()[0:8:4]
        lattice = float(lattice[0]) * self.scale, float(lattice[1]) * self.scale

        img = np.zeros((*self.shape, 1), dtype=np.uint8)
        lbl = np.zeros((self.n_classes, *self.shape), dtype=np.uint8)
               
        lines = [l.split() for l in lines[2:]]
        points = [l[1:3] for l in lines]
        classes = [l[-1] for l in lines]
    
        for i, p in enumerate(points):
            # convert to pixel coordinates by scaling with 1000 - radius is fixed to 0.01
            px = float(p[0]) * self.scale
            py = float(p[1]) * self.scale
            
            # crop window with ORIG_SHAPE aroudn SIM_ORIGIN
            ox, oy = self.origin
            cx, cy = int(self.shape[0]/2), int(self.shape[1]/2)
            
            if abs(px - ox) > cx - self.radius:
                continue
            if abs(py - oy) > cy - self.radius:
                continue

            cv2.circle(img, (int(px + cx - ox), int(py + cy - oy)), self.radius, (1, 1, 1), -1)
            cv2.circle(lbl[int(classes[i])], (int(px + cx - ox), int(py + cy - oy)), self.radius, (1, 1, 1), -1)

        # add unknown where there are no pixels
        lbl[0] = 1
        for c in range(1, self.n_classes):
            lbl[0] = np.clip(lbl[0] - lbl[c], 0, 1) # clip in case something overlaps
        
        # make label channel_last ordering
        lbl = np.moveaxis(lbl, 0, -1)

        # simulation data comes out horizontally flipped (y axis bottom-left corner)
        img = np.flip(img, 0)        
        lbl = np.flip(lbl, 0)

        if np.count_nonzero(classes) > 0:
            return img, lbl

        return img, None

    def __generate_autodiscover(self, txtfile):
        """
        Automatically discover the source of the txtfile and return its content

        Args:
            txtfile: File that contains the data
        Returns:
            An pair of image, label with dimensions of (1600, 1600, 1/n_classes) or None, None if source type could not be determined
        """
        with open(txtfile, 'r') as f:
            _ = f.readline()
            line1 = f.readline()

        if line1.startswith("Lattice="):
            return self.__generate_from_simulation(txtfile)
        elif line1.startswith("i x y z l"):
            return self.__generate_from_experimental(txtfile)
        else:
            return None, None

    def _get_labeled(self, txtfile, input_shape=None, output_shape=None, source='auto'):
        """
        Get the content of the txtfile as labeled data in the specified shape.
        """
        if source == 'simulation':
            img, lbl = self.__generate_from_simulation(txtfile)
        elif source == 'experiment':
            img, lbl = self.__generate_from_experimental(txtfile)
        elif source == 'auto':
            img, lbl = self.__generate_autodiscover(txtfile)
        else:
            raise AttributeError("Unknown source: %s" % source)

        if lbl is None:
            raise RuntimeError("File %s does not contain any labels!" % txtfile)
        
        # transform to desired shape
        if input_shape is not None:
            img = self._resize(img, input_shape)        
        if output_shape is not None:
            lbl = self._resize(lbl, output_shape)
    
        return img, lbl

    def _get_unlabeled(self, txtfile, input_shape=None, source='auto'):
        """
        Get the content of the txtfile as unlabeled data in the specified shape.
        """
        if source == 'simulation':
            img, _ = self.__generate_from_simulation(txtfile)
        elif source == 'experiment':
            img, _ = self.__generate_from_experimental(txtfile)
        elif source == 'auto':
            img, _ = self.__generate_autodiscover(txtfile)
        else:
            raise AttributeError("Unknown source: %s" % source)

        if input_shape is not None:
            img = self._resize(img, input_shape)
        
        return img, None
    
    
