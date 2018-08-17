import numpy as np
import glob
import os
from PIL import Image, ImageDraw

from universal_datagen.loader.loader_base import LoaderBase

ORIG_SHAPE = (1600, 1600)
RADIUS = 10
SIM_SCALE = 1000
SIM_ORIGIN = {
    1200 : (800, 800), 
    2700 : (800, 950), 
    5400 : (800, 1100)
}
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

    def __to_categorical(self, img):
        """
        Create a categorical version of the input which has non-overlapping classes.
        For each channel, only the areas where the class is active are 1, else 0.
        Args:
            img: The condensed image with dimensions [h, w]
        Returns:
            An image [h, w, c] that has class activations split in channels 
        """
        assert(len(img.shape) == 2)
        # flatten and save shape
        img = img.ravel() 
        n = img.shape[0] 
        
        categorical = np.zeros((n, self.n_classes), dtype=np.uint8)
        
        # Set to 1 everywhere where 
        categorical[np.arange(n), img] = 1
        categorical = np.reshape(categorical, (*self.shape, self.n_classes))

        return categorical

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
       
        img = Image.new('L', self.shape, color='black')
        draw_img = ImageDraw.Draw(img)
        lbl = Image.new('L', self.shape, color='black')
        draw_lbl = ImageDraw.Draw(lbl)

        for i, p in enumerate(points):
            px, py = int(float(p[0])), int(float(p[1]))
            draw_img.ellipse((px, py, px + 2 * self.radius - 1, py + 2 * self.radius - 1), fill=1)
            draw_lbl.ellipse((px, py, px + 2 * self.radius - 1, py + 2 * self.radius - 1), fill=int(classes[i]))

        del draw_img, draw_lbl # destroy objects to draw on

        # split into channels
        img = np.asarray(img, dtype=np.uint8) # 0, 1
        img = img[..., np.newaxis]
        lbl = np.asarray(lbl, dtype=np.uint8) # 0 - ?
        lbl = self.__to_categorical(lbl)

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
        
        num_particles = int(lines[0])
        lattice_line = lines[1]
        lattice = lattice_line[lattice_line.find('\"') +1:lattice_line.rfind('\"')].split()[0:8:4]
        lattice = float(lattice[0]) * self.scale, float(lattice[1]) * self.scale

        img = Image.new('L', self.shape, color='black')
        draw_img = ImageDraw.Draw(img)
        lbl = Image.new('L', self.shape, color='black')
        draw_lbl = ImageDraw.Draw(lbl)
               
        lines = [l.split() for l in lines[2:]]
        points = [l[1:3] for l in lines]
        classes = [l[-1] for l in lines]
    
        for i, p in enumerate(points):
            # convert to pixel coordinates by scaling with 1000 - radius is fixed to 0.01
            px = float(p[0]) * self.scale
            py = float(p[1]) * self.scale
            
            
            # get entry closest to num_particles
            ox, oy = self.origin.get(num_particles, self.origin[min(self.origin.keys(), key=lambda k: abs(k - num_particles))]) 
            cx, cy = int(self.shape[0]/2), int(self.shape[1]/2)
            
            # crop window with ORIG_SHAPE around SIM_ORIGIN           
            if abs(px - ox) > cx: 
                continue
            if abs(py - oy) > cy: 
                continue            
            
            # only paint particles that are visible
            pxc, pyc = int(px + cx - ox), int(py + cy - oy)
            
            draw_img.ellipse((pxc, pyc, pxc + 2 * self.radius - 1, pyc + 2 * self.radius - 1), fill=1)
            draw_lbl.ellipse((pxc, pyc, pxc + 2 * self.radius - 1, pyc + 2 * self.radius - 1), fill=int(classes[i]))

        del draw_img, draw_lbl

        # split into channels
        img = np.asarray(img, dtype=np.uint8) # 0, 1
        img = img[..., np.newaxis]
        lbl = np.asarray(lbl, dtype=np.uint8) # 0 - ?
        lbl = self.__to_categorical(lbl)

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
        if source == 'simulation':
            img, _ = self.__generate_from_simulation(file)
        elif source == 'experiment':
            img, _ = self.__generate_from_experimental(file)
        elif source == 'auto':
            img, _ = self.__generate_autodiscover(file)
        else:
            raise AttributeError("Unknown source: %s" % source)
        
        if shape is not None:
            img = self._resize(img, shape)        

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
        if source == 'simulation':
            _, lbl = self.__generate_from_simulation(file)
        elif source == 'experiment':
            _, lbl = self.__generate_from_experimental(file)
        elif source == 'auto':
            _, lbl = self.__generate_autodiscover(file)
        else:
            raise AttributeError("Unknown source: %s" % source)

        if shape is not None:
            lbl = self._resize(lbl, shape)
        
        return lbl

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
    
