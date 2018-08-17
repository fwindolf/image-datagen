from universal_datagen.generator.generator_base import GeneratorBase
from universal_datagen.loader.loader_image import AM2018ImageLoader

class AM2018ImageGenerator(GeneratorBase):
    """
    A generator for image based AM2018 data.
    """
    def __init__(self, data_paths, image_dims, label_dims, offset=1, crop_scale=None, ignore_unknown=False, num_data=None):
        """
        Create a new data generator base.
         Args:
            data_paths: The paths where the data can be found. Each path should contain files with increasing numbers from one sequence.
            image_dims: The dimensions of generated image data in (height, width, channels) format, with channels being the number of images stacked together.
            label_dims: The dimensions of generated label data in (height, width, n_classes) format, with n_classes including background (0) class
            offset: The offset between images and labels
            crop_scale: Scale the provided images before cropping to increase the image area covered by the crops
            ignore_unknown: Flag to ignore the first class (0) in the data. This will produce output (target) data with n_classes - 1 channels.
            num_data: Artificially reduce the number of data points to use.
        """

        super().__init__(data_paths=data_paths, image_dims=image_dims, label_dims=label_dims, offset=offset, ignore_unknown=ignore_unknown, num_data=num_data)

        self.loader = AM2018ImageLoader(crop_scale)
        self.data = self.loader.discover_data(data_paths, num_data)

    