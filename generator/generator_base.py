import numpy as np
from itertools import cycle

class GeneratorBase():
    """
    A base class for all generators.
    Provides abstraction for generating data in different structure.

    Internally, all image data is handeled in channel_last format.
    """
    def __init__(self, data_paths, image_dims, label_dims, flatten_label=False, ignore_unknown=False, num_data=None):
        """
        Create a new data generator base.
         Args:
            data_paths: The paths where the data can be found. Each path should contain files with increasing numbers from one sequence.
            image_dims: The dimensions of generated image data in (height, width, channels) format, with channels being the number of images stacked together.
            output_dims: The dimensions of generated label data in (height, width, n_classes) format, with n_classes including background (0) class
            flatten_label: Flag to enforce that the output comes in output_height*output_width dimension.
            ignore_unknown: Flag to ignore the first class (0) in the data. This will produce output (target) data with n_classes - 1 channels.
            num_data: Artificially reduce the number of data points to use.
        """
        assert(len(image_dims) == 3) # h, w, stack_size
        assert(len(label_dims) == 3) # h, w, num_classes
        
        self.data_paths = data_paths
        self.num_data = num_data

        self.input_height = image_dims[0]
        self.input_width = image_dims[1]
        self.stack_size = image_dims[2] + 1 # make up for dropping one data point for offset

        self.output_height = label_dims[0]
        self.output_width = label_dims[1]
        self.n_classes = label_dims[2]

        self.flatten_label = flatten_label
        self.ignore_unknown = ignore_unknown

        self.loader = None
        self.data = None 

    def _get_stacked(self, files, labeled, input_shape=None, output_shape=None):
        """
        Get the image data for a number of files, with the label for the last one. The files need to 
        come from the same sequence, else the data will be meaningless.
        Args:
            files: Any number of files that the loader can make sense of 
            labeled: Set to True to return a label for the sequence, else a image will be returned as label
            input_shape: The (h, w) of the returned images
            output_shape: The (h, w) of the returned label
        Return:
            A stacked image [input_shape x len(files) - 1]  and a label [output_shape x n_classes] 
            for labeled or [output_shape x 1] for unlabeled
        """
        assert(len(files) > 1) # Must be more than one to get a label

        # stack all the images up to the last one
        imgs = np.array([self.loader._get_unlabeled(f, input_shape, source='auto')[0] for f in files[:-1]])
        
        if labeled:             
            # get the label for the next image as label for this sequence
            _, lbl = self.loader._get_labeled(files[-1], input_shape, output_shape, source='auto')
        else:
            # get the next image and provide it as label of this sequence
            lbl, _ = self.loader._get_unlabeled(files[-1], output_shape, source='auto')
        
        # stack the images by squeezing along the channel dimension and using the array dimension as new channel
        if len(imgs.shape) > 3:
            imgs = np.squeeze(np.array(imgs), axis=3) # TODO: For input images with c > 1 this wont work!
        
        imgs = np.moveaxis(imgs, 0, -1) 

        return np.asarray(imgs), np.asarray(lbl)

    def _get_sequence(self, files, labeled, input_shape=None, output_shape=None):
        """
        Get the image data for a number of files, with a sequence of labels or image data.
        The labels have a temporal offset of 1.
        Args:
            files: Any number of files that the loader can make sense of 
            labeled: Set to True to return a label for the sequence, else a image will be returned as label
            input_shape: The (h, w) of the returned images
            output_shape: The (h, w) of the returned label
        Return:
            A stacked image [input_shape x 1 x len(files) - 1] and stacked 
            labels [output_shape x n_classes x stack_size - 1] or [output_shape x 1 x stack_size - 1] for unlabeled
        """
        assert(len(files) > 1) # Must be more than one to get a label

        if labeled:
            data = [self.loader._get_labeled(f, input_shape, output_shape, source='auto') for f in files]
            imgs, lbls = zip(*data)
            # from the beginning for inputs
            imgs = [img for img in imgs[:-1]]
            # offset 1 for the targets
            lbls = [lbl for lbl in lbls[1:]]
        else:
            data = [self.loader._get_unlabeled(f, input_shape, source='auto') for f in files]
            data, _ = zip(*data) # extract only the images
            # from the beginning for inputs
            imgs = [img for img in data[:-1]]
            # offset 1 for the targets
            lbls = [lbl for lbl in data[1:]]
            if output_shape is not None:
                lbls = [self.loader._resize(np.array(lbl), output_shape) for lbl in lbls]
            
        return np.asarray(imgs), np.asarray(lbls)
    
    def _get_pair(self, files, labeled, input_shape=None, output_shape=None):
        """
        Get a pair of either an image and the label with offset 1, or two sequential images.
        Args:
            files: Two files that the loader can make sense of 
            labeled: Set to True to return a label for the image, else the next image will be returned as label
            input_shape: The (h, w) of the returned images
            output_shape: The (h, w) of the returned label
        Return:
            A pair of an image [input_shape x channels] and a label [output_shape x n_classes] or 
            [output_shape x 1] for unlabeled
        """
        assert(len(files) == 2) # one for image, one for label/next image

        img, _ = self.loader._get_unlabeled(files[0], input_shape, source='auto')
        unlbl, lbl = self.loader._get_labeled(files[-1], output_shape, output_shape, source='auto')
        
        if labeled:
            return np.asarray(img), np.asarray(lbl)
        else:
            return np.asarray(img), np.asarray(unlbl)


    def __chunks(self, data, size):
        """
        Generate chunks of size from array data and throw away incomplete chunks
        Args:
            data: Array of data
            size: size of chunks
        Return:
            A generator that provides chunks from the array
        """
        for i in range(0, int(len(data) / size) * size, size):
            yield data[i:i + size]

    def __gather_data(self, shuffled=True):
        """
        Put together the data files in chunks to use in generator
        Args: 
            shuffled: True if the data should be shuffled
        Returns:
            Nested array of sequences of data files.
        Raises:
            AttributeError if the generators data reservoir is empty.
        """
        if self.data is None:
            raise AttributeError("No data available for generator!")

        data = []
        # generate arrays of stack_size for all sequences
        for seq in self.data:
            data.extend([chunk for chunk in self.__chunks(seq, self.stack_size)])

        if shuffled:
            data = np.random.permutation(data)
        
        return data

    def __get_chunk(self, files, structure, labeled, input_shape, output_shape):
        """
        Get a chunk of data for some files.
        Args:
            files: The files pointing to the location of the data
            structure: The structure of the returned data (sequence, stacked or pair)
            labeled: The flag the decides if the returned data is labeled
            input_shape: height, width of the returned image(s)
            output_shape: height, width of the returned label(s)
        Return:
            A tuple of x and y in the respective structure and dimensions
        Raises:
            AttributeError in case the supplied structure is invalid
        """
        assert(len(input_shape) == 2) # h, w
        assert(len(output_shape) == 2) # h, w

        if structure == 'sequence':
            x, y = self._get_sequence(files, labeled, input_shape, output_shape)
        elif structure == 'stacked':
            x, y = self._get_stacked(files, labeled, input_shape, output_shape)
        elif structure == 'pair':
            x, y = self._get_pair(files, labeled, input_shape, output_shape)
        else:
            raise AttributeError("Unknown data structure:", structure, ". Must be in [sequence, stacked, pair]")
        
        return x, y

    def __generator(self, data, structure, labeled, batch_size, num_crops, ordering='channel_first'):
        """
        Create a new generator for the provided data
        Args:
            data: Chunks of data files that the loader can interpret
            structure: The structure of the returned data (sequence, stacked or pair)
            labeled: The flag the decides if the returned data is labeled
            batch_size: The number of data chunks in one batch
            num_crops: The number of crops per chunk or None if no cropping
            ordering: The way the data is returned, either [..., h, w, c] for channel_last or [..., c, h, w] for channel_first
        Return:
            A generator that yields batches of X,Y from chunks of the data.
        """
        input_shape = (self.input_height, self.input_width)
        output_shape = (self.output_height, self.output_width)

        while True:
            X, Y = [], []
            for i in range(batch_size):
                if num_crops is None or i % num_crops == 0:
                    files = next(data)

                # use original size when cropping
                i_s = input_shape if num_crops is None else None
                o_s = output_shape if num_crops is None else None

                x, y = self.__get_chunk(files, structure, labeled, i_s, o_s)

                # crop if necessary
                if num_crops is not None:
                    seed = np.random.randint(10000) # new seed every iteration, but same crops for the stack
                    cropped = [self.loader._get_crop(x[i], y[i], input_shape, output_shape, seed=seed) for i in range(len(x))]
                    x_out, y_out = zip(*cropped)
                else:
                    x_out, y_out = x, y # dont destroy original x,y for num_crops > 1

                x_out, y_out = np.array(x_out), np.array(y_out)

                # post process                
                if self.flatten_label:
                    y_out = np.reshape(y_out, (y_out.shape[0], np.prod(output_shape), y_out.shape[3]))
                
                if labeled and self.ignore_unknown:                    
                    y_out = y_out[..., 1:]
                
                # make sure that the image data is in form [sz, h, w, c]
                if len(x_out.shape) < 4:
                    x_out = x_out[:, :, :, np.newaxis]

                # put into right ordering
                if ordering is 'channel_first':
                    x_out = np.moveaxis(x_out, -1, 1)
                    y_out = np.moveaxis(y_out, -1, 1)
                elif ordering is 'channel_last':
                    pass # already in right ordering
                else:
                    raise AttributeError("Unknown ordering: %s" % ordering)

                X.append(x_out)
                Y.append(y_out)
        
            yield np.asarray(X), np.asarray(Y)

    def generator(self, structure, labeled, batch_size, num_crops=None, split=None, ordering='channel_first'):
        """
        Create a generator or a tuple of generators 
        Args:
            structure: The structure of the returned data (sequence, stacked or pair)
            labeled: The flag the decides if the returned data is labeled
            batch_size: The number of data chunks in one batch
            num_crops: The number of crops per chunk or None if no cropping
            split: Split the data between training and validation, float in [0, 1]
            ordering: The way the data is returned, either [..., h, w, c] for channel_last or [..., c, h, w] for channel_first
        Return:
            One or two generators that yields batches of X,Y from chunks of the data.
        """
        data = self.__gather_data(shuffled=True)

        if split is not None:
            assert(split > 0. and split < 1.) # make sure split is a valid parameter
            split_idx = int(len(data) * (1 - split))
            tdata = data[:split_idx]
            vdata = data[split_idx:]

            tgen = self.__generator(cycle(tdata), structure, labeled, batch_size, num_crops, ordering)
            vgen = self.__generator(cycle(vdata), structure, labeled, batch_size, num_crops, ordering)
            return tgen, vgen
        else:
            return self.__generator(cycle(data), structure, labeled, batch_size, num_crops, ordering)