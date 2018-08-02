import numpy as np
from itertools import cycle

class GeneratorBase():
    """
    A base class for all generators.
    Provides abstraction for generating data in different structure.

    Internally, all image data is handeled in channel_last format.
    """
    def __init__(self, data_paths, image_dims, label_dims, ignore_unknown=False, num_data=None):
        """
        Create a new data generator base.
         Args:
            data_paths: The paths where the data can be found. Each path should contain files with increasing numbers from one sequence.
            image_dims: The dimensions of generated image data in (height, width, channels) format, with channels being the number of images stacked together.
            output_dims: The dimensions of generated label data in (height, width, n_classes) format, with n_classes including background (0) class
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
            structure: Structure of the returned data (sequence, stacked or pair)
            labeled: Decides if the returned data is labeled
            input_shape: height, width of the returned image(s)
            output_shape: height, width of the returned label(s)
        Return:
            A tuple of x and y in the respective structure and dimensions
        Raises:
            AttributeError in case the supplied structure is invalid
        """
        if input_shape is not None:
        assert(len(input_shape) == 2) # h, w
        if output_shape is not None:
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

    def __get_crop(self, x, y, num_crops, structure, input_shape, output_shape, seed=None):
        # crop if necessary
        if num_crops is not None:
            seed = np.random.randint(10000) if seed is None else seed # new seed every iteration, but same crops for the stack
            if structure == 'sequence':
                # crop along the stack_size dimension
                cropped = [self.loader._get_crop(x[i], y[i], input_shape, output_shape, seed=seed) for i in range(len(x))]
                x, y = zip(*cropped)
                x, y = np.array(x), np.array(y)
            elif structure == 'stacked':
                # crop along the stack_size dimension for x
                cropped = [self.loader._get_crop(x[i], y, input_shape, output_shape, seed=seed) for i in range(len(x))]
                x, y = zip(*cropped)
                x, y = np.array(x), np.array(y[0])
            elif structure == 'pair':
                x, y = self.loader._get_crop(x, y, input_shape, output_shape, seed=seed)           
        else:
            x, y = x, y 
        
        return x, y 

    def __correct_shape_pair(self, x, y, labeled, ordering, flatten_label):
        """
        Correct the shape of x and y for pair structured data.
        Args:
            x: Images
            y: Labels            
            labeled: Is the data with or without labels
            flatten_label: Reduce the h,w dimension and put channels to the last dimension
        Returns:
            A tuple of the shape corrected x, y inputs
        Raises:
            AssertionError in case of invalid dimensions or orderings
        """
        x_shape = x.shape
        y_shape = y.shape

        assert(len(x_shape) <= 3)
        assert(len(y_shape) <= 3)
    
        if len(x_shape) < 3: # channels lost
            x = x[:, :, np.newaxis]
        if not labeled and len(y_shape) < 3: # channels lost
            y = y[:, :, np.newaxis]
        
        assert(len(x.shape) == 3) # [h, w, c]
        assert(len(y.shape) == 3) # [h, w, c/classes]

        if ordering == 'channel_first':
            x = np.moveaxis(x, -1, 0)
            if not flatten_label: # flattened label always has channel last
                y = np.moveaxis(y, -1, 0)

        if flatten_label: 
            y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[-1])) # the label will become [h*w, c/classes]

        return x, y

    def __correct_shape_stacked(self, x, y, labeled, ordering, flatten_label):
        """
        Correct the shape of x and y for stacked data.
        Args:
            x: Images
            y: Labels            
            labeled: Is the data with or without labels
            flatten_label: Reduce the h,w dimension and put channels to the last dimension
        Returns:
            A tuple of the shape corrected x, y inputs
        Raises:
            AssertionError in case of invalid dimensions or orderings
        """
        x_shape = x.shape
        y_shape = y.shape

        assert(x_shape[0] == self.stack_size - 1)

        if len(x_shape) == 3: # channels lost
            x = x[:, :, :, np.newaxis]
        if not labeled and len(y_shape) == 2:
            y = y[:, :, np.newaxis]

        assert(len(x.shape) == 4) # [sz, h, w, c]
        assert(len(y.shape) == 3) # [h, w, c/classes]

        if x.shape[-1] == 1:
            x = np.moveaxis(np.squeeze(x), 1, -1) # [sz, h, w, 1] -> [h, w, sz]

        if ordering == 'channel_first':
            x = np.moveaxis(x, -1, 1)
            if not flatten_label:
                y = np.moveaxis(y, -1, 0)

        if flatten_label: 
            y = np.reshape(y, (y.shape[0] * y.shape[1], -1)) # the label will become [h*w, c/classes]

        
        return x, y

    def __correct_shape_sequence(self, x, y, labeled, ordering, flatten_label):
        """
        Correct the shape of x and y for stacked data.
        Args:
            x: Images
            y: Labels            
            labeled: Is the data with or without labels
            flatten_label: Reduce the h,w dimension and put channels to the last dimension
        Returns:
            A tuple of the shape corrected x, y inputs
        Raises:
            AssertionError in case of invalid dimensions or orderings
        """  
        x_shape = x.shape
        y_shape = y.shape

        assert(x_shape[0] == self.stack_size - 1)
        assert(y_shape[0] == self.stack_size - 1)

        if len(x_shape) == 3: # channels lost
            x = x[:, :, :, np.newaxis]
        if not labeled and len(y_shape) == 3: 
            y = y[:, :, :, np.newaxis]
        
        assert(len(x.shape) == 4) # [sz, h, w, c]
        assert(len(y.shape) == 4) # [sz, h, w, c/classes]

        if ordering == 'channel_first':
            x = np.moveaxis(x, -1, 1)
            if not flatten_label:
                y = np.moveaxis(y, -1, 1)

        if flatten_label: 
            y = np.reshape(y, (y.shape[0], y.shape[0] * y.shape[1], -1)) # the label will become [sz, h*w, c/classes]

        return x, y

    def __correct_shape(self, x, y, structure, labeled, ordering, flatten_label):
        """
        Correct the shape of x and y depending on structure and labeled.
        Args:
            x: Images
            y: Labels
            structure: Structure of the data, one of [sequence, stacked, pair]
            labeled: Is the data with or without labels
        Returns:
            A tuple of the shape corrected x, y inputs
        Raises:
            AssertionError in case of invalid dimensions or orderings
        """
        assert(ordering in ['channel_first', 'channel_last'])
        
        if structure == 'sequence':
            return self.__correct_shape_sequence(x, y, labeled, ordering, flatten_label)
        elif structure == 'stacked':
            return self.__correct_shape_stacked(x, y, labeled, ordering, flatten_label)
        elif structure == 'pair':
            return self.__correct_shape_pair(x, y, labeled, ordering, flatten_label)
        else:
            raise AttributeError("Unknow structure : %s" % structure)

    def __generator(self, data, structure, labeled, batch_size, num_crops, flatten_label, ordering='channel_first'):
        """
        Create a new generator for the provided data
        Args:
            data: Chunks of data files that the loader can interpret
            structure: Structure of the returned data (sequence, stacked or pair)
            labeled: Decides if the returned data is labeled
            batch_size: The number of data chunks in one batch
            num_crops: The number of crops per chunk or None if no cropping
            flatten_label: Reduce the h,w dimension and put channels to the last dimension
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

                x_out, y_out = self.__get_crop(x, y, num_crops, structure, input_shape, output_shape)

                x_out, y_out = np.array(x_out), np.array(y_out)

                if labeled and self.ignore_unknown:                    
                    y_out = y_out[..., 1:]
                
                # correct shape and apply ordering
                x_out, y_out = self.__correct_shape(x_out, y_out, structure, labeled, ordering, flatten_label)

                X.append(x_out)
                Y.append(y_out)
        
            yield np.asarray(X), np.asarray(Y)

    def generator(self, structure, labeled, batch_size, num_crops=None, split=None, flatten_label=False, ordering='channel_first'):
        """
        Create a generator or a tuple of generators 
        Args:
            structure: Structure of the returned data (sequence, stacked or pair)
            labeled: Decides if the returned data is labeled
            batch_size: The number of data chunks in one batch
            num_crops: The number of crops per chunk or None if no cropping
            split: Split the data between training and validation, float in [0, 1]
            flatten_label: Reduce the h,w dimension and put channels to the last dimension
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

            tgen = self.__generator(cycle(tdata), structure, labeled, batch_size, num_crops, flatten_label, ordering)
            vgen = self.__generator(cycle(vdata), structure, labeled, batch_size, num_crops, flatten_label, ordering)
            return tgen, vgen
        else:
            return self.__generator(cycle(data), structure, labeled, batch_size, num_crops, flatten_label, ordering)

    def iterator(self, structure, labeled, cropped, ordering='channel_first'):
        """
        Create a new iterator over the provided data.
        Args:
            structure: Structure of the returned data (sequence, stacked or pair)
            labeled: Decides if the returned data is labeled
            cropped: Decides if the returned data is cropped
            ordering: The way the data is returned, either [..., h, w, c] for channel_last or [..., c, h, w] for channel_first
        Return:
            An iterator that yields data in a sequential way (but in the structure that is provided)
        """
        assert(len(self.data) == 1) # can only handle one sequence
        data = self.data[0] # use the whole sequence

        input_shape = (self.input_height, self.input_width)
        output_shape = (self.output_height, self.output_width)

        seed = np.random.randint(1986512) # equal cropping for the whole data
        for i in range(len(data) - self.stack_size):
            files = data[i:i+self.stack_size]

            i_s = input_shape if not cropped else None
            o_s = output_shape if not cropped else None

            x, y = self.__get_chunk(files, structure, labeled, i_s, o_s)
            x, y = self.__get_crop(x, y, 1, structure, input_shape, output_shape, seed)

            if labeled and self.ignore_unknown:                    
                y = y[..., 1:]

            # correct shape and apply ordering
            x, y = self.__correct_shape(x, y, structure, labeled, ordering, False)

            yield x, y

    def __len__(self):
        return int(np.sum([len(seq) // self.stack_size for seq in self.data]))