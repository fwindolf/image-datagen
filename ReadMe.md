# Universal DataGenerator Module

## Generator

Either use one of the available generators to create data batches for training, validation or testing.

### Usage

```
from generator.generator_image import AM2018ImageGenerator

gen = AM2018ImageGenerator(['test/data/image/',], (200, 200, 3), (100, 100, 4))
train_gen = gen.generator(batch_size=4, structure='sequence', labeled=True)

model.train_generator(train_gen, ...)
```

For training/validation splits use the split parameter
```
train_gen = gen.generator(batch_size=4, structure='sequence', labeled=True)
```

### Shapes

- input_shape in format of [height, width, stack_size] determines the size of the provided images, while stack_size defines the number of images stacked on top of each other.
- output_shape in format of [height, width, num_classes] determines the size of the provided labels, while num_classes indicates how many classes from the data will be used. 

### Structures

- 'sequence' will provide images and labels in a sequence, with stack_size images/labels stacked among the first dimension ([sz, h, w, c], [sz, h, w, nc]).
- 'stacked' will provide images in a stacked way, but a single label for the following frame ([sz, h, w, c], [h, w, nc])
- 'pair' will provide a pair of an image and the corresponding label with offset ([h, w, c], [h, w, nc])

For every sequence based image data, the label will have an offset of 1 timestep from the last image, so the label corresponds to the next image after the end of the image sequence.

## Loader

The loaders get acquired by the generator and provide means of opening various formats of data.

Every loader has to implement the `_get_labeled()` and `_get_unlabeled()` methods as well as `discover_data` for gathering the
data from the provided data_paths. 



