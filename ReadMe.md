# Universal DataGenerator Module

## Generator

Either use one of the available generators to create data batches for training, validation or testing.

A generator base can provide data in different layouts (structure), with a fixed offset and for fixed shapes (set on initialization).
When creating a data generator from a base, the structure [pair, stacked, sequence], the format [labeled, unlabeled] as well as cropping, batch size,
data flattening, and the ordering [channel_last, channel_first] can be set.

#### Generator Base

| Parameter       | Meaning  |
|-----------------|----------|
| Data Paths      | Paths where the data can be found. Those will be provided to the `discover_data()`  function. |
| Image Dims      | Dimensions of the image data that is created by the generator in format (height, width, size of stacks). |
| Label Dims      | Dimensions of the label data that is created by the generator in format (height, width, number of classes). |
| Offset          | The offset between the image(s) and label(s). Offset of 0 would provide image/label data from the same file, an offset of 1 would use data from the next file for labels. |
| Ignore Unknown  | A flag indicating if the background class (0) should be contained in the data. |
| Num Data        | The number of files used for data generation, if None all found data is used. Evenly distributed among all provided data paths. |

#### Data Generator

| Parameter       | Meaning  |
|-----------------|----------|
| Structure       | Structure of the returned data (sequence, stacked, or pair). See Structures for more information. |
| Batch Size      | Size of the batches provided by the generator. The first dimension of the data.
| Labeled         | Flag to specify if only images (False) or images and labeles (True) are provided. |
| Num Crops       | The number of crops for one file. Set to None if original sized images/labels should be used. |
| Split           | Create two generators for training and validation from the data, where the validation data contains the given percentage of data. |
| Flatten Label   | Some networks use flat data for training. Provides data that is collapsed in height * width dimension. |
| Ordering        | Flag to specify if the data should be in channel_last or channel_first format. |

#### Data Iterator

The iterator can be used to traverse all data in a directory (eg. for inference). It provides data without the batchsize dimension.
It is only possible to create an iterator if the generator base was created with one data_path.

| Parameter       | Meaning  |
|-----------------|----------|
| Structure       | Structure of the returned data (sequence, stacked, or pair). See Structures for more information. |
| Labeled         | Flag to specify if only images (False) or images and labeles (True) are provided. |
| Cropped         | Flag to specify if the data should be cropped. Will use the same window for all data provided by this iterator. |
| Ordering        | Flag to specify if the data should be in channel_last or channel_first format. |

### Usage

```
from generator.generator_image import AM2018ImageGenerator

gen = AM2018ImageGenerator(['test/data/image/',], (200, 200, 3), (100, 100, 4), offset=1)
train_gen = gen.generator(batch_size=4, structure='sequence', labeled=True)

model.train_generator(train_gen, ...)
```

For training/validation splits use the split parameter
```
train_gen = gen.generator(batch_size=4, structure='sequence', labeled=True)

x, y = next(train_gen)
x.shape # [4, 3, 200, 200, 1]
y.shape # [4, 3, 200, 200, 4]
```

### Shapes

- input_shape in format of [height, width, stack_size] determines the size of the provided images, while stack_size defines the number of images stacked on top of each other (if applicable).
- output_shape in format of [height, width, num_classes] determines the size of the provided labels, while num_classes indicates how many classes from the data will be used. 

### Structures

- 'sequence' will provide images and labels in a sequence, with stack_size images/labels stacked among the first dimension ([sz, h, w, c], [sz, h, w, nc]).
- 'stacked' will provide images in a stacked way, but a single label for the following frame ([sz, h, w, c], [h, w, nc])
- 'pair' will provide a pair of an image and the corresponding label with offset ([h, w, c], [h, w, nc])

The offset between image and label data can be set in the offset parameter when creating the generator base. 

## Loader

The loaders get acquired by the generator and provide means of opening various formats of data.

Every loader has to implement the `_get_image()` and `_get_label()` methods, as well as `discover_data` for gathering the
data from the provided data_paths. 

`_get_image()` defines how to access the input data for a file name (the format is arbitrary and set in `discover_data`. Look into the `ImageLoader` class
to see how working with string replacements for accessing folder structure can be handled).
`_get_label()` defines how to access the label data for a file name.

