from data_generator.object_detection_2d_data_generator import DegenerateBatchError, DatasetError, DataGenerator
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

from copy import deepcopy
from PIL import Image
import warnings
import sklearn.utils
import numpy as np

import keras
from keras.utils import Sequence

class DataGeneratorSequence(Sequence):
    '''
    Generates batches of samples and (optionally) corresponding labels indefinitely.

    Can shuffle the samples consistently after each complete pass.

    Optionally takes a list of arbitrary image transformations to apply to the
    samples ad hoc.

    Arguments:
        batch_size (int, optional): The size of the batches to be generated.
        shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
            This option should always be `True` during training, but it can be useful to turn shuffling off
            for debugging or if you're using the generator for prediction.
        transformations (list, optional): A list of transformations that will be applied to the images and labels
            in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
            and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
            format.
        label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
            labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
            The general use case for this is to convert labels from their input format to a format that a given object
            detection model needs as its training targets.
        returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
            is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
            it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
            * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                matter whether or not you include this keyword in the set.
            * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
            * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                in training mode.
            * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
            * 'filenames': A list containing the file names (full paths) of the images in the batch.
            * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                are image IDs available.
            * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                usually treated to be neutral in a model evaluation.
            * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                that were applied to the original image to them. This makes it possible to let the model make predictions on a
                transformed image and then convert these predictions back to the original image. This is mostly relevant for
                evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                original images, not with respect to the transformed images. This means you will have to transform the predicted
                box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                image need to be applied in the order in which they are given in the respective list for that image.
            * 'original_images': A list containing the original images in the batch before any processing.
            * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                processing. Only available if ground truth is available.
            The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
            output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
        keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
            any transformations have been applied will be removed from the batch. If `True`, such images will be kept
            in the batch.
        degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
            `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
            after they were processed by transformations. Note that the generator checks for degenerate boxes after all
            transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
            Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
            are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

    __getitem__:
        The next batch as a tuple of items as defined by the `returns` argument.
    '''
    def __init__(self,
                 dataset,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove',
                 length = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations
        self.label_encoder = label_encoder
        self.returns = returns
        self.keep_images_without_gt = keep_images_without_gt
        self.degenerate_box_handling = degenerate_box_handling
        if length != None:
            self.length = length
        else:
            self.length = dataset.dataset_size

        if dataset.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if dataset.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [dataset.dataset_indices]
            if not (dataset.filenames is None):
                objects_to_shuffle.append(dataset.filenames)
            if not (dataset.labels is None):
                objects_to_shuffle.append(dataset.labels)
            if not (dataset.image_ids is None):
                objects_to_shuffle.append(dataset.image_ids)
            if not (dataset.eval_neutral is None):
                objects_to_shuffle.append(dataset.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=dataset.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (dataset.labels is None):
            for transform in transformations:
                transform.labels_format = self.dataset.labels_format

    def __len__(self):
        return self.length

    def __getitem__(self, current):
        while True:

            batch_X, batch_y = [], []

            current = current % self.dataset.dataset_size
            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = self.dataset.dataset_indices[current:current+self.batch_size]
            if not (self.dataset.images is None):
                for i in batch_indices:
                    batch_X.append(self.dataset.images[i])
                if not (self.dataset.filenames is None):
                    batch_filenames = self.dataset.filenames[current:current+self.batch_size]
                else:
                    batch_filenames = None
            elif not (self.dataset.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.dataset.hdf5_dataset['images'][i].reshape(self.dataset.hdf5_dataset['image_shapes'][i]))
                if not (self.dataset.filenames is None):
                    batch_filenames = self.dataset.filenames[current:current+self.batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.dataset.filenames[current:current+self.batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if not (self.dataset.labels is None):
                batch_y = deepcopy(self.dataset.labels[current:current+self.batch_size])
            else:
                batch_y = None

            if not (self.dataset.eval_neutral is None):
                batch_eval_neutral = self.dataset.eval_neutral[current:current+self.batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if not (self.dataset.image_ids is None):
                batch_image_ids = self.dataset.image_ids[current:current+self.batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in self.returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in self.returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.dataset.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not self.keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received.
                if self.transformations:

                    inverse_transforms = []

                    for transform in self.transformations:

                        if not (self.dataset.labels is None):

                            if ('inverse_transform' in self.returns) and ('return_inverter' in self.inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in self.returns) and ('return_inverter' in self.inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.dataset.labels is None):

                    xmin = self.dataset.labels_format['xmin']
                    ymin = self.dataset.labels_format['ymin']
                    xmax = self.dataset.labels_format['xmax']
                    ymax = self.dataset.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if self.degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif self.degenerate_box_handling == 'remove':
                            batch_y[i] = self.box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not self.keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.dataset.labels is None): batch_y.pop(j)
                    if not (self.dataset.image_ids is None): batch_image_ids.pop(j)
                    if not (self.dataset.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in self.returns: batch_original_images.pop(j)
                    if 'original_labels' in self.returns and not (self.dataset.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (self.label_encoder is None or self.dataset.labels is None):

                if ('matched_anchors' in self.returns) and isinstance(self.label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = self.label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = self.label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################

            ret = []
            if 'processed_images' in self.returns: ret.append(batch_X)
            if 'encoded_labels' in self.returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in self.returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in self.returns: ret.append(batch_y)
            if 'filenames' in self.returns: ret.append(batch_filenames)
            if 'image_ids' in self.returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in self.returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in self.returns: ret.append(batch_inverse_transforms)
            if 'original_images' in self.returns: ret.append(batch_original_images)
            if 'original_labels' in self.returns: ret.append(batch_original_labels)

            return ret

    def on_epoch_end(self):
        pass
