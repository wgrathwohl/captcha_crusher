"""
A binary to train captcha using a single GPU or cpu.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from layers import *
import input_data
from utils import _add_loss_summaries
import losses

FLAGS = tf.app.flags.FLAGS
# A hack to stop conflicting batch sizes
try:
    print(FLAGS.batch_size)
except:
    tf.app.flags.DEFINE_integer(
        'batch_size', 128,
        """Number of images to process in a batch."""
    )
tf.app.flags.DEFINE_string(
    'train_dir', '/home/will/Desktop/hard_captcha_mean_sub_rrelu3',
    """Directory where to write event logs and checkpoint."""
)
tf.app.flags.DEFINE_integer(
    'max_steps', 1000000,
    """Number of batches to run."""
)
tf.app.flags.DEFINE_boolean(
    'log_device_placement', False,
    """Whether to log device placement."""
)


COMPANION_LOSS = False
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
USE_CHECKPOINT = True
CHECKPOINT_PATH = '/home/will/Desktop/hard_captcha_mean_sub_rrelu2/model.ckpt-600'
NUM_EPOCHS_PER_DECAY = 3.0
INITIAL_LEARNING_RATE = 0.00001     # Initial learning rate.
NUM_CLASSES = 36
TRAIN_FNAMES = [
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_train3',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_train',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_train4',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_train3'
]
TEST_FNAMES = [
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_test',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_test2',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_test3',
    '/home/will/code/tf/data/serialized_hard_captchas/hard_captchas_test4'
]
COMP_LOSS_WEIGHT = 0.0
FULL_IMAGE_SIZE = (70, 200)
IMAGE_SIZE = (60, 190)
NUM_CHANNELS = 1
LABEL_BYTES = 1
NUM_LABELS = 6
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

AUX_LOSS_COEFF = .1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30087 + 58600 + 44867 + 11352
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3340 + 6511 + 4985 + 1358


def inference_captcha_easy(images, is_training, companion_loss=False, num_classes=NUM_CLASSES):
    """
    Build the captcha model. For the easy captchas. Obtains 99.7% accuracy for fully correct captchas

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        Logits.
    """

    test = not is_training
    # conv1
    n_filters_conv1 = 64
    conv1 = batch_normalized_conv_layer(images, "conv1", 1, n_filters_conv1, [5, 5], "MSFT", 0.004, test=test)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 64
    conv2 = batch_normalized_conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004, test=test)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # conv3
    n_filters_conv3 = 64
    conv3 = batch_normalized_conv_layer(pool2, "conv3", n_filters_conv2, n_filters_conv3, [3, 3], "MSFT", 0.004, test=test)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # reshape pool3 to linear
    reshape, dim = reshape_conv_layer(pool3)

    # local3
    n_outputs_local_3 = 500
    local3 = batch_normalized_linear_layer(reshape, "local3", dim, n_outputs_local_3, .01, 0.004, test=test)

    # local4
    n_outputs_local_4 = 500
    local4 = batch_normalized_linear_layer(local3, "local4", n_outputs_local_3, n_outputs_local_4, .01, .004, test=test)

    # one for each character
    softmax_linear1 = batch_normalized_linear_layer(local4, "softmax_linear1", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear2 = batch_normalized_linear_layer(local4, "softmax_linear2", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear3 = batch_normalized_linear_layer(local4, "softmax_linear3", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear4 = batch_normalized_linear_layer(local4, "softmax_linear4", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear5 = batch_normalized_linear_layer(local4, "softmax_linear5", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear6 = batch_normalized_linear_layer(local4, "softmax_linear6", n_outputs_local_4, num_classes, .001, 0.004, test=test)

    softmax_linear = [softmax_linear1, softmax_linear2, softmax_linear3, softmax_linear4, softmax_linear5, softmax_linear6]

    comp_logits = []
    if companion_loss:
        comp_logits.extend(
            [
                conv_companion_logits(pool1, "conv1", num_classes, .001, .004, .01),
                conv_companion_logits(pool2, "conv2", num_classes, .001, .004, .01),
                conv_companion_logits(pool3, "conv3", num_classes, .001, .004, .01),
                linear_companion_logits(local3, "local3", n_outputs_local_3, num_classes, .001, .004, .01),
                linear_companion_logits(local4, "local4", n_outputs_local_4, num_classes, .001, .004, .01)
            ]
        )

    return softmax_linear, comp_logits


def inference_captcha(images, is_training, companion_loss=False, num_classes=NUM_CLASSES):
    """
    Build the captcha model. For the main hard captcahs. Obtains 95% accuracy on all single outputs
    and 83% fully correct

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        Logits.
    """

    test = not is_training
    # conv1
    n_filters_conv1 = 32
    conv1 = batch_normalized_conv_layer(images, "conv1", 1, n_filters_conv1, [5, 5], "MSFT", 0.004, test=test)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 64
    conv2 = batch_normalized_conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004, test=test)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # conv3
    n_filters_conv3 = 128
    conv3 = batch_normalized_conv_layer(pool2, "conv3", n_filters_conv2, n_filters_conv3, [3, 3], "MSFT", 0.004, test=test)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # reshape pool3 to linear
    reshape, dim = reshape_conv_layer(pool3)

    # local3
    n_outputs_local_3 = 500
    local3 = batch_normalized_linear_layer(reshape, "local3", dim, n_outputs_local_3, .01, 0.004, test=test)

    # local4
    n_outputs_local_4 = 500
    local4 = batch_normalized_linear_layer(local3, "local4", n_outputs_local_3, n_outputs_local_4, .01, .004, test=test)

    # one for each character
    softmax_linear1 = batch_normalized_linear_layer(local4, "softmax_linear1", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear2 = batch_normalized_linear_layer(local4, "softmax_linear2", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear3 = batch_normalized_linear_layer(local4, "softmax_linear3", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear4 = batch_normalized_linear_layer(local4, "softmax_linear4", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear5 = batch_normalized_linear_layer(local4, "softmax_linear5", n_outputs_local_4, num_classes, .001, 0.004, test=test)
    softmax_linear6 = batch_normalized_linear_layer(local4, "softmax_linear6", n_outputs_local_4, num_classes, .001, 0.004, test=test)

    softmax_linear = [softmax_linear1, softmax_linear2, softmax_linear3, softmax_linear4, softmax_linear5, softmax_linear6]

    comp_logits = []
    if companion_loss:
        comp_logits.extend(
            [
                conv_companion_logits(pool1, "conv1", num_classes, .001, .004, .01),
                conv_companion_logits(pool2, "conv2", num_classes, .001, .004, .01),
                conv_companion_logits(pool3, "conv3", num_classes, .001, .004, .01),
                linear_companion_logits(local3, "local3", n_outputs_local_3, num_classes, .001, .004, .01),
                linear_companion_logits(local4, "local4", n_outputs_local_4, num_classes, .001, .004, .01)
            ]
        )


    return softmax_linear, comp_logits

def inference_captcha_fully_conv(images, is_training, companion_loss=False, num_classes=NUM_CLASSES):
    """
    Build the captcha model. Fully convolutional version with mean pooling layers. Obtains 96% accuracy on all single outputs
    and 86% fully correct

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        Logits.
    """

    test = not is_training
    # conv1
    n_filters_conv1 = 32
    conv1 = batch_normalized_conv_layer(images, "conv1", 1, n_filters_conv1, [5, 5], "MSFT", 0.004, test=test)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 64
    conv2 = batch_normalized_conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004, test=test)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # conv3
    n_filters_conv3 = 128
    conv3 = batch_normalized_conv_layer(pool2, "conv3", n_filters_conv2, n_filters_conv3, [3, 3], "MSFT", 0.004, test=test)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # local3
    n_outputs_conv4 = 256
    conv4 = batch_normalized_conv_layer(pool3, "conv4", n_filters_conv3, n_outputs_conv4, [3, 3], "MSFT", 0.004, test=test)

    # local4
    n_outputs_conv5 = 256
    conv5 = batch_normalized_conv_layer(conv4, "conv5", n_outputs_conv4, n_outputs_conv5, [3, 3], "MSFT", .004, test=test)

    # one for each character
    softmax_linear1 = global_pooling_output_layer(conv5, "softmax_linear1", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)
    softmax_linear2 = global_pooling_output_layer(conv5, "softmax_linear2", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)
    softmax_linear3 = global_pooling_output_layer(conv5, "softmax_linear3", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)
    softmax_linear4 = global_pooling_output_layer(conv5, "softmax_linear4", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)
    softmax_linear5 = global_pooling_output_layer(conv5, "softmax_linear5", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)
    softmax_linear6 = global_pooling_output_layer(conv5, "softmax_linear6", n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, "mean", test=test)

    softmax_linear = [softmax_linear1, softmax_linear2, softmax_linear3, softmax_linear4, softmax_linear5, softmax_linear6]

    comp_logits = []
    if companion_loss:
        comp_logits.extend(
            [
                conv_companion_logits(pool1, "conv1", num_classes, .001, .004, .01),
                conv_companion_logits(pool2, "conv2", num_classes, .001, .004, .01),
                conv_companion_logits(pool3, "conv3", num_classes, .001, .004, .01),
                linear_companion_logits(local3, "local3", n_outputs_local_3, num_classes, .001, .004, .01),
                linear_companion_logits(local4, "local4", n_outputs_local_4, num_classes, .001, .004, .01)
            ]
        )

    return softmax_linear, comp_logits


def inference_captcha_mean_subtracted(images, is_training, companion_loss=False, num_classes=NUM_CLASSES):
    """
    Build the captcha model. Fully convolutional version with mean pooling layers
    This version takes the output filter maps takes the average of them and subtracts that from all of the output
    tensors. Obtains 97.8% accuracy on all single outputs and 90% fully correct


    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        Logits.
    """

    test = not is_training
    # conv1
    n_filters_conv1 = 32
    conv1 = batch_normalized_conv_layer(images, "conv1", 1, n_filters_conv1, [5, 5], "MSFT", 0.004, test=test)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 64
    conv2 = batch_normalized_conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004, test=test)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # conv3
    n_filters_conv3 = 128
    conv3 = batch_normalized_conv_layer(pool2, "conv3", n_filters_conv2, n_filters_conv3, [3, 3], "MSFT", 0.004, test=test)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # local3
    n_outputs_conv4 = 256
    conv4 = batch_normalized_conv_layer(pool3, "conv4", n_filters_conv3, n_outputs_conv4, [3, 3], "MSFT", 0.004, test=test)

    # local4
    n_outputs_conv5 = 256
    conv5 = batch_normalized_conv_layer(conv4, "conv5", n_outputs_conv4, n_outputs_conv5, [3, 3], "MSFT", .004, test=test)

    raw_output_tensors = [
        batch_normalized_conv_layer(conv5, "output_tensor%d_raw" % (i+1), n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, test=test)
        for i in range(6)
    ]

    output_tensor_mean = tf.add_n(raw_output_tensors) * (1.0/6.0)

    output_tensors = [
        output_tensor - output_tensor_mean
        for output_tensor in raw_output_tensors
    ]

    softmax_linear = [
        global_pooling_layer(output_tensor, "softmax_linear%d" % (i+1))
        for i, output_tensor in enumerate(output_tensors)
    ]

    comp_logits = []
    if companion_loss:
        comp_logits.extend(
            [
                conv_companion_logits(pool1, "conv1", num_classes, .001, .004, .01),
                conv_companion_logits(pool2, "conv2", num_classes, .001, .004, .01),
                conv_companion_logits(pool3, "conv3", num_classes, .001, .004, .01),
                linear_companion_logits(local3, "local3", n_outputs_local_3, num_classes, .001, .004, .01),
                linear_companion_logits(local4, "local4", n_outputs_local_4, num_classes, .001, .004, .01)
            ]
        )
    return softmax_linear, comp_logits


def inference_captcha_mean_subtracted_residual(images, is_training, companion_loss=False, num_classes=NUM_CLASSES):
    """
    Will implement residual learning from the recent MSR paper.
    NOT FINISHED!!!!!!!


    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        Logits.
    """

    test = not is_training
    # conv1
    n_filters_conv1 = 32
    conv1 = batch_normalized_conv_layer(images, "conv1", 1, n_filters_conv1, [5, 5], "MSFT", 0.004, test=test)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 64
    conv2 = batch_normalized_conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004, test=test)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # conv3
    n_filters_conv3 = 128
    conv3 = batch_normalized_conv_layer(pool2, "conv3", n_filters_conv2, n_filters_conv3, [3, 3], "MSFT", 0.004, test=test)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # local3
    n_outputs_conv4 = 256
    conv4 = batch_normalized_conv_layer(pool3, "conv4", n_filters_conv3, n_outputs_conv4, [3, 3], "MSFT", 0.004, test=test)

    # local4
    n_outputs_conv5 = 256
    conv5 = batch_normalized_conv_layer(conv4, "conv5", n_outputs_conv4, n_outputs_conv5, [3, 3], "MSFT", .004, test=test)

    conv5_out = conv5 + conv4

    raw_output_tensors = [
        batch_normalized_conv_layer(conv5_out, "output_tensor%d_raw" % (i+1), n_outputs_conv5, num_classes, [3, 3], "MSFT", .004, test=test)
        for i in range(6)
    ]

    output_tensor_mean = tf.add_n(raw_output_tensors) * (1.0/6.0)

    output_tensors = [
        output_tensor - output_tensor_mean
        for output_tensor in raw_output_tensors
    ]

    softmax_linear = [
        global_pooling_layer(output_tensor, "softmax_linear%d" % (i+1))
        for i, output_tensor in enumerate(output_tensors)
    ]

    comp_logits = []
    if companion_loss:
        comp_logits.extend(
            [
                conv_companion_logits(pool1, "conv1", num_classes, .001, .004, .01),
                conv_companion_logits(pool2, "conv2", num_classes, .001, .004, .01),
                conv_companion_logits(pool3, "conv3", num_classes, .001, .004, .01),
                linear_companion_logits(local3, "local3", n_outputs_local_3, num_classes, .001, .004, .01),
                linear_companion_logits(local4, "local4", n_outputs_local_4, num_classes, .001, .004, .01)
            ]
        )


    return softmax_linear, comp_logits


def captcha_loss(logits, all_labels, all_comp_logits, comp_loss_weight):
    """
    Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
        logits: list of Logits from inference_captcha().
        labels: list of Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, NUM_CLASSES].
    labels = tf.split(1, 6, all_labels)
    sparse_labels = [tf.reshape(label, [FLAGS.batch_size, 1]) for label in labels]
    indices = tf.reshape(tf.range(0, FLAGS.batch_size, 1), [FLAGS.batch_size, 1])
    concateds = [tf.concat(1, [indices, sparse_label]) for sparse_label in sparse_labels]
    dense_labels = [tf.sparse_to_dense(concated,
                                      [FLAGS.batch_size, NUM_CLASSES],
                                      1.0, 0.0) for concated in concateds]

    # Calculate the average cross entropy loss across the batch.
    # add it to total loss
    cross_entropies = []
    for i, dense_label in enumerate(dense_labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits[i], dense_label, name='cross_entropy_per_example_%d' % i)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_%d' % i)
        tf.add_to_collection('losses', cross_entropy_mean)

    """
    This stuff here was supposed to add an extra loss to force output i to not predict the value of
    label i+1 or label i-1, but it didnt really help :(
    """
    # aux_cross_entropies = []
    # for dense_label, logit in zip(dense_labels, logits):
    #   aux_logits = losses.aux_logits(logit, dense_labels)
    #   aux_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #       logit, dense_label, name='aux_cross_entropy_per_example_%d' % i)
    #   aux_cross_entropy_mean = tf.reduce_mean(aux_cross_entropy, name="aux_cross_entropy_%d" % i)
    #   tf.add_to_collection('losses', AUX_LOSS_COEFF * aux_cross_entropy_mean)


    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _generate_image_and_label_batch(image, label, min_queue_examples):
    """
    Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [FLAGS.batch_size, NUM_LABELS])


def distorted_inputs():
    """
    Construct distorted input for captcha training using the Reader ops.

    Raises:
        ValueError: if no data_dir

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = TRAIN_FNAMES

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    print("Using files {} for training".format(TRAIN_FNAMES))

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    key, string = input_data.read_string(filename_queue, FULL_IMAGE_SIZE, NUM_CHANNELS, LABEL_BYTES, NUM_LABELS)
    labels, im = input_data.string_to_data_multilabel(string, NUM_LABELS, FULL_IMAGE_SIZE, NUM_CHANNELS, LABEL_BYTES)
    reshaped_image = tf.cast(im, tf.float32)

    height = IMAGE_SIZE[0]
    width = IMAGE_SIZE[1]

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])

    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, labels,
                                           min_queue_examples)

def inputs(eval_data):
    """
    Construct input for captcha evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Raises:
        ValueError: if no data_dir

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    if not eval_data:
        filenames = TRAIN_FNAMES
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print("Using files {} for training".format(TRAIN_FNAMES))
    else:
        filenames = TEST_FNAMES
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        print("Using files {} for testing".format(TEST_FNAMES))

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    key, string = input_data.read_string(filename_queue, FULL_IMAGE_SIZE, NUM_CHANNELS, LABEL_BYTES, NUM_LABELS)
    labels, im = input_data.string_to_data_multilabel(string, NUM_LABELS, FULL_IMAGE_SIZE, NUM_CHANNELS, LABEL_BYTES)
    reshaped_image = tf.cast(im, tf.float32)

    height = IMAGE_SIZE[0]
    width = IMAGE_SIZE[1]

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.04
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, labels,
                                           min_queue_examples)


def captcha_train(total_loss, global_step):
    """
    Train captcha model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True
    )
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)


    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_to_average = list(set(tf.trainable_variables() + filter(lambda v: "_mean" in v.name or "_variance" in v.name, tf.all_variables())))
    variables_averages_op = variable_averages.apply(variables_to_average)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    """Train FOOD-101 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels
        images, labels = distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, comp_logits = inference_captcha_mean_subtracted(images, True, COMPANION_LOSS)

        # Calculate loss.
        loss = captcha_loss(logits, labels, comp_logits, COMP_LOSS_WEIGHT)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = captcha_train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        if USE_CHECKPOINT:
            saver.restore(sess, CHECKPOINT_PATH)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 50 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
