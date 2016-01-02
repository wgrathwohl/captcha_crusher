"""Routine for decoding the captcha binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
import tensorflow as tf


def read_string(filename_queue, im_size=128, num_channels=3, label_bytes=1, num_labels=1):
    """Reads and parses string examples from a class of data files.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (d)
        width: number of columns in the result (d)
        depth: number of color channels in the result (num_channels)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..255.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    if isinstance(im_size, int):
    	  im_size = (im_size, im_size)
    assert isinstance(im_size, tuple), im_size

    class ReadRecord(object):
        pass
    result = ReadRecord()

    label_bytes = label_bytes * num_labels
    result.height = im_size[0]
    result.width = im_size[1]
    result.depth = num_channels
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the captcha format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    return key, value


def string_to_data(value, im_size, num_channels=3, label_bytes=1):
    """
    string encoding of [label|image]
    """

    image_bytes = im_size * im_size * num_channels

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    uint8image = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [im_size, im_size, num_channels]
    )

    return label, uint8image


def string_to_data_multilabel(value, num_labels, im_size, num_channels, label_bytes):
    """
    string encoding of [label1|label2|label3|....||image]
    """
    assert isinstance(im_size, tuple)

    image_bytes = im_size[0] * im_size[1] * num_channels

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the labels, which we convert from uint8->int32.
    labels = [tf.cast(tf.slice(record_bytes, [label_bytes * i], [label_bytes]), tf.int32) for i in range(num_labels)]
    all_labels = tf.concat(0, labels)  # tf.Variable(tf.zeros([num_labels]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    uint8image = tf.reshape(
        tf.slice(record_bytes, [label_bytes * num_labels], [image_bytes]),
        [im_size[0], im_size[1], num_channels]
    )

    return all_labels, uint8image
