"""Tests for captcha input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform

import tensorflow as tf

import input_data
import matplotlib.pyplot as plt
import numpy as np
import losses
IMAGE_SIZE = 128


def make_oh(labels, num_out):
    oh = []
    for label in labels:
        o = [0.0 for i in range(num_out)]
        o[label] = 1.0
        oh.append(o)
    return np.array(oh)

def softmax(l):
    top = np.exp(l)
    return top / np.expand_dims(np.sum(top, axis=1), 1)


def np_aux_logits(logits, labels_oh, zero_value=1e5, name=None):

    mask = np.ones(logits.shape)
    for i, label in enumerate(labels_oh):
        mask = mask * (label - 1)
    mask = mask + 1
    # mask is now 1 at value of any of the labels
    # and zero if if the label does not exist in this batch
    to_zero = zero_value * (mask - 1)

    pos_part = mask * logits

    return to_zero + pos_part

# labels = [[0, 1, 2, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
# labels_oh = [make_oh(label, 3) for label in labels]
# logits = [[1, 0, 0], [0, 1, 0], [-10, 1, 10], [1, 1, 1]]
# logits = np.array(logits)
# print(np_aux_loss(logits, labels_oh, 0))
labels = [[0, 1], [1, 1], [0, 3]]
labels_oh = np.array([make_oh(label, 4) for label in labels])
logits = [[10, -11, -12, -31], [10, -11, -12, -31]]
ex =     [[10, -11, -1e5, -1e5], [-1e5, -11, -1e5, -31]]
logits = np.array(logits)
v = np_aux_logits(logits, labels_oh)






class InputTest(tf.test.TestCase):

    def _write_10_classes(self):
        # for 10 classes
        fns = []
        for i in range(10):
            # make 10 examples per class
            records = [self._record(i, i, j, i*j) for j in range(10)]
            contents = "".join([record for record, _ in records])
            expected = [expected for _, expected in records]
            filename = os.path.join(self.get_temp_dir(), "food-test-%d" % i)
            # writes one data file per class. Each file contains only 1 type
            # of example
            with open(filename, "w") as f:
                f.write(contents)
            fns.append(filename)
        return fns

    def _record(self, label, red, green, blue):
        image_size = IMAGE_SIZE * IMAGE_SIZE
        record = "%s%s%s%s" % (chr(label), chr(red) * image_size,
                               chr(green) * image_size, chr(blue) * image_size)
        expected = [[[red, green, blue]] * IMAGE_SIZE] * IMAGE_SIZE
        return record, expected

    def testHingeLoss(self):
      labels_py = [[0, 1], [1, 1], [0, 3]]
      labels_py = np.array([make_oh(label, 4) for label in labels_py])
      logits_py = [[10, -11, -12, -31], [10, -11, -12, -31]]
      masked_logits_py = [[10, -11, -1e5, -1e5], [-1e5, -11, -1e5, -31]]

      with self.test_session():
          # Initialize variables for numpy implementation.
          labels_np = [np.array(label, dtype=np.float32) for label in labels_py]
          logits_np = np.array(logits_py, dtype=np.float32)
          masked_logits_np = np.array(masked_logits_py, dtype=np.float32)

          labels = [tf.Variable(label) for label in labels_np]
          logits = tf.Variable(logits_np)
          masked_logits = losses.aux_logits(logits, labels)

          tf.initialize_all_variables().run()

          mle = masked_logits.eval()
          #print(mle, "MLE")
          self.assertAllEqual(mle, v)
        
      


if __name__ == "__main__":
    tf.test.main()
