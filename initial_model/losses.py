"""
Implements loss new loss functions
"""
import tensorflow as tf
import numpy as np


def scale_labels(labels, margin=1):
    """
    Converts 0,1 labels to -margin,margin labels
    """
    return (2.0 * margin * labels) - margin


def hinge_loss(logits, labels, name=None):
    """
    Implements squared hinge loss
    """
    scaled_labels = scale_labels(labels)
    logits_labels = tf.mul(logits, scaled_labels)
    logits_labels_shifted = tf.minimum(logits_labels - 1.0, 0.0)
    squared_component_hinge_loss = tf.square(logits_labels_shifted)
    loss = tf.reduce_sum(squared_component_hinge_loss, 1)
    return loss


def aux_logits(logits, labels_oh, zero_value=1e5, name=None):
    """
    Adds extra penalty to ensure that outputs are not confused with the letters near them

    will be the average log probability of the labels that are in the label set but are not for the classifier

    NOTE: This didn't really work :( shucks 

    """
    mask = tf.ones(tf.shape(logits))
    for label in labels_oh:
        mask = tf.mul(mask, label - 1)
    mask = mask + 1

    to_zero = tf.mul(zero_value, (mask - 1))
    pos_part = tf.mul(mask, logits)
    return to_zero + pos_part


def np_aux_logits(logits, labels_oh, zero_value=1e5, name=None):
    """
    numpy version aux logits for testing
    """

    mask = np.ones(logits.shape)
    for label in labels_oh:
        mask = mask * (label - 1)
    mask = mask + 1
    # mask is now 1 at value of any of the labels
    # and zero if if the label does not exist in this batch
    to_zero = zero_value * (mask - 1)

    pos_part = mask * logits

    return to_zero + pos_part
