"""
Contains training and testing utility functions
"""

import tensorflow as tf
import re


# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def get_saver(moving_average_decay, nontrainable_restore_names=None):
    """
    Gets the saver that restores the variavles for testing

    by default, restores to moving exponential average versions of
    trainable variables

    if nontrainable_restore_names is set, then restores
    nontrainable variables that match (this can be used
    for restoring mean averages for batch normalization)
    """
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay
    )
    variables_to_restore = {}
    for v in tf.all_variables():
        # if the variable is trainable or its name has the desird substring
        if v in tf.trainable_variables() or nontrainable_restore_names is not None and nontrainable_restore_names in v.name:
            print(v.name)
            restore_name = variable_averages.average_name(v)
        else:
            restore_name = v.op.name
        variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)
    return saver


def microsoft_initilization_std(shape):
    """
    Convolution layer initialization as described in:
    http://arxiv.org/pdf/1502.01852v1.pdf
    """
    if len(shape) == 4:
        n = shape[0] * shape[1] * shape[3]
        return (2.0 / n)**.5
    elif len(shape) == 2:
        return (2.0 / shape[1])**.5
    else:
        assert False, "Only works on normal layers"


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, stddev="MSFT"):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    if stddev == "MSFT":
        # use microsoft initialization
        stddev = microsoft_initilization_std(shape)
    var = _variable_on_cpu(
        name, shape,
        tf.truncated_normal_initializer(stddev=stddev)
    )
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_loss_summaries(total_loss):
    """Add summaries for losses in the model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
