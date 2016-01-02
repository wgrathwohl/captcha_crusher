"""
houses neural network layers
"""
import tensorflow as tf
from utils import _variable_with_weight_decay, _variable_on_cpu, _activation_summary
from tensorflow.python.ops import control_flow_ops



def conv_layer(state_below, scope_name, n_inputs, n_outputs, filter_shape, stddev, wd):
    """
    A Standard convolutional layer
    """
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            wd=wd
        )
        conv = tf.nn.conv2d(state_below, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu("biases", [n_outputs], tf.constant_initializer(0.0))
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        output = tf.nn.relu(bias, name=scope.name)
        _activation_summary(output)
    return output

def batch_normalized_conv_layer(state_below, scope_name, n_inputs, n_outputs, filter_shape, stddev, wd, eps=.00001, test=False):
    """
    Convolutional layer with batch normalization
    """
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        conv = tf.nn.conv2d(state_below, kernel, [1, 1, 1, 1], padding='SAME')
        # get moments
        conv_mean, conv_variance = tf.nn.moments(conv, [0, 1, 2])
        # get mean and variance variables
        mean = _variable_on_cpu("bn_mean", [n_outputs], tf.constant_initializer(0.0), False)
        variance = _variable_on_cpu("bn_variance", [n_outputs], tf.constant_initializer(1.0), False)
        # assign the moments

        if not test:
            assign_mean = mean.assign(conv_mean)
            assign_variance = variance.assign(conv_variance)
            conv_bn = tf.mul((conv - conv_mean), tf.rsqrt(conv_variance + eps), name=scope.name+"_bn")
        else:
            conv_bn = tf.mul((conv - mean), tf.rsqrt(variance + eps), name=scope.name+"_bn")

        beta = _variable_on_cpu("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _variable_on_cpu("gamma", [n_outputs], tf.constant_initializer(1.0))
        bn = tf.add(tf.mul(conv_bn, gamma), beta)
        output = tf.nn.relu(bn, name=scope.name)
        if not test:
            output = control_flow_ops.with_dependencies(dependencies=[assign_mean, assign_variance], output_tensor=output)
        _activation_summary(output)

    return output

def batch_normalized_linear_layer(state_below, scope_name, n_inputs, n_outputs, stddev, wd, eps=.00001, test=False):
    """
    A linear layer with batch normalization
    """
    with tf.variable_scope(scope_name) as scope:
        weight = _variable_with_weight_decay(
            "weights", shape=[n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        act = tf.matmul(state_below, weight)
        # get moments
        act_mean, act_variance = tf.nn.moments(act, [0])
        # get mean and variance variables
        mean = _variable_on_cpu('bn_mean', [n_outputs], tf.constant_initializer(0.0), trainable=False)
        variance = _variable_on_cpu('bn_variance', [n_outputs], tf.constant_initializer(1.0), trainable=False)
        # assign the moments

        if not test:
            assign_mean = mean.assign(act_mean)
            assign_variance = variance.assign(act_variance)
            act_bn = tf.mul((act - act_mean), tf.rsqrt(act_variance + eps), name=scope.name+"_bn")
        else:
            act_bn = tf.mul((act - mean), tf.rsqrt(variance + eps), name=scope.name+"_bn")

        beta = _variable_on_cpu("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _variable_on_cpu("gamma", [n_outputs], tf.constant_initializer(1.0))
        bn = tf.add(tf.mul(act_bn, gamma), beta)
        output = tf.nn.relu(bn, name=scope.name)
        if not test:
            output = control_flow_ops.with_dependencies(dependencies=[assign_mean, assign_variance], output_tensor=output)
        _activation_summary(output)
    return output

def reshape_conv_layer(state_below):
    """
    Reshapes a conv layer activations to be linear. Assumes that batch dimension is 0
    """
    dims = state_below.get_shape().as_list()
    batch_size = dims[0]
    conv_dims = dims[1:]
    dim = 1
    for d in conv_dims:
        dim *= d
    reshape = tf.reshape(state_below, [batch_size, dim])
    return reshape, dim

def linear_layer(state_below, scope_name, n_inputs, n_outputs, stddev, wd):
    """
    Standard linear neural network layer
    """
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay(
            'weights', [n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        biases = _variable_on_cpu(
            'biases', [n_outputs], tf.constant_initializer(0.0)
        )
        output = tf.nn.xw_plus_b(state_below, weights, biases, name=scope.name)
        _activation_summary(output)
    return output

def linear_companion_logits(state_below, scope_name, n_inputs, n_classes, stddev, wd, b_init):
    """
    Attaches companion logits to a linear layer
    """
    return linear_layer(
        state_below, "{}_companion_logits".format(scope_name),
        n_inputs, n_classes, stddev, wd, b_init
    )

def conv_companion_logits(state_below, scope_name, n_classes, stddev, wd, b_init):
    """
    Attaches companion logits to a convolutional layer
    """
    reshaped, dim = reshape_conv_layer(state_below)
    return linear_layer(
        reshaped, "{}_companion_logits".format(scope_name),
        dim, n_classes, stddev, wd, b_init
    )

def global_pooling_layer(state_below, scope_name, pool_type="mean"):
    """
    Performs global pooling over a 2-d convolutional layer's output
    So BxHxWxD -> BxD 
    """

    if pool_type == "mean":
        f = tf.nn.avg_pool
    elif pool_type == "max":
        f = tf.nn.max_pool
    dims = state_below.get_shape().as_list()
    im_shape = dims[1:3]
    with tf.variable_scope(scope_name) as scope:
        pooled = f(
            state_below, ksize=[1, im_shape[0], im_shape[1], 1],
            strides=[1, im_shape[0], im_shape[1], 1], padding='SAME', name=scope.name
        )
        out_shape = pooled.get_shape().as_list()
        assert out_shape[1] == 1 and out_shape[2] == 1, out_shape
        num_channels = out_shape[-1]

        reshaped, dim = reshape_conv_layer(pooled)

        reshaped_shape = reshaped.get_shape().as_list()
        assert len(reshaped_shape) == 2, reshaped_shape
        assert reshaped_shape[-1]  == num_channels, reshaped_shape
        return reshaped

    return pooled

def global_pooling_output_layer(state_below, scope_name, num_inputs, num_outputs, filter_shape, stddev, wd, pool_type, test):
    """
    Output layer for fully convolutional network. Applies num_outputs filters and pools them to num_outputs logits
    """
    with tf.variable_scope(scope_name) as scope:
        conv_outputs = batch_normalized_conv_layer(
            state_below, "{}_conv_outputs".format(scope.name),
            num_inputs, num_outputs, filter_shape, stddev, wd, test=test
        )
        pooled = global_pooling_layer(conv_outputs, "{}_pooled".format(scope.name), pool_type)
    return pooled


def randomized_relu(state_below, irange, name=None, is_training=False):
    """
    Randomized rectified linear unit
    """
    if not is_training:
        # if testing, use standard relu
        return tf.nn.relu(state_below, name=name)
    else:
        # sample in irange around 1 for pos side
        pos_rand = tf.random_uniform(tf.shape(state_below), 1 - (irange / 2.0), 1 + (irange / 2.0))
        # sampel in irange around 0 for neg side
        neg_rand = tf.random_uniform(tf.shape(state_below), -irange / 2.0, irange / 2.0)

        pos = tf.mul(state_below,  pos_rand)
        neg = tf.mul(state_below, neg_rand)

        where_pos = tf.greater(state_below, 0.0)

        out = tf.select(where_pos, pos, neg, name=name)
        return out

