# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for MnesNet model.

[1] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Quoc V. Le
  MnasNet: Platform-Aware Neural Architecture Search for Mobile.
  arXiv:1807.11626
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import custom_layers
from args import FLAGS
from model_profiling import module_profiling
from tensorflow.python.keras import backend as K

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

# TODO(hongkuny): Consider rewrite an argument class with encoding/decoding.
BlockArgs = collections.namedtuple('BlockArgs', [
    'dw_ksize', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'non_local',
    'expand_ksize', 'project_ksize', 'swish',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.
  
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
  
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
  
    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
  
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.
  
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
  
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_multiplier
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


class MnasBlock(object):
    """A class of MnasNet Inveretd Residual Bottleneck.
  
    Attributes:
      has_se: boolean. Whether the block contains a Squeeze and Excitation layer
        inside.
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MnasNet block.
    
        Args:
          block_args: BlockArgs, arguments to create a MnasBlock.
          global_params: GlobalParams, a set of global parameters.
        """
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self.data_format = global_params.data_format
        assert global_params.data_format == 'channels_last'
        self._channel_axis = -1
        self._spatial_dims = [1, 2]
        self.has_se = (self._block_args.se_ratio is not None) and (
                self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)
        self._relu_fn = tf.nn.swish
        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds MnasNet block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = custom_layers.GroupedConv2D(
                filters,
                kernel_size=self._block_args.expand_ksize,
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False, data_format=self.data_format,
                use_keras=FLAGS.use_keras
            )
            self._bn0 = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
                fused=True)

        kernel_size = self._block_args.dw_ksize
        # Depth-wise convolution phase:
        self._depthwise_conv = custom_layers.MDConv(
            kernel_size,
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self.data_format,
            use_bias=False, dilated=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)

        if self.has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.keras.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True, data_format=self.data_format)
            self._se_expand = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True, data_format=self.data_format)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = custom_layers.GroupedConv2D(
            filters,
            kernel_size=self._block_args.project_ksize,
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False, data_format=self.data_format,
            use_keras=FLAGS.use_keras)
        self._bn2 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)
        if self._block_args.non_local:  # this line still work, even if this value becomes a float, or a list
            self._non_local_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False,
                data_format=self.data_format)

            self._non_local_bn = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
                gamma_initializer=tf.zeros_initializer() if FLAGS.nl_zero_init else tf.ones_initializer,  # this line is correct
                fused=True)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
    
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
    
        Returns:
          A output tensor, which should have the same shape as input.
        """
        macs = 0.
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        macs += module_profiling(tf.keras.layers.GlobalAveragePooling2D(), input_tensor, se_tensor, False)
        s_tensor = self._relu_fn(self._se_reduce(se_tensor))
        macs += module_profiling(self._se_reduce, se_tensor, s_tensor, False)
        e_tensor = self._se_expand(s_tensor)
        macs += module_profiling(self._se_expand, s_tensor, e_tensor, False)
        tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' % e_tensor.shape)
        return tf.sigmoid(e_tensor) * input_tensor, macs

    def _call_non_local(self, l, training=True, nl_ratio=1.0, nl_stride=1):
        def reduce_func(l, nl_stride):
            return l[:, ::nl_stride, ::nl_stride, :], 0

        total_macs = 0.
        tf.logging.info('Block input: %s shape: %s' % (l.name, l.shape))
        f, macs = non_local_op(l, embed=False, softmax=False, nl_ratio=nl_ratio, nl_stride=nl_stride,
                               reduce_func=reduce_func)
        total_macs += macs
        f_output = self._non_local_conv(f)
        macs = module_profiling(self._non_local_conv, f, f_output, False)
        total_macs += macs
        f = self._non_local_bn(f_output, training=training)
        l = l + f
        tf.logging.info('Non-local: %s shape: %s' % (l.name, l.shape))
        return l, total_macs

    def call(self, inputs, training=True):
        """Implementation of MnasBlock call().
    
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
    
        Returns:
          A output tensor.
        """
        total_macs = 0.
        tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            outputs_expand_conv = self._expand_conv(inputs)
            total_macs += module_profiling(self._expand_conv, inputs, outputs_expand_conv, False)  # compute macs
            x = self._relu_fn(self._bn0(outputs_expand_conv, training=training))
        else:
            x = inputs
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        outputs_depthwise_conv = self._depthwise_conv(x)
        total_macs += module_profiling(self._depthwise_conv, x, outputs_depthwise_conv, False)  # compute macs
        x = self._relu_fn(self._bn1(outputs_depthwise_conv, training=training))
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

        if self.has_se:
            with tf.variable_scope('se'):
                x, macs = self._call_se(x)
                total_macs += macs
                # raise NotImplementedError

        self.endpoints = {'expansion_output': x}

        outputs_project_conv = self._project_conv(x)
        total_macs += module_profiling(self._project_conv, x, outputs_project_conv, False)  # compute macs
        x = self._bn2(outputs_project_conv, training=training)

        if self._block_args.non_local:
            with tf.variable_scope('nl'):
                x, macs = self._call_non_local(x, training=training, nl_ratio=self._block_args.non_local[0],
                                               nl_stride=self._block_args.non_local[1])
                total_macs += macs

        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                x = tf.add(x, inputs)
                total_macs += module_profiling(tf.add, x, inputs, False)  # compute macs
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))

        return x, total_macs


def non_local_op(l, embed, softmax, nl_ratio=1.0, nl_stride=1, reduce_func=None):
    H, W, n_in = l.shape.as_list()[1:]
    reduced_HW = (H // nl_stride) * (W // nl_stride)
    if embed:
        raise NotImplementedError
    else:
        if nl_stride == 1:
            l_reduced = l
            reduce_macs = 0
        else:
            assert reduce_func is not None
            l_reduced, reduce_macs = reduce_func(l, nl_stride)

        theta, phi, g = l[:, :, :, :int(nl_ratio * n_in)], l_reduced[:, :, :, :int(nl_ratio * n_in)], l_reduced

    if (H * W) * reduced_HW * n_in * (1 + nl_ratio) < (
            H * W) * n_in ** 2 * nl_ratio + reduced_HW * n_in ** 2 * nl_ratio or softmax:
        f = tf.einsum('nabi,ncdi->nabcd', theta, phi)
        if softmax:
            raise NotImplementedError
        f = tf.einsum('nabcd,ncdi->nabi', f, g)
        # macs = (H * W) ** 2 * n_in * nl_ratio + (H * W) ** 2 * n_in
        macs = (H * W) * reduced_HW * n_in * (1 + nl_ratio)
    else:
        f = tf.einsum('nhwi,nhwj->nij', phi, g)
        f = tf.einsum('nij,nhwi->nhwj', f, theta)
        # macs = (H * W) * n_in ** 2 * 2 * nl_ratio
        macs = (H * W) * n_in ** 2 * nl_ratio + reduced_HW * n_in ** 2 * nl_ratio
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    return tf.reshape(f, tf.shape(l)), macs + reduce_macs


class MnasNetModel(tf.keras.Model):
    """A class implements tf.keras.Model for MnesNet model.

      Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `MnasNetModel` instance.

        Args:
          blocks_args: A list of BlockArgs to construct MnasNet block modules.
          global_params: GlobalParams, a set of global parameters.

        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(MnasNetModel, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        # Use relu in default for head and stem.
        self._relu_fn = tf.nn.swish
        self.endpoints = None
        self._build()

    def _build(self):
        """Builds a MnasNet model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MnasBlock(block_args, self._global_params))  # removed kernel mask here
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(MnasBlock(block_args, self._global_params))  # removed kernel mask here

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        stem_size = 32
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=round_filters(stem_size, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False, data_format=self._global_params.data_format
        )
        self._bn0 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        # Head part.
        self._conv_head = tf.keras.layers.Conv2D(
            filters=1280,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False, data_format=self._global_params.data_format)
        self._bn1 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)
        self._fc = tf.keras.layers.Dense(
            self._global_params.num_classes,
            kernel_initializer=dense_kernel_initializer)

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True):
        """Implementation of MnasNetModel call().

        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.

        Returns:
          output tensors.
        """
        outputs = None
        self.endpoints = {}
        total_macs = 0.
        # Calls Stem layers
        with tf.variable_scope('mnas_stem'):
            outputs_conv_stem = self._conv_stem(inputs)
            total_macs += module_profiling(self._conv_stem, inputs, outputs_conv_stem, False)  # compute macs
            outputs = self._relu_fn(self._bn0(outputs_conv_stem, training=training))

        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
        self.endpoints['stem'] = outputs
        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            with tf.variable_scope('mnas_blocks_%s' % idx):
                outputs, n_macs = block.call(outputs, training=training)
                total_macs += n_macs
                self.endpoints['block_%s' % idx] = outputs
                if block.endpoints:
                    for k, v in six.iteritems(block.endpoints):
                        self.endpoints['block_%s/%s' % (idx, k)] = v
        # Calls final layers and returns logits.
        with tf.variable_scope('mnas_head'):
            outputs_conv_head = self._conv_head(outputs)
            total_macs += module_profiling(self._conv_head, outputs, outputs_conv_head, False)  # compute macs
            outputs = self._relu_fn(self._bn1(outputs_conv_head, training=training))
            self.endpoints['cam_feature'] = outputs

            outputs_avg_pooling = self._avg_pooling(outputs)
            total_macs += module_profiling(self._avg_pooling, outputs, outputs_avg_pooling, False)  # compute macs
            outputs = outputs_avg_pooling

            if self._dropout:
                outputs = self._dropout(outputs, training=training)

            outputs_fc = self._fc(outputs)
            total_macs += module_profiling(self._fc, outputs, outputs_fc, False)  # compute macs
            outputs = outputs_fc
            self.endpoints['head'] = outputs
            self.endpoints['_fc'] = self._fc
        return outputs, total_macs
