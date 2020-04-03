# author: dstamoulis
#
# This code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Creates the ConvNet found model by parsing the NAS-decision values 
   from the provided NAS-search output dir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import numpy as np

import model_def
from args import FLAGS


class MnasNetDecoder(object):
    """A class of MnasNet decoder to get model configuration."""

    def _decode_block_string(self, block_string):
        """Gets a MNasNet block through a string notation of arguments.
    
        E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
        k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
        o - output filters, se - squeeze/excitation ratio
    
        Args:
          block_string: a string, a string representation of block arguments.
    
        Returns:
          A BlockArgs instance.
        Raises:
          ValueError: if the strides option is not correctly specified.
        """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            if op == 'nonlocal':
                op = 'nonlocal1.0'
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        def _parse_ksize(ss):
            return [int(k) for k in ss.split('.')]

        def _parse_nonlocal(ss):
            ss = ss.split("-")
            if len(ss) == 2:
                return [float(ss[0]), int(ss[1])]
            else:
                assert len(ss) == 1
                return [float(ss[0]), 1]

        BlockArgs = model_def.BlockArgs

        return BlockArgs(
            expand_ksize=_parse_ksize(options['a']) if 'a' in options else [1],
            dw_ksize=_parse_ksize(options['k']),
            project_ksize=_parse_ksize(options['p']) if 'p' in options else [1],
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])],
            swish=('sw' in block_string),
            non_local=_parse_nonlocal(options['nonlocal']) if 'nonlocal' in options else 0.0
        )

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
    
        Args:
          string_list: a list of strings, each string is a notation of MnasNet
            block.
    
        Returns:
          A list of namedtuples to represent MnasNet blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args


def parse_netarch_string(blocks_args, depth_multiplier=None):
    decoder = MnasNetDecoder()
    global_params = model_def.GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        num_classes=1000,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None)
    return decoder.decode(blocks_args), global_params


def build_model(images, training, override_params=None, arch=None):
    """A helper functiion to creates a ConvNet model and returns predicted logits.
  
    Args:
      images: input images tensor.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        model_def.GlobalParams.
  
    Returns:
      logits: the logits tensor of classes.
      endpoints: the endpoints for each layer.
    Raises:
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)
    assert os.path.isfile(arch)
    with open(arch, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    blocks_args, global_params = parse_netarch_string(lines)

    if override_params:
        global_params = global_params._replace(**override_params)

    with tf.variable_scope('single-path'):
        model = model_def.MnasNetModel(blocks_args, global_params)
        logits, macs = model(images, training=training)
        macs /= 1e6  # macs to M

    logits = tf.identity(logits, 'logits')

    return logits, model.endpoints, macs
