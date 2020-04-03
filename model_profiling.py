import tensorflow as tf

import custom_layers
import model_def


def module_profiling(self, input, output, verbose):
    """
    only support NHWC data format
    :param self:
    :param input:
    :param output:
    :param verbose:
    :return:
    """
    in_size = input.shape.as_list()
    out_size = output.shape.as_list()
    if isinstance(self, custom_layers.GroupedConv2D):
        n_macs = 0
        for _conv in self._convs:
            kernel_size = _conv.kernel.shape.as_list()
            n_macs += kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3] * out_size[1] * out_size[2]
    elif isinstance(self, custom_layers.MDConv):
        n_macs = 0
        for _conv in self._convs:
            kernel_size = _conv.depthwise_kernel.shape.as_list()
            n_macs += kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3] * out_size[1] * out_size[2]
    elif isinstance(self, tf.keras.layers.DepthwiseConv2D):
        kernel_size = self.depthwise_kernel.shape.as_list()
        n_macs = kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3] * out_size[1] * out_size[2]
    elif isinstance(self, tf.keras.layers.Conv2D):
        kernel_size = self.kernel.shape.as_list()
        n_macs = kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3] * out_size[1] * out_size[2]
    elif isinstance(self, tf.keras.layers.GlobalAveragePooling2D):
        assert in_size[-1] == out_size[-1]
        n_macs = in_size[1] * in_size[2] * in_size[3]
    elif isinstance(self, tf.keras.layers.Dense):
        n_macs = in_size[1] * out_size[1]
    elif self == tf.add:
        # n_macs = in_size[1] * in_size[2] * in_size[3]
        n_macs = 0  # other people don't take skip connections into consideration
    else:
        raise NotImplementedError
    return n_macs
