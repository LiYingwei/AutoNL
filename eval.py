import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from absl import app
from tensorpack import dataset, DataFromList, MultiThreadMapData, BatchData

import models
from args import FLAGS

CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_val_dataflow(datadir, parallel=1):
    assert datadir is not None
    ds = dataset.ILSVRC12Files(datadir, 'val', shuffle=False)

    def mapf(dp):
        fname, cls = dp
        with open(fname, "rb") as f:
            im_bytes = f.read()
        return im_bytes, cls

    ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=min(2000, ds.size()), strict=True)
    return ds


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

    return image


def preprocess_for_eval(image_bytes, image_size, scope=None):
    with tf.name_scope(scope, 'eval_image', [image_bytes, image_size, image_size]):
        image = _decode_and_center_crop(image_bytes, image_size)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image


def main(unused_argv):
    # get input
    image_ph = tf.placeholder(tf.string)
    image_proc = preprocess_for_eval(image_ph, FLAGS.input_image_size)
    images = tf.expand_dims(image_proc, 0)
    images -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype)
    images /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype)

    override_params = {'data_format': 'channels_last', 'num_classes': 1000}
    logits, _, _ = models.build_model(
        images, training=False,
        override_params=override_params,
        arch=FLAGS.arch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ckpt_path = os.path.join(FLAGS.model_dir, "bestmodel.ckpt")
    if not os.path.exists(ckpt_path + ".data-00000-of-00001"):
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    global_step = tf.train.get_global_step()
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)
    ema_vars = list(set(ema_vars))
    restore_vars_dict = ema.variables_to_restore(ema_vars)
    ckpt_restorer = tf.train.Saver(restore_vars_dict)
    ckpt_restorer.restore(sess, ckpt_path)

    c1, c5 = 0, 0
    ds = get_val_dataflow(os.path.join(FLAGS.valdir, ".."))
    ds.reset_state()
    preds = []
    labs = []
    for i, (image, label) in enumerate(ds):
        # image, label = images[0], labels[0]
        logits_val = sess.run(logits, feed_dict={image_ph: image})
        top5 = logits_val.squeeze().argsort()[::-1][:5]
        top1 = top5[0]
        if label == top1:
            c1 += 1
        if label in top5:
            c5 += 1
        preds.append(top1)
        labs.append(label)
        if (i + 1) % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Prec@1 {2:.1f}\t'
                  'Prec@5 {3:.1f}\t'.format(
                i + 1, len(ds), c1 / (i + 1.) * 100, c5 / (i + 1.) * 100))


if __name__ == '__main__':
    app.run(main)
