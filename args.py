from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', default=None,
                    help='The directory where the model and training/evaluation summaries are stored.')
flags.DEFINE_integer('input_image_size', default=224, help='Input image size.')
flags.DEFINE_float('moving_average_decay', default=0.9999, help='Moving average decay rate.')
flags.DEFINE_string('arch', default=None, help='The directory where the output of SP-NAS search is stored.')
flags.DEFINE_bool('use_keras', default=True, help='whether use keras')
flags.DEFINE_bool('nl_zero_init', default=True, help='whether use zero initializer to initialize non local bn')
flags.DEFINE_string('valdir', default='/mnt/cephfs_hl/uslabcv/yingwei/datasets/imagenet/val',
                    help='validation dir')