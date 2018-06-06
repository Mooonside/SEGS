import os
import tensorflow as tf

from datasets.pascal_voc_reader import get_dataset, get_next_batch, TRAIN_DIR
from datasets.pascal_voc_utils import pascal_voc_classes
from segs import deeplab_v3_plus, common_configure
from tf_ops.benchmarks import mAP, mIOU
from tf_ops.visualize import paint, compare
from tf_ops.wrap_ops import *
from tf_utils import partial_restore, add_gradient_summary, average_gradients,\
    add_var_summary, add_activation_summary, parse_device_name, add_iou_summary

arg_scope = tf.contrib.framework.arg_scope

LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

flags = tf.app.flags
flags = common_configure.DeepLabFlags(flags)
# pre settings
flags.DEFINE_string('data_dir', TRAIN_DIR, 'where training set is put')
flags.DEFINE_integer('reshape_height', 513, 'reshape height')
flags.DEFINE_integer('reshape_weight', 513, 'reshape weight')
flags.DEFINE_string('net_name', 'fcn8', 'which segmentation net to use')
flags.DEFINE_integer('num_classes', 21, '#classes')

# learning configs
flags.DEFINE_integer('epoch_num', 10, 'epoch_nums')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_float('weight_learning_rate', 1e-4, 'weight learning rate')
flags.DEFINE_float('bias_learning_rate', None, 'bias learning rate')
flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')
flags.DEFINE_float('learning_decay', 0.99, 'learning rate decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')

# deploy configs
flags.DEFINE_string('store_device', '/CPU:0', 'where to place the variables')
flags.DEFINE_string('run_device', '0', 'where to run the models')
flags.DEFINE_float('gpu_fraction', 0.8, 'gpu memory fraction')
flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
flags.DEFINE_float('weight_reg_scale', 4e-5, 'weight regularization scale')
flags.DEFINE_string('weight_reg_func', 'l2', 'use which func to regularize weight')
flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# model load & save configs
flags.DEFINE_string('summaries_dir', '/mnt/disk/chenyifeng/TF_Logs/SEGS/deeplabv3+/sgpu',
                           'where to store summary log')

flags.DEFINE_string('pretrained_ckpts', None,
                           'where to load pretrained model')

flags.DEFINE_string('last_ckpt', '/mnt/disk/chenyifeng/TF_Models/ptrain/SEGS/xception_voc_trainval',
                           'where to load last saved model')

flags.DEFINE_string('next_ckpt', '/mnt/disk/chenyifeng/TF_Models/atrain/SEGS/deeplabv3+/sgpu',
                           'where to store current model')

flags.DEFINE_integer('save_per_step', 1000, 'save model per xxx steps')

FLAGS = flags.FLAGS

if (FLAGS.reshape_height is None or FLAGS.reshape_weight is None) and FLAGS.batch_size != 1:
    assert 0, 'Can''t Stack Images Of Different Shapes, Please Speicify Reshape Size!'

store_device = parse_device_name(FLAGS.store_device)
run_device = parse_device_name(FLAGS.run_device)
weight_reg = regularizer(mode=FLAGS.weight_reg_func, scale=FLAGS.weight_reg_scale)
bias_reg = regularizer(mode=FLAGS.bias_reg_func, scale=FLAGS.bias_reg_scale)

# config devices
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
if FLAGS.run_device in '01234567':
    print('Deploying Model on {} GPU Card'.format(''.join(FLAGS.run_device)))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
else:
    print('Deploying Model on CPU')

# set up step
sess = tf.Session(config=config)

with tf.device(store_device):
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    # read data
    reshape_size = [FLAGS.reshape_height, FLAGS.reshape_weight]
    name_batch, image_batch, label_batch = get_next_batch(get_dataset(
        dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.epoch_num,
        reshape_size=reshape_size,
        normalize=False)
    )

# inference
with arg_scope([get_variable], device=store_device):
    with tf.device('/GPU:0'):
        outputs_to_scales_to_logits, mean_loss = deeplab_v3_plus._build_deeplab(image_batch,
                                                                           label_batch,
                                                                           ignore_labels=[255],
                                                                           FLAGS=FLAGS,
                                                                           is_training=True)

        score_map = outputs_to_scales_to_logits['semantic']['merged_logits']

        class_map = arg_max(score_map, axis=3, name='class_map')
        class_map = tf.image.resize_nearest_neighbor(class_map,
                                                     tf.shape(label_batch)[1:3],
                                                     align_corners=True)
        pixel_acc = mAP(class_map, label_batch)
        mean_IOU, IOUs = mIOU(class_map, label_batch, ignore_label=[0], num_classes=FLAGS.num_classes)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_losses)
        total_loss = mean_loss + reg_loss
        #
        # set up optimizer
        decay_learning_rate = tf.train.exponential_decay(FLAGS.weight_learning_rate, global_step,
                                                         decay_steps=10000, decay_rate=0.96, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate, momentum=FLAGS.momentum)
        ratio = (FLAGS.weight_learning_rate / FLAGS.bias_learning_rate) \
            if FLAGS.bias_learning_rate is not None else 2

        # solve for gradients
        weight_vars = tf.get_collection(weight_collections)
        bias_vars = tf.get_collection(bias_collections)

        weight_grads = optimizer.compute_gradients(total_loss, weight_vars)
        weight_grads = [(tf.clip_by_norm(grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                        for grad, var in weight_grads if grad is not None]

        bias_grads = optimizer.compute_gradients(total_loss, bias_vars)
        bias_grads = [(tf.clip_by_norm(ratio * grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                      for grad, var in bias_grads if grad is not None]

        # # for batch norm update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(weight_grads + bias_grads, global_step=global_step)

# add summaries
with tf.name_scope('summary_input_output'):
    tf.summary.image('image_batch', image_batch, max_outputs=1)
    tf.summary.image('label_batch', tf.cast(paint(label_batch), tf.uint8), max_outputs=1)
    tf.summary.image('predictions', tf.cast(paint(class_map), tf.uint8), max_outputs=1)
    tf.summary.image('contrast', tf.cast(compare(class_map, label_batch), tf.uint8), max_outputs=1)
    tf.summary.scalar('pixel_acc', pixel_acc)
    tf.summary.scalar('mean_loss', mean_loss)
    tf.summary.scalar('mean_iou', mean_IOU)
    tf.summary.scalar('reg_loss', reg_loss)
    # tf.summary.scalar('learning_rate', decay_learning_rate)

with tf.name_scope('summary_ious'):
    add_iou_summary(IOUs, pascal_voc_classes)

with tf.name_scope('summary_vars'):
    for weight in weight_vars:
        add_var_summary(weight)
    for bias in bias_vars:
        add_var_summary(bias)

with tf.name_scope('summary_grads'):
    for grad, var in weight_grads:
        add_gradient_summary(grad, var)
    for grad, var in bias_grads:
        add_gradient_summary(grad, var)

# with tf.name_scope('summary_activations'):
#     for activations in endpoints.keys():
#         add_activation_summary(endpoints[activations])

merge_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=3)

# initialize
ckpt = None
if FLAGS.last_ckpt is not None:
    ckpt = tf.train.latest_checkpoint(FLAGS.last_ckpt)
    if ckpt is not None:
        # set up save configuration
        saver.restore(sess, ckpt)
        print('Recovering From {}'.format(ckpt))
else:
    print('No previous Model Found in {}'.format(ckpt))
    if FLAGS.pretrained_ckpts is not None:
        # pre-train priority higher
        partial_restore_op = partial_restore(sess, tf.global_variables(), FLAGS.pretrained_ckpts)
        sess.run(partial_restore_op)
        print('Recovering From Pretrained Model {}'.format(FLAGS.pretrained_ckpts))

# for i in tf.global_variables():
#     print(i)
try:
    # start training
    local_step = 0
    sess.run(tf.local_variables_initializer())
    while True:  # train until OutOfRangeError
        batch_mIOU, batch_mAP, batch_tloss, batch_rloss, step, summary, _ = \
            sess.run([mean_IOU, pixel_acc, mean_loss, reg_loss, global_step, merge_summary, train_op])

        train_writer.add_summary(summary, step)
        local_step += 1

        # # save model per xxx steps
        # if local_step % FLAGS.save_per_step == 0 and local_step > 0:
        #     save_path = saver.save(sess, os.path.join(FLAGS.next_ckpt,
        #                                               '{:.3f}_{}'.format(batch_mAP, step)))
        #     print("Model saved in path: %s" % save_path)

        print("Step {} : mAP {:.3f}%  mIOU {:.3f}% loss {:.3f} reg {:.3f}"
              .format(step, batch_mAP * 100, batch_mIOU * 100, batch_tloss, batch_rloss))

except tf.errors.OutOfRangeError:
    print('Done training')
