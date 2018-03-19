import os

from datasets.pascal_voc_reader import pascal_inputs, TRAIN_DIR
from segs.factory import get_net
from tf_ops.benchmarks import mAP
from tf_ops.visualize import paint
from tf_ops.wrap_ops import *
from tf_utils import partial_restore, add_gradient_summary, add_var_summary, add_activation_summary

arg_scope = tf.contrib.framework.arg_scope

LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

# pre settings
tf.app.flags.DEFINE_string('data_dir', TRAIN_DIR, 'where training set is put')
tf.app.flags.DEFINE_integer('reshape_height', 224, 'reshape height')
tf.app.flags.DEFINE_integer('reshape_weight', 224, 'reshape weight')
tf.app.flags.DEFINE_string('net_name', 'fcn8', 'which segmentation net to use')
tf.app.flags.DEFINE_integer('num_classes', 21, '#classes')

# learning configs
tf.app.flags.DEFINE_integer('epoch_num', 10, 'epoch_nums')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_float('weight_learning_rate', 1e-3, 'weight learning rate')
tf.app.flags.DEFINE_float('bias_learning_rate', None, 'bias learning rate')
tf.app.flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')
tf.app.flags.DEFINE_float('learning_decay', 0.99, 'learning rate decay')
tf.app.flags.DEFINE_float('momentum', 0.99, 'momentum')

# deploy configs
tf.app.flags.DEFINE_string('store_device', 'cpu', 'where to place the variables')
tf.app.flags.DEFINE_string('run_device', '0', 'where to run the models')
tf.app.flags.DEFINE_float('gpu_fraction', 0.8, 'gpu memory fraction')
tf.app.flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
tf.app.flags.DEFINE_float('weight_reg_scale', 1e-5, 'weight regularization scale')
tf.app.flags.DEFINE_string('weight_reg_func', 'l2', 'use which func to regularize weight')
tf.app.flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
tf.app.flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# model load & save configs
tf.app.flags.DEFINE_string('summaries_dir', '/home/chenyifeng/TF_Logs/SEGS',
                           'where to store summary log')

tf.app.flags.DEFINE_string('pretrained_ckpts', '/home/chenyifeng/TF_Models/ptrain/vgg_16.ckpt',
                           'where to load pretrained model')

tf.app.flags.DEFINE_string('last_ckpt', '/home/chenyifeng/TF_Models/atrain/SEGS/fcn',
                           'where to load last saved model')

tf.app.flags.DEFINE_string('next_ckpt', '/home/chenyifeng/TF_Models/atrain/SEGS/fcn',
                           'where to store current model')

tf.app.flags.DEFINE_integer('save_per_step', 1000,
                            'save model per xxx steps')

FLAGS = tf.app.flags.FLAGS

if (FLAGS.reshape_height is None or FLAGS.reshape_weight is None) and FLAGS.batch_size != 1:
    assert 0, 'Can''t Stack Images Of Different Shapes, Please Speicify Reshape Size!'

if FLAGS.store_device.lower() == 'cpu':
    device = 'cpu'
elif FLAGS.store_device in ['0', '1', '2', '3']:
    device = FLAGS.device
else:
    device = None

config = tf.ConfigProto(log_device_placement=False)
if set(FLAGS.run_device).issubset({'0', '1', '2', '3'}):
    print('Deploying Model on {}'.format(''.join(FLAGS.run_device)))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

# set up step
sess = tf.Session(config=config)
global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

# read data
reshape_size = [FLAGS.reshape_height, FLAGS.reshape_weight]
name_batch, image_batch, label_batch = pascal_inputs(
    dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch_num, reshape_size=reshape_size)

weight_reg = regularizer(mode=FLAGS.weight_reg_func, scale=FLAGS.weight_reg_scale)
bias_reg = regularizer(mode=FLAGS.bias_reg_func, scale=FLAGS.bias_reg_scale)

# inference
net = get_net(FLAGS.net_name)
score_map, endpoints = net(image_batch, num_classes=FLAGS.num_classes,
                           weight_init=None, weight_reg=weight_reg,
                           bias_init=tf.zeros_initializer, bias_reg=bias_reg, device=device)

# solve for mAP and loss
class_map = arg_max(score_map, axis=3, name='class_map')
pixel_acc = mAP(class_map, label_batch)

# calculate loss
loss = softmax_with_logits(score_map, label_batch)
mean_loss = tf.reduce_mean(loss, name='mean_loss')
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = tf.reduce_sum(reg_losses)
total_loss = mean_loss + reg_loss

# set up optimizer
decay_learning_rate = tf.train.exponential_decay(FLAGS.weight_learning_rate, global_step,
                                                 decay_steps=100000, decay_rate=0.96, staircase=True)

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

# add summaries
with tf.name_scope('summary_input_output'):
    tf.summary.image('image_batch', image_batch, max_outputs=1)
    tf.summary.image('label_batch', tf.cast(paint(label_batch), tf.uint8), max_outputs=1)
    tf.summary.image('predictions', tf.cast(paint(class_map), tf.uint8), max_outputs=1)
    add_var_summary(loss)
    tf.summary.scalar('pixel_acc', pixel_acc)
    tf.summary.scalar('mean_loss', mean_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('learning_rate', decay_learning_rate)

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

with tf.name_scope('summary_activations'):
    for activations in endpoints.keys():
        add_activation_summary(endpoints[activations])


merge_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)

# set up train operation
train_op = optimizer.apply_gradients(weight_grads + bias_grads, global_step=global_step)

saver = tf.train.Saver(max_to_keep=3)

try:
    sess.run(tf.global_variables_initializer())

    ckpt = None
    if FLAGS.pretrained_ckpts is not None:
        # pre-train priority higher
        partial_restore_op = partial_restore(sess, tf.global_variables(), FLAGS.pretrained_ckpts)
        sess.run(partial_restore_op)
        print('Recovering From Pretrained Model {}'.format(FLAGS.pretrained_ckpts))

    elif FLAGS.last_ckpt is not None:
        ckpt = tf.train.latest_checkpoint(FLAGS.last_ckpt)
        if ckpt is not None:
            # set up save configuration
            saver.restore(sess, ckpt)
            print('Recovering From {}'.format(ckpt))
        else:
            print('No previous Model Found in {}'.format(ckpt))

    # start training
    local_step = 0
    while True:  # train until OutOfRangeError
        batch_mAP, batch_tloss, batch_rloss, step, summary, _ = \
            sess.run([pixel_acc, mean_loss, reg_loss, global_step, merge_summary, train_op])
        train_writer.add_summary(summary, step)
        local_step += 1

        # save model per xxx steps
        if local_step % FLAGS.save_per_step == 0 and local_step > 0:
            save_path = saver.save(sess, os.path.join(FLAGS.next_ckpt,
                                                      '{:.3f} _ {}'.format(batch_mAP, step)))
            print("Model saved in path: %s" % save_path)

        print("Step {} : mAP {:.3f}%  loss {:.3f} reg {:.3f}"
              .format(step, batch_mAP * 100, batch_tloss, batch_rloss))

except tf.errors.OutOfRangeError:
    print('Done training')
