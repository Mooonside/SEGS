import os

from datasets.pascal_voc_reader import get_dataset, get_next_batch, TRAIN_DIR
from datasets.pascal_voc_utils import pascal_voc_classes
from segs.factory import get_net
from tf_ops.benchmarks import mAP, mIOU
from tf_ops.visualize import paint, compare
from tf_ops.wrap_ops import *
from tf_utils import partial_restore, parse_device_name, average_gradients, \
    add_activation_summary, add_gradient_summary, add_var_summary, add_iou_summary

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
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_float('weight_learning_rate', 1e-3, 'weight learning rate')
tf.app.flags.DEFINE_float('bias_learning_rate', None, 'bias learning rate')
tf.app.flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')
tf.app.flags.DEFINE_float('learning_decay', 0.99, 'learning rate decay')
tf.app.flags.DEFINE_float('momentum', 0.99, 'momentum')

# deploy configs
tf.app.flags.DEFINE_string('store_device', 'cpu', 'where to place the variables')
tf.app.flags.DEFINE_string('run_device', '01', 'where to run the models')
tf.app.flags.DEFINE_float('gpu_fraction', 0.8, 'gpu memory fraction')
tf.app.flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
tf.app.flags.DEFINE_float('weight_reg_scale', 1e-5, 'weight regularization scale')
tf.app.flags.DEFINE_string('weight_reg_func', 'l2', 'use which func to regularize weight')
tf.app.flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
tf.app.flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# model load & save configs
tf.app.flags.DEFINE_string('summaries_dir', '/home/chenyifeng/TF_Logs/SEGS/fcn/mgpu',
                           'where to store summary log')

tf.app.flags.DEFINE_string('pretrained_ckpts', '/home/chenyifeng/TF_Models/ptrain/vgg_16.ckpt',
                           'where to load pretrained model')

tf.app.flags.DEFINE_string('last_ckpt', '/home/chenyifeng/TF_Models/atrain/SEGS/fcn/mgpu',
                           'where to load last saved model')

tf.app.flags.DEFINE_string('next_ckpt', '/home/chenyifeng/TF_Models/atrain/SEGS/fcn/mgpu',
                           'where to store current model')

tf.app.flags.DEFINE_integer('save_per_step', 1000,
                            'save model per xxx steps')

FLAGS = tf.app.flags.FLAGS

if (FLAGS.reshape_height is None or FLAGS.reshape_weight is None) and FLAGS.batch_size != 1:
    assert 0, 'Can''t Stack Images Of Different Shapes, Please Speicify Reshape Size!'

store_device = parse_device_name(FLAGS.store_device)
run_device = parse_device_name(FLAGS.run_device)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
if FLAGS.run_device in '01234567':
    print('Deploying Model on {} GPU Card'.format(''.join(FLAGS.run_device)))
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
else:
    print('Deploying Model on CPU')

weight_reg = regularizer(mode=FLAGS.weight_reg_func, scale=FLAGS.weight_reg_scale)
bias_reg = regularizer(mode=FLAGS.bias_reg_func, scale=FLAGS.bias_reg_scale)
net = get_net(FLAGS.net_name)
num_replicas = len(FLAGS.run_device)
ratio = (FLAGS.weight_learning_rate / FLAGS.bias_learning_rate) \
    if FLAGS.bias_learning_rate is not None else 2
FLAGS.batch_size = FLAGS.batch_size
# set up step
sess = tf.Session(config=config)

with arg_scope([get_variable], device='/CPU:0'):
    with tf.device('/CPU:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        reshape_size = [FLAGS.reshape_height, FLAGS.reshape_weight]
        dataset = get_dataset(
            dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch_num, reshape_size=reshape_size)

        decay_learning_rate = tf.train.exponential_decay(FLAGS.weight_learning_rate, global_step,
                                                         decay_steps=10000, decay_rate=0.96, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate, momentum=FLAGS.momentum)


        def inference_to_loss():
            # with tf.device(FLAGS.store_device):
            name_batch, image_batch, label_batch = get_next_batch(dataset)

            score_map, endpoints = net(image_batch, num_classes=FLAGS.num_classes,
                                       weight_init=None, weight_reg=weight_reg,
                                       bias_init=tf.zeros_initializer, bias_reg=bias_reg)
            class_map = arg_max(score_map, axis=3, name='class_map')
            pixel_acc = mAP(class_map, label_batch)
            mean_loss = softmax_with_logits(score_map, label_batch)
            mean_IOU, IOUs = mIOU(class_map, label_batch, ignore_label=[0], num_classes=FLAGS.num_classes)
            with tf.name_scope('summary_input_output'):
                tf.summary.image('tower_image_batch', image_batch, max_outputs=1)
                tf.summary.image('tower_label_batch', tf.cast(paint(label_batch), tf.uint8), max_outputs=1)
                tf.summary.image('tower_predictions', tf.cast(paint(class_map), tf.uint8), max_outputs=1)
                tf.summary.image('tower_contrast', tf.cast(compare(class_map, label_batch), tf.uint8), max_outputs=1)
                tf.summary.scalar('tower_pixel_acc', pixel_acc)
                tf.summary.scalar('tower_mean_iou', mean_IOU)
                tf.summary.scalar('tower_mean_loss', mean_loss)
            with tf.name_scope('tower_ious'):
                add_iou_summary(IOUs, pascal_voc_classes)
            return mean_loss, mean_IOU, pixel_acc, endpoints


        gather_weight_grads = []
        gather_bias_grads = []
        gather_loss = []
        gather_acc = []
        gather_iou = []
        reg_loss = None

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for gpu_id in range(num_replicas):
                with tf.name_scope('tower_{}'.format(gpu_id)):
                    device_name = parse_device_name(str(gpu_id))
                    with tf.device(device_name):
                        # mean while add to loss in its name_scope
                        mean_loss, mean_IOU, pixel_acc, endpoints = inference_to_loss()
                        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_loss = mean_loss + reg_loss

                        with tf.name_scope('summary_activations'):
                            for activations in endpoints.keys():
                                add_activation_summary(endpoints[activations])

                        gather_acc.append(pixel_acc)
                        gather_loss.append(mean_loss)
                        gather_iou.append(mean_IOU)

                        weight_vars = tf.get_collection(weight_collections)
                        bias_vars = tf.get_collection(bias_collections)

                        weight_grads = optimizer.compute_gradients(total_loss, weight_vars)
                        clip_weight_grads = [(tf.clip_by_norm(grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                                             for grad, var in weight_grads if grad is not None]

                        bias_grads = optimizer.compute_gradients(total_loss, bias_vars)
                        clip_bias_grads = [(tf.clip_by_norm(ratio * grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                                           for grad, var in bias_grads if grad is not None]
                        gather_weight_grads.append(clip_weight_grads)
                        gather_bias_grads.append(clip_bias_grads)
        gather_weight_grads = average_gradients(gather_weight_grads)
        gather_bias_grads = average_gradients(gather_bias_grads)
        grads = gather_weight_grads + gather_bias_grads

        # for batch norm update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(update_ops)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        with tf.name_scope('summary_grads'):
            for grad, var in grads:
                add_gradient_summary(grad, var)

        overall_loss = tf.reduce_mean(gather_loss, name='overall_loss')
        overall_acc = tf.reduce_mean(gather_acc, name='overall_acc')
        overall_iou = tf.reduce_mean(gather_iou, name='overall_iou')

        with tf.name_scope('summary_overall'):
            tf.summary.scalar('learning_rate', decay_learning_rate)
            tf.summary.scalar('overall_loss', overall_loss)
            tf.summary.scalar('overall_acc', overall_acc)
            tf.summary.scalar('overall_iou', overall_iou)
            tf.summary.scalar('reg_loss', reg_loss)

with tf.name_scope('summary_var'):
    for var in tf.global_variables():
        add_var_summary(var)

sess.run(tf.global_variables_initializer())
merge_summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=3)
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)

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

try:
    # start training
    local_step = 0
    while True:  # train until OutOfRangeError
        batch_mIOU, batch_mAP, batch_tloss, batch_rloss, step, summary, _ = \
            sess.run([overall_iou, overall_acc, overall_loss, reg_loss, global_step, merge_summary, train_op])
        train_writer.add_summary(summary, step)
        local_step += 1
        # save model per xxx steps
        if local_step % FLAGS.save_per_step == 0 and local_step > 0:
            save_path = saver.save(sess, os.path.join(FLAGS.next_ckpt,
                                                      '{:.3f}_{}'.format(batch_mAP, step)))
            print("Model saved in path: %s" % save_path)

        print("Step {} : mAP {:.3f}%  mIOU {:.3f}% loss {:.3f} reg {:.3f}"
              .format(step, batch_mAP * 100, batch_mIOU * 100, batch_tloss, batch_rloss))

except tf.errors.OutOfRangeError:
    print('Done training')
