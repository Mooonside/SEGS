import os

from datasets.pascal_voc_reader import get_dataset, get_next_batch, TRAIN_DIR
from datasets.pascal_voc_utils import pascal_voc_classes
from segs.factory import get_net
from tf_ops.benchmarks import validation_metrics
from tf_ops.visualize import paint, compare
from tf_ops.wrap_ops import *
from tf_utils import partial_restore, \
    add_var_summary, add_activation_summary, parse_device_name

arg_scope = tf.contrib.framework.arg_scope

LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

# pre settings
tf.app.flags.DEFINE_string('data_dir', TRAIN_DIR, 'where training set is put')
tf.app.flags.DEFINE_integer('reshape_height', 224, 'reshape height')
tf.app.flags.DEFINE_integer('reshape_weight', 224, 'reshape weight')
tf.app.flags.DEFINE_string('net_name', 'fcn8', 'which segmentation net to use')
tf.app.flags.DEFINE_integer('num_classes', 21, '#classes')

# learning configs
<<<<<<< HEAD
tf.app.flags.DEFINE_integer('epoch_num', 64, 'epoch_nums')
=======
tf.app.flags.DEFINE_integer('epoch_num',32, 'epoch_nums')
>>>>>>> dd4b38b50f1d687618c6c80ede3ce5d0c4f48b4b
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')

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
tf.app.flags.DEFINE_string('summaries_dir', '/home/chenyifeng/TF_Logs/SEGS/fcn/validation',
                           'where to store summary log')

tf.app.flags.DEFINE_string('outputs_dir', '/home/chenyifeng/TF_Outs/SEGS/fcn/validation',
                           'where to store summary log')

tf.app.flags.DEFINE_string('pretrained_ckpts', '/home/chenyifeng/TF_Models/ptrain/vgg_16.ckpt',
                           'where to load pretrained model')

tf.app.flags.DEFINE_string('last_ckpt', '/home/chenyifeng/TF_Models/atrain/SEGS/fcn/sgpu',
                           'where to load last saved model')


FLAGS = tf.app.flags.FLAGS

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
    run_device = '/GPU:0'
else:
    print('Deploying Model on CPU')
    run_device = '/CPU:0'

# set up step
sess = tf.Session(config=config)

with tf.device(run_device):
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    # read data
    reshape_size = [FLAGS.reshape_height, FLAGS.reshape_weight]
    name_batch, image_batch, label_batch = get_next_batch(get_dataset(
        dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch_num, reshape_size=reshape_size)
    )

# inference
with arg_scope([get_variable], device=store_device):
    with tf.device('/GPU:0'):
        net = get_net(FLAGS.net_name)
        score_map, endpoints = net(image_batch, num_classes=FLAGS.num_classes,
                                   weight_init=None, weight_reg=weight_reg,
                                   bias_init=tf.zeros_initializer, bias_reg=bias_reg)

        # solve for mAP and loss
        class_map = arg_max(score_map, axis=3, name='class_map')
        # count in background

        validation_metrics_dict = validation_metrics(predictions=class_map,
                                                     labels=label_batch,
                                                     ignore_label=[],
                                                     num_classes=FLAGS.num_classes,
                                                     updates_collections=tf.GraphKeys.UPDATE_OPS)
        mean_acc = validation_metrics_dict['mAP']
        mean_iou, ious = validation_metrics_dict['mIOU']
        mean_prc, prcs = validation_metrics_dict['mPRC']
        mean_rec, recs = validation_metrics_dict['mREC']
        # calculate loss
        mean_loss = softmax_with_logits(score_map, label_batch)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_losses)
        total_loss = mean_loss + reg_loss

        # solve for gradients
        weight_vars = tf.get_collection(weight_collections)
        bias_vars = tf.get_collection(bias_collections)


# add summaries
with tf.name_scope('summary_io'):
    tf.summary.image('image_batch', image_batch, max_outputs=FLAGS.batch_size)
    tf.summary.image('label_batch', tf.cast(paint(label_batch), tf.uint8), max_outputs=FLAGS.batch_size)
    tf.summary.image('predictions', tf.cast(paint(class_map), tf.uint8), max_outputs=FLAGS.batch_size)
    tf.summary.image('contrast', tf.cast(compare(class_map, label_batch), tf.uint8), max_outputs=FLAGS.batch_size)
    tf.summary.scalar('mean_acc', mean_acc)
    tf.summary.scalar('mean_IOU', mean_iou)
    tf.summary.scalar('mean_prc', mean_prc)
    tf.summary.scalar('mean_rec', mean_rec)
    tf.summary.scalar('mean_loss', mean_loss)
    tf.summary.scalar('reg_loss', reg_loss)

with tf.name_scope('class_wise_metrics'):
    for i in range(FLAGS.num_classes):
        tf.summary.scalar('{}_iou'.format(pascal_voc_classes[i]), ious[i])
        tf.summary.scalar('{}_prc'.format(pascal_voc_classes[i]), prcs[i])
        tf.summary.scalar('{}_rec'.format(pascal_voc_classes[i]), recs[i])

with tf.name_scope('summary_vars'):
    for weight in weight_vars:
        add_var_summary(weight)
    for bias in bias_vars:
        add_var_summary(bias)


with tf.name_scope('summary_activations'):
    for activations in endpoints.keys():
        add_activation_summary(endpoints[activations])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


merge_summary = tf.summary.merge_all()


train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=3)
sess.run(tf.global_variables_initializer())

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
    sess.run(tf.local_variables_initializer())
    while True:  # train until OutOfRangeError
        name_batch_v, image_batch_v, label_batch_v, class_map_v, _ , \
        batch_mIOU, batch_mAP, batch_tloss, batch_rloss, summary = \
            sess.run([name_batch, image_batch, label_batch, class_map, update_ops,
                      mean_iou, mean_acc, mean_loss, reg_loss, merge_summary])

        if not os.path.exists(FLAGS.outputs_dir):
            os.makedirs(FLAGS.outputs_dir)

        for idx, name in enumerate(name_batch_v):
            np.save(os.path.join(FLAGS.outputs_dir, name.decode()+'_image.npy'), image_batch_v[idx,:])
            np.save(os.path.join(FLAGS.outputs_dir, name.decode()+'_label.npy'), label_batch_v[idx,:])
            np.save(os.path.join(FLAGS.outputs_dir, name.decode()+'_prediction.npy'), class_map_v[idx,:])

        train_writer.add_summary(summary, local_step)
        local_step += 1
        #
        print("Step {} : mAP {:.3f}%  mIOU {:.3f}% loss {:.3f} reg {:.3f}"
              .format(local_step, batch_mAP * 100, batch_mIOU * 100, batch_tloss, batch_rloss))

except tf.errors.OutOfRangeError:
    print('Done Validating')
