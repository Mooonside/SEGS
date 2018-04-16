import os
import tensorflow as tf
from numpy import arange
from tf_ops.wrap_ops import weight_collections, bias_collections
from tensorflow.python.ops import control_flow_ops



class DeployConfig(object):
    def __init__(self, FLAGS):
        self.num_replicas = FLAGS.num_replicas
        self.optimizer_device = '/CPU:0'
        self.variables_device = '/CPU:0'
        self.inputs_device = '/CPU:0'
        self.train_device = ['/GPU:{}'.format(i) for i in range(num_replicas)]
        self.name_scope = ['/Replicas:{}'.format(i) for i in range(num_replicas)]

        print('Deploying Model on {} GPUS'.format(num_replicas))
        os.environ['CUDA_VISIBLE_DEVICES'] = '012345678'[:num_replicas]

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = FLAGS.allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        self.sess_config = config


class Replica(object):
    def __init__(self, output, name_scope, device):
        self.output = output
        self.name_scope = name_scope
        self.device = device


config = DeployConfig


def infer_multi_gpu(func, config, *args, **kwargs):
    replicas = []
    for gpu_id in range(config.num_replicas):
        with tf.name_scope(config.name_scope[gpu_id]) as ns:
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=False if gpu_id ==0 else True):
                with tf.device(config.train_device[gpu_id]) as ds:
                    # mean while add to loss in its name_scope
                    loss, output = func(args, kwargs)
                    replicas.append(Replica(output, ns, ds))
    return replicas


def train_multi_gpu(func, config, optimizer, global_step, FLAGS, *args, **kwargs):
    replicas = infer_multi_gpu(func, config, *args, **kwargs)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


    weight_grads = []
    bias_grads = []
    total_loss = []

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, replicas[0].name_scope)

    for gpu_id in range(config.num_replicas):
        name_scope = replicas[gpu_id].name_scope
        device = replicas[gpu_id].device

        with name_scope:
            with device:
                loss = tf.get_collection(tf.GraphKeys.LOSSES, scope=replicas[gpu_id].name_scope)
                loss = tf.add_n([loss, reg_loss])

                # solve for gradients
                weight_vars = tf.get_collection(weight_collections)
                bias_vars = tf.get_collection(bias_collections)

                replica_weight_grads = optimizer.compute_gradients(loss, weight_vars)
                replica_weight_grads = [(tf.clip_by_norm(grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                                        for grad, var in replica_weight_grads if grad is not None]

                replica_bias_grads = optimizer.compute_gradients(loss, bias_vars)
                replica_bias_grads = [(tf.clip_by_norm(FLAGS.ratio * grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                                      for grad, var in replica_bias_grads if grad is not None]
                print(gpu_id, len(replica_weight_grads), len(replica_bias_grads))

                weight_grads += replica_weight_grads
                replica_bias_grads += replica_bias_grads
                total_loss.append(loss)

    weight_grads = sorted(weight_grads, key=lambda x : x[1].name)
    bias_grads = sorted(bias_grads, key=lambda x : x[1].name)

    assert len(weight_grads) % config.num_replicas == 0 and \
        len(bias_grads) % config.num_replicas == 0

    with tf.device(config.optimizer_device):
        grad_vars = []
        for i in range(0, len(weight_grads), config.num_replicas):
            partial = weight_grads[i:i+config.num_replicas]
            variable = set([ v for _, v in partial])
            assert len(variable) == 1
            variable = variable.pop()

            grads = [ g for g, _ in partial]
            grad = tf.reduce_mean(grads, axis=0)

            grad_vars.append((grad, variable))

        update_grad = optimizer.apply_gradients(grad_vars, global_step=global_step)
        update_ops.append(update_grad)
        update_op = tf.group(*update_ops)

        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

    return train_op