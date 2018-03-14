import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def inspect_ckpt(ckpt_path):
    """
    inspect var names and var shapes in a checkpoint
    :param ckpt_path: checkpoint name (give prefix if multiple files for one ckpt)
    :return: None
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    var_names = sorted(var_to_shape_map.keys())
    for var_name in var_names:
        var = reader.get_tensor(var_name)
        shape = var_to_shape_map[var_name]
        # dtype = var_to_dtype_map[var_name]
        # print(var_name, shape, dtype)
        print(var_name, shape, var)


def rename_vars_in_ckpt(ckpt_path, name_map, output_path):
    """
    rename vars in ckpt according to a dict name_map
    :param ckpt_path: original ckpt path
    :param name_map: dict {org_name: new_name}
    :param output_path: renamed ckpt path
    :return: None
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    var_names = sorted(var_to_dtype_map.keys())

    sess = tf.Session()

    for var_name in var_names[-1:]:
        var = reader.get_tensor(var_name)
        dtype = var_to_dtype_map[var_name]

        newname = name_map[var_name]
        tf.get_variable(name=newname, dtype=dtype, initializer=var)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, save_path=output_path)
    print('Renamed Model Saved')


def partial_restore(sess, cur_var_lists, ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    var_names = set(var_to_dtype_map.keys())
    var_names = var_names.intersection(set(cur_var_lists))
    saver = tf.train.Saver(var_list=var_names)
    saver.restore(sess, ckpt_path)

