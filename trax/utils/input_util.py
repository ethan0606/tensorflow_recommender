import os

import tensorflow as tf


def extract_file_names(base_dir, user_name):
    os.environ['HADOOP_USER_NAME'] = user_name
    if base_dir.startswith('hdfs://'):
        file_names = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + base_dir)]
        return file_names
    else:
        file_names = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
        return [f for f in file_names if 'part' in f and not f.endswith('_COPYING_')
                and not f.endswith('SUCCESS') and not f.endswith('.crc')]


def parse_tf_example(example_proto, feature_spec):
    return tf.io.parse_single_example(serialized=example_proto, features=feature_spec)


def input_fn(file_path,
             feature_spec,
             batch_size=10,
             num_epochs=1,
             prefetch_factor=1,
             shuffle_factor=10,
             num_parallel_calls=4,
             user_name=''):
    file_names = extract_file_names(file_path, user_name)
    ds = tf.compat.v1.data.TFRecordDataset(file_names)
    ds = ds.map(lambda tf_example: parse_tf_example(tf_example, feature_spec),
                num_parallel_calls=num_parallel_calls)
    if shuffle_factor > 0:
        ds = ds.shuffle(buffer_size=batch_size * shuffle_factor)
    ds = ds.repeat(num_epochs).batch(batch_size)
    if prefetch_factor > 0:
        ds = ds.prefetch(buffer_size=batch_size * prefetch_factor)
    return ds.make_one_shot_iterator().get_next()


def get_sample():
    file_path = os.path.abspath('../data/agaricus/train')
    feature_map = {
        'features': tf.io.FixedLenFeature([126], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32)
    }
    features = input_fn(file_path, feature_map)
    return features
