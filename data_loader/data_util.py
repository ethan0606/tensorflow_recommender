import os
import tensorflow as tf


def parse_tf_example(example_proto, feature_map=None, label_name='label'):
    features = tf.io.parse_example(serialized=example_proto, features=feature_map)
    labels = features.pop(label_name)
    return features, labels


class FileExtractor:
    @staticmethod
    def get_filenames(directory):
        res = []
        if directory.startswith('hdfs://'):
            filenames = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + directory)]
        else:
            filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
        for f in filenames:
            if 'part' in f and not f.endswith("_COPYING_") and not f.endswith("SUCCESS") and not f.endswith(".crc"):
                res.append(f)
        return res


class DataSetIterator:

    @staticmethod
    def get_label_features(directory,
                           feature_map,
                           label_name,
                           batch_size=10,
                           num_parallel_calls=2,
                           num_epochs=1):
        filenames = FileExtractor.get_filenames(directory)
        ds = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=num_parallel_calls)
        ds = ds.shuffle(buffer_size=2 * batch_size + 1)
        ds = ds.batch(batch_size)
        ds = ds.repeat(num_epochs)
        ds = ds.map(lambda x: parse_tf_example(x, feature_map, label_name), num_parallel_calls=num_parallel_calls)
        features, label = ds.make_one_shot_iterator().get_next()
        return features, label


if __name__ == '__main__':
    filenames = FileExtractor.get_filenames("/Users/yifanguo/Desktop/dataset/taobao_ad/tfrecord/train")
    print(filenames)
    print(len(filenames))

# https://blog.csdn.net/u014061630/article/details/80776975
