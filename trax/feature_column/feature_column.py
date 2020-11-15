import tensorflow as tf


class FeatureColumn:
    def __init__(self, features):
        self.features = features

    def concat(self, feature_columns):
        tensors = []
        for i in feature_columns:
            tensors.append(i.transform(self.features))
        return tf.concat(values=tensors, axis=-1)

    def get_label(self, label_columns):
        tensors = []
        for i in label_columns:
            tensors.append(i.transform(self.features))
        return tensors

    @staticmethod
    def make_parse_example_spec(feature_columns):
        feature_spec = dict()
        for i in feature_columns:
            feature_spec.update(i.make_io_spec())
        return feature_spec

    @staticmethod
    def make_placeholder_spec(feature_columns):
        placeholder_spec = dict()
        for i in feature_columns:
            placeholder_spec.update(i.make_placeholder())
            return placeholder_spec
