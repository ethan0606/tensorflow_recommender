import tensorflow as tf


class Feature:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def transform(self, features):
        return features[self.name]

    def make_placeholder(self):
        return {
            self.name: tf.compat.v1.placeholder(name=self.name, dtype=self.dtype, shape=self.shape)
        }

    def make_io_spec(self):
        return {
            self.name: tf.io.FixedLenFeature(dtype=self.dtype, shape=self.shape)
        }


class NumericFeature(Feature):
    def __init__(self, name, shape, dtype, norm=None):
        self.norm = norm
        super(NumericFeature, self).__init__(name, shape, dtype)

    def transform(self, features):
        if self.norm == 'log1p':
            return tf.math.log1p(features[self.name])
        return features[self.name]


class CategoryFeature(Feature):
    def __init__(self, name, shape, dtype, embedding_name, num_classes, dense_dim, combiner=None):
        self.embedding_name = embedding_name
        self.num_classes = num_classes
        self.dense_dim = dense_dim
        self.combiner = combiner
        super(CategoryFeature, self).__init__(name, shape, dtype)

    def transform(self, features):
        with tf.compat.v1.variable_scope('category_features', reuse=tf.compat.v1.AUTO_REUSE):
            embedding_table = tf.compat.v1.get_variable(name=self.embedding_name,
                                                        shape=[self.num_classes, self.dense_dim],
                                                        dtype=tf.float32,
                                                        initializer=tf.initializers.random_uniform)
            look_up = tf.nn.embedding_lookup(params=embedding_table, ids=features[self.name])
            if self.combiner == 'sum':
                return tf.reduce_sum(look_up, axis=1)
            if self.combiner == 'mean':
                return tf.reduce_mean(look_up, axis=1)
            return look_up


class SparseCategoryFeature(CategoryFeature):
    def __init__(self, name, shape, dtype, embedding_name, num_classes, dense_dim, combiner=None, weights=None):
        self.weights = weights
        super(SparseCategoryFeature, self).__init__(name, shape, dtype, embedding_name, num_classes, dense_dim,
                                                    combiner)

    def transform(self, features):
        with tf.compat.v1.variable_scope('category_features', reuse=tf.compat.v1.AUTO_REUSE):
            embedding_table = tf.compat.v1.get_variable(name=self.embedding_name,
                                                        shape=[self.num_classes, self.dense_dim],
                                                        dtype=tf.float32,
                                                        initializer=tf.initializers.random_uniform)
            look_up = tf.nn.embedding_lookup_sparse(params=embedding_table,
                                                    sp_ids=features[self.name],
                                                    sp_weight=None if self.weights is None else features[self.weights],
                                                    combiner=self.combiner)
            return look_up

    def make_io_spec(self):
        io_spec = dict()
        io_spec[self.name] = tf.io.VarLenFeature(dtype=self.dtype)
        if self.weights is not None:
            io_spec[self.weights] = tf.io.VarLenFeature(dtype=self.dtype)

    def make_placeholder(self):
        placeholders = dict()
        placeholders[self.name] = tf.compat.v1.sparse_placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
        if self.weights is not None:
            placeholders[self.weights] = tf.compat.v1.sparse_placeholder(dtype=self.dtype,
                                                                         shape=self.shape,
                                                                         name=self.weights)
        return placeholders
