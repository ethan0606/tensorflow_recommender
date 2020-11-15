import tensorflow as tf

c = tf.constant([[1, 2, 0]])
e = tf.keras.layers.Embedding(10, 2, mask_zero=True)
t = e(c)
print(t)
r = tf.keras.layers.AveragePooling1D()(t)
print(r)
