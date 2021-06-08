import tensorflow as tf

def cross(vector1, vector2, axis=-1, name=None):
    vector1_x, vector1_y, vector1_z = tf.unstack(vector1, axis=axis)
    vector2_x, vector2_y, vector2_z = tf.unstack(vector2, axis=axis)
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return tf.stack((n_x, n_y, n_z), axis=axis)
