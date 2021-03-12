# Borrowed from https://github.com/xuwenzhe/EECS442-Challenge-Surface-Normal-Estimation
import numpy as np
import tensorflow as tf


VARIABLE_COUNTER = 0


NUM_CH = [64,128,256,512,1024]
KER_SZ = 3

########## TABLE 1 ARCHITECTURE PARAMETERS #########
color_encode = "ABCDEFG"
block_in  = [128,128,128,128,256,256,256]
block_out = [64,128,128,256,256,256,128]
block_inter = [64,32,64,32,32,64,32]
block_conv1 = [1,1,1,1,1,1,1]
block_conv2 = [3,3,3,3,3,3,3]
block_conv3 = [7,5,7,5,5,7,5]
block_conv4 = [11,7,11,7,7,11,7]
####################### TABLE 1 END#################


def variable(name, shape, initializer,regularizer=None):
	global VARIABLE_COUNTER
	with tf.device('/gpu:0'):
		VARIABLE_COUNTER += np.prod(np.array(shape))
		return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=tf.float32, trainable=True)


def conv_layer(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,relu=True):
	input_channels = input_tensor.get_shape().as_list()[-1]
	with tf.variable_scope(name) as scope:
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))
		conv_layer = tf.nn.bias_add(conv, biases)
		if bn:
			conv_layer = batch_norm_layer(conv_layer,scope,training)
		if relu:
			conv_layer = tf.nn.relu(conv_layer, name=scope.name)
# 	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer


def max_pooling(input_tensor,name,factor=2):
	pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)
# 	print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),pool.get_shape().as_list()))
	return pool


def batch_norm_layer(input_tensor,scope,training):
	return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)


def hourglass_normal_prediction(netIN,training):
	print('-'*30)
	print('Hourglass Architecture')
	print('-'*30)
	global VARIABLE_COUNTER
	VARIABLE_COUNTER = 0;
	layer_name_dict = {}
	def layer_name(base_name):
		if base_name not in layer_name_dict:
			layer_name_dict[base_name] = 0
		layer_name_dict[base_name] += 1
		name = base_name + str(layer_name_dict[base_name])
		return name


	bn = True
	def hourglass_stack_no_incep(stack_in):
		c0 = conv_layer(stack_in,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c1 = conv_layer(c0,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c2 = conv_layer(c1,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)

		p0 = max_pooling(c2,layer_name('pool'))
		c3 = conv_layer(p0,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)

		p1 = max_pooling(c3,layer_name('pool'))
		c4 = conv_layer(p1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)

		p2 = max_pooling(c4,layer_name('pool'))
		c5 = conv_layer(p2,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)

		p3 = max_pooling(c5,layer_name('pool'))
		c6 = conv_layer(p3,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)

		c7 = conv_layer(c6,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)
		c8 = conv_layer(c7,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)

		r0 = tf.image.resize_images(c8,[c8.get_shape().as_list()[1]*2, c8.get_shape().as_list()[2]*2])
		cat0 = tf.concat([r0,c5],3)

		c9 = conv_layer(cat0,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)
		c10 = conv_layer(c9,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)

		r1 = tf.image.resize_images(c10,[c10.get_shape().as_list()[1]*2, c10.get_shape().as_list()[2]*2])
		cat1 = tf.concat([r1,c4],3)

		c11 = conv_layer(cat1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)
		c12 = conv_layer(c11,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)

		r2 = tf.image.resize_images(c12,[c12.get_shape().as_list()[1]*2, c12.get_shape().as_list()[2]*2])
		cat2 = tf.concat([r2,c3],3)

		c13 = conv_layer(cat2,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)
		c14 = conv_layer(c13,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)

		r3 = tf.image.resize_images(c14,[c14.get_shape().as_list()[1]*2, c14.get_shape().as_list()[2]*2])
		cat3 = tf.concat([r3,c2],3)

		c15 = conv_layer(cat3,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		c16 = conv_layer(c15,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)
		stack_out_d = conv_layer(c16, layer_name('conv'), 1, 3, bn=False, training=training, relu=False)
		return stack_out_d

	out0_n = hourglass_stack_no_incep(netIN)
	return out0_n
