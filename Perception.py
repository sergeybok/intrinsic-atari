
import tensorflow as tf 
from tensorflow.contrib import slim
import numpy as np 
from tensorflow.contrib.layers import xavier_initializer

conv4_shape_1 = None
conv4_shape_2 = None
conv4_shape_3 = None

conv3_shape_1 = None
conv3_shape_2 = None
conv3_shape_3 = None

conv2_shape_1 = None
conv2_shape_2 = None
conv2_shape_3 = None

conv1_shape_1 = None
conv1_shape_2 = None
conv1_shape_3 = None

seed = 1234

def CNN(input, height, width,in_channel, out_channel,afn=tf.nn.elu,weights=[]):

	img = tf.reshape(input, shape=[-1, height, width, in_channel])
	# filter [filter ht, filter wd, inchannels, outchannels]
	if len(weights) == 0:
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+1)
		conv1_weights = tf.Variable(xavier_init([8,8,in_channel,64]), name='CNN_1_Weights')
		weights.append(conv1_weights)
		conv1_bias = tf.Variable(tf.constant(0.01, shape=[64]), name='CNN_1_Bias')
		weights.append(conv1_bias)
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+2)
		conv2_weights = tf.Variable(xavier_init([6,6,64,64]), name='CNN_2_Weights')
		weights.append(conv2_weights)
		conv2_bias = tf.Variable(tf.constant(0.01, shape=[64]), name='CNN_2_Bias')
		weights.append(conv2_bias)
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+3)
		conv3_weights = tf.Variable(xavier_init([5,5,64,128]), name='CNN_3_Weights')
		weights.append(conv3_weights)
		conv3_bias = tf.Variable(tf.constant(0.01, shape=[128]), name='CNN_3_Bias')
		weights.append(conv3_bias)
		xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+4)
		conv4_weights = tf.Variable(xavier_init([3,3,128,out_channel]), name='CNN_4_Weights')
		weights.append(conv4_weights)
		conv4_bias = tf.Variable(tf.constant(0.01, shape=[out_channel]), name='CNN_4_Bias')
		weights.append(conv4_bias)

	conv1 = afn(tf.nn.conv2d(input=img,filter=weights[0],strides=[1,5,4,1],padding='VALID',name='cnn1')+weights[1])
	conv2 = afn(tf.nn.conv2d(input=conv1,filter=weights[2],strides=[1,3,3,1],padding='VALID',name='cnn2')+weights[3])
	conv3 = afn(tf.nn.conv2d(input=conv2,filter=weights[4],strides=[1,2,2,1],padding='VALID',name='cnn3')+weights[5])
	conv4 = afn(tf.nn.conv2d(input=conv3,filter=weights[6],strides=[1,1,1,1],padding='VALID',name='cnn4')+weights[7])


	#batch, in_height, in_width, in_channels
	global conv4_shape_1 
	global conv4_shape_2 
	global conv4_shape_3  
	conv4_shape_1 = conv4.shape[1].value
	conv4_shape_2 = conv4.shape[2].value
	conv4_shape_3 = conv4.shape[3].value
	

	global conv3_shape_1 
	global conv3_shape_2 
	global conv3_shape_3  
	conv3_shape_1 = conv3.shape[1].value
	conv3_shape_2 = conv3.shape[2].value
	conv3_shape_3 = conv3.shape[3].value
	
	global conv2_shape_1 
	global conv2_shape_2 
	global conv2_shape_3 
	conv2_shape_1 = conv2.shape[1].value
	conv2_shape_2 = conv2.shape[2].value
	conv2_shape_3 = conv2.shape[3].value
	
	global conv1_shape_1 
	global conv1_shape_2 
	global conv1_shape_3 
	conv1_shape_1 = conv1.shape[1].value
	conv1_shape_2 = conv1.shape[2].value
	conv1_shape_3 = conv1.shape[3].value


	#conv4 = slim.conv2d(inputs=conv3,num_outputs=out_channel,activation_fn=tf.nn.sigmoid,
	#					kernel_size=[7,7],stride=[1,1],
	#					padding='VALID',scope=("CNN_4"))

	return tf.contrib.layers.flatten(conv4), weights
	#return conv3


def Deconv(input, height, width,network_name,in_channel=None,out_channel = None,afn=tf.nn.elu):

	#print('1 {0} \n 2 {1} \n 3 {2}'.format(conv4_shape_1,conv4_shape_2,conv4_shape_3))

	img = tf.reshape(input,shape=[-1,conv4_shape_1,conv4_shape_2,conv4_shape_3]) 
	# This depends on the cnn output before flatten
	batch_size = tf.shape(img)[0]
	scope = network_name + '_Deconv'

	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+11)
	dconv1_weights = tf.Variable(xavier_init([3,3,128,conv4_shape_3]), name='DCNN_1_Weights')
	dconv1_bias = tf.Variable(tf.constant(0.01, shape=[128]), name='DCNN_1_Bias')
	dconv1 = afn(tf.nn.conv2d_transpose(img,filter=dconv1_weights,output_shape=[batch_size,conv3_shape_1,conv3_shape_2,conv3_shape_3],
									strides=[1,1,1,1],padding='VALID',name='dcnn1')+dconv1_bias)
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+12)
	dconv2_weights = tf.Variable(xavier_init([5,5,64,128]), name='DCNN_2_Weights')
	dconv2_bias = tf.Variable(tf.constant(0.01, shape=[64]), name='DCNN_2_Bias')
	dconv2 = afn(tf.nn.conv2d_transpose(dconv1,filter=dconv2_weights,output_shape=[batch_size,conv2_shape_1,conv2_shape_2,conv2_shape_3],
									strides=[1,2,2,1],padding='VALID',name='dcnn2')+dconv2_bias)

	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+13)
	dconv3_weights = tf.Variable(xavier_init([6,5,64,64]), name='DCNN_3_Weights')
	dconv3_bias = tf.Variable(tf.constant(0.01, shape=[64]), name='DCNN_3_Bias')
	dconv3 = afn(tf.nn.conv2d_transpose(dconv2,filter=dconv3_weights,output_shape=[batch_size,conv1_shape_1,conv1_shape_2,conv1_shape_3],
									strides=[1,3,3,1],padding='VALID',name='dcnn3')+dconv3_bias)
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+14)
	dconv4_weights = tf.Variable(xavier_init([8,8,out_channel,64]), name='DCNN_4_Weights')
	dconv4_bias = tf.Variable(tf.constant(0.01, shape=[out_channel]), name='DCNN_4_Bias')
	dconv4 = tf.nn.sigmoid(tf.nn.conv2d_transpose(dconv3,filter=dconv4_weights,output_shape=[batch_size,height,width,out_channel],
									strides=[1,5,4,1],padding='VALID',name='dcnn4')+dconv4_bias)

	return dconv4, [dconv1_weights,dconv1_bias,dconv2_weights,dconv2_bias,dconv3_weights,dconv3_bias]


def Predictor(state, action,action_space, height, width, out_channel, network_name, state_size=None):
	scope = network_name+'_Predictor'
	#state_size = tf.shape(state)[1]
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+21)
	in_state_weights = tf.Variable(xavier_init([state_size,state_size]), name=(scope+'_state_W'))
	in_state_bias = tf.Variable(tf.constant(0.01, shape=[state_size]), name=(scope+'_state_b'))
	
	xavier_init = tf.contrib.layers.xavier_initializer(seed=seed+22)
	in_action_weights = tf.Variable(xavier_init([action_space,state_size]), name=(scope+'_action_W'))
	in_action_bias = tf.Variable(tf.constant(0.01, shape=[state_size]), name=(scope+'_action_b'))
	
	deconv_input_state = tf.matmul(state,in_state_weights) + in_state_bias
	deconv_input_action = tf.matmul(action,in_action_weights) + in_action_bias

	pred_w = [in_state_weights,in_state_bias,in_action_weights,in_action_bias]

	deconv_input = deconv_input_state * deconv_input_action
	deconv_output, dconv_w = Deconv(deconv_input,height,width,network_name, out_channel=out_channel)
	pred_w += dconv_w

	return deconv_output, pred_w




"""

Start with 
[-1, 210, 160, 4]

conv with stride [5,4] filter [8,8] valid

[-1, 41, 39, x]

conv with stride [3,3] filter [6,6] valid

[-1, 12, 12, x]

conv with stride [2,2] filter [5,5] valid

[-1, 4, 4, x] 

conv with stride [1,1] filter [3,3] valid 

[-1, 2, 2, x]



"""







